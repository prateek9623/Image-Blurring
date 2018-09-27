#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "time.h";
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<cmath>
#include <iomanip>

const int RADIUS = 5;
const int FILTER_SIZE = RADIUS * 2 + 1;
const char* inputImageName = "input.jpg";
const char* outputImageName = "output.jpg";

using namespace std;
using namespace cv;

__constant__ double  gaussianFilter[FILTER_SIZE*FILTER_SIZE];
__constant__ double PI = 3.14159265358979323846;
__constant__ double SIGMA = 1.0;

__global__ void generateGaussianFilter(double *dGausFilter) {
	int indexX = threadIdx.x + blockDim.x*blockIdx.x;
	int indexY = threadIdx.y + blockDim.y*blockIdx.y;
	__shared__ double sum;
	double s = 2 * SIGMA*SIGMA;
	double r = sqrtf((indexX-RADIUS)*(indexX-RADIUS) + (indexY-RADIUS) * (indexY-RADIUS));
	int index = indexX + indexY*blockDim.x;
	double value = expf(-(r * r) / s) / (PI * s);
	atomicAdd(&sum, value);
	__syncthreads();
	dGausFilter[index] = value / sum;
}
__device__ void memSetSharedMem(int x, int y, uchar3 *sharedData, const uchar3 *globalData, int maxX, int maxY) {
	int posX;
	int posY;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				if (x <= 0 && y <= 0) {
					sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + posX + maxX * (y + posY)];
				}
				else {
					sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y - RADIUS + posY)];
				}
			}
	}
	else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				if (y <= 0) {
					sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - posX) + maxX * (y + posY)];
				}
				else {
					sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y - RADIUS + posY)];
				}
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				if (x <= 0) {
					sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y + posY)];
				}
				else {
					sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y + posY)];
				}
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				if (x == maxX - 1 && y == maxY - 1) {
					sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x - posX) + maxX * (y - posY)];
				}
				else {
					sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y + posY)];
				}
			}
	}
	else if (threadIdx.x == 0) {
		posY = threadIdx.y + RADIUS;
		for (posX = 0; posX <= RADIUS; posX++) {
			if (x <= 0) {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + posX + maxX * y];
			}
			else {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * y];
			}

		}
	}
	else if (threadIdx.y == 0) {
		posX = threadIdx.x + RADIUS;
		for (posY = 0; posY <= RADIUS; posY++) {
			if (y <= 0) {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y + posY)];
			}
			else {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y - RADIUS + posY)];
			}

		}
	}
	else if (threadIdx.x == blockDim.x - 1) {
		posY = threadIdx.y + RADIUS;
		for (posX = 0; posX <= RADIUS; posX++) {
			if (x == maxX) {
				sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[x - posX + maxX * y];
			}
			else {
				sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * y];
			}
		}
	}
	else if (threadIdx.y == blockDim.y - 1) {
		posX = threadIdx.x + RADIUS;
		for (posY = 0; posY <= RADIUS; posY++) {
			if (y == maxY - 1) {
				sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y - posY)];
			}
			else {
				sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y + posY)];
			}
		}
	}
	else {
		sharedData[threadIdx.x + RADIUS + (threadIdx.y + RADIUS) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * y];
	}
}

__global__ void gaussianBlur(const uchar3 *matrix, uchar3 *avgMatrix, int maxX, int maxY) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	extern __shared__ uchar3 sharedData[];

	if (x < maxX  && y < maxY) {
		memSetSharedMem(x, y, sharedData, matrix, maxX, maxY);
		__syncthreads();
		int3 sum{ 0,0,0 };
		for (int r = 0; r < FILTER_SIZE; r++) {
			for (int c = 0; c < FILTER_SIZE; c++) {
				int indexOffset = (threadIdx.x + c) + (blockDim.x + 2 * RADIUS)*(threadIdx.y + r);
				//printf("%d ", sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)]);
				sum.x += sharedData[indexOffset].x*gaussianFilter[c + FILTER_SIZE * r];
				sum.y += sharedData[indexOffset].y*gaussianFilter[c + FILTER_SIZE * r];
				sum.z += sharedData[indexOffset].z*gaussianFilter[c + FILTER_SIZE * r];
			}
			//printf("\n");
		}
		int index = x + maxX * y;
		avgMatrix[index].x = (uchar)(sum.x);
		avgMatrix[index].y = (uchar)(sum.y);
		avgMatrix[index].z = (uchar)(sum.z);
	}
}

void generateFilter() {
	double *dGausFilter;
	double *gausFilter = new double[FILTER_SIZE*FILTER_SIZE];
	if (cudaMalloc(&dGausFilter, sizeof(double)*(FILTER_SIZE*FILTER_SIZE) )!= cudaSuccess) {
		cerr<<"Couldn't allocate memory for filter";
		cudaFree(dGausFilter);
		exit(1);
	}
	if (cudaMemset(dGausFilter, 0, sizeof(double)*(FILTER_SIZE*FILTER_SIZE)) != cudaSuccess) {
		cerr << "Couldn't initialize memory for filter";
		cudaFree(dGausFilter);
		exit(1);
	}
	
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(FILTER_SIZE, FILTER_SIZE);
	generateGaussianFilter<<<gridSize,blockSize>>>(dGausFilter);
	cudaDeviceSynchronize();

	if(cudaMemcpy(gausFilter,dGausFilter,sizeof(double)*FILTER_SIZE*FILTER_SIZE,cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		cerr << "Couldn't copy filter from device to host";
		cudaFree(dGausFilter);
		delete gausFilter;
		exit(1);
	}
	double sum = 0;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			//cout << *((gausFilter + i * FILTER_SIZE) + j) << " ";
			sum += *((gausFilter + i * FILTER_SIZE) + j);
		}
		//cout << endl;
	}
	cout << sum;
	cudaFree(dGausFilter);
	if (cudaMemcpyToSymbol(gaussianFilter, gausFilter, sizeof(double)*FILTER_SIZE*FILTER_SIZE) != cudaSuccess) {
		cudaFree(dGausFilter);
		delete gausFilter;
		exit(1);
	}
	cudaDeviceSynchronize();
	delete gausFilter;
}

void doGaussianBlur() {
	Mat inputImage = imread(inputImageName, CV_LOAD_IMAGE_UNCHANGED);
	Mat outputImage = imread(inputImageName, CV_LOAD_IMAGE_UNCHANGED);


	if (inputImage.empty() || outputImage.empty()) {
		cerr << "Couldn't open file::" << inputImageName;
		exit(1);
	}

	uchar3* inputImageData = (uchar3*)inputImage.ptr<uchar3>(0);
	uchar3* outputImageData = (uchar3*)outputImage.ptr<uchar3>(0);

	uchar3* dInputImageData;
	uchar3* dOutputImageData;

	if (cudaMalloc(&dInputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
		cerr << "Couldn't allocate memory for input image";
		cudaFree(dInputImageData);
		exit(1);
	};

	if (cudaMalloc(&dOutputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
		cerr << "Couldn't allocate memory for output image";
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	};

	if (cudaMemcpy(dInputImageData, inputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols, cudaMemcpyHostToDevice) != cudaSuccess) {
		cerr << "Couldn,t initialiZe device for input image";
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}

	if (cudaMemset(dOutputImageData, 0, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
		cerr << "Couldn,t initialiZe device Average Matrix";
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}
	const dim3 blockSize(32, 32, 1);
	const dim3 gridSize((inputImage.cols + blockSize.x - 1) / blockSize.x, (inputImage.rows + blockSize.y - 1) / blockSize.y, 1);

	int sharedMemSpace = (blockSize.x + 2 * RADIUS)*(blockSize.y + 2 * RADIUS) * sizeof(uchar3);
	//printf("%d %d %d %d %d %d %d %d", blockSize.x, blockSize.y, gridSize.x, gridSize.y, inputImage.rows, inputImage.cols, outputImage.rows, outputImage.cols);
	gaussianBlur <<<gridSize, blockSize, sharedMemSpace >> > (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows);
	//findAverage <<<gridSize, blockSize >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows, count);

	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		cerr << "kernel launch failed: " << cudaGetErrorString(cudaGetLastError());
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}

	if (cudaMemcpy(outputImageData, dOutputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cerr << "Couldn't copy original matrix memory from device to host " << cudaGetErrorString(cudaGetLastError());;
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}
	cudaFree(dInputImageData);
	cudaFree(dOutputImageData);

	imwrite(outputImageName, outputImage);
}

int main()
{
	cudaFree(0);
	generateFilter();
	doGaussianBlur();
}

