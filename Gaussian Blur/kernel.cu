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


void showImage(Mat img, char *title) {
	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, img);
	waitKey(0);
}

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
	int sharedBlockIndex;
	int globalMemIndex;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				sharedBlockIndex = posX + posY * (blockDim.x + 2 * RADIUS);
				if (x <= 0 && y <= 0) {
					globalMemIndex = x + posX + maxX * (y + posY);
					//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x+posX + maxX * (y+posY)];
				}
				else {
					globalMemIndex = (x - RADIUS + posX) + maxX * (y - RADIUS + posY);
					//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y - RADIUS + posY)];
				}
				sharedData[sharedBlockIndex] = globalData[globalMemIndex];
			}
	}
	else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				sharedBlockIndex = posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS);
				if (y <= 0) {
					globalMemIndex = (x - posX) + maxX * (y + posY);
					//sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x-posX) + maxX * (y+posY)];
				}
				else {
					globalMemIndex = (x + posX) + maxX * (y - RADIUS + posY);
					//sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y - RADIUS + posY)];
				}
				sharedData[sharedBlockIndex] = globalData[globalMemIndex];
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				sharedBlockIndex = posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS);
				if (x <= 0) {
					globalMemIndex = (x + posX) + maxX * (y + posY);
					//sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x+posX) + maxX * (y+posY)];
				}
				else {
					globalMemIndex = (x - RADIUS + posX) + maxX * (y + posY);
					//sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y + posY)];
				}
				sharedData[sharedBlockIndex] = globalData[globalMemIndex];
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
		for (posY = 0; posY <= RADIUS; posY++)
			for (posX = 0; posX <= RADIUS; posX++) {
				sharedBlockIndex = posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS);
				if (x == maxX - 1 && y == maxY - 1) {
					globalMemIndex = (x - posX) + maxX * (y - posY);
					//sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x-posX) + maxX * (y-posY)];
				}
				else {
					globalMemIndex = (x + posX) + maxX * (y + posY);
					//sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y + posY)];
				}
				sharedData[sharedBlockIndex] = globalData[globalMemIndex];
			}
	}
	else if (threadIdx.x == 0) {
		posY = threadIdx.y + RADIUS;
		for (posX = 0; posX <= RADIUS; posX++) {
			sharedBlockIndex = posX + posY * (blockDim.x + 2 * RADIUS);
			if (x <= 0) {
				globalMemIndex = x + posX + maxX * y;
				//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x+posX + maxX * y];
			}
			else {
				globalMemIndex = (x - RADIUS + posX) + maxX * y;
				//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * y];
			}
			sharedData[sharedBlockIndex] = globalData[globalMemIndex];
		}
	}
	else if (threadIdx.y == 0) {
		posX = threadIdx.x + RADIUS;
		for (posY = 0; posY <= RADIUS; posY++) {
			sharedBlockIndex = posX + posY * (blockDim.x + 2 * RADIUS);
			if (y <= 0) {
				globalMemIndex = x + maxX * (y + posY);
				//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y+posY)];
			}
			else {
				globalMemIndex = x + maxX * (y - RADIUS + posY);
				//sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y - RADIUS + posY)];
			}
			sharedData[sharedBlockIndex] = globalData[globalMemIndex];
		}
	}
	else if (threadIdx.x == blockDim.x - 1) {
		posY = threadIdx.y + RADIUS;
		for (posX = 0; posX <= RADIUS; posX++) {
			sharedBlockIndex = posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS);
			if (x == maxX) {
				globalMemIndex = x - posX + maxX * y;
				//sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[x-posX + maxX * y];
			}
			else {
				globalMemIndex = (x + posX) + maxX * y;
				//sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * y];
			}
			sharedData[sharedBlockIndex] = globalData[globalMemIndex];
		}
	}
	else if (threadIdx.y == blockDim.y - 1) {
		posX = threadIdx.x + RADIUS;
		for (posY = 0; posY <= RADIUS; posY++) {
			sharedBlockIndex = posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS);
			if (y == maxY - 1) {
				globalMemIndex = x + maxX * (y - posY);
				//sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y-posY)];
			}
			else {
				globalMemIndex = x + maxX * (y + posY);
				//sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y + posY)];
			}
			sharedData[sharedBlockIndex] = globalData[globalMemIndex];
		}
	}
	else {
		sharedBlockIndex = threadIdx.x + RADIUS + (threadIdx.y + RADIUS) * (blockDim.x + 2 * RADIUS);
		globalMemIndex = x + maxX * y;
		sharedData[sharedBlockIndex] = globalData[globalMemIndex];
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

void doGaussianBlur(int imageCount, char* imageName[]) {

	vector<Mat> inputImageObject;
	//vector<Mat> outputImageObject;
	vector<uchar3*> deviceInputImageData;
	vector<uchar3*> hostInputImageData;
	vector<uchar3*> deviceOutputImageData;
	vector<uchar3*> hostOutputImageData;
	for (int i = 0; i < imageCount - 1; i++) {
		Mat inputImage = imread(imageName[i + 1], CV_LOAD_IMAGE_UNCHANGED);
		//Mat outputImage = imread(imageName[i + 1], CV_LOAD_IMAGE_UNCHANGED);


		if (inputImage.empty()) {
			cerr << "Couldn't open file::" << imageName[i + 1];
			continue;
		}
		inputImageObject.push_back(inputImage);
		//outputImageObject.push_back(outputImage);

		uchar3 *inputImageData;
		uchar3 *outputImageData;

		cudaMallocHost(&inputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols);
		cudaMallocHost(&outputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols);

		for (int rows = 0; rows < inputImage.rows; rows++)
			for (int cols = 0; cols < inputImage.cols; cols++) {
				inputImageData[cols + inputImage.cols*rows] = *(uchar3*)inputImageObject[i].ptr<uchar3>(rows, cols);
			}

		//uchar3* inputImageData = (uchar3*)inputImageObject[i].ptr<uchar3>(0);
		//uchar3* outputImageData = (uchar3*)outputImageObject[i].ptr<uchar3>(0);

		hostInputImageData.push_back(inputImageData);
		hostOutputImageData.push_back(outputImageData);

		uchar3* dInputImageData;
		uchar3* dOutputImageData;

		if (cudaMalloc(&dInputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
			cerr << "Couldn't allocate memory for input image " << imageName[i + 1];
			delete inputImageData;
			delete outputImageData;
			hostInputImageData.pop_back();
			hostOutputImageData.pop_back();
			cudaFree(dInputImageData);
			continue;
		};

		if (cudaMalloc(&dOutputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
			cerr << "Couldn't allocate memory for output image " << imageName[i + 1];
			delete inputImageData;
			delete outputImageData;
			hostInputImageData.pop_back();
			hostOutputImageData.pop_back();
			cudaFree(dOutputImageData);
			cudaFree(dInputImageData);
			continue;
		};
		deviceInputImageData.push_back(dInputImageData);
		deviceOutputImageData.push_back(dOutputImageData);
	}

	vector <cudaStream_t> streamPool = vector<cudaStream_t>(deviceInputImageData.size());

	for (int i = 0; i < deviceInputImageData.size(); i++) {
		cudaStreamCreate(&streamPool[i]);
	}

	for (int i = 0; i < deviceInputImageData.size(); i++) {

		if (cudaMemcpyAsync(deviceInputImageData[i], hostInputImageData[i], sizeof(uchar3)*inputImageObject[i].rows*inputImageObject[i].cols, cudaMemcpyHostToDevice, streamPool[i]) != cudaSuccess) {
			cerr << "Couldn,t initialiZe device for input image "<<imageName[i+1];
			cudaFree(deviceInputImageData[i]);
			cudaFree(deviceOutputImageData[i]);
			continue;
		}
		if (cudaMemsetAsync(deviceOutputImageData[i], 0, sizeof(uchar3)*inputImageObject[i].rows*inputImageObject[i].cols, streamPool[i]) != cudaSuccess) {
			cerr << "Couldn,t initialiZe device for input image " << imageName[i + 1];
			cudaFree(deviceInputImageData[i]);
			cudaFree(deviceOutputImageData[i]);
			continue;
		}

		const dim3 blockSize(32, 32, 1);
		const dim3 gridSize((inputImageObject[i].cols + blockSize.x - 1) / blockSize.x, (inputImageObject[i].rows + blockSize.y - 1) / blockSize.y, 1);

		int sharedMemSpace = (blockSize.x + 2 * RADIUS)*(blockSize.y + 2 * RADIUS) * sizeof(uchar3);
		//printf("%d %d %d %d %d %d %d %d", blockSize.x, blockSize.y, gridSize.x, gridSize.y, inputImage.rows, inputImage.cols, outputImage.rows, outputImage.cols);
		gaussianBlur <<<gridSize, blockSize, sharedMemSpace,streamPool[i] >>> (deviceInputImageData[i], deviceOutputImageData[i], inputImageObject[i].cols, inputImageObject[i].rows);
		//findAverage <<<gridSize, blockSize >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows, count);


		//if (cudaGetLastError() != cudaSuccess) {
		//	cerr << "kernel launch failed: " << cudaGetErrorString(cudaGetLastError());
		//	cudaFree(deviceInputImageData[i]);
		//	cudaFree(deviceOutputImageData[i]);
		//	exit(1);
		//}


	}


	//if (cudaMemcpy(dInputImageData, inputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols, cudaMemcpyHostToDevice) != cudaSuccess) {
	//	cerr << "Couldn,t initialiZe device for input image";
	//	cudaFree(dOutputImageData);
	//	cudaFree(dInputImageData);
	//	exit(1);
	//}

	//if (cudaMemset(dOutputImageData, 0, sizeof(uchar3)*inputImage.rows*inputImage.cols) != cudaSuccess) {
	//	cerr << "Couldn,t initialiZe device Average Matrix";
	//	cudaFree(dOutputImageData);
	//	cudaFree(dInputImageData);
	//	exit(1);
	//}
	//const dim3 blockSize(32, 32, 1);
	//const dim3 gridSize((inputImage.cols + blockSize.x - 1) / blockSize.x, (inputImage.rows + blockSize.y - 1) / blockSize.y, 1);

	//int sharedMemSpace = (blockSize.x + 2 * RADIUS)*(blockSize.y + 2 * RADIUS) * sizeof(uchar3);
	////printf("%d %d %d %d %d %d %d %d", blockSize.x, blockSize.y, gridSize.x, gridSize.y, inputImage.rows, inputImage.cols, outputImage.rows, outputImage.cols);
	//gaussianBlur <<<gridSize, blockSize, sharedMemSpace >> > (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows);
	////findAverage <<<gridSize, blockSize >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows, count);

	//cudaDeviceSynchronize();

	//if (cudaGetLastError() != cudaSuccess) {
	//	cerr << "kernel launch failed: " << cudaGetErrorString(cudaGetLastError());
	//	cudaFree(dOutputImageData);
	//	cudaFree(dInputImageData);
	//	exit(1);
	//}

	//if (cudaMemcpy(outputImageData, dOutputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
	//	cerr << "Couldn't copy original matrix memory from device to host " << cudaGetErrorString(cudaGetLastError());;
	//	cudaFree(dOutputImageData);
	//	cudaFree(dInputImageData);
	//	exit(1);
	//}
	for (int i = 0; i < streamPool.size(); i++)
	{
		if (cudaMemcpyAsync(hostOutputImageData[i], deviceOutputImageData[i], sizeof(uchar3)*inputImageObject[i].rows*inputImageObject[i].cols, cudaMemcpyDeviceToHost, streamPool[i]) != cudaSuccess) {
			cerr << "Couldn't copy original matrix memory from device to host " << cudaGetErrorString(cudaGetLastError());
			cudaFree(deviceInputImageData[i]);
			cudaFree(deviceOutputImageData[i]);
			exit(1);
		}
		cudaStreamSynchronize(streamPool[i]);

		cv::Mat image(inputImageObject[i].rows, inputImageObject[i].cols, CV_8UC3, hostOutputImageData[i]);
		showImage(image, "output");
		imwrite("output"+std::to_string(i+1)+".jpg",image);
	}
	cudaDeviceSynchronize();



	hostInputImageData.clear();
	hostOutputImageData.clear();
	hostInputImageData.shrink_to_fit();
	hostOutputImageData.shrink_to_fit();

	for (int i = 0; i < deviceInputImageData.size();i++ ) {
		cudaFree(deviceInputImageData[i]);
	}

	for (int i = 0; i < deviceOutputImageData.size();i++)
	{	
		cudaFree(deviceOutputImageData[i]);
	}

	deviceInputImageData.clear();
	deviceOutputImageData.clear();
	deviceOutputImageData.shrink_to_fit();
	deviceInputImageData.shrink_to_fit();
}

int main(int argc, char* argv[])
{
	cudaFree(0);
	generateFilter();
	doGaussianBlur(argc,argv);
}

