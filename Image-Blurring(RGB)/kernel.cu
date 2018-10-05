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

const int RADIUS = 1;
const char* inputImageName = "input.jpg";
const char* outputImageName = "output.jpg";
using namespace cv;
using namespace std;


void showImage(Mat img, char *title) {
	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, img);
	waitKey(0);
}

__device__ void memSetSharedMem(int x, int y, uchar3 *sharedData,const uchar3 *globalData,int maxX,int maxY) {
	int posX;
	int posY;
	int sharedBlockIndex;
	int globalMemIndex;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for ( posY = 0; posY <= RADIUS; posY++)
			for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posY = 0; posY <= RADIUS; posY++)
			for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posY = 0; posY <= RADIUS; posY++)
			for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posY = 0; posY <= RADIUS; posY++)
			for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posY = 0; posY <= RADIUS; posY++) {
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
		for ( posX = 0; posX <= RADIUS; posX++) {
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
		for ( posY = 0; posY <= RADIUS; posY++) {
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

__global__ void findAverage(const uchar3 *matrix, uchar3 *avgMatrix,int maxX,int maxY) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	extern __shared__ uchar3 sharedData[];

	if (x < maxX  && y < maxY) {
		memSetSharedMem(x, y, sharedData, matrix,maxX,maxY);
		__syncthreads();
		int3 sum { 0,0,0 };
		//if (threadIdx.x == 0 && blockIdx.x == 3 && threadIdx.y == 2 && blockIdx.y == 3)
		{
			for (int r = 0; r < 2 * RADIUS + 1; r++) {
				for (int c = 0; c < 2 * RADIUS + 1; c++) {
					int indexOffset = (threadIdx.x + c) + (blockDim.x + 2 * RADIUS)*(threadIdx.y + r);
					//printf("%d ", sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)]);
					sum.x += sharedData[indexOffset].x;
					sum.y += sharedData[indexOffset].y;
					sum.z += sharedData[indexOffset].z;
				}
				//printf("\n");
			}
			int index = x + maxX * y;
			avgMatrix[index].x = (uchar)(sum.x / ((RADIUS * 2 + 1)*(RADIUS * 2 + 1)));
			avgMatrix[index].y =(uchar) (sum.y / ((RADIUS * 2 + 1)*(RADIUS * 2 + 1)));
			avgMatrix[index].z =(uchar) (sum.z / ((RADIUS * 2 + 1)*(RADIUS * 2 + 1)));
		}
	}

}
//__global__ void findAverage(const uchar3 *matrix, uchar3 *avgMatrix, int maxX, int maxY,  int count) {
//	int x = threadIdx.x + blockIdx.x*blockDim.x;
//	int y = threadIdx.y + blockIdx.y*blockDim.y;
//	int index = x + maxX*y;
//	
//	if (x < maxX && y < maxY) 
//	{
//		int3 sum = { 0,0,0 };
//		for (int offsetY = y - RADIUS; offsetY <= y + RADIUS && offsetY < maxY; offsetY++) {
//			for (int offsetX = x - RADIUS; offsetX <= x + RADIUS && offsetX < maxX; offsetX++) {
//				if (offsetX >= 0 && offsetY >= 0)
//				{
//					int indexOffset = offsetY * maxX + offsetX;
//					sum.x += (int)matrix[indexOffset].x;
//					sum.y += (int)matrix[indexOffset].y;
//					sum.z += (int)matrix[indexOffset].z;
//				}
//			}
//		}
//		avgMatrix[index].x =  (uchar)(sum.x/count);
//		avgMatrix[index].y = (uchar)(sum.y / count);
//		avgMatrix[index].z = (uchar)(sum.z / count);
//		//matrix[index];
//		//__syncthreads();
//		//printf("%d ", avgMatrix[index]);
//	}
//}


int main()
{
	cudaFree(0);
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
	findAverage <<<gridSize, blockSize, sharedMemSpace >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows);
	//findAverage <<<gridSize, blockSize >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows, count);

	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		cerr << "kernel launch failed: " << cudaGetErrorString(cudaGetLastError());
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}

	if (cudaMemcpy(outputImageData, dOutputImageData, sizeof(uchar3)*inputImage.rows*inputImage.cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cerr << "Couldn't copy original matrix memory from device to host";
		cudaFree(dOutputImageData);
		cudaFree(dInputImageData);
		exit(1);
	}
	cudaFree(dInputImageData);
	cudaFree(dOutputImageData);

	imwrite(outputImageName, outputImage);
	//showImage(inputImage, "Original Image ");
	//showImage(outputImage, "Blur Image");
	//delete[] inputImageData;
	//delete[] outputImageData;
	return 0;
}
