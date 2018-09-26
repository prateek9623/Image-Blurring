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

__device__ void memSetSharedMem(int x, int y, uchar3 *sharedData, uchar3 *globalData, int maxX, int maxY) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				if (x <= 0 && y <= 0) {
					sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x+posX + maxX * (y+posY)];
				}
				else {
					sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y - RADIUS + posY)];
				}
			}
	}
	else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				if (y <= 0) {
					sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x-posX) + maxX * (y+posY)];
				}
				else {
					sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y - RADIUS + posY)];
				}
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				if (x <= 0) {
					sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x+posX) + maxX * (y+posY)];
				}
				else {
					sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y + posY)];
				}
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				if (x == maxX - 1 && y == maxY - 1) {
					sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x-posX) + maxX * (y-posY)];
				}
				else {
					sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y + posY)];
				}
			}
	}
	else if (threadIdx.x == 0) {
		int posY = threadIdx.y + RADIUS;
		for (int posX = 0; posX <= RADIUS; posX++) {
			if (x <= 0) {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x+posX + maxX * y];
			}
			else {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * y];
			}

		}
	}
	else if (threadIdx.y == 0) {
		int posX = threadIdx.x + RADIUS;
		for (int posY = 0; posY <= RADIUS; posY++) {
			if (y <= 0) {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y+posY)];
			}
			else {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y - RADIUS + posY)];
			}

		}
	}
	else if (threadIdx.x == blockDim.x - 1) {
		int posY = threadIdx.y + RADIUS;
		for (int posX = 0; posX <= RADIUS; posX++) {
			if (x == maxX) {
				sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[x-posX + maxX * y];
			}
			else {
				sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * y];
			}
		}
	}
	else if (threadIdx.y == blockDim.y - 1) {
		int posX = threadIdx.x + RADIUS;
		for (int posY = 0; posY <= RADIUS; posY++) {
			if (y == maxY - 1) {
				sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y-posY)];
			}
			sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y + posY)];
		}
	}
	else {
		sharedData[threadIdx.x + RADIUS + (threadIdx.y + RADIUS) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * y];
	}
}

__global__ void findAverage(uchar3 *matrix, uchar3 *avgMatrix, int maxX, int maxY, int count) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	extern __shared__ uchar3 sharedData[];



	if (x < maxX  && y < maxY) {
		memSetSharedMem(x, y, sharedData, matrix, maxX, maxY);
		__syncthreads();
		int3 sum { 0,0,0 };;
		int sharedMaxX = blockDim.x + 2 * RADIUS;
		//if (threadIdx.x == 0 && blockIdx.x == 3 && threadIdx.y == 2 && blockIdx.y == 3)
		{
			for (int r = 0; r < 2 * RADIUS + 1; r++) {
				for (int c = 0; c < 2 * RADIUS + 1; c++) {
					//printf("%d ", sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)]);
					sum.x += sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)].x;
					sum.y += sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)].y;
					sum.z += sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)].z;
				}
				//printf("\n");
			}
			avgMatrix[x + maxX * y].x = (uchar)(sum.x / count);
			avgMatrix[x + maxX * y].y =(uchar) (sum.y / count);
			avgMatrix[x + maxX * y].z =(uchar) (sum.z / count);
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
	int count = (RADIUS * 2 + 1)*(RADIUS * 2 + 1);

	int sharedMemSpace = (blockSize.x + 2 * RADIUS)*(blockSize.y + 2 * RADIUS);
	//printf("%d %d %d %d %d %d %d %d", blockSize.x, blockSize.y, gridSize.x, gridSize.y, inputImage.rows, inputImage.cols, outputImage.rows, outputImage.cols);
	findAverage <<<gridSize, blockSize, sharedMemSpace * sizeof(uchar3) >>> (dInputImageData, dOutputImageData, inputImage.cols, inputImage.rows, count);
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
