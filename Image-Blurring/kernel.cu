#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "time.h";
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

const int RADIUS = 1;
const int MATRIX_SIZE = 18;
const int MAX = 10;

using namespace std;

void fillRandom(int *matrix, int maxX, int maxY, int range, unsigned long seed)
{
	srand(seed);
	for (int i = 0; i < maxX; i++)
		for (int j = 0; j < maxY; j++)
			*((matrix + i * maxY) + j) = rand() % MAX;

	//*((matrix + i * maxY) + j) = 10;
}

__device__ void memSetSharedMem(int x, int y, int *sharedData, int *globalData, int maxX, int maxY) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y - RADIUS + posY)];
			}
	}
	else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				sharedData[posX + threadIdx.x + RADIUS + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y - RADIUS + posY)];
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * (y + posY)];
			}
	}
	else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
		for (int posY = 0; posY <= RADIUS; posY++)
			for (int posX = 0; posX <= RADIUS; posX++) {
				sharedData[posX + RADIUS + threadIdx.x + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * (y + posY)];
			}
	}
	else if (threadIdx.x == 0) {
		int posY = threadIdx.y + RADIUS;
		for (int posX = 0; posX <= RADIUS; posX++) {
			sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x - RADIUS + posX) + maxX * y];
		}
	}
	else if (threadIdx.y == 0) {
		int posX = threadIdx.x + RADIUS;
		for (int posY = 0; posY <= RADIUS; posY++) {
			sharedData[posX + posY * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y - RADIUS + posY)];
		}
	}
	else if (threadIdx.x == blockDim.x - 1) {
		int posY = threadIdx.y + RADIUS;
		for (int posX = 0; posX <= RADIUS; posX++) {
			sharedData[posX + RADIUS + threadIdx.x + posY * (blockDim.x + 2 * RADIUS)] = globalData[(x + posX) + maxX * y];
		}
	}
	else if (threadIdx.y == blockDim.y - 1) {
		int posX = threadIdx.x + RADIUS;
		for (int posY = 0; posY <= RADIUS; posY++) {
			sharedData[posX + (posY + RADIUS + threadIdx.y) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * (y + posY)];
		}
	}
	else {
		sharedData[threadIdx.x + RADIUS + (threadIdx.y + RADIUS) * (blockDim.x + 2 * RADIUS)] = globalData[x + maxX * y];
	}
}

__global__ void findAverage(int *matrix, int *avgMatrix, int maxX, int maxY, int count) {
	int x = threadIdx.x + blockIdx.x*blockDim.x + RADIUS;
	int y = threadIdx.y + blockIdx.y*blockDim.y + RADIUS;

	extern __shared__ int sharedData[];



	if (x < maxX - RADIUS && y < maxY - RADIUS) {
		memSetSharedMem(x, y, sharedData, matrix, maxX, maxY);
		__syncthreads();
		int sum = 0;
		int sharedMaxX = blockDim.x + 2 * RADIUS;
		//if (threadIdx.x == 0 && blockIdx.x == 3 && threadIdx.y == 2 && blockIdx.y == 3)
		{
			for (int r = 0; r < 2 * RADIUS + 1; r++) {
				for (int c = 0; c < 2 * RADIUS + 1; c++) {
					//printf("%d ", sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)]);
					sum += sharedData[(threadIdx.x + c) + (sharedMaxX)*(threadIdx.y + r)];
				}
				//printf("\n");
			}
			avgMatrix[x + maxX * y] = sum / count;
		}
	}

}
//__global__ void findAverage(int *matrix, int *avgMatrix, int maxX, int maxY, int radius, int count) {
//	int x = threadIdx.x + blockIdx.x*blockDim.x;
//	int y = threadIdx.y + blockIdx.y*blockDim.y;
//	int index = x + maxX*y;
//	
//	if (x < maxX && y < maxY) {
//		int sum = 0;
//		int cou = 0;
//		for (int offsetY = y - radius; offsetY <= y + radius && offsetY < maxY; offsetY++) {
//			for (int offsetX = x - radius; offsetX <= x + radius && offsetX < maxX; offsetX++) {
//				if (offsetX >= 0 && offsetY >= 0)
//				{
//					int indexOffset = offsetY * maxX + offsetX;
//					sum += matrix[indexOffset];
//				}
//			}
//		}
//		avgMatrix[index] = sum / count;
//		//__syncthreads();
//		//printf("%d ", avgMatrix[index]);
//	}
//}


int main()
{
	int *matrix = new int[MATRIX_SIZE*MATRIX_SIZE];
	int *avgMatrix = new int[MATRIX_SIZE*MATRIX_SIZE];

	int *dMatrix;
	int *dAvgMatrix;

	cudaFree(0);

	fillRandom((int*)matrix, MATRIX_SIZE, MATRIX_SIZE, 10, time(NULL));
	int totalElements = MATRIX_SIZE * MATRIX_SIZE;

	if (cudaMalloc(&dMatrix, sizeof(int)*totalElements) != cudaSuccess) {
		cerr << "Couldn't allocate memory for matrix";
		cudaFree(dMatrix);
	};

	if (cudaMalloc(&dAvgMatrix, sizeof(int)*totalElements) != cudaSuccess) {
		cerr << "Couldn't allocate memory for Average Matrix";
		cudaFree(dAvgMatrix);
	};

	if (cudaMemcpy(dMatrix, matrix, sizeof(int)*totalElements, cudaMemcpyHostToDevice) != cudaSuccess) {
		cerr << "Couldn,t initialiZe device Original Matrix";
		cudaFree(dMatrix);
		cudaFree(dAvgMatrix);
	}

	if (cudaMemcpy(dAvgMatrix, matrix, sizeof(int)*totalElements, cudaMemcpyHostToDevice) != cudaSuccess) {
		cerr << "Couldn,t initialiZe device Average Matrix";
		cudaFree(dMatrix);
		cudaFree(dAvgMatrix);
	}

	const dim3 blockSize(16, 16);
	const dim3 gridSize((MATRIX_SIZE - 2 * RADIUS + blockSize.x - 1) / blockSize.x, (MATRIX_SIZE - 2 * RADIUS + blockSize.y - 1) / blockSize.y);
	int count = (RADIUS * 2 + 1)*(RADIUS * 2 + 1);

	int sharedMemSpace = (blockSize.x + 2 * RADIUS)*(blockSize.y + 2 * RADIUS);

	findAverage << <gridSize, blockSize, sharedMemSpace * sizeof(int) >> > (dMatrix, dAvgMatrix, MATRIX_SIZE, MATRIX_SIZE, count);
	//findAverage <<<gridSize, blockSize >>> (dMatrix, dAvgMatrix, MATRIX_SIZE, MATRIX_SIZE, RADIUS, count);

	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		cerr << "kernel launch failed: " << cudaGetErrorString(cudaGetLastError());
		cudaFree(dMatrix);
		cudaFree(dAvgMatrix);
		exit(1);
	}

	if (cudaMemcpy(avgMatrix, dAvgMatrix, sizeof(int)*totalElements, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cerr << "Couldn't copy original matrix memory from device to host";
		cudaFree(dMatrix);
		cudaFree(dAvgMatrix);
		exit(1);
	}
	cudaFree(dAvgMatrix);
	cudaFree(dMatrix);

	cout << endl << endl;
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			cout << *((matrix + i * MATRIX_SIZE) + j) << " ";
		}
		cout << endl;
	}

	cout << endl << endl;
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			cout << *((avgMatrix + i * MATRIX_SIZE) + j) << " ";
		}
		cout << endl;
	}

	delete[] avgMatrix;
	delete[] matrix;
	return 0;
}
