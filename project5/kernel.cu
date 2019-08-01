#include "app.h"

Point *p = 0;
double *w = 0;
double *r = 0;


__global__ void f(double* weights, Point* points, double* result, int size, int weightSize)
{

	int i = blockIdx.x * 1000 + threadIdx.x;

	if (i < size)
	{

		int j;
		double value = 0;

		for (j = 0; j < weightSize; j++) {
			value += (weights[j] * points[i].coordinates[j]);
		}
		result[i] = value;
	}
}

cudaError_t calculateWithCuda(double* weights,double* results , int size ,int weightSize )
{
	
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	////alocate the weights arrays in GPU
	//cudaStatus = cudaMalloc((double**)&dev_a, weightSize * sizeof(double));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//}

	////alocate the points array in GPU
	//cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(Point));

	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//}

	////alocate the result array in GPU
	//cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//}



	// Copy input weights vector from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(w, weights, weightSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	//// Copy points vector from host memory to GPU buffer.
	//cudaStatus = cudaMemcpy(dev_b, points, size * sizeof(Point), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//}

	// Launch a kernel on the GPU with one thread for each element.
	f <<<size/1000, 1000 >>>( w, p, r,size,weightSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "f launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, r, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	return cudaStatus;
}

cudaError_t setArraysInGPU(Point* points,double* weights  ,int n,int k)
{


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//alocate the weights arrays in GPU
	cudaStatus = cudaMalloc((double**)&w, (k+1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	//alocate the points array in GPU
	cudaStatus = cudaMalloc((void**)&p, n * sizeof(Point));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	//alocate the result array in GPU
	cudaStatus = cudaMalloc((void**)&r, n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}



	// Copy input weights vector from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(w, weights, (k+1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// Copy points vector from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(p, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	

	return cudaStatus;
}

void freeArraysInGPU()
{
	cudaFree(p);
	cudaFree(w);
	cudaFree(r);
}