
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include <cmath>
__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void kDot(const float *m1, const float *m2, const int m1_columns, const int m2_columns, float* output) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	//const int r = (int)id / m2_columns;
	//const int c = id % m2_columns;
	double t_output = 0.f;

	for (int k = 0; k < m1_columns; ++k) {
		t_output += m1[blockIdx.x *  m1_columns + k] * m2[k * m2_columns + threadIdx.x];
	}
	output[id] = t_output;
}



int main(void)
{

	const int arraySize = 5;
	//Training size is the instance of the data_set.
	const int TRAINING_SIZE = 4;
	//Training dim is the dimension, which is the features number of the given data.
	const int TRAINING_DIM = 4;
	//Since the train size of the traning data is [4][4], than to create 8 nodes that layer 1 required
	//to be [4][8]
	const int L1_SIZE = 8;

	float h_X[TRAINING_SIZE*TRAINING_DIM] = { 5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		6.2, 3.4, 5.4, 2.3,
		5.9, 3.0, 5.1, 1.8 };

	float W0_size[L1_SIZE * TRAINING_DIM];
	float output[TRAINING_DIM * L1_SIZE];
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++) {
		W0_size[i] = 0.1 * (2.0*rand() / RAND_MAX - 1.0);
	}


	int N = 1 << 20;
	float *d_x, *d_y, *out;

	cudaMalloc(&d_x, TRAINING_SIZE*TRAINING_DIM * sizeof(float));
	cudaMalloc(&d_y, L1_SIZE * TRAINING_DIM * sizeof(float));
	cudaMalloc(&out, TRAINING_DIM * L1_SIZE * sizeof(float));

	cudaMemcpy(d_x, h_X, TRAINING_SIZE*TRAINING_DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, W0_size, L1_SIZE * TRAINING_DIM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(out, output, TRAINING_DIM * L1_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	kDot <<< TRAINING_DIM, L1_SIZE >> > (d_x, d_y, TRAINING_DIM, L1_SIZE, out);

	cudaMemcpy(output, out, TRAINING_DIM * L1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	for (auto i : output) {
		std::cout << i << std::endl;
	}

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(out);
	//free(h_X);
	//free(W0_size);
	//free(output);
}