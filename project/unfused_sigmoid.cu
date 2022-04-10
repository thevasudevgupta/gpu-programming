// nvcc sigmoid.cu && ./a.out
#include <iostream>
#include <cuda.h>

// e^x / ∑e^x

__global__ void exp(int *input, int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > size) { return; }
    output[id] = exp(input[id]);
}

__global__ void reduce_sum(int *input, int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > size) { return; }

    // TODO: replace with reduction
    int temp = atomicAdd(input[id], INT_MAX);
    __syncthreads();
    output[0] = temp;
}

__global__ void sigmoid(int *exponents, int *reduction, int *output, int size) {
    // reduction is array of size 1
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > size) { return; }
    return exponents[id] / reduction[0];
}

__global__ void print(int *input, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");
}


void main() {
    int size;
    // int input = <read from file>

    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    int *d_exponents;
    cudaMalloc(&d_exponents, size * sizeof(int));
    exp<<<1, size>>>(d_input, d_exponents, size);

    int *d_reduction;
    cudaMalloc(&d_reduction, 1 * sizeof(int));
    reduce_sum<<<1, size>>>(d_exponent, d_reduction, size);

    sigmoid<<<1, size>>>(d_exponents, d_reduction, d_output, size);
    print<<<1, 1>>>(d_output, size);
}
