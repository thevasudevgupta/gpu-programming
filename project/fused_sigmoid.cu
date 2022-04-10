// nvcc fused_sigmoid.cu && ./a.out
#include <iostream>
#include <cuda.h>

// e^x / ∑e^x

__global__ void sigmoid(int *input, int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > size) { return; }

    int temp = exp(input[id]);
    output[id] = temp
    __syncthreads();

    // TODO: replace with reduction
    int reduction = atomicAdd(output[id], INT_MAX);
    __syncthreads();

    output[id] = temp / reduction;
}

__global__ void print(int *input, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");
}

void main() {
    int size, *input;
    // input = <read from file>

    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // assuming size < 1024
    sigmoid<<<1, size>>>(d_input, d_output, size);
    print<<<1, 1>>>(d_output, size);
}
