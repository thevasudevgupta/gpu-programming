// nvcc sigmoid.cu && ./a.out
#include <iostream>
#include <cuda.h>

__global__ void multiply(int a, int *X, int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { output[id] = a * X[id]; }
    // 1 read + 1 write
}


__global__ void add(int *vector1, int *vector2, int size, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { output[id] = vector1[id] + vector[id]; }
    // 2 read + 1 write
}


void unfused_op(int a, int *X, int *B, int size, int *output) {
    int *temp;
    cudaMalloc(&temp, size * sizeof(int));
    multiply<<<1, size>>>(a, X, temp, size);
    add<<<1, size>>>(temp, B, size, output);
    cudaFree(temp);

    // temp buffer is allocated (extra)
    // 3 read + 2 write => 5
}


__global__ void fused_op(int a, int *X, int *B, int size, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { output[id] = a * X[id] + B[id]; }
    // 2 reads + 1 write => 3
}


__global__ void print(int *input, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");
}


void main() {
    int size;
    int *X, *B;
    int a;
    // read from file

    int *d_X, *d_B, *d_output;
    cudaMalloc(&d_X, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_X, X, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    unfused_op(a, d_X, d_B, size, d_output);
    cudaDeviceSynchronize();

    print<<<1, 1>>>(d_output, size);

    fused_op<<<1, size>>>(a, d_X, d_B, size, d_output);
    cudaDeviceSynchronize();

    print<<<1, 1>>>(d_output, size); 
}
