// nvcc kernel_fusion.cu && ./a.out
#include <iostream>
#include <cuda.h>


__global__ kernel1(int *b, int *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { c[id] = b[id]; }
}

__global__ kernel2(int *c, int *d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > 0 && id < size) { d[id] = c[id - 1]; }
}

__global__ fused_kernel(int *b, int *c, int *d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        c[id] = b[id];

        if (id > 0) { d[id] = c[id - 1]; }
    }
}

__global__ fused_kernel(int *b, int *c, int *d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        c[id] = b[id];

        __syncthreads();
        if (id > 0) { d[id] = c[id - 1]; }
    }
}


__global__ void multiply(int a, int *X, int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { output[id] = a * X[id]; }
    // 1 read + 1 write
}


__global__ void add(int *vector1, int *vector2, int size, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) { output[id] = vector1[id] + vector2[id]; }
    // 2 read + 1 write
}


void unfused_op(int a, int *X, int *B, int size, int *output) {
    int *temp;
    cudaMalloc(&temp, size * sizeof(int));
    int num_blocks = ceil(float(size) / 1024);
    multiply<<<num_blocks, 1024>>>(a, X, temp, size);
    add<<<num_blocks, 1024>>>(temp, B, size, output);
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


int main() {
    int size = 100000000;

    int *X = (int *) malloc(size * sizeof(int));
    int *B = (int *) malloc(size * sizeof(int));

    // generate dummy inputs
    for (int i = 0; i <  size; i++) {
        X[i] = i;
        B[i] = i * 2;
    }
    int a = 32;

    int *d_X, *d_B, *d_output;
    cudaMalloc(&d_X, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_X, X, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; cudaEventCreate(&start);
    cudaEventCreate(&stop); float milliseconds = 0;
    cudaEventRecord(start,0);

    unfused_op(a, d_X, d_B, size, d_output); cudaDeviceSynchronize();

    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by unfused op is: %.6f ms\n", milliseconds);

    // print<<<1, 1>>>(d_output, size); cudaDeviceSynchronize();

    cudaEvent_t start2, stop2; cudaEventCreate(&start2);
    cudaEventCreate(&stop2); float milliseconds2 = 0;
    cudaEventRecord(start2, 0);

    int num_blocks = ceil(float(size) / 1024);
    fused_op<<<num_blocks, 1024>>>(a, d_X, d_B, size, d_output); cudaDeviceSynchronize();

    cudaEventRecord(stop2,0); cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("Time taken by fused op is: %.6f ms\n", milliseconds2);

    // print<<<1, 1>>>(d_output, size); cudaDeviceSynchronize();
}
