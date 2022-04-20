#include <iostream>
#include <cuda.h>


__global__ void simple_kernel(int a, int *input, int* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) { output[idx] = a * input[idx]; }
}


int main() {

    // allocate memory in CPU
    int a = 2, size = 1024;
    int *input = (int *) malloc(size * sizeof(int));
    int *output = (int *) malloc(size * sizeof(int));

    // initialize input / read from disk
    for (int i = 0; i < size; i++) { input[i] = i; }

    // initize some pointers on CPU for storing address of GPU memory
    int *d_input, *d_output;

    // allocate memory on GPU & store GPU memory address in CPU memory
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    // copy data from CPU -> GPU
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // do computation on GPU
    int num_blocks = ceil(float(size) / 1024);
    simple_kernel<<<num_blocks, 1024>>>(a, d_input, d_output);

    // copy results back to CPU
    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
}
