// #include <stdio.h>
// int main() {
//     printf("Hello World.\n");
//     return 0;
// }

#include <stdio.h>
#include <cuda.h>

__global__ void dkernel() {
    printf("Hello World.\n");
}

int main() {
    dkernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
