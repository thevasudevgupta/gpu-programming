#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; //the handle for printing the output

// 1, 2, 3		// 1, 2, 3
// 5, 6, 7		// 5, 6, 7
// 2, 4, 7		// 2, 4, 7

// (1 * 1) + (2 * 5) + (3 * 2)
// 
// 


// complete the following kernel...
__global__ void per_row_column_kernel(long int *A, long int *B, long int *C, long int m, long int n){
	// (A + B.T) * (B.T - A)
	// (A * B.T) - (A * A) + (B.T * B.T) - (B.T * A)
	if (blockIdx.x > m || threadIdx.x > n) { return; }

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	// lets use id as the row index for A or column index for B

	C[id * n + i] = 0 // initialize C to 0
	for (long int i = 0; i < n; i++) {
		// [i * n + j] == [i, j]

		// c1 = A[id][i] * B[id][i]
		long int c1 = A[id * n + i] * B[id * n + i]; // A * B.T

		// c2 = A[id][i] * A[i][id]
		long int c2 = A[id * n + i] * A[i * n + id]; // A * A

		// c3 = B[i][id] * B[id][i]
		long int c3 = B[i * n + id] * B[id * n + i]; // B.T * B.T

		// c4 = B[i][id] * A[id][i]
		long int c4 = B[i * n + id] * A[id * n + i]; // B.T * A

		C[id * n + i] += c1 + c2 + c3 + c4;
	}
}


// complete the following kernel...
__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){

}

// complete the following kernel...
__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){

}

/**
 * Prints any 1D array in the form of a matrix 
 * */
void printMatrix(long int *arr, long int rows, long int cols, char* filename) {

	outfile.open(filename);
	for(long int i = 0; i < rows; i++) {
		for(long int j = 0; j < cols; j++) {
			outfile<<arr[i * cols + j]<<" ";
		}
		outfile<<"\n";
	}
	outfile.close();
}

// m -> number of rows
// n -> number of columns

int main(int argc,char **argv){

	// variable declarations
	long int m,n;	
	cin >> m >> n;

	// host_arrays
	long int *h_a, *h_b, *h_c;

	// device arrays
	long int *d_a, *d_b, *d_c;

	// Allocating space for the host_arrays
	h_a = (long int *) malloc(m * n * sizeof(long int));
	h_b = (long int *) malloc(m * n * sizeof(long int));	
	h_c = (long int *) malloc(m * n * sizeof(long int));	

	// Allocating memory for the device arrays
	cudaMalloc(&d_a, m * n * sizeof(long int));
	cudaMalloc(&d_b, m * n * sizeof(long int));
	cudaMalloc(&d_c, m * n * sizeof(long int));

	//Read the input matrix A 
	for (long int i = 0; i < m * n; i++) {
		cin >> h_a[i];
	}

	// Read the input matrix B 
	for (long int i = 0; i < m * n; i++) {
		cin >> h_b[i];
	}

	// Transfer the input host arrays to the device 
	cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

	long int gridDimx, gridDimy;
	//Launch the kernels
	/**
	 * Kernel 1 - per_row_column_kernel
	 * To be launched with 1D grid, 1D block
	 * */
	// 10, 12
	gridDimx = ceil(float(m) / 1024);
	dim3 grid1(gridDimx, 1, 1);
	dim3 block1(1024, 1, 1);
	per_row_column_kernel<<<grid1, block1>>>(d_a, d_b, d_c, m, n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n, "kernel1.txt");

	/**
	 * Kernel 2 - per_column_row_kernel
	 * To be launched with 1D grid, 2D block
	 * */
	// gridDimx = ceil(float(n) / 1024);
	// dim3 grid2(gridDimx,1,1);
	// dim3 block2(32,32,1);
	// per_column_row_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
	// cudaDeviceSynchronize();
	// cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	// printMatrix(h_c, m, n,"kernel2.txt");

	/**
	 * Kernel 3 - per_element_kernel
	 * To be launched with 2D grid, 2D block
	 * */
	// gridDimx = ceil(float(n) / 16);
	// gridDimy = ceil(float(m) / 64);
	// dim3 grid3(gridDimx,gridDimy,1);
	// dim3 block3(64,16,1);
	// per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
	// cudaDeviceSynchronize();
	// cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	// printMatrix(h_c, m, n,"kernel3.txt");

	return 0;
}
