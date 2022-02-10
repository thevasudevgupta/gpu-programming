// nvcc main.cu && ./a.out < sample.txt

#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; //the handle for printing the output

// (A + B.T) * (B.T - A)
// A -> (m, n)
// B -> (n, m)
// C -> (m, n)
// m -> number of rows
// n -> number of columns

// complete the following kernel...
__global__ void per_row_column_kernel(long int *A, long int *B, long int *C, long int m, long int n){
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= m) { return; }
	// lets use id as the row index for A or column index for B

	for (long int kk = 0; kk < n; kk++) {
		long int value = (A[id * n + kk] + B[kk * m + id]) * (B[kk * m + id] - A[id * n + kk]);
		C[id * n + kk] = value;
	}
}

// complete the following kernel...
__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	unsigned int id = blockIdx.x * blockDim.y * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
	if (id >= n) { return; }
	// lets use id as the column index for A or column index for B
	// printf("id: %d\n", id);

	for (long int kk = 0; kk < m; kk++) {
		long int value = (A[kk * n + id] + B[id * m + kk]) * (B[id * m + kk] - A[kk * n + id]);
		C[kk * n + id] = value;
	}
}


// complete the following kernel...
__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	unsigned int id = blockIdx.y * gridDim.x * blockDim.y * blockDim.x + blockIdx.x * blockDim.y * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
	if (id >= m * n) { return; }
	// printf("id: %d\n", id);

	unsigned int ii = id / n; // row index for A
	unsigned int jj = id % n; // column index for A
	long int value = (A[ii * n + jj] + B[jj * m + ii]) * (B[jj * m + ii] - A[ii * n + jj]);
	C[id] = value;
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

void printMatrix(long int *arr, long int rows, long int cols) {

	for(long int i = 0; i < rows; i++) {
		for(long int j = 0; j < cols; j++) {
			cout << arr[i * cols + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}


int main(int argc,char **argv){

	//variable declarations
	long int m,n;	
	cin>>m>>n;	

	//host_arrays 
	long int *h_a,*h_b,*h_c;

	//device arrays 
	long int *d_a,*d_b,*d_c;
	
	//Allocating space for the host_arrays 
	h_a = (long int *) malloc(m * n * sizeof(long int));
	h_b = (long int *) malloc(m * n * sizeof(long int));	
	h_c = (long int *) malloc(m * n * sizeof(long int));	

	//Allocating memory for the device arrays 
	cudaMalloc(&d_a, m * n * sizeof(long int));
	cudaMalloc(&d_b, m * n * sizeof(long int));
	cudaMalloc(&d_c, m * n * sizeof(long int));

	//Read the input matrix A 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_a[i];
	}

	//Read the input matrix B 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_b[i];
	}

	//Transfer the input host arrays to the device 
	cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

	long int gridDimx, gridDimy;
	//Launch the kernels 
	/**
	 * Kernel 1 - per_row_column_kernel
	 * To be launched with 1D grid, 1D block
	 * */
	gridDimx = ceil(float(m) / 1024);
	dim3 grid1(gridDimx,1,1);
	dim3 block1(1024,1,1);
	per_row_column_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	// printMatrix(h_c, m, n);
	printMatrix(h_c, m, n,"kernel1.txt");

	/**
	 * Kernel 2 - per_column_row_kernel
	 * To be launched with 1D grid, 2D block
	 * */
	gridDimx = ceil(float(n) / 1024);
	dim3 grid2(gridDimx,1,1);
	dim3 block2(32,32,1);
	per_column_row_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	// printMatrix(h_c, m, n);
	printMatrix(h_c, m, n,"kernel2.txt");

	/**
	 * Kernel 3 - per_element_kernel
	 * To be launched with 2D grid, 2D block
	 * */
	gridDimx = ceil(float(n) / 16);
	gridDimy = ceil(float(m) / 64);
	dim3 grid3(gridDimx,gridDimy,1);
	dim3 block3(64,16,1);
	per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	// printMatrix(h_c, m, n);
	printMatrix(h_c, m, n,"kernel3.txt");
	return 0;
}
