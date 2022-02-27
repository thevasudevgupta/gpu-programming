// nvcc main.cu && ./a.out evaluation-script/testcases/input/input1.txt output.txt

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

// write kernels here...
__global__ void transpose(int *A, int *X, int a, int b) {
	// A -> a x b
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= a * b) { return; }
	int ii = id / b;
	int jj = id % b;

	X[jj * a + ii] = A[ii * b + jj];
}


__global__ void matmul_after_transpose_B(int *A, int *B, int *X, int a, int b, int c) {
	// fully coelesced!
	// A -> a x b
	// B -> c x b
	// X -> a x c
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= a * c) { return; }

	int ii = id / c;
	int jj = id % c;

	X[ii * c + jj] = 0;
	for (int kk = 0; kk < b; kk++) {
		X[ii * c + jj] += A[ii * b + kk] * B[jj * b + kk];
	}
}


__global__ void add_(int *A, int *B, int a, int b) {
	// A = A + B
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= a * b) { return; }
	A[id] += B[id];
}


void transfer_to_host_and_print(int *d_matrix, int m, int n) {
	
	int *h_matrix = (int *) malloc(m * n * sizeof(int));
	cudaMemcpy(h_matrix, d_matrix, m * n * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << h_matrix[i * n + j] << " ";
		}
		cout << endl;
	}
}

// A -> p x q
// B -> q x p
// C -> q x r
// D -> s x r
// X -> p x s

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
	int num_blocks;

	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);

	// call the kernels for doing required computations...

	// C @ D.T
	// memory for storing intermediate states
	int *C_DT;
	cudaMalloc(&C_DT, q * s * sizeof(int));

	num_blocks = ceil(float(q * s) / 1024);
	matmul_after_transpose_B<<<num_blocks, 1024>>>(d_matrixC, d_matrixD, C_DT, q, r, s);
	cudaDeviceSynchronize();
	// transfer_to_host_and_print(C_DT, q, s);

	// // // memory for storing intermediate states
	int *temp_d_matrix;
	cudaMalloc(&temp_d_matrix, max(p * q, q * s) * sizeof(int));

	// // // A = A + B.T
	num_blocks = ceil((float)(p * q) / 1024);
	transpose<<<num_blocks, 1024>>>(d_matrixB, temp_d_matrix, q, p); // B -> B.T
	// transfer_to_host_and_print(temp_d_matrix, p, q);
	// temp_d_matrix -> p, q
	add_<<<num_blocks, 1024>>>(d_matrixA, temp_d_matrix, p, q);
	cudaDeviceSynchronize();
	// transfer_to_host_and_print(d_matrixA, p, q);

	// // d_matrixB is useless now
	cudaFree(d_matrixB);

	// temp_d_matrix is useless now
	// let's reuse temp_d_matrix for storing transpose of C_DT
	num_blocks = ceil((float)(s * q) / 1024);
	transpose<<<num_blocks, 1024>>>(C_DT, temp_d_matrix, q, s); // C_DT -> C_DT.T
	cudaDeviceSynchronize();
	// transfer_to_host_and_print(temp_d_matrix, s, q);
	// temp_d_matrix -> s, q

	// C_DT is useless now as we have stored it's transpose in temp_d_matrix
	cudaFree(C_DT);
	cudaMalloc(&d_matrixX, p * s * sizeof(int));

	// transfer_to_host_and_print(d_matrixA, p, q);
	// cout << endl;
	// transfer_to_host_and_print(temp_d_matrix, s, q);

	// // (A + B.T) @ C @ D.T
	num_blocks = ceil((float)(p * s) / 1024);
	matmul_after_transpose_B<<<num_blocks, 1024>>>(d_matrixA, temp_d_matrix, d_matrixX, p, q, s);
	cudaDeviceSynchronize();

	// cout << endl;
	transfer_to_host_and_print(d_matrixX, p, s);

	// copy the result back...
	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
}


// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}