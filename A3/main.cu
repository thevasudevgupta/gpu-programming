// nvcc main.cu && ./a.out sample.txt output.txt
// nvcc main.cu && ./a.out Evaluation_script/testcases/input/input1.txt output.txt && python3 verify.py Evaluation_script/testcases/output/output1.txt output.txt

#include <stdio.h>
#include <cuda.h>

using namespace std;


__device__ volatile int t = 0;


__global__ void initialize(volatile int *array, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { array[idx] = value; }
}


__global__ void simulate(volatile int *task_schedule_status, int *priority, int *executionTime,  volatile int *priority_to_core_map, volatile int *core_free_status, volatile  int *core_busy_time, int *result, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) { return; }

    // current task can't execute untill the previous task is already scheduled
    while (idx > 0 && task_schedule_status[idx - 1] == 0) ;

    int p = priority[idx];
    int core_idx = priority_to_core_map[p];
    // task is not allocated any core yet! let's allocate core then!
    if (core_idx == -1) {

        // find available core with min core idx
        int tmp_core_idx = 0;
        while (tmp_core_idx < m && core_free_status[tmp_core_idx] == 1) { tmp_core_idx += 1; }
        if (tmp_core_idx > m) { core_idx = -1; }
        else { core_idx = tmp_core_idx; }

        priority_to_core_map[p] = core_idx;
    }

    if (core_free_status[core_idx] == 1) {
        t = core_busy_time[core_idx];

        // free the cores whenever needed
        for (int i = 0; i < m; i++) {
            if (t >= core_busy_time[i]) {
                core_free_status[i] = 0;
            }
        }
    }

    result[idx] = t + executionTime[idx];
    // we want all the tasks to wait until that core is free
    // printf("t=%d | task-%d is scheduled on core-%d untill %d units time\n", t, idx, core_idx, result[idx]);

    core_busy_time[core_idx] = result[idx];
    core_free_status[core_idx] = 1;

    task_schedule_status[idx] = 1; // unlock next thread
}


//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    // m -> no of cores
    // n -> no of tasks
    // exectutionTime -> {task: execution time} ; shape: n
    // priority -> {task: priority} ; shape: n
    // result -> {task: end time} ; shape: n

    // allocating memory on GPU
    int *d_executionTime, *d_priority, *d_result;
    cudaMalloc(&d_executionTime, n * sizeof(int));
    cudaMalloc(&d_priority, n * sizeof(int));
    cudaMalloc(&d_result, n * sizeof(int));

    // copy arrays from CPU to GPU
    cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);

    // ###################################################################
    int num_blocks;

    volatile int *d_task_schedule_status;
    cudaMalloc(&d_task_schedule_status, n * sizeof(int));
    num_blocks = ceil(float(n) / 1024);
    initialize<<<num_blocks, 1024>>>(d_task_schedule_status, n, 0);
    cudaDeviceSynchronize();
    // 0 -> task is not scheduled yet
    // 1 -> task has been scheduled

    volatile int *d_priority_to_core_map;
    cudaMalloc(&d_priority_to_core_map, m * sizeof(int));
    num_blocks = ceil(float(m) / 1024);
    initialize<<<num_blocks, 1024>>>(d_priority_to_core_map, m, -1);
    cudaDeviceSynchronize();

    volatile int *d_core_free_status;
    cudaMalloc(&d_core_free_status, m * sizeof(int));
    num_blocks = ceil(float(m) / 1024);
    initialize<<<num_blocks, 1024>>>(d_core_free_status, m, 0);
    cudaDeviceSynchronize();

    volatile int *d_core_busy_time;
    cudaMalloc(&d_core_busy_time, m * sizeof(int));
    num_blocks = ceil(float(m) / 1024);
    initialize<<<num_blocks, 1024>>>(d_core_busy_time, m, 0);
    cudaDeviceSynchronize();
    // ###################################################################

    // TODO: think can we have threads within same warp?
    // num_blocks = ceil(float(n) / 1024);
    // simulate<<<num_blocks, 1024>>>(d_task_schedule_status, d_priority, d_executionTime, d_priority_to_core_map, d_core_free_status, d_core_busy_time, d_result, n, m);
    simulate<<<n, 1>>>(d_task_schedule_status, d_priority, d_executionTime, d_priority_to_core_map, d_core_free_status, d_core_busy_time, d_result, n, m);
    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // free up GPU memory
    cudaFree(d_executionTime);
    cudaFree(d_priority);
    cudaFree(d_result);
}


int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}
