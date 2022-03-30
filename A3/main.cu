#include <stdio.h>
#include <cuda.h>
// #include <iostream>

using namespace std;

// // following memory should be accessible to all threads
// priority -> allotted core
// core -> free/busy
// task -> allotted core
// result
// some tracker deciding which thread to block

__global__ void fill_sum(int size, int *A, int *B, int *C) {
    // A = B + C
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { A[idx] = B[idx] + C[idx]; }
}


__global__ void initialize(int *array, int size, int value) {
    // array[i] = value
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { array[idx] = value; }
}


void initialize_state(int *d_tasks_start_time, int *d_priority_hashmap, int *d_core_free_status, int *d_task_core_mapping, int m, int n) {
    cudaMalloc(&d_tasks_start_time, n * sizeof(int));
    cudaMalloc(&d_priority_hashmap, m * sizeof(int));
    cudaMalloc(&d_core_free_status, m * sizeof(int));
    cudaMalloc(&d_task_core_mapping, n * sizeof(int));

    num_blocks = ceil(float(n) / 1024);
    initialize<<<num_blocks, 1024>>>(d_tasks_start_time, n, -1);
    initialize<<<num_blocks, 1024>>>(d_task_core_mapping, n, -1);

    num_blocks = ceil(float(m) / 1024);
    initialize<<<num_blocks, 1024>>>(d_priority_hashmap, m, -1); // {priority: core}
    initialize<<<num_blocks, 1024>>>(d_core_free_status, n, 0); // 0: free core ; 1: busy

    cudaDeviceSynchronize();
}


int find_useful_core(int p, int *priority_to_core_map, int m, int *core_free_status) {
    int core_idx = priority_to_core_map[p];

    // core is not mapped yet!
    if (core_idx == -1) {
        int idx = 0;
        while (idx < m && core_free_status[idx] == 1) { idx += 1; }

        // when no core is free
        if (idx > m) { core_idx = -1; }
        else { core_idx = idx; }
    }
    else {
        if (core_free_status[core_idx] == 1) { core_idx = -1; }
    }
    return core_idx;
}


// mantain the state of what of tasks are already scheduled?
// [0, 0, 0, 0, 0] -> [1, 0, 0, 0, 0] -> [1, 1, 0, 0, 0] -> ...
// next tasks is going to get scheduled only when previous task is already scheduled
// + if the core is free

// how to make the cores free?
// make the core free only when you want to schedule the task

// for any task
// find useful core
// if that core is free
    // ans: previous task start time + execution time
// otherwise
    // find which task is executing on that core
    // ans: ans-for-that-task + execution time

// TODO: think if code would work incase threads belong to same warp
// TODO: think how to release the core once task is executed


// [0, 0, 0, 0, 0]
// [1, 0, 0, 0, 0]

__global__ void simulate(int *task_schedule_status, int *executionTime, int *priority, int *priority_to_core_map, int *tasks_start_time, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) { return; }

    // current task can't execute untill the previous task is already scheduled
    while (idx > 0 && task_schedule_status[idx - 1] == 0) ;

    int core_idx = priority_to_core_map[priority[idx]]

    // task is not allocated any core yet! let's allocate core then!
    if (core_idx == -1) {

        // find available core with min core idx
        int tmp_core_idx = 0;
        while (tmp_core_idx < m && core_free_status[tmp_core_idx] == 1) { tmp_core_idx += 1; }
        if (tmp_core_idx > m) { core_idx = -1; }
        else { core_idx = tmp_core_idx; }

        priority_to_core_map[priority[idx]] = core_idx;
    }
    // otherwise: whether core idx we get, task should be executed on that!

    core_to_task[core_idx] = task_idx;

    // core idx would be some no between 0 ... N-1
    // it indicates the core on which task must be executed

    // if that core is free => ans: previous task start time + current task execution time
    if (core_free_status[core_idx] == 0) {
        if (idx == 0) { tasks_start_time[idx] = 0; }
        else { tasks_start_time[idx] = tasks_start_time[idx - 1]; }        
    }
    else {
        // otherwise => find which task is executing on that core
        // => ans: ans-for-that-task + current task execution time
        tasks_start_time[idx] = result[core_to_task[core_idx]];
    }

    result[idx] = tasks_start_time[idx] + executionTime[idx];
    // we want all the tasks to wait until that core is free

    task_schedule_status[idx] = 1; // unlock next thread
    core_free_status[core_idx] = 1;
}

// 0 -> task is not scheduled yet
// 1 -> task has been scheduled
int volatile *task_schedule_status
initialize<<<n, 1>>>(task_schedule_status, n, 0);

// [0, 0, 0, 0]
// all threads should be locked untill previous task is scheduled

// [1, 0, 0, 0]

num_blocks = ceil(float(n) / 1024);
simulate<<<num_blocks, 1024>>>();

void simulate() {
    // 
    int timeout = 0;
    for (int i = 0; i < n; i++) { timeout += executionTime[i]; }

    bool time_updated = false;
    int t = 0;
    int task_idx = 0;
    while (true) {
        if (t > timeout) { break; }
        printf("t=%d :: ", t);

        int p = priority[task_idx];

        if (time_updated) {
            for (int i = 0; i < task_idx; i++) {
                if (t == tasks_start_time[i] + executionTime[i]) {
                    // cout << "inside" << endl;
                    core_free_status[task_core_mapping[i]] = 0;
                }
            }
        }

        int core_idx = find_useful_core(p, priority_to_core_map, m, core_free_status);

        // no core is available
        // task has to be blocked until free core becomes available
        if (core_idx == -1) {
            t++;
            time_updated = true;
            printf("doing nothing!!\n");
            continue;
        }

        printf("scheduling task-%d on core-%d\n", task_idx, core_idx);

        // core is busy now!
        core_free_status[core_idx] = 1;
        task_core_mapping[task_idx] = core_idx;

        // priority should be mapped to this core now!
        if (priority_to_core_map[p] == -1) {
            priority_to_core_map[p] = core_idx;
        }

        tasks_start_time[task_idx] = t;

        task_idx++;
        time_updated = false;

        // once all tasks are scheduled and result is computed, we can stop the script
        if (task_idx == n) { break; }
    }
    //
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

    int *d_tasks_start_time, *d_priority_hashmap, *d_core_free_status, *d_task_core_mapping;
    initialize_state(d_tasks_start_time, d_priority_hashmap, d_core_free_status, d_task_core_mapping, m, n);

    // TODO: think can we have threads within same warp?
    simulate<<<n, 1>>>();
    cudaDeviceSynchronize();

    num_blocks = ceil(float(n) / 1024);
    fill_sum<<<num_blocks, 1024>>>(n, d_result, d_tasks_start_time, d_executionTime);
    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_tasks_start_time);
    cudaFree(d_priority_hashmap);
    cudaFree(d_core_free_status);
    cudaFree(d_task_core_mapping);

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
