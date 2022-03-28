// g++ -std=c++11 main_cpu.cu && ./a.out Evaluation_script/testcases/input/input1.txt output.txt

#include <stdio.h>
// #include <cuda.h>

using namespace std;


int find_useful_core(int p, int *priority_hashmap, int m, int *core_task_mapping) {
    int core_idx = priority_hashmap[p];

    // core is not mapped yet!
    if (core_idx == -1) {
        int core_idx = 0;
        while (core_idx < m && core_task_mapping[core_idx] != -1) {
            core_idx += 1;
        }

        // when no core is free
        if (core_idx >= m) {
            core_idx = -1;
        }
    }
    else {
        if (core_task_mapping[core_idx] == -1) {
            core_idx = -1;
        }
    }

    return core_idx;
}



// Complete the following function
void operations (int m, int n, int *executionTime, int *priority, int *result)  {
    // m -> no of cores
    // n -> no of tasks
    // exectutionTime -> {task: execution time} ; shape: n
    // priority -> {task: priority} ; shape: n
    // result -> {task: end time} ; shape: n

    // // allocating memory on GPU
    // int *d_executionTime, *d_priority, *d_result;
    // cudaMalloc(&d_executionTime, n * sizeof(int));
    // cudaMalloc(&d_priority, n * sizeof(int));
    // cudaMalloc(&result, n * sizeof(int));

    // // copy arrays from CPU to GPU
    // cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);

    int *tasks_start_time, *priority_hashmap, *core_task_mapping, *task_core_mapping;

    for (int i = 0; i < n; i++) { result[i] = -1; }
    for (int i = 0; i < m; i++) { priority_hashmap[i] = -1; } // {priority: core}
    for (int i = 0; i < m; i++) { core_task_mapping[i] = -1; } // {core: task}; -1 for free core
    for (int i = 0; i < m; i++) { tasks_start_time[i] = -1; }
    for (int i = 0; i < n; i++) { task_core_mapping[i] = -1; }

    int t = 0;
    int task_idx = 0;
    while (true) {
        int p = priority[task_idx];

        for (int i = 0; i < task_idx; i++) {
            if (t == tasks_start_time[i] + executionTime[i]) {
                int core_idx = task_core_mapping[task_idx];
                core_task_mapping[core_idx] = -1;
            }
        }

        int core_idx = find_useful_core(p, priority_hashmap, m, core_task_mapping);

        // no core is available
        // task has to be blocked until free core becomes available
        if (core_idx == -1) {
            t++;
            continue;
        }

        // core is busy now!
        core_task_mapping[core_idx] = task_idx;
        task_core_mapping[task_idx] = core_idx;

        // priority should be mapped to this core now!
        if (priority_hashmap[p] == -1) {
            priority_hashmap[p] = core_idx;
        }

        tasks_start_time[task_idx] = t;

        t++;
        task_idx++;

        // once all tasks are scheduled and result is computed, we can stop the script
        if (task_idx == n) { break; }
    }

    for (int i = 0; i < n; i++) {
        result[i] = tasks_start_time[i] + executionTime[i];
    }

    // // copy results back to host
    // cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaFree(d_executionTime);
    // cudaFree(d_priority);
    // cudaFree(d_result);
}


int main(int argc, char **argv)
{
    int m, n;
    // Input file pointer declaration
    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename , "r");

    // Checking if file ptr is NULL
    if (inputfilepointer == NULL)  {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &m);      // scaning for number of cores
    fscanf(inputfilepointer, "%d", &n);      // scaning for number of tasks

    // Taking execution time and priorities as input	
    int *executionTime = (int *) malloc (n * sizeof(int));
    int *priority = (int *) malloc (n * sizeof(int));
    for (int i = 0; i < n; i++)  {
        fscanf(inputfilepointer, "%d", &executionTime[i]);
    }
    for (int i = 0; i < n; i++)  {
        fscanf(inputfilepointer, "%d", &priority[i]);
    }

    // Allocate memory for final result output 
    int *result = (int *) malloc (n * sizeof(int));
    for (int i = 0; i < n; i++)  {
        result[i] = 0;
    }

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds = 0;
    // cudaEventRecord(start, 0);

    //==========================================================================================================

	operations (m, n, executionTime, priority, result); 

    //===========================================================================================================

    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Time taken by function to execute is: %.6f ms\n", milliseconds);

    // Output file pointer declaration
    char *outputfilename = argv[2];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    // Total time of each task: Final Result
    for (int i=0; i < n; i++)  {
        fprintf(outputfilepointer, "%d ", result[i]);
    }

    fclose(outputfilepointer);
    fclose(inputfilepointer);

    // free(executionTime);
    // free(priority);
    // free(result);
}
