// g++ -std=c++11 main_cpu.cpp && ./a.out sample.txt output.txt

#include <iostream>

using namespace std;


int find_useful_core(int p, int *priority_hashmap, int m, int *core_free_status) {
    int core_idx = priority_hashmap[p];

    // core is not mapped yet!
    if (core_idx == -1) {
        int idx = 0;
        while (idx < m && core_free_status[idx] == 1) { idx += 1; }

        // when no core is free
        if (idx > m) { core_idx = -1; }
        else { core_idx = idx; }
    }
    else {
        // printf("yayyy %d\n", core_free_status[core_idx]);
        if (core_free_status[core_idx] == 1) { core_idx = -1; }
    }

    // cout << "core free status: ";
    // for (int i = 0; i < m; i++) { cout << core_free_status[i] << " "; }
    // cout << endl;
    // cout << "priority to core: ";
    // for (int i = 0; i < m; i++) { cout << priority_hashmap[i] << " "; }
    // cout << endl;

    return core_idx;
}



// Complete the following function
void operations (int m, int n, int *executionTime, int *priority, int *result)  {
    // m -> no of cores
    // n -> no of tasks
    // exectutionTime -> {task: execution time} ; shape: n
    // priority -> {task: priority} ; shape: n
    // result -> {task: end time} ; shape: n

    int *tasks_start_time = (int *) malloc (n * sizeof(int));
    int *priority_hashmap = (int *) malloc (m * sizeof(int));
    int *core_free_status = (int *) malloc (m * sizeof(int));
    int *task_core_mapping = (int *) malloc (n * sizeof(int));

    for (int i = 0; i < n; i++) { result[i] = -1; }
    for (int i = 0; i < m; i++) { priority_hashmap[i] = -1; } // {priority: core}
    for (int i = 0; i < m; i++) { core_free_status[i] = 0; } // 0: free core ; 1: busy
    for (int i = 0; i < n; i++) { tasks_start_time[i] = -1; }
    for (int i = 0; i < n; i++) { task_core_mapping[i] = -1; }

    int timeout = 0;
    for (int i = 0; i < n; i++) { timeout += executionTime[i]; }

    bool time_updated = false;
    int t = 0;
    int task_idx = 0;
    while (true) {
        if (t > timeout) { break; }
        printf("t=%d :: ", t);

        // for (int i = 0; i < n; i++) { cout << task_core_mapping[i] << " "; }
        // cout << endl;

        int p = priority[task_idx];
        // cout << "task_idx = " << task_idx << endl;

        if (time_updated) {
            for (int i = 0; i < task_idx; i++) {
                if (t == tasks_start_time[i] + executionTime[i]) {
                    // cout << "inside" << endl;
                    core_free_status[task_core_mapping[i]] = 0;
                }
            }
        }

        int core_idx = find_useful_core(p, priority_hashmap, m, core_free_status);
        // cout << "core_idx = " << core_idx << endl;

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

        // cout << "core free status: ";
        // for (int i = 0; i < m; i++) { cout << core_free_status[i] << " "; }
        // cout << endl;

        // priority should be mapped to this core now!
        if (priority_hashmap[p] == -1) {
            priority_hashmap[p] = core_idx;
        }

        tasks_start_time[task_idx] = t;

        task_idx++;
        time_updated = false;

        // once all tasks are scheduled and result is computed, we can stop the script
        if (task_idx == n) { break; }
    }

    for (int i = 0; i < n; i++) {
        result[i] = tasks_start_time[i] + executionTime[i];
    }
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

    //==========================================================================================================

	operations (m, n, executionTime, priority, result); 

    //===========================================================================================================

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

    free(executionTime);
    free(priority);
    free(result);
}
