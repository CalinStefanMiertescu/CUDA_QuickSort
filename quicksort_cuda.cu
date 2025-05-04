#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_ARRAY_SIZE 1000000
#define THREADS_PER_BLOCK 256
#define MAX_DEPTH 24  

// Swap two elements
__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function for quicksort
__device__ int partition(int* data, int low, int high) {
    int pivot = data[high];  // Choose the rightmost element as pivot
    int i = low - 1;  // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than the pivot
        if (data[j] < pivot) {
            i++;  // Increment index of smaller element
            swap(&data[i], &data[j]);
        }
    }
    swap(&data[i + 1], &data[high]);
    return (i + 1);
}

// Non-recursive quicksort kernel for small arrays
__device__ void quicksortIterative(int* data, int low, int high) {
    // Create an auxiliary stack
    int stack[MAX_DEPTH];

    // Initialize top of stack
    int top = -1;

    // Push initial values of low and high to stack
    stack[++top] = low;
    stack[++top] = high;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop high and low
        high = stack[top--];
        low = stack[top--];

        // Set pivot element at its correct position
        int p = partition(data, low, high);

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > low) {
            stack[++top] = low;
            stack[++top] = p - 1;
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < high) {
            stack[++top] = p + 1;
            stack[++top] = high;
        }
    }
}

// CUDA kernel for quicksort
__global__ void quicksortKernel(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // Each thread works on a different segment of the array
        // For simplicity, we'll divide array into segments
        int segmentSize = (n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
        int start = tid * segmentSize;
        int end = min(start + segmentSize - 1, n - 1);

        if (start < n && end >= start) {
            quicksortIterative(data, start, end);
        }
    }
}

// Merge function to combine sorted segments
__global__ void mergeKernel(int* data, int* temp, int n, int segmentSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSegments = (n + segmentSize - 1) / segmentSize;

    if (tid < totalSegments / 2) {
        int seg1Start = tid * 2 * segmentSize;
        int seg1End = min(seg1Start + segmentSize - 1, n - 1);
        int seg2Start = seg1End + 1;
        int seg2End = min(seg2Start + segmentSize - 1, n - 1);

        if (seg2Start < n) {
            // Merge two adjacent segments
            int i = seg1Start;
            int j = seg2Start;
            int k = seg1Start;

            while (i <= seg1End && j <= seg2End) {
                if (data[i] <= data[j]) {
                    temp[k++] = data[i++];
                }
                else {
                    temp[k++] = data[j++];
                }
            }

            while (i <= seg1End) {
                temp[k++] = data[i++];
            }

            while (j <= seg2End) {
                temp[k++] = data[j++];
            }

            // Copy back the merged array to original array
            for (int idx = seg1Start; idx <= seg2End; idx++) {
                data[idx] = temp[idx];
            }
        }
    }
}

// Host function to read input from file and sort using CUDA
void quicksortCUDA(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    // Read array size
    int n;
    fscanf(file, "%d", &n);

    if (n <= 0 || n > MAX_ARRAY_SIZE) {
        printf("Invalid array size: %d\n", n);
        fclose(file);
        return;
    }

    // Read array elements
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (fscanf(file, "%d", &h_data[i]) != 1) {
            printf("Error reading element %d\n", i);
            free(h_data);
            fclose(file);
            return;
        }
    }
    fclose(file);

    // Allocate device memory
    int* d_data;
    int* d_temp;
    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Start timing
    clock_t start = clock();

    // Launch quicksort kernel
    quicksortKernel << <blocksPerGrid, threadsPerBlock >> > (d_data, n);
    cudaDeviceSynchronize();

    // Merge sorted segments using multiple passes
    int segmentSize = (n + blocksPerGrid * threadsPerBlock - 1) / (blocksPerGrid * threadsPerBlock);

    while (segmentSize < n) {
        int mergeBlocks = (n + 2 * segmentSize - 1) / (2 * segmentSize);
        mergeBlocks = (mergeBlocks + threadsPerBlock - 1) / threadsPerBlock;

        mergeKernel << <mergeBlocks, threadsPerBlock >> > (d_data, d_temp, n, segmentSize);
        cudaDeviceSynchronize();

        segmentSize *= 2;
    }

    // End timing
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify sorting (optional)
    bool sorted = true;
    for (int i = 1; i < n; i++) {
        if (h_data[i - 1] > h_data[i]) {
            sorted = false;
            break;
        }
    }

    printf("Array %s sorted.\n", sorted ? "is" : "is not");
    printf("Sorting time: %.6f seconds\n", elapsed);

    // Save sorted array to output file
    FILE* outFile = fopen("output.txt", "w");
    if (outFile == NULL) {
        printf("Error opening output file\n");
    }
    else {
        fprintf(outFile, "%d\n", n);
        for (int i = 0; i < n; i++) {
            fprintf(outFile, "%d ", h_data[i]);
        }
        fclose(outFile);
        printf("Sorted array written to output.txt\n");
    }

    // Clean up
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_temp);
}

int main(int argc, char** argv) {
    const char* filename = "input.txt";  // Default to input.txt

    if (argc > 1) {
        filename = argv[1];  // Use command-line argument if provided
    }

    quicksortCUDA(filename);
    return 0;
}