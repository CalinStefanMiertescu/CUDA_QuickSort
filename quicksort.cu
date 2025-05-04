#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS	128 
#define N		512
#define MAX_LEVELS	300

int*	r_values;
int*	d_values;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// initialize data set
void Init(int* values, int i) {
	srand( time(NULL) );
        printf("\n------------------------------\n");

        if (i == 0) {
        // Uniform distribution
                printf("Data set distribution: Uniform\n");
                for (int x = 0; x < N; ++x) {
                        values[x] = rand() % 100;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 1) {
        // Gaussian distribution
        #define MEAN    100
        #define STD_DEV 5
                printf("Data set distribution: Gaussian\n");
                float r;
                for (int x = 0; x < N; ++x) {
                        r  = (rand()%3 - 1) + (rand()%3 - 1) + (rand()%3 - 1);
                        values[x] = int( round(r * STD_DEV + MEAN) );
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 2) {
        // Bucket distribution
                printf("Data set distribution: Bucket\n");
                int j = 0;
                for (int x = 0; x < N; ++x, ++j) {
                        if (j / 20 < 1)
                                values[x] = rand() % 20;
                        else if (j / 20 < 2)
                                values[x] = rand() % 20 + 20;
                        else if (j / 20 < 3)
                                values[x] = rand() % 20 + 40;
                        else if (j / 20 < 4)
                                values[x] = rand() % 20 + 60;
                        else if (j / 20 < 5)
                                values[x] = rand() % 20 + 80;
                        if (j == 100)
                                j = 0;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 3) {
        // Sorted distribution
                printf("Data set distribution: Sorted\n");
                /*for (int x = 0; x < N; ++x)
                        printf("%d ", values[x]);
		*/
        }
	else if (i == 4) {
        // Zero distribution
                printf("Data set distribution: Zero\n");
                int r = rand() % 100;
                for (int x = 0; x < N; ++x) {
                        values[x] = r;
                        //printf("%d ", values[x]);
                }
        }
        printf("\n");
}

// Kernel function
__global__ static void quicksort(int* values) {
	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}
 
// program main
int main(int argc, char **argv) {
	printf("Quicksort starting with %d numbers. ", N);
	size_t size = N * sizeof(int);
	r_values = (int*)malloc(size);
	d_values = NULL;
	CUDA_CHECK(cudaMalloc((void**)&d_values, size));
	const unsigned int cThreadsPerBlock = 128;
	
	/* Types of data sets to be sorted:
         *      1. Normal distribution
         *      2. Gaussian distribution
         *      3. Bucket distribution
         *      4. Sorted Distribution
         *      5. Zero Distribution
         */

	for (int i = 0; i < 5; ++i) {
                // initialize data set
                Init(r_values, i);

	 	// copy data to device	
		CUDA_CHECK(cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice));

		printf("Beginning kernel execution...\n");

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));
		CUDA_CHECK(cudaEventRecord(start, 0));
	
		// execute kernel
 		quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
	 	CUDA_CHECK(cudaGetLastError());
	 	CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaEventRecord(stop, 0));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float gpuTime = 0;
		CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

 		printf( "\nKernel execution completed in %f ms\n", gpuTime );
	 	
	 	// copy data back to host
		CUDA_CHECK(cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost));
 	
	 	// test print
 		/*for (int i = 0; i < N; i++) {
 			printf("%d ", r_values[i]);
 		}
 		printf("\n");
		*/

		// test
                printf("\nTesting results...\n");
                for (int x = 0; x < N - 1; x++) {
                        if (r_values[x] > r_values[x + 1]) {
                                printf("Sorting failed.\n");
                                break;
                        }
                        else
                                if (x == N - 2)
                                        printf("SORTING SUCCESSFUL\n");
                }

		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));
	}
 	
 	// free memory
	CUDA_CHECK(cudaFree(d_values));
 	free(r_values);
 	
	CUDA_CHECK(cudaDeviceReset());
	return 0;
}