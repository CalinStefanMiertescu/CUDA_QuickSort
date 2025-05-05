#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREADS	128 // numarul maxim de thread-uri
#define N		512
#define MAX_LEVELS	300

int*	r_values;
int*	d_values;

// Macro pentru verificarea erorilor CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Eroare CUDA la %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// initializare set de date
void Init(int* values, int i) {
	srand( time(NULL) );
        printf("\n------------------------------\n");

        if (i == 0) {
        // Distributie uniforma
                printf("Distributia setului de date: Uniforma\n");
                for (int x = 0; x < N; ++x) {
                        values[x] = rand() % 100;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 1) {
        // Distributie gaussiana
        #define MEAN    100
        #define STD_DEV 5
                printf("Distributia setului de date: Gaussiana\n");
                float r;
                for (int x = 0; x < N; ++x) {
                        r  = (rand()%3 - 1) + (rand()%3 - 1) + (rand()%3 - 1);
                        values[x] = int( round(r * STD_DEV + MEAN) );
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 2) {
        // Distributie in galeti
                printf("Distributia setului de date: Galeti\n");
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
        // Distributie sortata
                printf("Distributia setului de date: Sortata\n");
                /*for (int x = 0; x < N; ++x)
                        printf("%d ", values[x]);
		*/
        }
	else if (i == 4) {
        // Distributie zero (toate elementele egale)
                printf("Distributia setului de date: Zero\n");
                int r = rand() % 100;
                for (int x = 0; x < N; ++x) {
                        values[x] = r;
                        //printf("%d ", values[x]);
                }
        }
        printf("\n");
}

// Functie kernel CUDA
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
	                        // interschimba start[idx] si start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // interschimba end[idx] si end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}
 
int main(int argc, char **argv) {
	printf("Quicksort incepe cu %d numere. ", N);
	size_t size = N * sizeof(int);
	r_values = (int*)malloc(size);
	d_values = NULL;
	CUDA_CHECK(cudaMalloc((void**)&d_values, size));
	const unsigned int cThreadsPerBlock = 128;
	
	/* Tipuri de seturi de date de sortat:
         *      1. Distributie normala
         *      2. Distributie gaussiana
         *      3. Distributie in galeti
         *      4. Distributie sortata
         *      5. Distributie zero
         */

	for (int i = 0; i < 5; ++i) {
                // initializare set de date
                Init(r_values, i);

	 	// copiere date pe dispozitiv	
		CUDA_CHECK(cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice));

		printf("Incepe executia kernel-ului...\n");

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));
		CUDA_CHECK(cudaEventRecord(start, 0));
	
		// executie kernel
 		quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
	 	CUDA_CHECK(cudaGetLastError());
	 	CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaEventRecord(stop, 0));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float gpuTime = 0;
		CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

 		printf( "\nExecutia kernel-ului s-a finalizat in %f ms\n", gpuTime );
	 	
	 	// copiere date inapoi pe host
		CUDA_CHECK(cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost));
 	
	 	// test
                printf("\nTestare rezultate...\n");
                for (int x = 0; x < N - 1; x++) {
                        if (r_values[x] > r_values[x + 1]) {
                                printf("Sortarea a esuat.\n");
                                break;
                        }
                        else
                                if (x == N - 2)
                                        printf("SORTARE REUSITA\n");
                }

		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));
	}
 	
	// eliberare memorie
	CUDA_CHECK(cudaFree(d_values));
 	free(r_values);
 	
	CUDA_CHECK(cudaDeviceReset());
	return 0;
}