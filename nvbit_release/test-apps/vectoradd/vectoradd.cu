#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

/*
pwd
/home/omiles/582/record_and_replay/nvbit_release
export LD_PRELOAD=tools/omiles_instr/omiles_instr.so 
export CUDA_VISIBLE_DEVICES=1
*/

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    //if (id == 50) printf("Result: %f\n\n", a[id] + b[id]);
    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char *argv[]) {
    // Size of vectors
    int n = 100000;
    if (argc > 1) n = atoi(argv[1]);

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    // Device output vector
    double *d_c;
    
    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    printf("d_a = %p\n", d_a);
    printf("d_b = %p\n", d_b);
    printf("d_c = %p\n", d_c);
    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        // if (i == 37) printf("h_a[37] = %f\n", h_a[i]);
        h_b[i] = cos(i) * cos(i);
        h_c[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
    //printf("Result in vector add: %d\n", r3);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    CUDA_SAFECALL((vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    double sum = 0;
    for (i = 0; i < n; i++) sum += h_c[i];
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
