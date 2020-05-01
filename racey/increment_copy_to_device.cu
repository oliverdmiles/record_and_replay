#include <stdio.h>

__global__ void kernel(int *x, int *y) {
  if (threadIdx.x % 32 == 1) {
    *x = *x - threadIdx.x;
  } else if (threadIdx.x % 32 == 17) {
    *x = *x * threadIdx.x;
    *y = *y + threadIdx.x;
  } else if (threadIdx.x % 32 == 31) {
    *x = *x - threadIdx.x;
  } else {
    *y = *y + *x;
  }
}

int main(int argc, char **argv) {

  int blocks = atoi(argv[1]);
  int threads = atoi(argv[2]);

  int x = atoi(argv[3]);
  int y = atoi(argv[4]);

  int *d_x, *d_y;

  cudaMalloc((void **)&d_x, sizeof(int));
  cudaMalloc((void **)&d_y, sizeof(int));

  cudaMemcpy(d_x, &x, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &y, sizeof(int), cudaMemcpyHostToDevice);

  printf("Before running kernel:\n");
  printf("    x: %x\n", x);
  printf("    y: %x\n", y);

  kernel<<<blocks, threads>>>(d_x, d_y);
  cudaDeviceSynchronize();

  cudaMemcpy((void *)&x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)&y, d_y, sizeof(int), cudaMemcpyDeviceToHost);

  printf("After running kernel:\n");
  printf("    x: %x\n", x);
  printf("    y: %x\n", y);

  cudaFree(d_x);
  cudaFree(d_y);
}