#include <chrono>
#include <stdio.h>
#include <thread>

__managed__ unsigned long x = 0xb456ac093f74;
__managed__ unsigned long y = 0xcd1e04389017;

__global__ void atomic_kernel(char *temp, short test) {
  uint32_t blockID =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  uint32_t threadID = blockID * (blockDim.x * blockDim.y * blockDim.z) +
                      (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

  // using char because only one byte in size
  // printf("Thread id: %d\n", threadID);
  if (threadIdx.x % 32 == 1) {
    x = x - threadID;
  } else if (threadIdx.x % 32 == 17) {
    x = x * threadID;
    y = y + threadID;
  } else if (threadIdx.x % 32 == 31) {
    x = x - threadID;
  } else {
    y = y + x;
  }
}

int main() {
  printf("Before running kernel\n");
  printf("    x: %llx\n", x);
  printf("    y: %llx\n", y);
  atomic_kernel<<<BLOCKS, THREADS>>>(0, 5);
  // std::this_thread::sleep_for(std::chrono::seconds(1));
  cudaDeviceSynchronize();
  cudaDeviceReset();
  printf("After running kernel\n");
  printf("    x: %llx\n", x);
  printf("    y: %llx\n", y);
  return 0;
}
