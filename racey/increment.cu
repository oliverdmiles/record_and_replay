#include <stdio.h>
#include <chrono>
#include <thread>

__managed__ unsigned int x = 0;
__managed__ unsigned int y = 0;

__global__ void atomic_kernel() {
    if (threadIdx.x%32 == 1 || threadIdx.x%32 == 17 || threadIdx.x%32 == 31) {
        //printf("bx:%u\n", x);
        x = x + 1;
        // printf("ax:%u\n", x);
        // printf("by:%u\n", y);
        atomicInc((unsigned int*)&y, 4096);
        //printf("ay:%u\n", y);
    }
}

int main() {
    atomic_kernel<<<BLOCKS,THREADS>>>();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    cudaDeviceSynchronize();
    printf("x:%u y:%u\n", x, y);
    atomic_kernel<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    printf("x:%u y:%u\n", x, y);
    cudaDeviceReset();
    return 0;
}
