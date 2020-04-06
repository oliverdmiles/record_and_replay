#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"
#include "global_variables.hpp"

__device__ void lock(int threadID) {
  while (0 != (atomicCAS(&mutex, 0, 1))) {
  }
}

__device__ void unlock(int threadID) { atomicExch(&mutex, 0); }

__device__ uint32_t get_TID() {
  uint32_t blockID =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  return blockID * (blockDim.x * blockDim.y * blockDim.z) +
         (threadIdx.z * (blockDim.x * blockDim.y)) +
         (threadIdx.y * blockDim.x) + threadIdx.x;
}



/* Instrumentation function used to log memory accesses*/
extern "C" __device__ __noinline__ 
void mem_record(int pred, uint32_t op_type, uint32_t reg_high,
           uint32_t target_reg_high, uint32_t reg_low, uint32_t target_reg_low,
           int32_t imm) {

  if (!pred) {
    return;
  }
  int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
  uint64_t addr = base_addr + imm;
  uint32_t volatile load = op_type << 1;
  uint64_t val =
      load ? (uint64_t)(*(uint64_t *)(addr))
           : (((uint64_t)target_reg_high) << 32) | ((uint64_t)target_reg_low);

  uint32_t threadID = get_TID();

  int active_mask = __ballot(1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  record_data rd;
  for (int i = 0; i < 32; i++) {
    rd.addr[i] = __shfl(addr, i);
    rd.value[i].u64 = __shfl(val, i);
  }

  if (first_laneid == laneid) {
    rd.is_mem_instr = true;
    rd.time = atomicInc(&count, 4294967295);
    rd.type_load_tid = op_type | threadID;
    channel_dev.push(&rd, sizeof(record_data));
  }
} NVBIT_EXPORT_FUNC(mem_record);

extern "C" __device__ __noinline__ 
void sync_record(int pred, int32_t op_type) {

  if (!pred) {
    return;
  }

  uint32_t threadID = get_TID();

  int active_mask = __ballot(1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  record_data rd;

  if (first_laneid == laneid) {
    rd.is_mem_instr = false;
    rd.time = atomicInc(&count, 4294967295);
    rd.type_load_tid = op_type | threadID;
    channel_dev.push(&rd, sizeof(record_data));
  }
} NVBIT_EXPORT_FUNC(sync_record);

extern "C" __device__ __noinline__ 
void mem_replay(int pred) {
  if (!pred) {
    return;
  }

  int loc = 1;

  uint32_t threadID = get_TID();

  int active_mask = __ballot(1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  if (laneid == first_laneid) {
    lock(threadID);
    int *currInterest = deviceArr[start];
    if (currInterest[0] == loc || true) {
      int curr_idx = currInterest[2];
      while (currInterest[NUM_METADATA + curr_idx] != threadID) {
        unlock(threadID);
        lock(threadID);
        curr_idx = currInterest[2];
      }
      printf("The thread being accessed is: %d\n", threadID);
      currInterest[2] += 1;

      if (currInterest[1] == currInterest[2]) {
        start += 1;
      }
    }
    unlock(threadID);
  }
} NVBIT_EXPORT_FUNC(mem_replay);