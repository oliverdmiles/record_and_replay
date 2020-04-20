#include "global_variables.hpp"
#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

__device__ bool lock() { return 0 == (atomicCAS(&mutex, 0, 1)); }

__device__ void unlock() { atomicExch(&mutex, 0); }

__device__ uint32_t get_TID() {
  uint32_t blockID =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  return blockID * (blockDim.x * blockDim.y * blockDim.z) +
         (threadIdx.z * (blockDim.x * blockDim.y)) +
         (threadIdx.y * blockDim.x) + threadIdx.x;
}

/* Instrumentation function used to log memory accesses */
extern "C" __device__ __noinline__ void
mem_record(int pred, uint32_t op_type, uint32_t reg_high,
           uint32_t target_reg_high, uint32_t reg_low, uint32_t target_reg_low,
           int32_t imm, int32_t cnt) {
  if (!pred) {
    return;
  }
  int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
  uint64_t addr = base_addr + imm;
  uint64_t val = (((uint64_t)nvbit_read_reg(target_reg_high)) << 32) |
                 ((uint64_t)nvbit_read_reg(target_reg_low));

  uint32_t is_extended = op_type & 0x1;
  uint32_t threadID = get_TID();
  val = is_extended ? val : val & 0xffffffff;

  record_data rd;
  rd.addr = addr;
  rd.value.u64 = val;
  rd.is_mem_instr = true;
  rd.time = atomicInc(&count, 4294967295);
  rd.type_load_tid = (op_type & 0xFFFFFFFE) | threadID;
  channel_dev.push(&rd, sizeof(record_data));
}
NVBIT_EXPORT_FUNC(mem_record);

extern "C" __device__ __noinline__ void sync_record(int pred, int32_t op_type) {

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
    rd.type_load_tid = threadID;
    channel_dev.push(&rd, sizeof(record_data));
  }
}
NVBIT_EXPORT_FUNC(sync_record);

extern "C" __device__ __noinline__ void
mem_replay(int pred, uint32_t is_extended, uint32_t reg_high,
           uint32_t target_reg_high, uint32_t reg_low, uint32_t target_reg_low,
           int32_t imm, int32_t count) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot(1);
  const int laneid = get_laneid();

  /* information about thread and access */
  uint64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
  uint64_t addr = base_addr + imm;

  uint32_t threadID = get_TID();

  /* information about waiting and order */
  bool waiting = true;
  bool isDependent = false;
  uint64_t depIdx = 0;

  // printf("Started thread %d\n", threadID);
  for (uint64_t i = 0; i < numDependecies; ++i) {
    if (deviceArr[i][0] == addr) {
      depIdx = i;
      isDependent = true;
      break;
    }
  }
  if (!isDependent) {
    printf("No instructions match: %lu\n", addr);
    return;
  }

  while (waiting) {
    uint64_t numThreadNext = deviceArr[depIdx][2];
    if (numThreadNext == deviceArr[depIdx][1]) {
      break;
    }
    uint64_t indexOfNextThread = NUM_METADATA + 3 * numThreadNext;
    if (deviceArr[depIdx][indexOfNextThread] == threadID && lock()) {
      printf("Curent thread: %d, Address %lu\n", threadID, addr);
      /* START OF WRITING OR READING */
      // is is_load supposed to be 64 bits? This uses 2 registers vs 1
      uint64_t is_load =
          deviceArr[depIdx][NUM_METADATA + 3 * indexOfNextThread + 1];
      uint32_t value_low =
          deviceArr[depIdx][NUM_METADATA + 3 * indexOfNextThread + 2] &
          0xffffffff;
      uint32_t value_high =
          (deviceArr[depIdx][NUM_METADATA + 3 * indexOfNextThread + 2] >> 32) &
          0xffffffff;

      if (is_load) {
        // load here
        printf("Before load low: %llx\n", nvbit_read_reg(target_reg_low));
        printf("Before load high: %llx\n", nvbit_read_reg(target_reg_high));
        nvbit_write_reg(target_reg_low, value_low);
        if (is_extended) {
          nvbit_write_reg(target_reg_high, value_high);
        }
        printf("After load low: %llx\n", nvbit_read_reg(target_reg_low));
        printf("After load high: %llx\n", nvbit_read_reg(target_reg_high));
      } else {
        // store here
        void *ptr = (void *)addr;
        if (is_extended)
          *(uint64_t *)(ptr) =
              ((uint64_t)value_high << 32) | (uint64_t)value_low;
        else
          *(uint32_t *)(ptr) = value_low;
      }

      /* END OF WRITING OR READING */
      deviceArr[depIdx][2] += 1;

      waiting = false;
      unlock();
    }
  }
}
NVBIT_EXPORT_FUNC(mem_replay);
