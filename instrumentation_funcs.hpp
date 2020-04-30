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

  uint64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
  uint64_t addr = base_addr + imm;

  uint64_t val = (uint64_t)((uint32_t)(nvbit_read_reg(target_reg_low)));
  if (target_reg_low != 255) {
    val = (((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_high)))) << 32) |
                   ((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_low))));
  }
  
  uint32_t is_extended = op_type & 0x1;
  uint32_t threadID = get_TID();
  val = is_extended ? val : val & 0xffffffff;
  uint32_t size = (op_type >> 28) & 3;

  //size is actually 4
  if (size == 2) 
    val &= 0xFFFFFFFF;
  //size is actually 2
  else if (size == 1)
    val &= 0xFFFF;
  //size is actually 1
  else if (size == 0)
    val &= 0xFF;

  record_data rd;
  rd.addr = addr;
  rd.value.u64 = val;
  rd.is_mem_instr = true;
  rd.time = atomicInc(&count, 4294967295);
  rd.type_load_tid = (op_type & 0xFFFFFFFE) | threadID;
  channel_dev.push(&rd, sizeof(record_data));
}
NVBIT_EXPORT_FUNC(mem_record);

extern "C" __device__ __noinline__ void
mem_replay(int pred, uint32_t op_info, uint32_t reg_high,
           uint32_t target_reg_high, uint32_t reg_low, uint32_t target_reg_low,
           int32_t imm, int32_t count) {
  if (!pred) {
    return;
  }

  /* position in warp */
  const int laneid = get_laneid();

  /* information about thread and access */
  uint64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
  uint64_t addr = base_addr + imm;

  uint32_t threadID = get_TID();

  /* information about waiting and order */
  bool waiting = true;
  bool isDependent = false;
  uint64_t depIdx = 0;

  for (uint64_t i = 0; i < numDependecies; ++i) {
    if (deviceArr[i][0] == addr) {
      depIdx = i;
      isDependent = true;
      break;
    }
  }
  if (!isDependent) {
    return;
  }



  while (waiting) {
    uint64_t numThreadNext = deviceArr[depIdx][2];
    if (numThreadNext == deviceArr[depIdx][1]) {
      break;
    }
    uint64_t indexOfNextThread = NUM_METADATA + 3 * numThreadNext;
    if (deviceArr[depIdx][indexOfNextThread] == threadID && lock()) {
      /* START OF WRITING OR READING */
      uint64_t is_load = deviceArr[depIdx][indexOfNextThread + 1];
      uint32_t size = op_info & 0xf;
      uint32_t value_low =
          deviceArr[depIdx][indexOfNextThread + 2] & 0xffffffff;
      uint32_t value_high =
          (deviceArr[depIdx][indexOfNextThread + 2] >> 32) & 0xffffffff;

      if (!is_load) {
        nvbit_write_reg(target_reg_low, value_low);
        if (op_info > 4 && target_reg_low != 255) {
          nvbit_write_reg(target_reg_high, value_high);
        }
      } else {
        void *ptr = (void *)addr;
        if (size > 4) { 
          *(uint64_t *)(ptr) =
              ((uint64_t)value_high << 32) | (uint64_t)value_low;
        }
        else if (size == 4) {
          *(uint32_t *)(ptr) = (uint32_t)value_low;
        } else if (size == 2) {
          *(uint16_t *)(ptr) = (uint16_t)value_low;
        } else {
          *(uint8_t *)(ptr) = (uint8_t)value_low;
        }
      }

      /* END OF WRITING OR READING */
      deviceArr[depIdx][2] += 1;
      waiting = false;
      atomicInc(&resolved, 4294967295);
      unlock();
    }
  }
}
NVBIT_EXPORT_FUNC(mem_replay);
