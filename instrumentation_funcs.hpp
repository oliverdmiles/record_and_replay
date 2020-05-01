#include "global_variables.hpp"
#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

__device__ bool lock() { return 0 == (atomicCAS(&mutex, 0, 1)); }

__device__ void unlock() { atomicExch(&mutex, 0); }

__device__ bool lock2(uint32_t * dep_lock, uint32_t my_tid) { return my_tid == (atomicCAS(dep_lock, my_tid, 0xffffffff)); }

__device__ void unlock2(uint32_t * dep_lock, uint32_t next_tid) { atomicExch(dep_lock, next_tid); }

__device__ uint32_t get_TID() {
  uint32_t blockID =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  return blockID * (blockDim.x * blockDim.y * blockDim.z) +
         (threadIdx.z * (blockDim.x * blockDim.y)) +
         (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__ uint64_t replicate_load(uint64_t addr, uint32_t size, uint32_t target_reg_high, uint32_t target_reg_low) {
  uint64_t value = 0;
  if (target_reg_low != 255) {
    if (size > 4) {
      uint64_t insert_full = *((uint64_t *)(addr));
      uint32_t insert_temp = insert_full & 0xffffffff;
      nvbit_write_reg(target_reg_low, insert_temp);
      insert_temp = (insert_full >> 32) & 0xffffffff;
      nvbit_write_reg(target_reg_high, insert_temp);
      value = insert_full;
    }
    else if (size == 4) {
      uint32_t insert_low = *((uint32_t *)(addr));
      nvbit_write_reg(target_reg_low, insert_low);
      value = (uint64_t)insert_low;
    }
    else if (size == 2) {
      uint32_t insert_low = 0;
      uint16_t insert_temp = *((uint16_t *)(addr));
      insert_low = ((uint32_t)(insert_temp)) | insert_low;
      nvbit_write_reg(target_reg_low, insert_low);
      value = (uint64_t)insert_temp;
    }
    else {
      uint32_t insert_low = 0;
      uint8_t insert_temp = *((uint8_t *)(addr));
      insert_low = ((uint32_t)(insert_temp)) | insert_low;
      nvbit_write_reg(target_reg_low, insert_low);
      value = (uint64_t)insert_temp;
    }
  }
  return atomicInc(&count, 4294967295);
}

__device__ uint64_t replicate_store(uint64_t addr, uint32_t size, uint32_t target_reg_high, uint32_t target_reg_low) {
  uint64_t value = 0;
  if (size > 4) {
    if (target_reg_low == 255) target_reg_high = 255;
    value = (((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_high)))) << 32) |
              ((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_low))));
    *(uint64_t *)(addr) = value;
  } else if (size == 4) {
    *(uint32_t *)(addr) = ((uint32_t)(nvbit_read_reg(target_reg_low)));
    value = (uint64_t)((uint32_t)(nvbit_read_reg(target_reg_low)));
  } else if (size == 2) {
    uint32_t insert_temp = ((uint32_t)(nvbit_read_reg(target_reg_low)));
    uint16_t insert_low = insert_temp & 0xffff;
    *(uint16_t *)(addr) = (uint16_t)insert_low;
    value = (uint64_t)(insert_low);
  } else {
    uint32_t insert_temp = ((uint32_t)(nvbit_read_reg(target_reg_low)));
    uint8_t insert_low = insert_temp & 0xff;
    *(uint8_t *)(addr) = (uint8_t)insert_low;
    value = (uint64_t)(insert_low);
  }
  return atomicInc(&count, 4294967295);
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

  uint32_t size = (op_type >> 28) & 3;
  uint64_t val = 0;
  
  record_data rd;
  //replicate and get unique id for instruction
  if (op_type & 0x40000000)
    rd.time = replicate_load(addr, 1 << size, target_reg_high, target_reg_low);
  else 
    rd.time = replicate_store(addr, 1 << size, target_reg_high, target_reg_low);

  //this section is bad but couldn't get the functions to return the right value 
  if ((op_type & 0x40000000) && target_reg_low != 255) {
    if (size == 3)
      val = ((uint64_t)*(uint64_t *)(addr));
    else if (size == 2)
      val = ((uint64_t)*(uint32_t *)(addr));
    else if (size == 1)
      val = ((uint64_t)*(uint16_t *)(addr));
    else
      val = ((uint64_t)*(uint8_t *)(addr));
  }
  else if (target_reg_low != 255) {
    val = (((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_high)))) << 32) |
                   ((uint64_t)((uint32_t)(nvbit_read_reg(target_reg_low))));
    if (size == 2)
      val &= 0xFFFFFFFF;
    else if (size == 1)
      val &= 0xFFFF;
    else if (size == 0)
      val &= 0xFF;
  }
  
  uint32_t is_extended = op_type & 0x1;
  uint32_t threadID = get_TID();
  val = is_extended ? val : val & 0xffffffff;

  rd.addr = addr;
  rd.value.u64 = val;
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

  uint32_t my_threadID = get_TID();

  /* information about waiting and order */
  bool waiting = true;
  bool isDependent = false;
  uint32_t depIdx = 0;

  for (uint64_t i = 0; i < numDependecies; ++i) {
    if (deviceArr[i][0] == addr) {
      depIdx = i;
      isDependent = true;
      break;
    }
  }
  
  uint32_t is_load_and_size = op_info & 0x10;

  if (!isDependent) {
    if (is_load_and_size) {
      replicate_load(addr, op_info & 0xf, target_reg_high, target_reg_low);
    }
    else {
      replicate_store(addr, op_info & 0xf, target_reg_high, target_reg_low);
    }
    return;
  }
  
  uint32_t * const & meta_info = (uint32_t*)&deviceArr[depIdx][1];

  while (waiting) {
    uint32_t numThreadNext = meta_info[0];
    uint32_t nextThread2run = NUM_METADATA + 2 * numThreadNext;
    uint32_t * const & nextThread_info = (uint32_t *) &deviceArr[depIdx][nextThread2run];
  #ifdef MULTILOCK
    if (nextThread_info[0] == my_threadID && lock2(&meta_info[1], my_threadID)) {
        //printf("got lock for dpidx %d with curthread idx %d on addr &meta_info[1] %p\n", depIdx, numThreadNext, &meta_info[1]);
  #else
    if (nextThread_info[0] == my_threadID && lock()) {
  #endif
      /* START OF WRITING OR READING */
      uint32_t value_low = nextThread_info[2];
      uint32_t value_high = nextThread_info[3];

      if (is_load_and_size && target_reg_low != 255) {
        is_load_and_size = op_info & 0xf;
        nvbit_write_reg(target_reg_low, value_low);
        if (is_load_and_size > 4) {
          nvbit_write_reg(target_reg_high, value_high);
        }
      } else {
        is_load_and_size = op_info & 0xf;
        if (is_load_and_size > 4) {
          *(uint64_t *)(addr) =
              ((uint64_t)value_high << 32) | (uint64_t)value_low;
        } else if (is_load_and_size == 4) {
          *(uint32_t *)(addr) = (uint32_t)value_low;
        } else if (is_load_and_size == 2) {
          *(uint16_t *)(addr) = (uint16_t)value_low;
        } else {
          *(uint8_t *)(addr) = (uint8_t)value_low;
        }
      }
      //if (my_threadID == 896 && depIdx == 25) printf("My vh:%x vl:%x\n", value_high, value_low);
      /* END OF WRITING OR READING */
      meta_info[0] += 1;
      waiting = false;
    #ifdef MULTILOCK
      unlock2(&meta_info[1], nextThread_info[4]);
    #else
      unlock();
    #endif
    }
  }
}
NVBIT_EXPORT_FUNC(mem_replay);
