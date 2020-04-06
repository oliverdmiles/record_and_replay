/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <map>
#include <queue>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* Enum used to indicate which mode is currently being run */
enum recordReplayPhase { RECORD, REPLAY };

/* Constant set used to determine if an instruction is a synchronization
 * operation */
const std::map<std::string, uint32_t> sync_instrs_to_id = {
    {"BAR", 0}, {"MEMBAR", 1}, {"ATOM", 2},  {"BARRIER", 3},   {"FENCE", 4},
    {"RED", 5}, {"VOTE", 6},   {"MATCH", 7}, {"ACTIVEMASK", 8}};

const std::map<uint32_t, std::string> id_to_sync_instrs = {
    {0, "BAR"}, {1, "MEMBAR"}, {2, "ATOM"},  {3, "BARRIER"},   {4, "FENCE"},
    {5, "RED"}, {6, "VOTE"},   {7, "MATCH"}, {8, "ACTIVEMASK"}};

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int phase = 0;
std::string filename = "recorded_data.txt";
FILE *fptr;

/* information collected in the instrumentation function */
union Data {
  double d;
  uint64_t u64;
  uint32_t u32[2];
  int64_t i64;
  int32_t i32[2];
};

typedef struct {
  bool is_mem_instr;
  uint32_t time;
  /* bit 31 is type (0 is shared 1 is global)
   * bit 30 is load (0 is store 1 is load)
   * bits 29-0 are thread id
   * would like to record memory operation size and embed as well - probably
   * need 2 bits but maybe 1 */
  uint32_t type_load_tid;
  uint64_t addr[32];
  Data value[32];
} record_data;

/* vector to store all accesses for a single kernel call  */
std::vector<record_data> accesses;

/* array to access on memory */
__managed__ int *deviceArr;
__managed__ int start = 0;

__device__ uint32_t count = 0;
__device__ int mutex = 0;

__device__ void lock(int threadID) {
  while (0 != (atomicCAS(&mutex, 0, 1))) {}
}

__device__ void unlock(int threadID) {
  atomicExch(&mutex, 0);
}

__device__ uint32_t get_TID() {
  uint32_t blockID =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  return blockID * (blockDim.x * blockDim.y * blockDim.z) +
                      (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;
}

/* Instrumentation function used to log memory accesses*/
extern "C" __device__ __noinline__ void
mem_record(int pred, uint32_t op_type, uint32_t reg_high,
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
}
NVBIT_EXPORT_FUNC(mem_record);

extern "C" __device__ __noinline__ void sync_record(int pred,
                                                    uint32_t op_type) {

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
}
NVBIT_EXPORT_FUNC(sync_record);

extern "C" __device__ __noinline__ void mem_replay(int pred) {
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
    printf("Looking at thread %d\n", threadID);
    if (deviceArr[start + 1] != loc || false) {}

    if (deviceArr[start + 2] == threadID) {
      while (deviceArr[start + 1] > 1) {
        unlock(threadID);
        lock(threadID);
      }
      printf("The last thread id: %d\n", threadID);
      start += 3;

    } else {
      printf("Not the last thread id: %d\n", threadID);
      deviceArr[start + 1] -= 1;
    }

    unlock(threadID);
  }
}
NVBIT_EXPORT_FUNC(mem_replay);

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(
      phase, "RECORD_REPLAY_PHASE", 0,
      "if unset, executes record phase. Otherwise, executes replay phase");
  GET_VAR_STR(filename, "RECORD_FILE",
              "Name of the file where record output will be written");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());
}

std::string getOpcodeBase(Instr *instr) {
  std::string opcode = instr->getOpcode();
  std::size_t first_dot = opcode.find(".");
  if (first_dot != std::string::npos) {
    opcode = opcode.erase(first_dot);
  }
  return opcode;
}

bool isInstrOfInterst(Instr *instr) {
  /* Memory instructions */
  if (instr->getMemOpType() != Instr::NONE) {
    return true;
  }

  /* Synchronization operations */
  std::string opcode = getOpcodeBase(instr);
  if (sync_instrs_to_id.find(opcode) != sync_instrs_to_id.end()) {
    return true;
  }

  return false;
}

/* Function used to insert mem_record before every memory instruction
 * Used in nvbit_at_function_first_load */
void addRecordInstrumentation(CUcontext &ctx, CUfunction &f) {
  const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
  uint32_t cnt = 0;
  /* iterate on all the static instructions in the function */
  for (auto instr : instrs) {
    printf("%s\n", instr->getOpcode());
    if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
        !isInstrOfInterst(instr)) {
      cnt++;
      continue;
    }
    if (verbose) {
      instr->printDecoded();
    }

    if (instr->getMemOpType() == Instr::memOpType::SHARED ||
        instr->getMemOpType() == Instr::memOpType::GLOBAL) {
      uint32_t op_type = 0;
      /* 00 is shared store
       * 01 is shared load
       * 10 is global store
       * 11 is global load */
      if (instr->getMemOpType() == Instr::memOpType::GLOBAL)
        op_type = 2;
      if (instr->isLoad()) {
        nvbit_insert_call(instr, "mem_record", IPOINT_BEFORE);
        op_type |= 1;
      } else {
        nvbit_insert_call(instr, "mem_record", IPOINT_BEFORE);
      }
      op_type <<= 30;
      nvbit_add_call_arg_pred_val(instr);
      nvbit_add_call_arg_const_val32(instr, op_type);

      const Instr::operand_t *op0 = instr->getOperand(0);
      const Instr::operand_t *op1 = instr->getOperand(1);
      const Instr::operand_t *temp = op0;
      if (op0->type != Instr::MREF) {
        op0 = op1;
        op1 = temp;
      }
      /* reg high / target high */
      if (instr->isExtended()) {
        nvbit_add_call_arg_reg_val(instr, (int)op0->value[0] + 1);
        nvbit_add_call_arg_reg_val(instr, (int)op1->value[0] + 1);
      } else {
        nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
        nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
      }
      /* reg low */
      nvbit_add_call_arg_reg_val(instr, (int)op0->value[0]);
      /* target low */
      nvbit_add_call_arg_reg_val(instr, (int)op1->value[0]);
      /* immediate */
      nvbit_add_call_arg_const_val32(instr, (int)op0->value[1]);
    } else if (instr->getMemOpType() == Instr::memOpType::NONE) {
      nvbit_insert_call(instr, "sync_record", IPOINT_BEFORE);
      nvbit_add_call_arg_pred_val(instr);

      uint32_t optype = sync_instrs_to_id.at(getOpcodeBase(instr)) << 28;

      nvbit_add_call_arg_const_val32(instr, optype);
    }
    cnt++;
  }
}

/* Function used to insert mem_record before every memory instruction
 * Used in nvbit_at_function_first_load */
void addReplayInstrumentation(CUcontext &ctx, CUfunction &f) {
  const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
  uint32_t cnt = 0;
  /* iterate on all the static instructions in the function */
  for (auto instr : instrs) {
    if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
        !isInstrOfInterst(instr)) {
      cnt++;
      continue;
    }
    if (verbose) {
      instr->printDecoded();
    }

    if (instr->getMemOpType() == Instr::memOpType::SHARED ||
        instr->getMemOpType() == Instr::memOpType::GLOBAL) {
      nvbit_insert_call(instr, "mem_replay", IPOINT_BEFORE);
      nvbit_add_call_arg_pred_val(instr);
    } else if (instr->getMemOpType() == Instr::memOpType::NONE) {
    }
  }
}

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {
  if (verbose) {
    printf("Inspecting function %s at address 0x%lx\n",
           nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
  }

  if (phase == recordReplayPhase::RECORD) {
    addRecordInstrumentation(ctx, f);
  } else {
    addReplayInstrumentation(ctx, f);
  }
}

/* Function used to indicate that there are no more values being added to the
 * channel */
__global__ void flush_channel() {
  /* push memory access with negative clock value to communicate the kernel is
   * completed */
  record_data rd;
  rd.time = UINT32_MAX;
  channel_dev.push(&rd, sizeof(record_data));

  /* flush channel */
  channel_dev.flush();
}

/* Function used to handle the beginning and end of kernel launches during the
 * record phase Used in nvbit_at_cuda_event */
void handleRecordKernelEvent(CUcontext &ctx, int is_exit, nvbit_api_cuda_t cbid,
                             const char *name, void *params,
                             CUresult *pStatus) {
  cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

  if (!is_exit) {
    recv_thread_receiving = true;

  } else {
    /* make sure current kernel is completed */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    /* make sure we prevent re-entry on the nvbit_callback when issuing
     * the flush_channel kernel */
    skip_flag = true;

    /* issue flush of channel so we are sure all the memory accesses
     * have been pushed */
    flush_channel<<<1, 1>>>();
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    /* unset the skip flag */
    skip_flag = false;

    /* wait here until the receiving thread has not finished with the
     * current kernel */
    while (recv_thread_receiving) {
      pthread_yield();
    }

    /* Write data at the conclusion of the function */
    fprintf(fptr, "%s\n", nvbit_get_func_name(ctx, p->f));
    for (size_t i = 0; i < accesses.size(); ++i) {
      record_data rd = accesses[i];
      uint32_t time = (uint32_t)rd.time;
      if (rd.is_mem_instr) {
        uint32_t base_tid = rd.type_load_tid & 0x3fffffff;
        char type = (rd.type_load_tid & (uint32_t)(1 << 31)) ? 'G' : 'S';
        char load = (rd.type_load_tid & (1 << 30)) ? 'L' : 'S';
        for (int i = 0; i < 32; ++i) {
          fprintf(fptr, "%u %lu %d %c %c %f\n", time, rd.addr[i], base_tid + i,
                  load, type, rd.value[i].d);
        }
      } else {
        uint32_t base_tid = rd.type_load_tid & 0x0fffffff;
        std::string opcode = id_to_sync_instrs.at(rd.type_load_tid >> 28);
        fprintf(fptr, "SYNC! %u %d %s\n", time, base_tid, opcode.c_str());
      }
    }

    /* Clear the vector so that information is not written twice */
    accesses.clear();
  }
}

/* Function used to handle the beginning and end of kernel launches during the
 * record phase Used in nvbit_at_cuda_event */
void handleReplayKernelEvent(int is_exit) {
  if (is_exit) {
    /* make sure current kernel is completed */
    cudaDeviceSynchronize();
  }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {

    if (phase == recordReplayPhase::RECORD) {
      handleRecordKernelEvent(ctx, is_exit, cbid, name, params, pStatus);
    } else {
      handleReplayKernelEvent(is_exit);
    }
  }
}

/* Function used to receive data from the device on the host */
void *recv_thread_fun(void *) {
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;
    if (recv_thread_receiving &&
        (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        record_data *rd = (record_data *)&recv_buffer[num_processed_bytes];

        /* when we get this cta_id_x it means the kernel has completed
         */
        if (rd->time == UINT32_MAX) {
          recv_thread_receiving = false;
          break;
        }

        accesses.push_back(*rd);
        num_processed_bytes += sizeof(record_data);
      }
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
  if (phase == recordReplayPhase::RECORD) {
    printf("Recording...\n");
    fptr = fopen(filename.c_str(), "w");
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
  } else {
    printf("Replaying...\n");
    // fptr = fopen(filename.c_str(), "r");
    fptr = fopen("tester_output.txt", "r");

    int Nrows = 3;
    int Ncols;
    fscanf(fptr, "%d", &Ncols);
    int* hostArr = new int[Ncols*Nrows];

    cudaError_t mallocErr =
        cudaMalloc((void **)&deviceArr, (Nrows * Ncols) * sizeof(int));
    if (cudaSuccess != mallocErr) {
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,
              __LINE__, cudaGetErrorString(mallocErr));
      exit(EXIT_FAILURE);
    }

    for (int i = 0; i < Ncols; ++i) {
      int addr, num_threads_total, last_thread;
      fscanf(fptr, "%d %d %d", &addr, &num_threads_total, &last_thread);
      hostArr[i] = addr;
      hostArr[i + 1] = num_threads_total;
      hostArr[i + 2] = last_thread;
    }

    cudaError_t cpyErr = cudaMemcpy(deviceArr, hostArr, (Nrows * Ncols) * sizeof(int),
                                    cudaMemcpyHostToDevice);
    if (cudaSuccess != cpyErr) {
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,
              __LINE__, cudaGetErrorString(cpyErr));
      exit(EXIT_FAILURE);
    }

    delete[] hostArr;

    fclose(fptr);
  }
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (phase == recordReplayPhase::RECORD) {
    if (recv_thread_started) {
      recv_thread_started = false;
      pthread_join(recv_thread, NULL);
    }
    fclose(fptr);
    printf("Recording complete!\n");
  } else {
    cudaFree(deviceArr);
    printf("Replaying complete\n");
  }
}
