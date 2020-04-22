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
#include <functional>
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

#include "channel_functions.hpp"
#include "global_variables.hpp"
#include "host_nvbit_funcs.hpp"
#include "instrumentation_funcs.hpp"

std::string getOpcodeBase(Instr *instr) {
  std::string opcode = instr->getOpcode();
  std::size_t first_dot = opcode.find(".");
  if (first_dot != std::string::npos) {
    opcode = opcode.erase(first_dot);
  }
  return opcode;
}

bool isInstrOfInterest(Instr *instr) {
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

uint64_t getFunctionHash(CUcontext ctx, cuLaunchKernel_params *p) {
  std::string func_name(nvbit_get_func_name(ctx, p->f));
  std::hash<std::string> func_hasher;
  return func_hasher(func_name);
}

/* Function used to insert mem_record before every memory instruction
 * Used in nvbit_at_function_first_load */
void addRecordInstrumentation(CUcontext &ctx, CUfunction &f) {
  const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
  uint32_t cnt = 0;
  /* iterate on all the static instructions in the function */
  for (auto instr : instrs) {
    if (verbose) {
      //instr->printDecoded();
      instr->print();
    }
    if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
        !isInstrOfInterest(instr)) {
      cnt++;
      continue;
    }

    if (instr->getMemOpType() == Instr::memOpType::SHARED ||
        instr->getMemOpType() == Instr::memOpType::GLOBAL) {

      /* Instrument loads before and after. Instrument stores before. */
      for (int i = 0; i < 2; ++i) {
        uint32_t op_type = 0;
        /* 00 is shared store
         * 01 is shared load
         * 10 is global store
         * 11 is global load */
        if (instr->getMemOpType() == Instr::memOpType::GLOBAL)
          op_type = 2;
        if (instr->isLoad()) {
          nvbit_insert_call(instr, "mem_record", (ipoint_t)i);
          op_type |= 1;
        } else if (i == 0) { /* if i = 0, then we are instrumenting before */
          nvbit_insert_call(instr, "mem_record", (ipoint_t)i);
        }
        op_type <<= 30;
        if (instr->isExtended()) {
          op_type |= 0x1;
        }
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
          nvbit_add_call_arg_const_val32(instr, (int)op1->value[0] + 1);
        } else {
          nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
          nvbit_add_call_arg_const_val32(instr, (int)Instr::RZ);
        }
        /* reg low */
        nvbit_add_call_arg_reg_val(instr, (int)op0->value[0]);
        /* target low */
        nvbit_add_call_arg_const_val32(instr, (int)op1->value[0]);
        /* immediate */
        nvbit_add_call_arg_const_val32(instr, (int)op0->value[1]);
        nvbit_add_call_arg_const_val32(instr, cnt);
      }

    } else if (instr->getMemOpType() == Instr::memOpType::NONE) {
      // nvbit_insert_call(instr, "sync_record", IPOINT_BEFORE);
      // nvbit_add_call_arg_pred_val(instr);

      // uint32_t optype = sync_instrs_to_id.at(getOpcodeBase(instr)) << 28;

      // nvbit_add_call_arg_const_val32(instr, optype);
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
    // instr->print();
    if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
        !isInstrOfInterest(instr)) {
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

      uint32_t is_extended = instr->isExtended();
      nvbit_add_call_arg_const_val32(instr, is_extended);

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
        nvbit_add_call_arg_const_val32(instr, (int)op1->value[0] + 1);
      } else {
        nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
        nvbit_add_call_arg_const_val32(instr, (int)Instr::RZ);
      }
      /* reg low */
      nvbit_add_call_arg_reg_val(instr, (int)op0->value[0]);
      /* target low */
      nvbit_add_call_arg_const_val32(instr, (int)op1->value[0]);
      /* immediate */
      nvbit_add_call_arg_const_val32(instr, (int)op0->value[1]);
      nvbit_add_call_arg_const_val32(instr, cnt);

      nvbit_remove_orig(instr);

    } else if (instr->getMemOpType() == Instr::memOpType::NONE) {
    }
    cnt++;
  }
}

/* Function used to handle the beginning and end of kernel launches during the
 * record phase Used in nvbit_at_cuda_event */
void handleRecordKernelEvent(CUcontext &ctx, int is_exit, const char *name,
                             cuLaunchKernel_params *p) {
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

    uint64_t file_prefix = getFunctionHash(ctx, p);
    if (replay_files.find(file_prefix) == replay_files.end()) {
      replay_files[file_prefix] = 0;
    }
    int file_suffix = replay_files[file_prefix]++;
    std::string file_name = "record_output/" + std::to_string(file_prefix) +
                            "_" + std::to_string(file_suffix) + ".record";
    fptr = fopen(file_name.c_str(), "w");

    /* wait here until the receiving thread has not finished with the
     * current kernel */
    while (recv_thread_receiving) {
      pthread_yield();
    }

    /* Write data at the conclusion of the function */
    for (size_t i = 0; i < accesses.size(); ++i) {
      record_data rd = accesses[i];
      uint32_t time = (uint32_t)rd.time;
      if (rd.is_mem_instr) {
        // uint32_t base_tid = rd.type_load_tid & 0x3fffffff;
        uint32_t tid = rd.type_load_tid & 0x3fffffff;
        char type = (rd.type_load_tid & (uint32_t)(1 << 31)) ? 'G' : 'S';
        char load = (rd.type_load_tid & (1 << 30)) ? 'L' : 'S';
        fprintf(fptr, "%u %lu %d %c %c %p\n", time, rd.addr, tid, load, type,
                rd.value.u64);
      } else {
        // TODO: Add in when ready
        // uint32_t tid = rd.type_load_tid & 0x0fffffff;
        // std::string opcode = id_to_sync_instrs.at(rd.type_load_tid >> 28);
        // fprintf(fptr, "SYNC! %u %d %s\n", time, base_tid, opcode.c_str());
      }
    }
    fclose(fptr);

    /* Clear the vector so that information is not written twice */
    accesses.clear();
  }
}

/* Function used to handle the beginning and end of kernel launches during the
 * record phase Used in nvbit_at_cuda_event */
void handleReplayKernelEvent(CUcontext &ctx, int is_exit, const char *name,
                             cuLaunchKernel_params *params) {
  if (!is_exit) {
    uint64_t file_prefix = getFunctionHash(ctx, params);
    if (replay_files.find(file_prefix) == replay_files.end()) {
      replay_files[file_prefix] = 0;
    }
    int file_suffix = replay_files[file_prefix];
    replay_files[file_prefix]++;

    std::string filename = "dependency_output/" + std::to_string(file_prefix) +
                           "_" + std::to_string(file_suffix) + ".dependencies";
    fptr = fopen(filename.c_str(), "r");

    // Create host array of device pointers
    fscanf(fptr, "%lu", &numDependecies);
    uint64_t **hostArr = new uint64_t *[numDependecies];

    for (uint64_t i = 0; i < numDependecies; ++i) {
      uint64_t addr, num_threads;
      fscanf(fptr, "%lu %lu", &addr, &num_threads);
      uint64_t numSpots = 3 * num_threads + NUM_METADATA;
      uint64_t *subArray = new uint64_t[numSpots];
      subArray[0] = addr;
      subArray[1] = num_threads;
      subArray[2] = 0;

      uint64_t curr_thread;
      char load_or_store;
      uint64_t value;
      for (uint64_t j = 0; j < 3*num_threads; j += 3) {
        fscanf(fptr, "%lu %s %llx", &curr_thread, &load_or_store, &value);
        subArray[NUM_METADATA + j] = curr_thread;
        if (load_or_store == 'L') {
          subArray[NUM_METADATA + j + 1] = 1;
        } else {
          subArray[NUM_METADATA + j + 1] = 0;
        }

        subArray[NUM_METADATA + j + 2] = value;
      }

      CUDA_SAFECALL(
          cudaMalloc((void **)&hostArr[i], numSpots * sizeof(uint64_t)));
      CUDA_SAFECALL(cudaMemcpy(hostArr[i], subArray,
                               numSpots * sizeof(uint64_t),
                               cudaMemcpyHostToDevice));

      delete[] subArray;
    }

    // Copy to a device array of device pointers
    CUDA_SAFECALL(cudaMalloc(&deviceArr, numDependecies * sizeof(uint64_t *)));

    CUDA_SAFECALL(cudaMemcpy(deviceArr, hostArr, numDependecies * sizeof(uint64_t *),
                             cudaMemcpyHostToDevice));

    delete[] hostArr;

    fclose(fptr);
  } else {
    /* make sure current kernel is completed */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    CUDA_SAFECALL(cudaFree(deviceArr));
    assert(cudaGetLastError() == cudaSuccess);
  }
}
