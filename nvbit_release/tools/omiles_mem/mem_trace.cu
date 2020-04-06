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
 #include <stdint.h>
 #include <stdio.h>
 #include <unistd.h>
 #include <string>
 #include <map>
 
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
 
 /* global control variables for this tool */
 uint32_t instr_begin_interval = 0;
 uint32_t instr_end_interval = UINT32_MAX;
 int verbose = 0;
 
 /* opcode to id map and reverse map  */
 std::map<std::string, int> opcode_to_id_map;
 std::map<int, std::string> id_to_opcode_map;
 
 /* information collected in the instrumentation function */
 typedef struct {
     int cta_id_x;
     int cta_id_y;
     int cta_id_z;
     int warp_id;
     int opcode_id;
     uint64_t addrs[32];
 } mem_access_t;
 
 union Data {
     double d;
     uint64_t u64;
     uint32_t u32[2];
     int64_t i64;
     int32_t i32[2];
 };
 
 typedef struct {
     clock_t time;
     //bit 31 is type (0 is shared 1 is global)
     //bit 30 is load (0 is store 1 is load)
     //bits 29-0 are thread id
     //would like to record memory operation size and embed as well - probably need 2 bits but maybe 1
     uint32_t type_load_tid;
     uint64_t addr[32];
     Data value[32];
 } record_data;
 
 /* Instrumentation function that we want to inject, please note the use of
  * 1. extern "C" __device__ __noinline__
  *    To prevent "dead"-code elimination by the compiler.
  * 2. NVBIT_EXPORT_FUNC(dev_func)
  *    To notify nvbit the name of the function we want to inject.
  *    This name must match exactly the function name.
  */
 extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                        uint32_t reg_high,
                                                        uint32_t reg_low,
                                                        int32_t imm) {
     if (!pred) {
         return;
     }
 
     int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
     uint64_t addr = base_addr + imm;
 
     int active_mask = __ballot(1);
     const int laneid = get_laneid();
     const int first_laneid = __ffs(active_mask) - 1;
 
     mem_access_t ma;
     /* collect memory address information */
     for (int i = 0; i < 32; i++) {
         ma.addrs[i] = __shfl(addr, i);
     }
 
     int4 cta = get_ctaid();
     ma.cta_id_x = cta.x;
     ma.cta_id_y = cta.y;
     ma.cta_id_z = cta.z;
     ma.warp_id = get_warpid();
     ma.opcode_id = opcode_id;
 
     /* first active lane pushes information on the channel */
     if (first_laneid == laneid) {
         channel_dev.push(&ma, sizeof(mem_access_t));
     }
 }
 NVBIT_EXPORT_FUNC(instrument_mem);
 
 extern "C" __device__ __noinline__ void mem_record(int pred, uint32_t op_type,
    uint32_t reg_high,
    uint32_t target_reg_high,
    uint32_t reg_low,
    uint32_t target_reg_low,
    int32_t imm) {
        
    if (!pred) {
        return;
    }
    int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
    uint64_t addr = base_addr + imm;
    uint32_t volatile load = op_type << 1;
    uint64_t val = load ? (uint64_t) (*(uint64_t *)(addr)) : (((uint64_t)target_reg_high) << 32) | ((uint64_t)target_reg_low);

    uint32_t blockID = blockIdx.x + blockIdx.y * gridDim.x 
        + gridDim.x * gridDim.y * blockIdx.z;
    uint32_t threadID = blockID * (blockDim.x * blockDim.y * blockDim.z) 
        + (threadIdx.z * (blockDim.x * blockDim.y)) 
        + (threadIdx.y * blockDim.x) + threadIdx.x;

    int active_mask = __ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    record_data rd;
    for (int i = 0; i < 32; i++) {
        rd.addr[i] = __shfl(addr, i);
        rd.value[i].u64 = __shfl(val, i);
    }
 
    clock_t system_time = clock();
 
    if (first_laneid == laneid) { 
        rd.time = system_time;
        rd.type_load_tid = op_type | threadID;
        channel_dev.push(&rd, sizeof(record_data));
        /*
        printf("Time: %d\n", (uint32_t)rd.time);
        printf("TID: %x\n", rd.type_load_tid & 0x3fffffff);
        for (int i = 0; i < 32; ++i) {
            printf("%d|%llx\t", i, rd.addr[i]);
            printf("%d|%f\n", i, rd.value[i].d);
        }
        */
    }
 }
 NVBIT_EXPORT_FUNC(mem_record);
 
 void nvbit_at_init() {
     setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
     GET_VAR_INT(
         instr_begin_interval, "INSTR_BEGIN", 0,
         "Beginning of the instruction interval where to apply instrumentation");
     GET_VAR_INT(
         instr_end_interval, "INSTR_END", UINT32_MAX,
         "End of the instruction interval where to apply instrumentation");
     GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
     std::string pad(100, '-');
     printf("%s\n", pad.c_str());
 }
 
 /* instrument each memory instruction adding a call to the above instrumentation
  * function */
 void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {
     const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
     if (verbose) {
         printf("Inspecting function %s at address 0x%lx\n",
                nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
     }
 
     uint32_t cnt = 0;
     /* iterate on all the static instructions in the function */
     for (auto instr : instrs) {
         if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
             instr->getMemOpType() == Instr::NONE) {
             cnt++;
             continue;
         }
         if (verbose) {
             instr->printDecoded();
         }
 
         if (instr->getMemOpType() == Instr::memOpType::SHARED || instr->getMemOpType() == Instr::memOpType::GLOBAL) {
             uint32_t op_type = 0;
             //00 is shared store
             //01 is shared load
             //10 is global store
             //11 is global load
             if (instr->getMemOpType() == Instr::memOpType::GLOBAL) op_type = 2;
             if (instr->isLoad()) {
                 nvbit_insert_call(instr, "mem_record", IPOINT_BEFORE);
                 op_type |= 1;
             }
             else {
                 nvbit_insert_call(instr, "mem_record", IPOINT_BEFORE);
             } 
             op_type <<= 30;
             nvbit_add_call_arg_pred_val(instr);
             nvbit_add_call_arg_const_val32(instr, op_type);
 
             const Instr::operand_t* op0 = instr->getOperand(0);
             const Instr::operand_t* op1 = instr->getOperand(1);
             const Instr::operand_t* temp = op0;
             if (op0->type != Instr::MREF) {
                 op0 = op1;
                 op1 = temp;
             }
             //reg high / target high
             if (instr->isExtended()) {
                 nvbit_add_call_arg_reg_val(instr, (int)op0->value[0] + 1);
                 nvbit_add_call_arg_reg_val(instr, (int)op1->value[0] + 1);
             }
             else {
                 nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
                 nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
             }
             //reg low
             nvbit_add_call_arg_reg_val(instr, (int)op0->value[0]);
             //target low
             nvbit_add_call_arg_reg_val(instr, (int)op1->value[0]);
             //immediate
             nvbit_add_call_arg_const_val32(instr, (int)op0->value[1]);
             
         }
         cnt++;
     }
 }
 
 __global__ void flush_channel() {
     /* push memory access with negative cta id to communicate the kernel is
      * completed */
    record_data rd;
    rd.type_load_tid = UINT32_MAX;
    channel_dev.push(&rd, sizeof(record_data));
 
     /* flush channel */
     channel_dev.flush();
 }
 
 void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                          const char *name, void *params, CUresult *pStatus) {
     if (skip_flag) return;
 
     if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
         cbid == API_CUDA_cuLaunchKernel) {
         cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
 
         if (!is_exit) {
             int nregs;
             CUDA_SAFECALL(
                 cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));
 
             int shmem_static_nbytes;
             CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
                                           CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                           p->f));
 
             printf(
                 "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                 "%d - shmem %d - cuda stream id %ld\n",
                 nvbit_get_func_name(ctx, p->f), p->gridDimX, p->gridDimY,
                 p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
                 shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
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
         }
     }
 }
 
void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    printf("Output Stuff: time addr tid store/load shared/global value\n");
    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving &&
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                record_data *rd =
                    (record_data *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                */
                if (rd->type_load_tid == UINT32_MAX) {
                    recv_thread_receiving = false;
                    break;
                }
                //likely need something to indicate type for output
                uint32_t time = (uint32_t) rd->time;
                uint32_t base_tid = rd->type_load_tid & 0x3fffffff;
                char type = (rd->type_load_tid & (uint32_t)(1<<31)) ? 'G' : 'S';
                char load = (rd->type_load_tid & (1<<30)) ? 'L' : 'S';
                for (int i = 0; i < 32; ++i) {
                    printf("%u %lu %d %c %c %f\n", time, rd->addr[i], base_tid + i, load, type, rd->value[i].d);
                }
                num_processed_bytes += sizeof(record_data);
            }
        }
    }
    
    free(recv_buffer);
    return NULL;
}
 
 void nvbit_at_ctx_init(CUcontext ctx) {
     recv_thread_started = true;
     channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
     pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
 }
 
 void nvbit_at_ctx_term(CUcontext ctx) {
     if (recv_thread_started) {
         recv_thread_started = false;
         pthread_join(recv_thread, NULL);
     }
 }
 