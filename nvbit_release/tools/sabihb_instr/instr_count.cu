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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

#define WARP_SIZE 32

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;

FILE *myfile;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 1;
int count_warp_level = 1;
int exclude_pred_off = 0;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     int count_warp_level,
						     int opType,
						     int load,
						     int store,
						     int r1,
						     int r2,
						     int r3) {
	
    static const char* memOpStr[] = {"NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT"};
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot(1);
    /* compute the predicate mask */
    const int predicate_mask = __ballot(predicate);
    /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    const int lane_id = get_laneid();
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    /* count all the active thread */
    const int num_threads = __popc(predicate_mask);
    
    /* only the first active thread will perform the atomic */
    if (first_laneid == lane_id) {
        if (count_warp_level) {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0) atomicAdd((unsigned long long *)&counter, 1);
        } else {
            atomicAdd((unsigned long long *)&counter, num_threads);
        }
    }
    //int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadID = WARP_SIZE * get_global_warp_id() + get_laneid(); 
    if (load == 1 & threadID < 100) {
        int32_t addr = nvbit_read_reg(r2) + r3;
	unsigned long* value = (unsigned long*) addr;
        //printf("r1=%d r2=%d r3=%d\n", r1, r2, r3);
        //printf("Thread %d %s load [r%i] + %i = %llu to r%i\n", threadID, memOpStr[opType], r2, r3, 100, r1);
    } else if (store == 1 && threadID == 100) {
	//int32_t addr = nvbit_read_reg(r3);
	//int32_t* value = (int32_t*) addr;
	printf("op1: %p\n", nvbit_read_reg(r1));
	printf("op1+1: %p\n", nvbit_read_reg(r1+1));
	printf("op2: %p\n", nvbit_read_reg(r2));
	printf("op2+1: %p\n", nvbit_read_reg(r2+1));
	printf("op3: %p\n", nvbit_read_reg(r3));
	printf("op3+1: %p\n", nvbit_read_reg(r3+1));
	double combined;
	unsigned int first = nvbit_read_reg(r3);
	printf("%p\n", first);
	unsigned long first_long = (unsigned long) first;
	printf("%p\n", first_long);
	unsigned long bytes = first + ((unsigned long)nvbit_read_reg(r3 + 1) << 32);
	memcpy(&combined, &bytes, sizeof(double));
	printf("adding %p and %p\n", first, (unsigned long) nvbit_read_reg(r3+1) << 32);
	printf("combined pointer: %p\n", bytes);
	printf("double: %f\n", combined);
	//printf("Thread %d %s store to r%i + %i = %p value %d (r%d)\n", 
			//threadID, memOpStr[opType], r1, r2, nvbit_read_reg(r1) + r2,
			//*value, r3);
    } //else {
        //printf("ERROR: not load or store\n");
    //}
    //printf("%d %s load:%d, store:%d %d %d %d\n", threadID, memOpStr[opType], load, store, nvbit_read_reg(r1), nvbit_read_reg(r2), nvbit_read_reg(r3));
    //printf("%d %d %d\n", nvbit_read_reg(r1), nvbit_read_reg(r2), nvbit_read_reg(r3));
    //fprintf(myfile, "%llu\n", *(p->dptr));

}
NVBIT_EXPORT_FUNC(count_instrs);

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    myfile = fopen("/home/omiles/recording.txt", "w");
    printf("Opened file to write\n");

    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        ker_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", verbose, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* nvbit_at_function_first_load() is executed every time a function is loaded
 * for the first time. Inside this call-back we typically get the vector of SASS
 * instructions composing the loaded CUfunction. We can iterate on this vector
 * and insert call to instrumentation functions before or after each one of
 * them. */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction func) {
    /* Get the vector of instruction composing the loaded CUFunction "func" */
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);

    /* If verbose we print function name and number of" static" instructions */
    if (verbose) {
        printf("Inspecting %s - num instrs %ld\n",
               nvbit_get_func_name(ctx, func), instrs.size());
    }

    printf("checking for memory instructions...\n");
    
    /* We iterate on the vector of instruction */
    for (auto i : instrs) {
        /* Check if the instruction falls in the interval where we want to
         * instrument */
        if (i->getMemOpType() != Instr::memOpType::NONE) {
	               /* If verbose we print which instruction we are instrumenting (both
             * offset in the function and SASS string) */
            if (verbose == 1) {
                i->print();
		//printf("%s\n",i->getSass());
            } else if (verbose == 2) {
                i->printDecoded();
            }

            /* Insert a call to "count_instrs" before the instruction "i" */
            nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
            if (exclude_pred_off) {
                /* pass predicate value */
                nvbit_add_call_arg_pred_val(i);
            } else {
                /* pass always true */
                nvbit_add_call_arg_const_val32(i, 1);
            }

            /* add count warps option */
            nvbit_add_call_arg_const_val32(i, count_warp_level);

	    // add memory type
	    nvbit_add_call_arg_const_val32(i, i->getMemOpType());
	    nvbit_add_call_arg_const_val32(i, i->isLoad());
	    nvbit_add_call_arg_const_val32(i, i->isStore());

            printf("adding call to count instr at %i\n", i->getIdx());
	    printf("Instruction opcode is %s\n", i->getOpcode());
	    printf("number of operands is %d\n", i->getNumOperands());
	    const Instr::operand_t* op1 = i->getOperand(0);
	    const Instr::operand_t* op2 = i->getOperand(1);
	    const char* type1 = Instr::operandTypeStr[op1->type];
	    const char* type2 = Instr::operandTypeStr[op2->type];
	    printf("operands are %s and %s\n", type1, type2);
	    if (!strcmp(type1, "MREF")) {
		printf("Op1 values: %f, %f\n", op1->value[0], op1->value[1]);
   	        nvbit_add_call_arg_const_val32(i, op1->value[0]);
   	        nvbit_add_call_arg_const_val32(i, op1->value[1]);
		printf("Op2 value: %f\n", op2->value[0]);
   	        nvbit_add_call_arg_const_val32(i, op2->value[0]);
	    } else if (!strcmp(type1, "REG")) {
		printf("Op1 value: %f\n", op1->value[0]);
   	        nvbit_add_call_arg_const_val32(i, op1->value[0]);
		printf("Op2 values: %f, %f\n", op2->value[0], op2->value[1]);
   	        nvbit_add_call_arg_const_val32(i, op2->value[0]);
   	        nvbit_add_call_arg_const_val32(i, op2->value[1]);
	    } else {
		printf("NOT EXPECTED OPERAND TYPE: %s\n", type1);
	    }

        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    printf("Calling at cuda event\n");
    if (is_exit && cbid == API_CUDA_cuMemAlloc_v2) {
	cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
	printf("Saving: %llu\n", *(p->dptr));
	//fprintf(myfile, "%llu\n", *(p->dptr));
    } 
}
