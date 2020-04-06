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

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;
int record = 0;
FILE *myfile;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     int count_warp_level) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot(1);
    /* compute the predicate mask */
    const int predicate_mask = __ballot(predicate);
    /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    const int laneid = get_laneid();
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    /* count all the active thread */
    const int num_threads = __popc(predicate_mask);
    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        int blockID = blockIdx.x + blockIdx.y *gridDim.x 
                        + gridDim.x*gridDim.y*blockIdx.z;
        int threadID = blockID * (blockDim.x * blockDim.y * blockDim.z) 
                        + (threadIdx.z * (blockDim.x * blockDim.y)) 
                        + (threadIdx.y * blockDim.x) + threadIdx.x;
        if (count_warp_level) {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0) atomicAdd((unsigned long long *)&counter, 1);
        } else {
            atomicAdd((unsigned long long *)&counter, num_threads);
        }
    }
}
NVBIT_EXPORT_FUNC(count_instrs);

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    
    if (record) {
        myfile = fopen("/home/omiles/example.txt", "w");
        printf("Hello world\n");
    } else {
        myfile = fopen("/home/omiles/example.txt", "r");
        printf("Hello world\n");
    }

    size_t lim;
    cudaDeviceGetLimit(&lim, cudaLimit::cudaLimitStackSize);
    printf("%d\n", lim);
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

void nvbit_at_term() {
    fclose(myfile);
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
        printf("inspecting %s - num instrs %ld\n",
               nvbit_get_func_name(ctx, func), instrs.size());
    }
    int counter = 0;
    /* We iterate on the vector of instruction */
    for (auto i : instrs) {
        /* Check if the instruction falls in the interval where we want to
         * instrument */
        if (i->getIdx() >= instr_begin_interval &&
            i->getIdx() < instr_end_interval && counter == 0) {
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

    printf("Id: %d\n", cbid);
    if (is_exit) {
        if (cbid == API_CUDA_cuMemAlloc_v2) {

            CUmemLocation oop;
            oop.type = CU_MEM_LOCATION_TYPE_DEVICE;
            oop.id = 0;

            cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;

            unsigned long long f = 0;
            CUresult r3 = cuMemGetAccess(&f, &oop, *(p->dptr));
            printf("Result: %d\n", r3);
            
            // CUdeviceptr buff = 0UL;
            // size_t buff_size = (1UL << 30);

            // CUDAAPI::CUmemLocation oop;
            // oop.type = CUDAAPI::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            // oop.id = 0;

            // CUDAAPI::CUmemAccessDesc halp;
            // halp.location = oop;
            // halp.flags = CUDAAPI::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            // printf("Flags type: %d\n", halp.flags);
            // CUresult r = CUDAAPI::cuMemSetAccess(buff, buff_size, &halp, 1);
            // printf("Result: %d\n", r);

            // unsigned long long f;
            // CUresult r3 = cuMemGetAccess (&f, &oop, buff);
            // printf("Result: %d\n", r3);

            // int device_count;
            // CUresult r2 = CUDAAPI::cuDeviceGetCount(&device_count);
            // printf("Result: %d\n", r2);
            // printf("Device count: %d\n", device_count);

            // cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
            if (record) {
                printf("Saving: %llu\n", *(p->dptr));
                fprintf(myfile, "%llu\n", *(p->dptr));
            } else {
                // fscanf(myfile,"%llu", p->dptr);


                // CUDAAPI::CUmemLocation oop;
                // oop.type = CUDAAPI::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                // oop.id = 0;
    
    
                // printf("Device pointer: %llu\n", *(p->dptr));
                // unsigned long long f = 0;
                // CUresult r3 = cuMemGetAccess(&f, &oop, *(p->dptr));
                // printf("Result: %d\n", r3);




                // CUDAAPI::CUmemLocation oop;
                // oop.type = CUDAAPI::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                // oop.id = 0;


                // CUDAAPI::CUmemAccessDesc halp;
                // halp.location = oop;
                // halp.flags = CUDAAPI::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                // // CUDAAPI::CUmemLocation oop = {CUDAAPI::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE, 1};
                // // CUDAAPI::CUmemAccessDesc halp = {oop, CUDAAPI::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
                // CUresult r = CUDAAPI::cuMemSetAccess(*(p->dptr), p->bytesize, &halp, 1);
                // printf("Result: %d\n", r);
                // printf("Buffer: %d\n", p->bytesize);
            }
            return;
        }
    } else {
        // printf("%d\n", ctx.get_limit());
        if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
            cuMemcpyHtoD_v2_ptds_params *p = (cuMemcpyHtoD_v2_ptds_params *)params;
            printf("Trying to copy: %llu\n", p->dstDevice);

            // CUdeviceptr buff = 0UL;
            // int N = 50000;
            // size_t  size = N * sizeof(float);
            // CUresult r4 = cuMemAddressReserve(&buff, size, size, 0, 0);
            // printf("Result: %d\n", r4);

            // CUmemLocation oop;
            // oop.type = CU_MEM_LOCATION_TYPE_DEVICE;
            // oop.id = 0;

            // unsigned long long f = 0;
            // CUresult r3 = cuMemGetAccess(&f, &oop, buff);
            // printf("Result: %d\n", r3);
        }
    }


    // /* Identify all the possible CUDA launch events */
    // if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
    //     cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
    //     cbid == API_CUDA_cuLaunchKernel) {
    //     /* cast params to cuLaunch_params since if we are here we know these are
    //      * the right parameters type */
    //     cuLaunch_params *p = (cuLaunch_params *)params;

    //     if (!is_exit) {
    //         /* if we are entering in a kernel launch:
    //          * 1. Lock the mutex to prevent multiple kernels to run concurrently
    //          * (overriding the counter) in case the user application does that
    //          * 2. Select if we want to run the instrumented or original
    //          * version of the kernel
    //          * 3. Reset the kernel instruction counter */

    //         pthread_mutex_lock(&mutex);
    //         if (kernel_id >= ker_begin_interval &&
    //             kernel_id < ker_end_interval) {
    //             nvbit_enable_instrumented(ctx, p->f, true);
    //         } else {
    //             nvbit_enable_instrumented(ctx, p->f, false);
    //         }
    //         counter = 0;
    //     } else {
    //         /* if we are exiting a kernel launch:
    //          * 1. Wait until the kernel is completed using
    //          * cudaDeviceSynchronize()
    //          * 2. Get number of thread blocks in the kernel
    //          * 3. Print the thread instruction counters
    //          * 4. Release the lock*/
    //         CUDA_SAFECALL(cudaDeviceSynchronize());
    //         tot_app_instrs += counter;
    //         int num_ctas = 0;
    //         if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
    //             cbid == API_CUDA_cuLaunchKernel) {
    //             cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
    //             num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
    //         }
    //         printf(
    //             "kernel %d - %s - #thread-blocks %d,  kernel "
    //             "instructions %ld, total instructions %ld\n",
    //             kernel_id++, nvbit_get_func_name(ctx, p->f), num_ctas, counter,
    //             tot_app_instrs);
    //         pthread_mutex_unlock(&mutex);
    //     }
    // }
}
