#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

extern FILE *fptr;

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {

        if (phase == recordReplayPhase::RECORD) {
            handleRecordKernelEvent(ctx, is_exit, cbid, name, params, pStatus);
        } else {
            handleReplayKernelEvent(is_exit);
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

void nvbit_at_ctx_init(CUcontext ctx) {
    if (phase == recordReplayPhase::RECORD)
        record();
    else
        replay();
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
