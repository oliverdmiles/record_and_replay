
# C compiler
CC = gcc
CC_FLAGS = -g  -O2

NTHREADS ?= 64
NBLOCKS ?= 2
SM ?= 50
# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include  -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared -DTHREADS=$(NTHREADS) -DBLOCKS=$(NBLOCKS) -rdc=true


# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

OBJS=atomic_increment increment base_with_syncthreads dynamic_parallelism flag_sync_across_blocks single_stream  warp_lock base branch_race dynamic_parallelism_with_streams implicit_intra_warp_sync memcpy_async two_streams increment_multi increment_large increment_copy_to_device

all: $(OBJS)

$(OBJS):
	$(NVCC) $(NVCC_FLAGS) $@.cu -DSHARED -o $@.shared
	$(NVCC) $(NVCC_FLAGS) $@.cu -DGLOBAL -o $@.global

clean:
	rm -f *.o *.shared *.global
