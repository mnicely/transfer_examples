
NVCC	:=nvcc --cudart=static -ccbin g++
CFLAGS	:=-O3 -std=c++14 -DUSE_NVTX
ARCHES	:=-gencode arch=compute_70,code=\"compute_70,sm_70\" -gencode arch=compute_75,code=\"compute_75,sm_75\" -gencode arch=compute_80,code=\"compute_80,sm_80\"
INC_DIR	:=
LIB_DIR	:=
LIBS	:=

SOURCES := cudaMemcpy \
		   cudaMemcpyAsync \
		   cudaManagedMemory \
		   cudaMemPrefetchAsync


all: $(SOURCES)
.PHONY: all

cudaMemcpy: cudaMemcpy.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

cudaMemcpyAsync: cudaMemcpyAsync.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

cudaManagedMemory: cudaManagedMemory.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

cudaMemPrefetchAsync: cudaMemPrefetchAsync.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES)