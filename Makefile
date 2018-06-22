CUDA_HOME   = /Soft/cuda/8.0.61

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = bomba.exe
BOMBA	    = bomba_gpu.o


default: $(EXE)

bomba_gpu.o: bomba_gpu.cu
	$(NVCC) -c -o $@ bomba_gpu.cu $(NVCC_FLAGS)

$(EXE): $(BOMBA)
	$(NVCC) $(BOMBA) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
