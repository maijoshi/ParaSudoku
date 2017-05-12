EXECUTABLE := cudaSudoku

CU_FILES   := parallelsudoku.cu
CU_DEPS    :=
CC_FILES   := main.cpp
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -I/usr/local/cuda/include -std=c++11 -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -std=c++11 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/parallelsudoku.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
