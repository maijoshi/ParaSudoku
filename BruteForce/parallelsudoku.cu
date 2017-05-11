#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"



__global__ void BGKernel(int *prev_boards, int* new_boards, int 
	total_num_boards, int* empty_space_ind, int* empty_space_num) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	

}

void 
Sudoku::main() {
    BGKernel<<<boards, threadsPerBoard>>>(prev_boards, new_boards, 
    	total_num_boards, empty_space_ind, empty_space_num);
}
