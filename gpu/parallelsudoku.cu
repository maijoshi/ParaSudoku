#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../Board.hpp"
// trying filling one cell at first
// still use Board?
__device__
bool noConflicts(int matrix[size][size], int row, int col, int num) {
    
    for (int i = 0; i < size; i++) {
        if (matrix[i][col] == num)
            return false;
    }
    
    for (int j = 0; j < size; j++) {
        if (matrix[row][j] == num)
            return false;
    }
    
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            if (matrix[(row/box_size)*box_size + i][(col/box_size)*box_size + j] == num)
                return false;
        }
    }
    
    return true;
}

__device__
int findNextEmptyCellIndex(int matrix[size][size], int start) {
    int i;
    for (i = start; i < size*size; i++) {
        if (matrix[i/size][i%size] == 0) {
            return i;
        }
    }
    return i;
}

__global__
void backtracking(int crook_result[size][size]) {
    Board tmp(crook_result);
    
    bool done = false;
    prev_boards
    int* new_boards = 
    // if (!multiStack) stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));


    vector<thread> threads;
    
    int index = findNextEmptyCellIndex(tmp.board, 0);
    int threadWorkload = (int)queue.size()/Thread_num;
    int end = (id == Thread_num) ? queue.size():(1+id)*threadWorkload;
    stack<pair<int, Board>> stk_local(deque<pair<int, Board>>(queue.begin()+id*threadWorkload, queue.begin()+end));
    
    while (!done) {
        cnt++;
        if (stk_local.size()) {
            int index = stk_local.top().first;
            Board b = stk_local.top().second;
            stk_local.pop();
            if (b.getTotalUnfilledCellsNum() == 0) {
                crook_result = b;
                done = true;
                break;
            }
            backtrackingUtil(stk_local, b, index, DEPTH, true);
        }
        else break;
    }
}

// __global__ void BGKernel(int *prev_boards, int* new_boards, int 
// 	total_num_boards, int* empty_space_ind, int* empty_space_num) {

// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
// }

// void 
// Sudoku::main() {
//     BGKernel<<<boards, threadsPerBoard>>>(prev_boards, new_boards, 
//     	total_num_boards, empty_space_ind, empty_space_num);
// }
