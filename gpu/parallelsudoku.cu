#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../Board.hpp"
#include "parallelsudoku.cuh"
#define UPDIV(n, d) (((n)+(d)-1)/(d))
const int threadsPerBlock = 128;
// trying filling one cell at first
// still use Board?
__device__
bool noConflicts(int matrix[size * size], int row, int col, int num) {
    if (num > size) return false;
    for (int i = 0; i < size; i++) {
        if (i == row)   continue;
        if (matrix[i * size + col] == num) {
            return false;
        }
    }
    
    for (int j = 0; j < size; j++) {
        if (j == col)   continue;
        if (matrix[row * size + j] == num) {
            return false;
        }
    }
    
    // for (int i = 0; i < box_size; i++) {
    //     for (int j = 0; j < box_size; j++) {
    //         if (i == row && j == col)   continue;
    //         if (matrix[((row/box_size)*box_size + i) * size + (col/box_size)*box_size + j] == num) {
    //             cout << "box conflict: i=" << i <<"j="<<j<< endl;
    //             return false;
    //         }
    //     }
    // }
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

// each thread work on a board in the frontier
__global__
void SolvingKernel(int* boards, int boardCnt, int* solution, bool solved) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localBoard[size*size];
    int emptyIndex[size*size];
    int emptyCnt = 0;
    if (idx < boardCnt) { //
        int start = idx * size * size;
        for (int i = start; i < (idx+1) * size * size; i++) {
            localBoard[i-start] = boards[i];
            if (!localBoard[i-start]) {
                emptyIndex[emptyCnt++] = i;
            }
        }
        
        int depth = 0;
        while (depth >= 0 && depth < emptyCnt) {
            int next = emptyIndex[depth];
            int row = next / size;
            int col = next % size;
            localBoard[next]++;

            if (noConflicts(localBoard, row, col, localBoard[next])) {
                depth++;
            }
            else if (localBoard[next] >= size) {
                localBoard[next] = 0;
                depth--;
            }
        }
        
        if (depth == emptyCnt) {
            memcpy(solution, localBoard, size*size*sizeof(int));
            
        }
        solved = depth == emptyCnt;
    }
    
}


__global__
BoardGenerationKernel(int* old_board, int* new_boards) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // for 

}

// __global__
// void backtrackingKernel(int crook_result[size][size]) {
//     // Board tmp(crook_result);
    
//     bool done = false;
//     int* new_boards[size][size];

//     // if (!multiStack) stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));


//     // vector<thread> threads;
    
//     // int index = findNextEmptyCellIndex(tmp.board, 0);
//     // int threadWorkload = (int)queue.size()/Thread_num;
//     // int end = (id == Thread_num) ? queue.size():(1+id)*threadWorkload;
//     // stack<pair<int, Board>> stk_local(deque<pair<int, Board>>(queue.begin()+id*threadWorkload, queue.begin()+end));
    
    // while (!done) {
    //     cnt++;
    //     if (stk_local.size()) {
    //         int index = stk_local.top().first;
    //         Board b = stk_local.top().second;
    //         stk_local.pop();
    //         if (b.getTotalUnfilledCellsNum() == 0) {
    //             crook_result = b;
    //             done = true;
    //             break;
    //         }
    //         backtrackingUtil(stk_local, b, index, DEPTH, true);
    //     }
    //     else break;
//     // }
// }

// __global__ void BGKernel(int *prev_boards, int* new_boards, int 
// 	total_num_boards, int* empty_space_ind, int* empty_space_num) {

// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
// }

void 
cudaSudokuSolver(int* old_boards, int num, int* solution, bool* solved) {
    int block = UPDIV(1, threadsPerBlock);
    SolvingKernel<<<block, threadsPerBlock>>>(old_boards, 1, solution, solved);
}
