#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../Board.hpp"
#include "parallelsudoku.cuh"
#include "../CycleTimer.h"
#define UPDIV(n, d) (((n)+(d)-1)/(d))
const int threadsPerBlock = 32;
// trying filling one cell at first
// still use Board?
__device__
bool noConflicts(int matrix[size * size], int row, int col, int num) {
    if (num <= 0 || num > size) return false;
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
    
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            int mat_row = (row/box_size)*box_size + i;
            int mat_col = (col/box_size)*box_size + j;
            if (mat_row == row && mat_col == col)   continue;
            if (matrix[mat_row * size + mat_col] == num) {
                // cout << "box conflict: i=" << i <<"j="<<j<< endl;
                return false;
            }
        }
    }
    return true;
}

__device__
int findNextEmptyCellIndex(int matrix[size*size], int start) {
    int i;
    for (i = start; i < size*size; i++) {
        if (matrix[i] == 0) {
            return i;
        }
    }
    return i;
}

// each thread work on a board in the frontier
__global__
void SolvingKernel(int* boards, int boardCnt, int* solution, int numThreads, int *test) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int localBoard[size*size];
    for (int idx = tidx; idx < boardCnt; idx += numThreads) {
        int emptyCnt = 0;
        int emptyIndex[size*size];
        int start = idx * size * size;
        for (int i = start; i < (idx+1) * size * size; i++) {
            localBoard[i-start] = boards[i];
            if (!localBoard[i-start]) {
                emptyIndex[emptyCnt++] = i-start;
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
            // if (idx == 3) *test = next;
        }
        
        if (depth == emptyCnt) {
            *test = boardCnt;
            memcpy(solution, localBoard, size*size*sizeof(int));
        }
    }
    
}




__global__
void BoardGenerationKernel(int* prev_boards, int* board_num, int* new_boards, int numThreads) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int* localBoard = (int*) malloc(sizeof(int)*size*size);

    int prev_board_num = *board_num;
    if (!prev_board_num) prev_board_num++;
    for (int idx = tidx; idx < prev_board_num; idx+=numThreads) {
        int start = idx * size * size;
        for (int i = start; i < (idx+1) * size * size; i++) {
            localBoard[i-start] = prev_boards[i];
        }

        int emptyIdx = findNextEmptyCellIndex(localBoard, 0);
        int cnt = 0;
        for (int k = 1; k <= size; k++) {
            localBoard[emptyIdx] = k;
            if (noConflicts(localBoard, emptyIdx/size, emptyIdx%size, k)) {
                cnt++;
                int offset = atomicAdd(board_num, 1);
                for (int ii = 0; ii < size*size; ii++) {
                    new_boards[size*size*offset+ii] = localBoard[ii];
                }
            }
        }
    }

}

void 
BoardGenerator(int* prev_boards, int* prev_board_num, int* new_boards) {
    int block = UPDIV(1, threadsPerBlock);
    for (int i = 0; i < 18; i++) {
        BoardGenerationKernel<<<block, threadsPerBlock>>>(prev_boards, prev_board_num, new_boards, block*threadsPerBlock);
        prev_boards = new_boards;
    }
}
void 
cudaSudokuSolver(int* boards, int board_num, int* solution, int* test) {
    int block = UPDIV(board_num, threadsPerBlock);
    double stime = CycleTimer::currentSeconds();
    SolvingKernel<<<block, threadsPerBlock>>>(boards, board_num, solution, block*threadsPerBlock, test);
    cudaDeviceSynchronize();
    cout << CycleTimer::currentSeconds() - stime << endl;
}