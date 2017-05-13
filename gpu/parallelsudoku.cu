#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../Board.hpp"
#include "parallelsudoku.cuh"
#include "../CycleTimer.h"
#define UPDIV(n, d) (((n)+(d)-1)/(d))
const int threadsPerBlock = 512;

// function to examine if there are conflicts or not if cell [row][col] is num
__device__
bool noConflicts(int matrix[boardSize * boardSize], int row, int col, int num) {
    if (num <= 0 || num > boardSize) return false;
    for (int i = 0; i < boardSize; i++) {
        if (i == row)   continue;
        if (matrix[i * boardSize + col] == num) {
            return false;
        }
    }
    
    for (int j = 0; j < boardSize; j++) {
        if (j == col)   continue;
        if (matrix[row * boardSize + j] == num) {
            return false;
        }
    }
    
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            int mat_row = (row/box_size)*box_size + i;
            int mat_col = (col/box_size)*box_size + j;
            if (mat_row == row && mat_col == col)   continue;
            if (matrix[mat_row * boardSize + mat_col] == num) {
                return false;
            }
        }
    }
    return true;
}

// find the next empty cell index 
__device__
int findNextEmptyCellIndex(int matrix[boardSize*boardSize], int start) {
    int i;
    for (i = start; i < boardSize*boardSize; i++) {
        if (matrix[i] == 0) {
            return i;
        }
    }
    return i;
}

// the kernel that solves sudoku problem on each board
// each thread works on a board in the boards array
__global__
void SolvingKernel(int* boards, int boardCnt, int* solution, int numThreads) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int localBoard[boardSize*boardSize];
    for (int idx = tidx; idx < boardCnt; idx += numThreads) {
        int emptyCnt = 0;
        int emptyIndex[boardSize*boardSize];
        int start = idx * boardSize * boardSize;
        for (int i = start; i < (idx+1) * boardSize * boardSize; i++) {
            localBoard[i-start] = boards[i];
            if (!localBoard[i-start]) {
                emptyIndex[emptyCnt++] = i-start;
            }
        }
        int depth = 0;
        while (depth >= 0 && depth < emptyCnt) {
            int next = emptyIndex[depth];
            int row = next / boardSize;
            int col = next % boardSize;
            localBoard[next]++;
            if (noConflicts(localBoard, row, col, localBoard[next])) depth++;
            else if (localBoard[next] >= boardSize) {
                localBoard[next] = 0;
                depth--;
            }
        }
        if (depth == emptyCnt) {
            memcpy(solution, localBoard, boardSize*boardSize*sizeof(int));
            break;
        }
    }
    
}



// kernel that generates new boards from previous ones
// call this kernel multiple times to have 
__global__
void BoardGenerationKernel(int* prev_boards, int* board_num, int prev_board_num, int* new_boards, int numThreads) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int* localBoard = (int*) malloc(sizeof(int)*boardSize*boardSize);

    if (!prev_board_num) prev_board_num++;
    for (int idx = tidx; idx < prev_board_num; idx+=numThreads) {
        int start = idx * boardSize * boardSize;
        for (int i = start; i < (idx+1) * boardSize * boardSize; i++) {
            localBoard[i-start] = prev_boards[i];
        }
        int emptyIdx = findNextEmptyCellIndex(localBoard, 0);
        if (emptyIdx == boardSize*boardSize)  return;
        for (int k = 1; k <= boardSize; k++) {
            localBoard[emptyIdx] = k;
            if (noConflicts(localBoard, emptyIdx/boardSize, emptyIdx%boardSize, k)) {
                int offset = atomicAdd(board_num, 1);
                for (int ii = 0; ii < boardSize*boardSize; ii++) {
                    new_boards[boardSize*boardSize*offset+ii] = localBoard[ii];
                }
            }
        }
    }

}

void 
BoardGenerator(int* prev_boards, int* prev_board_num, int* new_boards, int memSize) {
    int i;
    int num = 1;
    for (i = 0; i < DEPTH; i++) {
        int block = UPDIV(num, threadsPerBlock);
        cudaMemset(prev_board_num, 0, sizeof(int));
        BoardGenerationKernel<<<block, threadsPerBlock>>>(prev_boards, prev_board_num, num, new_boards, block*threadsPerBlock);
        int* tmp = prev_boards;
        prev_boards = new_boards;
        new_boards = tmp;
        cudaMemcpy(&num, prev_board_num, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

void 
cudaSudokuSolver(int* boards, int board_num, int* solution) {
    int block = UPDIV(board_num, threadsPerBlock);
    double stime = CycleTimer::currentSeconds();
    SolvingKernel<<<block, threadsPerBlock>>>(boards, board_num, solution, block*threadsPerBlock);
    cudaDeviceSynchronize();
    cout << "cudaSudokuSolver takes time: " << CycleTimer::currentSeconds() - stime << endl;
}