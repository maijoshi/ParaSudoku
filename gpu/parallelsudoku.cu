#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../Board.hpp"
#include "parallelsudoku.cuh"
#include "../CycleTimer.h"
#define UPDIV(n, d) (((n)+(d)-1)/(d))
const int threadsPerBlock = 1024;
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
void SolvingKernel(int* boards, int boardCnt, int* solution, int numThreads) {
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
            // printf("orz\n");
            memcpy(solution, localBoard, size*size*sizeof(int));
            break;
        }
    }
    
}




__global__
void BoardGenerationKernel(int* prev_boards, int* board_num, int prev_board_num, int* new_boards, int numThreads) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int* localBoard = (int*) malloc(sizeof(int)*size*size);

    // int prev_board_num = *board_num;
    // memcpy(&prev_board_num, board_num, sizeof(int));
    // *board_num = 0;
    if (!prev_board_num) prev_board_num++;
    for (int idx = tidx; idx < prev_board_num; idx+=numThreads) {
        int start = idx * size * size;
        for (int i = start; i < (idx+1) * size * size; i++) {
            localBoard[i-start] = prev_boards[i];
        }

        int emptyIdx = findNextEmptyCellIndex(localBoard, 0);
        if (emptyIdx == size*size)  return;
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
            // else {
            //     // *(test+1) = k;
            //     *test = emptyIdx;
            //     // sprintf(test, "emptyIdx=%d", emptyIdx);
            // }
        }
    }

}

void 
BoardGenerator(int* prev_boards, int* prev_board_num, int* new_boards, int memSize) {
    // int block = UPDIV(1, threadsPerBlock);
    int i;
    int num = 1;
    for (i = 0; i < 10; i++) {
        int block = UPDIV(num, threadsPerBlock);
        cudaMemset(prev_board_num, 0, sizeof(int));
        if (i%2 == 0) 
            BoardGenerationKernel<<<block, threadsPerBlock>>>(prev_boards, prev_board_num, num, new_boards, block*threadsPerBlock);
        else BoardGenerationKernel<<<block, threadsPerBlock>>>(new_boards, prev_board_num, num, prev_boards, block*threadsPerBlock);
        // int* tmp = prev_boards;
        // prev_boards = new_boards;
        // new_boards = tmp;
        cudaMemcpy(&num, prev_board_num, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(host_test, test, sizeof(int), cudaMemcpyDeviceToHost);
        cout << "iter " << i << ", board number=" << num << endl;
    }
    if (i % 2) new_boards = prev_boards;
}
void 
cudaSudokuSolver(int* boards, int board_num, int* solution) {
    int block = UPDIV(board_num, threadsPerBlock);
    cout << "board_num=" << board_num << endl;
    double stime = CycleTimer::currentSeconds();
    SolvingKernel<<<block, threadsPerBlock>>>(boards, board_num, solution, block*threadsPerBlock);
    cudaDeviceSynchronize();
    cout << CycleTimer::currentSeconds() - stime << endl;
}