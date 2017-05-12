#ifndef CUDA_SUDOKU_CUDA_CUH
#define CUDA_SUDOKU_CUDA_CUH
// void SolvingKernel(int* boards, int boardCnt);
void cudaSudokuSolver(int* old_board, int num, int* solution, bool* b);
#endif