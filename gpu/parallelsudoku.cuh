#ifndef CUDA_SUDOKU_CUDA_CUH
#define CUDA_SUDOKU_CUDA_CUH
// void SolvingKernel(int* boards, int boardCnt);
void cudaSudokuSolver(int* old_board, int num, int* solution, int* test);
void BoardGenerator(int* prev_boards, int* prev_board_num, int* new_boards, int memSize, char* test);
#endif