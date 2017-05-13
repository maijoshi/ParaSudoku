#ifndef CUDA_SUDOKU_CUDA_CUH
#define CUDA_SUDOKU_CUDA_CUH
const int DEPTH = 5;
const int boardSize = 9;
void cudaSudokuSolver(int* old_board, int num, int* solution);
void BoardGenerator(int* prev_boards, int* prev_board_num, int* new_boards, int memSize);
#endif