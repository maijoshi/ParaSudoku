#include <stdlib.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "parallelsudoku.cuh"
#define UPDIV(n, d) (((n)+(d)-1)/(d))
using namespace std;

int main(int argc, char* argv[]) {
    // read sudoku board from file
    int board[boardSize*boardSize];
    ifstream myfile ("../test/sudoku.txt");
    for (int i = 0; i < boardSize*boardSize; i++) {
        for (int j = 0; j < boardSize*boardSize; j++) 
            myfile >> board[i*boardSize+j];
    }
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++)
            cout << board[i*boardSize+j] << " ";
            cout << endl;
    }

    int *new_boards;
    int *old_boards;
    int *solution;
    int *board_num;
    int host_solution[boardSize*boardSize];

    const int memSize = 81*pow(9, DEPTH);
    // alloc device memory
    cudaMalloc(&new_boards, memSize * sizeof(int));
    cudaMalloc(&old_boards, memSize * sizeof(int));
    cudaMalloc(&solution, boardSize * boardSize * sizeof(int));
    cudaMalloc(&board_num, sizeof(int));

    cudaMemset(board_num, 0, sizeof(int));
    cudaMemset(new_boards, 0, memSize * sizeof(int));
    cudaMemset(old_boards, 0, memSize * sizeof(int));
    cudaMemset(solution, 0, boardSize * boardSize * sizeof(int));
    
    cudaMemcpy(old_boards, board, boardSize * boardSize * sizeof(int), cudaMemcpyHostToDevice);

    BoardGenerator(new_boards, board_num, new_boards, memSize);
    int host_board_num = 1;
    cudaMemcpy(&host_board_num, board_num, sizeof(int), cudaMemcpyDeviceToHost);
    cudaSudokuSolver(new_boards, host_board_num, solution);
    
    // print out solution
    memset(host_solution, 0, boardSize*boardSize * sizeof(int));
    cudaMemcpy(host_solution, solution, boardSize*boardSize*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++)
            cout << host_solution[i*boardSize+j] << " ";
            cout << endl;
    }
    // free device memory
    cudaFree(new_boards);
    cudaFree(&old_boards);
    cudaFree(&solution);
    cudaFree(&board_num);
    return 0;
}