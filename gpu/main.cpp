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

const int boardSize = 9;
const int threadsPerBlock = 128;

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
    int DEPTH = 5;
    int *test;
    char *c;

    const int memSize = 81*pow(9, DEPTH);
    cout << memSize << endl;

    cudaMalloc(&new_boards, memSize * sizeof(int));
    cudaMalloc(&old_boards, memSize * sizeof(int));
    cudaMalloc(&solution, boardSize * boardSize * sizeof(int));
    cudaMalloc(&board_num, sizeof(int));
    cudaMalloc(&test, sizeof(int));
    cudaMalloc(&c, sizeof(char)*100000);
int *host_new_boards=(int*)malloc(memSize*sizeof(int));

    // // initialize memory
    cudaMemset(board_num, 0, sizeof(int));
    cudaMemset(new_boards, 0, memSize * sizeof(int));
    cudaMemset(old_boards, 0, memSize * sizeof(int));
    cudaMemset(solution, 0, boardSize * boardSize * sizeof(int));

    // // copy the initial board to the old boards
    cudaMemcpy(old_boards, board, boardSize * boardSize * sizeof(int), cudaMemcpyHostToDevice);

    BoardGenerator(new_boards, board_num, new_boards, memSize, c);
    int host_board_num = 1;
    cudaMemcpy(&host_board_num, board_num, sizeof(int), cudaMemcpyDeviceToHost);
    cout << host_board_num << endl;
    cudaMemcpy(host_new_boards, new_boards, host_board_num*boardSize*boardSize*sizeof(int), cudaMemcpyDeviceToHost);
    // ofstream outputFile;
    // outputFile.open("output_cuda");
    // for (int ii = 0; ii < host_board_num; ii++) {
    //     for (int i = 0; i < boardSize; i++) {
    //         for (int j = 0; j < boardSize; j++) {
    //             // cout << ii << " " << ii*boardSize*boardSize+i*boardSize+j << endl;
    //             if (ii*boardSize*boardSize+i*boardSize+j < host_board_num*boardSize*boardSize)
    //             outputFile << host_new_boards[ii*boardSize*boardSize+i*boardSize+j] << " ";
    //         }
    //         outputFile << endl;
    //     }
    //     outputFile << endl;
    // }

    cudaSudokuSolver(new_boards, host_board_num, solution, test);
    cout << "depth=" << endl;
    memset(host_solution, 0, boardSize*boardSize * sizeof(int));
    cudaMemcpy(host_solution, solution, boardSize*boardSize*sizeof(int), cudaMemcpyDeviceToHost);
    int t;
    cudaMemcpy(&t, test, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "depth=" << t << endl;
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++)
            cout << host_solution[i*boardSize+j] << " ";
            cout << endl;
    }
    cudaFree(new_boards);
    cudaFree(&old_boards);
    cudaFree(&solution);
    cudaFree(&board_num);
    return 0;
}