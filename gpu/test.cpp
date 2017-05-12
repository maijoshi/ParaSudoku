#include <iostream>
#include <fstream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
const int size = 9;
const int box_size = 3;
int boardSize = 9;
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

int main(int argc, char* argv[]) {
    int* board = (int*)malloc(sizeof(int)*size*size);
    ifstream myfile ("../test/sudoku.txt");
    for (int i = 0; i < boardSize*boardSize; i++) {
        for (int j = 0; j < boardSize*boardSize; j++) 
            myfile >> board[i*boardSize+j];
    }
    int localBoard[size*size];
    int emptyIndex[size*size];
    int emptyCnt = 0;
    for (int i = 0; i < size * size; i++) {
        localBoard[i] = board[i];
        if (!localBoard[i]) {
            emptyIndex[emptyCnt++] = i;
        }
    }
    
    int depth = 0;
    int* solution = (int*)malloc(sizeof(int)*size*size);
    cout << emptyCnt << endl;
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
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++)
            cout << solution[i*boardSize+j] << " ";
            cout << endl;
    }
}