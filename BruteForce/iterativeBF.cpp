//
//  main.cpp
//  sudoku
//
//  Created by Sophie on 5/7/17.
//  Copyright ? 2017 Sophie. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <stack>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "../CycleTimer.h"
#include "../Board.hpp"

#define ROW 0
#define COL 1
#define BOX 2
#define NDEBUG
#define MAX_ITER 1
using namespace std;

int DEPTH;
static int Thread_num;
unordered_map<int, vector<vector<vector<int> > > > comb_map;

vector<vector<int>> comb(int n, int r)
{
    vector<bool> v(n);
    vector<vector<int>> combinations;
    fill(v.begin(), v.begin() + r, true);
    
    do {
        vector<int> combination;
        for (int i = 0; i < n; ++i) {
            if (v[i]) {
                combination.push_back(i);
            }
        }
        combinations.push_back(combination);
    } while (prev_permutation(v.begin(), v.end()));
    return combinations;
}

bool noConflicts(int matrix[size][size], int row, int col, int num) {
    
    for (int i = 0; i < size; i++) {
        if (matrix[i][col] == num)
            return false;
    }
    
    for (int j = 0; j < size; j++) {
        if (matrix[row][j] == num)
            return false;
    }
    
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            if (matrix[(row/box_size)*box_size + i][(col/box_size)*box_size + j] == num)
                return false;
        }
    }
    
    return true;
}
int cnt = 0;

int findNextEmptyCellIndex(int matrix[size][size], int start) {
    int i;
    for (i = start; i < size*size; i++) {
        if (matrix[i/size][i%size] == 0) {
            return i;
        }
    }
    return i;
}

void backtracking(Board &crook_result) {
    stack<Board> stk;
    Board tmp(crook_result);
    stk.push(tmp);
    bool done = false;
    
    while (!done) {
        cnt++;
        Board b = stk.top();
        stk.pop();
        if (b.getTotalUnfilledCellsNum() == 0) {
            crook_result = b;
            break;
        }
        int i;
        for (i = 0; i < size * size; i++) {
            int row = i/size;
            int col = i%size;
            if (!b.board[row][col]) {
                int k;
                for (k = 1; k <= size; k++) {
                    if (noConflicts(b.board, row, col, k)) {
                        b.board[row][col] = k;
                        stk.push(b);
                    }
                }
                break;
            }
        }
    }
}

int main(int numArgs, char* args[]) {
    Board bb;
    double time = 0.0;
    double crook_time = 0.0;
    
    if (numArgs < 2) {
        DEPTH = 15;
        Thread_num = 1;
    }
    else {
        DEPTH = atoi(args[1]);
        Thread_num = atoi(args[2]);
    }
    
    // get combinations
    for (int i = 1; i <= 9; i++)
        for (int j = 1; j <= i; j++)
            comb_map[i].push_back(comb(i, j));
    
    // read from file input
    ifstream myfile ("../test/sudoku.txt");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            myfile >> bb.board[i][j];
        }
    }
    bb.printBoard();
    
    Board b;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        b = bb;
        double startTime = CycleTimer::currentSeconds();
        backtracking(b);
        double endTime = CycleTimer::currentSeconds();
        time += endTime - startTime;
    }
    cout << "average time=" << time/MAX_ITER << endl;
    cout << "average crook time=" << crook_time/MAX_ITER << endl;
    b.printBoard();
    
    return 0;
}
