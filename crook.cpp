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
#include "CycleTimer.h"
#include "pthread.h"

#define ROW 0
#define COL 1
#define BOX 2
#define NDEBUG
#define MAX_ITER 1
using namespace std;

const int size = 9;
int box_size = 3;
int DEPTH;
int Thread_num;
unordered_map<int, vector<vector<vector<int> > > > comb_map;

class Board {
public:
    unsigned markup[size][size];
    int board[size][size];
    Board() {};
    
    void printBoard() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                cout << board[i][j] << " ";
            cout << endl;
        }
    }
    void printMarkup() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                cout << i << ", " << j << ": ";
                for (int k = 0; k < size; k++)
                    if ((markup[i][j] >> k) & 1)
                        cout << k+1 << " ";
                cout << endl;
            }
        }
    }
    bool markupContains(int i, int j, int val) {
        return (markup[i][j] >> (val-1)) & 1;
    }
    bool removeFromMarkup(int i, int j, int val) {
        bool ret = markupContains(i, j, val);
        markup[i][j] &= ~(1 << (val-1));
        return ret;
    }
    int getPossibilitiesCnt(int i, int j) {
        int x = markup[i][j];
        int cnt = 0;
        for (int k = 0; k < size; k++) {
            cnt += (x >> k) & 1;
        }
        return cnt;
    }
    void initialMarkup() {
        unsigned m = 0;
        for (int k = 0; k < size; k++)
            m |= 1 << k;
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                markup[i][j] = board[i][j]? 0:m;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (board[i][j]) {
                    for (int k = 0; k < size; k++) {
                        removeFromMarkup(i, k, board[i][j]);
                        removeFromMarkup(k, j, board[i][j]);
                        int box_x = k/box_size+i/box_size*box_size;
                        int box_y = k%box_size+j/box_size*box_size;
                        removeFromMarkup(box_x, box_y, board[i][j]);
                    }
                }
            }
        }
    }
    
    int getTotalUnfilledCellsNum() {
        int cnt = 0;
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                cnt += (board[i][j] == 0);
        return cnt;
    }
    
    int getUnfilledCellsNum(int choice, int i) {
        int cnt = 0;
        for (int k = 0; k < size; k++) {
            if (choice == ROW) {
                if (!board[i][k]) cnt++;
            }
            else if (choice == COL) {
                if (!board[k][i]) cnt++;
            }
            else if (choice == BOX) {
                if (!board[box_size*i/box_size+k/box_size][box_size*i%box_size+k%box_size]) cnt++;
            }
        }
        return cnt;
    }
    
    vector<int> getUnfilledCellsIndex(int choice, int i) {
        vector<int> ret;
        for (int k = 0; k < size; k++) {
            if (choice == ROW) {
                if (!board[i][k]) ret.push_back(k);
            }
            else if (choice == COL) {
                if (!board[k][i]) ret.push_back(k);
            }
            else if (choice == BOX) {
                if (!board[box_size*i/box_size+k/box_size][box_size*i%box_size+k%box_size]) ret.push_back(k);
            }
        }
        return ret;
    }
    
    
    vector<int> getPossibleValues(int i, int j) {
        int x = markup[i][j];
        vector<int> ret;
        for (int k = 0; k < size; k++) {
            if ((x >> k) & 1) {
                ret.push_back(k+1);
            }
        }
        return ret;
    }
    
    void setBoardVal(int row, int col, int val) {
        board[row][col] = val;
        // cout << "set " << row << ", " << col << "to " << val << endl;
        markup[row][col] = 0;
        for (int i = 0; i < size; i++) {
            removeFromMarkup(row, i, val);
            removeFromMarkup(i, col, val);
            int box_x = i/box_size+row/box_size*box_size;
            int box_y = i%box_size+col/box_size*box_size;
            removeFromMarkup(box_x, box_y, val);
        }
    }
    bool elimination();
    bool loneRangers();
    bool findTwins();
    bool findTriplets();
    bool findPreemptiveSet(int setSize);
};

bool Board::elimination() {
    #pragma omp parallel for 
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            int m = 0;
            int index = 0;
            for (int k = 0; k < size; k++) {
                if ((markup[i][j] >> k) & 1) {
                    m++;
                    index = k;
                }
            }
            if (m == 1) {
                setBoardVal(i, j, index+1);
            }
        }
    return (getTotalUnfilledCellsNum() == 0);
}

bool Board::loneRangers() {
    // check row
    for (int i = 0; i < size; i++) {
        for (int k = 1; k <= size; k++) {
            int cnt = 0;
            int col = -1;
            for (int j = 0; j < size; j++) {
                if (markupContains(i, j, k)) {
                    cnt++;
                    col = j;
                }
            }
            if (cnt == 1) {
                // found lone ranger
                setBoardVal(i, col, k);
#ifndef NDEBUG
                cout << "lone ranger - checked row:" << endl;
                printMarkup();
#endif
                return true;
            }
        }
    }
    
    // check column
    for (int j = 0; j < size; j++) {
        for (int k = 1; k <= size; k++) {
            int cnt = 0;
            int row = -1;
            for (int i = 0; i < size; i++) {
                if (markupContains(i, j, k)) {
                    cnt++;
                    row = i;
                    if (cnt > 1)    break;
                }
            }
            if (cnt == 1) {
                // found lone ranger
                setBoardVal(row, j, k);
#ifndef NDEBUG
                cout << "lone ranger - checked col:" << endl;
                printMarkup();
#endif
                return true;
            }
        }
    }
    
    
    // check box
    for (int i = 0; i < size; i+=box_size) {
        for (int j = 0; j < size; j+=box_size) {
            for (int k = 1; k <= size; k++) {
                int cnt = 0;
                int index = -1;
                bool next_box = false;
                for (int ii = 0; ii < box_size; ii++) {
                    for (int jj = 0; jj < box_size; jj++) {
                        if (markupContains(i+ii, j+jj, k)) {
                            cnt++;
                            index = ii * box_size + jj;
                            if (cnt > 1) {
                                next_box = true;
                                break;
                            }
                        }
                    }
                    if (next_box)   break;
                }
                if (cnt == 1) {
                    // found lone ranger
                    setBoardVal(index/box_size+i, index%box_size+j, k);
                    return true;
                }
            }
        }
    }
    
    return false;
}

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

bool Board::findPreemptiveSet(int setSize) {
    // check row
    for (int row = 0; row < size; row++) {
        // choose set of setSize cells
        vector<int> unfilled = getUnfilledCellsIndex(ROW, row);
        int num = (int) unfilled.size();
        if (num < setSize)  continue;
        vector<vector<int>> indexes = comb_map[num][setSize-1];//comb(num, setSize);
        for (int i = 0; i < indexes.size(); i++) {
            int res = 0;
            vector<int> tmp;
            for (int in : indexes[i]) {
                tmp.push_back(unfilled[in]);
                res |= markup[row][unfilled[in]];
            }
            int cnt = 0;
            for (int k = 0; k < size; k++) {
                if ((res >> k) & 1) cnt++;
            }
            if (cnt == setSize) {
                // found preemptive set
#ifndef NDEBUG
                cout << "pset found: row=" << row << " ";
                for (int in: indexes[i])
                    cout << unfilled[in] << " ";
                cout << endl;
#endif
                vector<int> pset;
                bool b = false;
                for (int k = 0; k < size; k++) {
                    if ((res >> k) & 1) {
                        for (int jj = 0; jj < size; jj++)
                            if (find(tmp.begin(), tmp.end(), jj) == tmp.end())
                                if (removeFromMarkup(row, jj, k+1))
                                    b = true;
                    }
                }
                return b;
            }
        }
        
        
        // check col
        for (int col = 0; col < size; col++) {
            // choose set of setSize cells
            vector<int> unfilled = getUnfilledCellsIndex(COL, col);
            int num = (int) unfilled.size();
            if (num < setSize)  continue;
            vector<vector<int>> indexes = comb_map[num][setSize-1];//comb(num, setSize);
            for (int i = 0; i < indexes.size(); i++) {
                int res = 0;
                vector<int> tmp;
                for (int in : indexes[i]) {
                    tmp.push_back(unfilled[in]);
                    res |= markup[unfilled[in]][col];
                }
                int cnt = 0;
                for (int k = 0; k < size; k++) {
                    if ((res >> k) & 1) cnt++;
                }
                if (cnt == setSize) {
                    // found preemptive set
                    
#ifndef NDEBUG
                    cout << "pset found: col=" << col << " ";
                    for (int in: indexes[i])
                        cout << unfilled[in] << " ";
                    cout << endl;
#endif
                    vector<int> pset;
                    bool b = false;
                    for (int k = 0; k < size; k++) {
                        if ((res >> k) & 1) {
                            for (int jj = 0; jj < size; jj++)
                                if (find(tmp.begin(), tmp.end(), jj) == tmp.end())
                                    if (removeFromMarkup(jj, col, k+1))
                                        b = true;
                        }
                    }
                    return b;
                }
            }
        }
    }
    return false;
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
mutex mtx;
bool backtrackingUtil(stack<pair<int, Board>> &stk, Board b, int index, int depth) {
    int row = index/size;
    int col = index%size;
    
    if (index >= size*size) {
        mtx.lock();
        stk.push(pair<int, Board>(index, b));
        mtx.unlock();
        return true;
    }
    
    if (!depth) {
        mtx.lock();
        stk.push(pair<int, Board>(index, b));
        mtx.unlock();
        return false;
    }
    
    for (int k = 1; k <= size; k++) {
        if (noConflicts(b.board, row, col, k)) {
            b.board[row][col] = k;
            if (backtrackingUtil(stk, b, findNextEmptyCellIndex(b.board, index + 1), depth - 1))
                return true;
        }
    }
    
    return false;
}

void backtrackingThreadWork(Board &crook_result, stack<pair<int, Board>> &stk, bool done) {
    while (!done) {
        cnt++;
        while (!stk.size());
        mtx.lock();
        int index = stk.top().first;
        Board b = stk.top().second;
        stk.pop();
        mtx.unlock();
        if (b.getTotalUnfilledCellsNum() == 0) {
            crook_result = b;
            break;
        }
        backtrackingUtil(stk, b, index, 5);
    }
}
void backtracking(Board &crook_result) {
    stack<pair<int, Board>> stk;
    Board tmp(crook_result);
    
    bool done = false;
    stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));
    double sTime = CycleTimer::currentSeconds();
    vector<thread> threads;
    for (int i = 0; i < Thread_num; i++) {
        threads.push_back(thread([&done, &stk, &crook_result](){
            while (!done) {
                cnt++;
                mtx.lock();
                if (stk.size()) {
                    int index = stk.top().first;
                    Board b = stk.top().second;
                    stk.pop();
                    mtx.unlock();
                    if (b.getTotalUnfilledCellsNum() == 0) {
                        crook_result = b;
                        done = true;
                        break;
                    }
                    backtrackingUtil(stk, b, index, DEPTH);
                }
                else mtx.unlock();
            }
        }));
    }
    double eTime = CycleTimer::currentSeconds();
    cout << eTime - sTime << endl;
    for (auto& thread:threads)
        thread.join();
    eTime = CycleTimer::currentSeconds();
    cout << eTime - sTime << endl;
    
    //    cout << cnt;
}

int main(int numArgs, char* args[]) {
    Board bb;
    double time = 0.0;
    double crook_time = 0.0;
    
    if (numArgs < 2) {
        DEPTH = 20;
        Thread_num = 25;
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
    ifstream myfile ("test/sudoku.txt");
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
        b.initialMarkup();
        bool done = false;
        bool change = false;
        bool use_crook = true;//false;
        while (!done && use_crook) {
            // step 1: elimination
            done = b.elimination();
            if (done) break;
            change = false;
#ifndef NDEBUG
            cout << "after elimination: " << done << endl;
            b.printBoard();
            b.printMarkup();
#endif
            // step 2: lone ranger
            if (b.loneRangers()) { // if any changes made, back to step 1
#ifndef NDEBUG
                cout << "after lone ranger search: " << endl;
                b.printBoard();
#endif
                continue;
            }
            else done = false;
            
            // step 3: find preemptive set with different sizes
            for (int i = 2; i < size; i++) {
                if (b.findPreemptiveSet(i)) { // if any changes made, back to step 1
#ifndef NDEBUG
                    cout << "after findPreemptiveSet " << i << ": " << endl;
                    b.printMarkup();
                    b.printBoard();
#endif
                    change = true;
                    break;
                }
            }
            if (!change) break; // no progress, leave it to backtracking
        }
        
        double middleTime = CycleTimer::currentSeconds();
        // if solution not found yet, use brute force
        if (!done) {
            backtracking(b);
        }
        double endTime = CycleTimer::currentSeconds();
        time += endTime - startTime;
        crook_time += middleTime - startTime;
    }
    //    cout << cnt;
    cout << "average time=" << time/MAX_ITER << endl;
    cout << "average crook time=" << crook_time/MAX_ITER << endl;
    b.printBoard();
    
    return 0;
}

