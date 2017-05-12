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

#include "CycleTimer.h"
#include "pthread.h"
#include "Board.hpp"

//#define NDEBUG
#define MAX_ITER 1
using namespace std;


int DEPTH;
static int Thread_num;




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
bool backtrackingUtil(deque<pair<int, Board>> &vec, Board b, int index, int depth) {
    int row = index/size;
    int col = index%size;
    
    if (index >= size*size) {
        vec.push_back(pair<int, Board>(index, b));
        return true;
    }
    
    if (!depth) {
        vec.push_back(pair<int, Board>(index, b));
        return false;
    }
    
    for (int k = 1; k <= size; k++) {
        if (noConflicts(b.board, row, col, k)) {
            b.board[row][col] = k;
            if (backtrackingUtil(vec, b, findNextEmptyCellIndex(b.board, index + 1), depth - 1))
                return true;
        }
    }
    
    return false;
}


bool backtrackingUtil(stack<pair<int, Board>> &stk, Board b, int index, int depth, bool multiStack) {
    int row = index/size;
    int col = index%size;
    
    if (index >= size*size) {
        if (!multiStack) mtx.lock();
        stk.push(pair<int, Board>(index, b));
        if (!multiStack) mtx.unlock();
        return true;
    }
    
    if (!depth) {
        if (!multiStack) mtx.lock();
        stk.push(pair<int, Board>(index, b));
        if (!multiStack) mtx.unlock();
        return false;
    }
    
    for (int k = 1; k <= size; k++) {
        if (noConflicts(b.board, row, col, k)) {
            b.board[row][col] = k;
            if (backtrackingUtil(stk, b, findNextEmptyCellIndex(b.board, index + 1), depth - 1, multiStack))
                return true;
        }
    }
    
    return false;
}

//void backtrackingThreadWork(Board &crook_result, stack<pair<int, Board>> &stk, bool done) {
//    while (!done) {
//        cnt++;
//        while (!stk.size());
//        mtx.lock();
//        int index = stk.top().first;
//        Board b = stk.top().second;
//        stk.pop();
//        mtx.unlock();
//        if (b.getTotalUnfilledCellsNum() == 0) {
//            crook_result = b;
//            break;
//        }
//        backtrackingUtil(stk, b, index, 5);
//    }
//}
//void backtracking(Board &crook_result) {
//    stack<pair<int, Board>> stk;
//    deque<pair<int, Board>> queue;
//    bool multiStack = false;
//    
//    Board tmp(crook_result);
//    
//    bool done = false;
//    if (!multiStack) stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));
//    
//    vector<thread> threads;
//    
//    double startTime = CycleTimer::currentSeconds();
//    int index = findNextEmptyCellIndex(tmp.board, 0);
//    if (multiStack) backtrackingUtil(queue, tmp, index, DEPTH);
//    
//    double eTime = CycleTimer::currentSeconds();
//    cout << eTime - startTime << endl;
//    for (int id = 0; id < Thread_num; id++) {
//        if (multiStack) {
//            threads.push_back(thread([&done, &queue, id, &crook_result](){
//                //#pragma omp parallel
//                //            {
//                double sTime = CycleTimer::currentSeconds();
//                int threadWorkload = (int)queue.size()/Thread_num;
//                int end = (id == Thread_num) ? queue.size():(1+id)*threadWorkload;
//                stack<pair<int, Board>> stk_local(deque<pair<int, Board>>(queue.begin()+id*threadWorkload, queue.begin()+end));
//                //                for (int ii = id*threadWorkload; ii < (id+1)*threadWorkload; ii++) {
//                //                    stk_local.push(queue[ii]);
//                //                    if (id == Thread_num-1 && queue.size()%Thread_num) {
//                //                        for (int jj = (id+1)*threadWorkload; jj<queue.size();jj++ ) {
//                //                            stk_local.push(queue[jj]);
//                //                        }
//                //                    }
//                //                }
//                
//                while (!done) {
//                    cnt++;
//                    if (stk_local.size()) {
//                        int index = stk_local.top().first;
//                        Board b = stk_local.top().second;
//                        stk_local.pop();
//                        if (b.getTotalUnfilledCellsNum() == 0) {
//                            crook_result = b;
//                            done = true;
//                            break;
//                        }
//                        backtrackingUtil(stk_local, b, index, DEPTH, true);
//                    }
//                    else break;
//                }
//                //                mtx.lock();
//                //cout << CycleTimer::currentSeconds() - sTime << endl;
//                //                mtx.unlock();
//            }));
//        }
//        //        }
//        else {
//            threads.push_back(thread([&done, id, &stk, &crook_result](){
//                //        threads.push_back(thread([&done, &vec, id, &crook_result](){
//                //            int threadWorkload = (int)vec.size()/Thread_num;
//                //            stack<pair<int, Board>> stk;
//                //            for (int ii = id*threadWorkload; ii < (id+1)*threadWorkload; ii++) {
//                //                stk.push(vec[ii]);
//                //            }
//                
//                while (!done) {
//                    cnt++;
//                    mtx.lock();
//                    if (stk.size()) {
//                        int index = stk.top().first;
//                        Board b = stk.top().second;
//                        stk.pop();
//                        mtx.unlock();
//                        if (b.getTotalUnfilledCellsNum() == 0) {
//                            crook_result = b;
//                            done = true;
//                            break;
//                        }
//                        backtrackingUtil(stk, b, index, DEPTH, false);
//                    }
//                    else mtx.unlock();
//                }
//            }));
//        }
//    }
//    
//    eTime = CycleTimer::currentSeconds();
//    //cout << eTime - sTime << endl;
//    for (auto& thread:threads)
//        thread.join();
//    eTime = CycleTimer::currentSeconds();
//    //cout << eTime - sTime << endl;
//    
//    //    cout << cnt;
//}
void backtracking(Board &crook_result) {
    stack<pair<int, Board>> stk;
    deque<pair<int, Board>> vec;
    bool multiStack = true;
    
    Board tmp(crook_result);
    
    bool done = false;
    if (!multiStack) stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));
    double sTime = CycleTimer::currentSeconds();
    vector<thread> threads;
    
    
    int index = findNextEmptyCellIndex(tmp.board, 0);
    if (multiStack) backtrackingUtil(vec, tmp, index, DEPTH);
    
    
    for (int id = 0; id < Thread_num; id++) {
//        if (multiStack) {
//            threads.push_back(thread([&done, &vec, id, &crook_result](){
//                int threadWorkload = (int)vec.size()/Thread_num;
//                int end = (id == Thread_num) ? vec.size():(1+id)*threadWorkload;
//                stack<pair<int, Board>> stk((vec.begin()+id*threadWorkload, vec.begin()+end));
//                for (int ii = id*threadWorkload; ii < (id+1)*threadWorkload; ii++) {
//                    stk.push(vec[ii]);
//                    if (id == Thread_num-1 && vec.size()%Thread_num) {
//                        for (int jj = (id+1)*threadWorkload; jj<vec.size();jj++ ) {
//                            stk.push(vec[jj]);
//                        }
//                    }
//                }
//                
//                while (!done) {
//                    cnt++;
//                    if (stk.size()) {
//                        int index = stk.top().first;
//                        Board b = stk.top().second;
//                        stk.pop();
//                        if (b.getTotalUnfilledCellsNum() == 0) {
//                            crook_result = b;
//                            done = true;
//                            break;
//                        }
//                        backtrackingUtil(stk, b, index, DEPTH, true);
//                    }
//                    else break;
//                }
//            }));
//            
//        }
//        else {
            threads.push_back(thread([&done, &vec, id, &stk, &crook_result](){
                //        threads.push_back(thread([&done, &vec, id, &crook_result](){
                //            int threadWorkload = (int)vec.size()/Thread_num;
                //            stack<pair<int, Board>> stk;
                //            for (int ii = id*threadWorkload; ii < (id+1)*threadWorkload; ii++) {
                //                stk.push(vec[ii]);
                //            }
                
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
                        backtrackingUtil(stk, b, index, DEPTH, false);
                    }
                    else mtx.unlock();
                }
            }));
//        }
    }
    double eTime = CycleTimer::currentSeconds();
    cout << eTime - sTime << endl;
    for (auto& thread:threads)
        thread.join();
    eTime = CycleTimer::currentSeconds();
    cout << eTime - sTime << endl;
    
    //    cout << cnt;
}
//void backtracking(Board &crook_result) {
//    //    stack<pair<int, Board>> stk;
//    vector<pair<int, Board>> vec;
//
//    Board tmp(crook_result);
//
//    bool done = false;
//    //    stk.push(pair<int, Board>(findNextEmptyCellIndex(tmp.board, 0), tmp));
//    double sTime = CycleTimer::currentSeconds();
//    vector<thread> threads;
//
//
//    int index = findNextEmptyCellIndex(tmp.board, 0);
//    backtrackingUtil(vec, tmp, index, DEPTH);
//
//
//    for (int id = 0; id < Thread_num; id++) {
//        //        threads.push_back(thread([&done, &stk, &crook_result](){
//        threads.push_back(thread([&done, &vec, id, &crook_result](){
//            int threadWorkload = (int)vec.size()/Thread_num;
//            stack<pair<int, Board>> stk;
//            for (int ii = id*threadWorkload; ii < (id+1)*threadWorkload; ii++) {
//                stk.push(vec[ii]);
//            }
//
//            while (!done) {
//                cnt++;
//                mtx.lock();
//                if (stk.size()) {
//                    int index = stk.top().first;
//                    Board b = stk.top().second;
//                    stk.pop();
//                    mtx.unlock();
//                    if (b.getTotalUnfilledCellsNum() == 0) {
//                        crook_result = b;
//                        done = true;
//                        break;
//                    }
//                    backtrackingUtil(stk, b, index, DEPTH);
//                }
//                else mtx.unlock();
//            }
//        }));
//    }
//    double eTime = CycleTimer::currentSeconds();
//    cout << eTime - sTime << endl;
//    for (auto& thread:threads)
//        thread.join();
//    eTime = CycleTimer::currentSeconds();
//    cout << eTime - sTime << endl;
//
//    //    cout << cnt;
//}

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

int main(int numArgs, char* args[]) {
    Board bb;
    double time = 0.0;
    double crook_time = 0.0;
    
    if (numArgs < 2) {
        DEPTH = 5;
        Thread_num = 5;
    }
    else {
        DEPTH = atoi(args[1]);
        Thread_num = atoi(args[2]);
    }
    
    // get combinations
    for (int i = 1; i <= 9; i++)
        for (int j = 1; j <= i; j++)
            bb.comb_map[i].push_back(comb(i, j));
    
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
