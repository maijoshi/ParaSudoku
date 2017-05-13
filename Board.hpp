//
//  Board.hpp
//
//  Created by Sophie on 5/11/17.
//  Copyright © 2017 Sophie. All rights reserved.
//
#ifndef Board_hpp
#define Board_hpp


#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#define ROW 0
#define COL 1
#define BOX 2
using namespace std;

const int size = 9;
const int box_size = 3;

class Board {
public:
    unsigned markup[size][size];
    int board[size][size];
    Board() {};
    unordered_map<int, vector<vector<vector<int> > > > comb_map;
    
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

#endif /* Board_hpp */
