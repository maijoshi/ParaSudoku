//
//  main.cpp
//  sudoku
//
//  Created by Sophie on 5/7/17.
//  Copyright © 2017 Sophie. All rights reserved.
//

#include <iostream>
#include <vector>
using namespace std;

const int size = 9;
int box_size = 3;

class Board {
public:
    unsigned markup[size][size];
    int board[size][size];
    Board() {};
    void removeFromMarkup(int i, int j, int val) {
        markup[i][j] &= ~(1 << (val-1));
    }
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
                    if ((markup[i][j] >> k) & 1)   cout << k+1 << " ";
                cout << endl;
            }
        }
    }
    bool markupContains(int i, int j, int val) {
        return (markup[i][j] >> (val-1)) & 1;
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
    
    int getUnfilledCellsNum() {
        int cnt = 0;
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                cnt += (board[i][j] == 0);
        return cnt;
    }
    
    bool elimination() {
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
        return (getUnfilledCellsNum() == 0);
    }
    
    void setBoardVal(int row, int col, int val) {
        board[row][col] = val;
        cout << "set " << row << ", " << col << "to " << val << endl;
        markup[row][col] = 0;
        for (int i = 0; i < size; i++) {
            removeFromMarkup(row, i, val);
            removeFromMarkup(i, col, val);
            int box_x = i/box_size+row/box_size*box_size;
            int box_y = i%box_size+col/box_size*box_size;
            removeFromMarkup(box_x, box_y, val);
        }
    }
    
    bool loneRangers() {
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
                    cout << "lone ranger - checked row:" << endl;
                    printMarkup();
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
                    cout << "lone ranger - checked col:" << endl;
                    printMarkup();
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
                        cout << "lone ranger - checked box:" << endl;
                        printMarkup();
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
};



int main() {
    Board b;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cin >> b.board[i][j];
        }
    }
    b.initialMarkup();
    b.printMarkup();
    bool done = false;
    
    while (!done) {
        done = b.elimination();
        if (done) break;
        cout << "after elimination: " << done << endl;
        b.printBoard();
        if (b.loneRangers()) {
            cout << "after lone ranger search: " << endl;
            b.printBoard();
            continue;
        }
        else done = false;
    }
    cout << "final result: " << done << endl;
    b.printBoard();
    return 0;
}
