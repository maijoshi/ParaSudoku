//
//  Board.cpp
//
//  Created by Sophie on 5/11/17.
//  Copyright © 2017 Sophie. All rights reserved.
//

#include "Board.hpp"
#include <algorithm>

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