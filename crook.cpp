//
//  main.cpp
//  sudoku
//
//  Created by Sophie on 5/7/17.
//  Copyright © 2017 Sophie. All rights reserved.
//

#include <iostream>
#include <vector>
#define ROW 0
#define COL 1
#define BOX 2
using namespace std;

const int size = 9;
int box_size = 3;

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
                    if ((markup[i][j] >> k) & 1)   cout << k+1 << " ";
                cout << endl;
            }
        }
    }
    bool markupContains(int i, int j, int val) {
        return (markup[i][j] >> (val-1)) & 1;
    }
    void removeFromMarkup(int i, int j, int val) {
        markup[i][j] &= ~(1 << (val-1));
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
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                cnt += (board[i][j] == 0);
        return cnt;
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
    bool elimination();
    bool loneRangers();
    bool findTwins();
    bool findTriplets();
    bool findPreemptiveSet(int setSize);
};

bool Board::elimination() {
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

//bool Board::findTwins() { // maybe when I have time
//    // check row
//    for (int i = 0; i < size; i++) {
//        for (int n1 = 1; n1 <= size; n1++) {
//            for (
//        }
//        for (int j = 0;)
//    }
//    return false;
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
                cout << (i + 1) << " ";
                combination.push_back(i);
            }
        }
        cout << endl;
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
        vector<vector<int>> indexes = comb(num, setSize);
        for (int i = 0; i < indexes.size(); i++) {
            int res = 0;
            for (int in : indexes[i]) {
                res |= markup[row][unfilled[in]];
            }
            int cnt = 0;
            for (int k = 0; k < size; k++) {
                if ((res >> k) & 1) cnt++;
            }
            if (cnt == setSize) {
                // found preemptive set
                vector<int> pset;
                for (int k = 0; k < size; k++) {
                    if ((res >> k) & 1) {
                        for (int jj = 0; jj < size; jj++)
                            removeFromMarkup(row, jj, k);
                    }
                }
                
            }
        }
//        for (int j = 0; j < size; j++) {
//            if (getPossibilitiesCnt(i, j) == 2) {
//                for (int k = j+1; k < size; k++) {
//                    if (markup[i][k] == markup[i][j]) {
//                        vector<int> twinNums = getPossibleValues(i, j);
//                        // found twins
//                        for (int jj = 0; jj < size; jj++) {
//                            if (jj == k || jj == j) continue;
//                            removeFromMarkup(i, jj, twinNums[0]);
//                            removeFromMarkup(i, jj, twinNums[1]);
//                        }
//                    }
//                }
//            }
//        }
    }
    return false;
}

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
        b.findPreemptiveSet(2);
        cout << "after findPreemptiveSet 2: " << endl;
        b.printMarkup();
        b.printBoard();
        
        b.findPreemptiveSet(3);
        cout << "after findPreemptiveSet 3: " << endl;
        b.printMarkup();
        b.printBoard();
    }
    cout << "final result: " << done << endl;
    b.printBoard();
    return 0;
}
