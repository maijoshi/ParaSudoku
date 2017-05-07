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

int size = 9;
int box_size = 3;

class Board {
public:
    vector<vector<vector<int>>> markup;
    vector<vector<int>> board;
    Board():board(size, vector<int>(size)), markup(size, vector<vector<int>>(size, vector<int>())) {};
    void removeFromMarkup(int i, int j, int val) {
//        for (auto iter = markup[i][j].begin(); iter != markup[i][j].end(); iter++)
//            if (*iter == val) {
//                markup[i][j].erase(iter);
//                break;
//            }
            if(!board[i][j])    markup[i][j][val-1] = 0;
//                int cnt = 0;
//                int index;
//                for (int k = 0; k < size; k++)
//                    if (markup[i][j][k]) {cnt++; index = k;}
//                if (cnt == 1) board[i][j] = markup[i][j][k];
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
                for (auto k : markup[i][j])
                    if (k)  cout << k << " ";
                cout << endl;
            }
        }
    }
    void initialMarkup() {
        vector<int> all_possible;
        for (int i = 1; i <= size; i++)
            all_possible.push_back(i);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                if (!board[i][j])    markup[i][j] = all_possible;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (board[i][j]) {
                    for (int k = 0; k < size; k++) {
                        removeFromMarkup(i, k, board[i][j]);
//                        removeFromMarkup(k, j, board[i][j]);
                        int box_x = k/box_size+i/box_size*box_size;
                        int box_y = k%box_size+j/box_size*box_size;
//                        removeFromMarkup(box_x, box_y, board[i][j]);
                    }
                }
            }
        }
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
    return 0;
}
