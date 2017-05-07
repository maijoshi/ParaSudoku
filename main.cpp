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

int ROW_NUM = 2;
int COL_NUM = 2;
int main() {
    vector<vector<int>> mat(ROW_NUM,vector<int>(COL_NUM));
    for (int i = 0; i < ROW_NUM; i++) {
        for (int j = 0; j < COL_NUM; j++) {
            cin >> mat[i][j];
        }
    }
    cout << mat[ROW_NUM-1][COL_NUM-1];
    return 0;
}
