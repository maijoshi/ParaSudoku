#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#define ROW_NUM 9//4
#define COL_NUM 9//4

bool noConflicts(int matrix[ROW_NUM][COL_NUM], int row, int col, int num) {
	
	for (int i = 0; i < ROW_NUM; i++) {
		if (matrix[i][col] == num)
			return false;
	}

	for (int j = 0; j < COL_NUM; j++) {
		if (matrix[row][j] == num) 
			return false;
	}

	return true;
}

bool sudokuSolver(int matrix[ROW_NUM][COL_NUM], int row, int col) {
	
	if (row >= ROW_NUM || col >= COL_NUM) 
		return true;

	if (matrix[row][col]) {
		if (col < COL_NUM - 1) {
			if (sudokuSolver(matrix, row, col + 1)) 
				return true;
		}
		else {
			if (sudokuSolver(matrix, row + 1, 0))
				return true;
		}

		return false;
	}
		

	for (int n = 1; n <= ROW_NUM; n++) {
		if (noConflicts(matrix, row, col, n)) {
			matrix[row][col] = n;

			if (col < COL_NUM - 1) {
				if (sudokuSolver(matrix, row, col + 1)) 
					return true;
			}
			else {
				if (sudokuSolver(matrix, row + 1, 0))
					return true;
			}

			matrix[row][col] = 0;
		}
	}

	return false;

}

void printSolution(int matrix[ROW_NUM][COL_NUM]) {
	for (int i = 0; i < ROW_NUM; i++) {
		for (int j = 0; j < COL_NUM; j++) {
			printf(" %d", matrix[i][j]);
		}
		printf("\n");
	}
}

int main() {
	int matrix[ROW_NUM][COL_NUM];
	int num;
	int i, j = 0;

	ifstream myfile ("sudoku.txt");

	while (myfile.is_open() && myfile >> num) {
		matrix[i][j] = num;

		if (j < COL_NUM - 1)
			j++;
		else {
			i++;
			j = 0;
		}
	}

	sudokuSolver(matrix, 0, 0);
	printSolution(matrix);
	return 0;
}
