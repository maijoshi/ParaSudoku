#include <iostream>
#include <vector>
#include <fstream>
#include "../CycleTimer.h"

using namespace std;

#define ROW_NUM 16
#define COL_NUM 16
#define BLOCK_SIZE 4
#define MAX_ITER 1

int cnt = 0;

bool noConflicts(int matrix[ROW_NUM][COL_NUM], int row, int col, int num) {
	
	for (int i = 0; i < ROW_NUM; i++) {
		if (matrix[i][col] == num)
			return false;
	}

	for (int j = 0; j < COL_NUM; j++) {
		if (matrix[row][j] == num) 
			return false;
	}

	for (int i = 0; i < BLOCK_SIZE; i++) {
		for (int j = 0; j < BLOCK_SIZE; j++) {
			if (matrix[(row/BLOCK_SIZE)*BLOCK_SIZE + i][(col/BLOCK_SIZE)*BLOCK_SIZE + j] == num) 
				return false;
		}
	}

	return true;
}

bool sudokuSolver(int matrix[ROW_NUM][COL_NUM], int row, int col) {
	cnt++;
        cout << row << ", " << col << endl;
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

	ifstream myfile ("../test/sudoku.txt");

	double time = 0.0;
	cnt = 0;
	for (int i = 0; i < MAX_ITER; i++) {
		while (myfile.is_open() && myfile >> num) {
			matrix[i][j] = num;

			if (j < COL_NUM - 1)
				j++;
			else {
				i++;
				j = 0;
			}
		}
		
		double startTime = CycleTimer::currentSeconds();
		sudokuSolver(matrix, 0, 0);
		double endTime = CycleTimer::currentSeconds();
		time += endTime - startTime;
	}
	printSolution(matrix);
	printf("average time: %f\n", time);
	cout << cnt;
	return 0;
}
