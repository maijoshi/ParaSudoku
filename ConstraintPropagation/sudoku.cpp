#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>



using namespace std;

#define ROW_NUM 9
#define COL_NUM 9
#define BLOCK_SIZE 3

vector<string> cross(string rows, string cols) {
	vector<string> squares;

	cout << "rows: " << rows.size() << endl;
	cout << "cols: " << cols.size() << endl;

	for (int i = 0; i < rows.size(); i++) {
		for (int j = 0; j < cols.size(); j++) {

			string str;
			str = rows[i];
			str += cols[j];
			//cout << "str " << str << endl;
			squares.push_back(str);
		}
	}

	return squares; 
}

vector<vector<string> > createUnitList(string rows, string cols) {
	vector<vector<string> > totalVector;

	for (int c = 0; c < COL_NUM; c++) {
        stringstream ss;
        ss << cols[c];
        string s = ss.str();
		totalVector.push_back(cross(rows, s));
	}

	for (int r = 0; r < ROW_NUM; r++) {
        stringstream ss;
        ss << rows[r];
        string s = ss.str();
		totalVector.push_back(cross(s, cols));
	}
    return totalVector;
}

string digits = "123456789";
string rows = "ABCDEFGHI";
string cols = digits; 
vector<string> squares = cross(rows, cols); 
vector<vector<string> > unitlist = createUnitList(rows, cols);


/************************************** CONSTRAINT PROPAGATION ***********************************************/

unordered_map<string, string> eliminate(unordered_map<string, string> values, string s, int d) {
	unordered_map<string, string> empty;
	bool exists = false;
	cout << "eliminate" << endl;

	for (int i = 0; i < values[s].size(); i++) {
		char ch = '0' + d;
		if ((values[s][i] == ch))
			exists = true;
	}
	if (!exists)
		return values; 

	string other_values;
	for (int i = 0; i < values[s].size(); i++) {
		char c = '0' + d;
		if ((values[s][i] != c))
			other_values += values[s][i];
	}
	values[s] = other_values;

	if (values[s].size() == 0) 
		return empty;
	else if (values[s].size() == 1) {
		int d2;
		stringstream str;
		str << values[s];
		str >> d2;
		cout << "d2: " << d2 << endl; /*
		for (int i = 0; i < peers[s].size(); i++) {
			if (eliminate(values, peers[s][i], d2).empty())
				return empty;
		} */
	}

    return empty;
    

}




unordered_map<string, string> assign(unordered_map<string, string> values, string s, int d) {
	string other_values;
	unordered_map<string, string> empty;
	cout << "assign" << endl;

	for (int i = 0; i < values[s].size(); i++) {
		char c = '0' + d;
		if ((values[s][i] != c))
			other_values += values[s][i];
	}

	for (int i = 0; i < other_values.size(); i++) {
		if ((eliminate(values, s, other_values[i])).empty())
			return empty;
	}

	return values;
}

/************************************** PARSE GRID ***********************************************/

unordered_map<string, int> grid_values(int matrix[ROW_NUM][COL_NUM]) {
	unordered_map<string, int> grid_values;

	cout << "grid_values" << endl;
	for (int i = 0; i < ROW_NUM; i++) {
		for (int j = 0; j < COL_NUM; j++) {
			grid_values.insert(pair<string,int>(squares[i*ROW_NUM+j], matrix[i][j]));
			//cout << squares[i*ROW_NUM+j] << " " << matrix[i][j] << endl;
		}
	}

	return grid_values;
}


unordered_map<string, string> parseGrid(int matrix[ROW_NUM][COL_NUM]) {
	unordered_map<string, string> values;
	unordered_map<string, string> empty;
	cout << "parseGrid" << endl;

	for (std::vector<string>::iterator i = squares.begin(); i != squares.end(); ++i) {
    	values.insert(pair<std::string,string>(*i, digits));
    }

    for (auto it : grid_values(matrix)) {
    	if ((it.second != 0) && (assign(values, it.first, it.second).empty()))
    		return empty;
    }

    return values;

}

/***********************************************************************************************/


void printSolution(int matrix[ROW_NUM][COL_NUM]) {
	for (int i = 0; i < ROW_NUM; i++) {
		for (int j = 0; j < COL_NUM; j++) {
			printf(" %d", matrix[i][j]);
		}
		printf("\n");
	}
}

/***********************************************************************************************/


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

	parseGrid(matrix);
	//sudokuSolver();
	printSolution(matrix);
}


