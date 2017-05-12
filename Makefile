all: default 

default: crook.cpp Board.hpp Board.cpp 
	g++ -std=c++11 -fopenmp -O3 -g -o Solver crook.cpp Board.hpp Board.cpp 
rm:
	rm -rf sudokuSolver bfs  *~ *.*~
