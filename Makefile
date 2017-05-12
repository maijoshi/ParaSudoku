all: default 

default: crook.cpp Board.hpp Board.cpp 
	g++ -std=c++11 -fopenmp -O3 -g crook.cpp Board.hpp Board.cpp -o Solver 
rm:
	rm -rf sudokuSolver bfs  *~ *.*~
