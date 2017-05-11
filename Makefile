all: default 

default: crook.cpp
	g++ -std=c++11 -fopenmp -O3 -g -o Solver crook.cpp 
rm:
	rm -rf sudokuSolver bfs  *~ *.*~
