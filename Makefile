all: default 

default: crook.cpp
	g++ -std=c++11 -fopenmp -O3 -g -o sudokuSolver crook.cpp 
rm:
	rm -rf sudokuSolver bfs  *~ *.*~
