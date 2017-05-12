all: default 

default: crook.cpp 
	g++ -std=c++11 -fopenmp -O3 -g crook.cpp -o Solver 
rm:
	rm -rf sudokuSolver bfs  *~ *.*~
