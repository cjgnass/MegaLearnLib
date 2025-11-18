CXX = g++ 
CXXFLAGS =  -std=c++23 -I ./eigen

all: main

main: main.cpp linear_regression.o
	$(CXX) $(CXXFLAGS) main.cpp linear_regression.o -o main -g

linear_regression.o: ./algorithms/linear_regression.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/linear_regression.cpp -o linear_regression.o

