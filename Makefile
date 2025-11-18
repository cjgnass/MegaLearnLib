CXX = g++ 
CXXFLAGS =  -std=c++23 -I ./eigen

all: main

main: main.cpp regression.o naive_bayes_classifier.o
	$(CXX) $(CXXFLAGS) main.cpp regression.o naive_bayes_classifier.o -o main -g

regression.o: ./algorithms/regression.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/regression.cpp -o regression.o

naive_bayes_classifier.o: ./algorithms/naive_bayes_classifier.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/naive_bayes_classifier.cpp -o naive_bayes_classifier.o

clean: 
	rm -f *.o main
