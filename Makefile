CXX = g++ 
CXXFLAGS =  -std=c++23 -I ./eigen

all: main

main: main.cpp tools.o linear_regression.o naive_bayes_classifier.o logistic_regression.o support_vector_machine.o
	$(CXX) $(CXXFLAGS) main.cpp linear_regression.o naive_bayes_classifier.o logistic_regression.o support_vector_machine.o -o main -g

tools.o: ./tools.cpp 
	$(CXX) $(CXXFLAGS) -c ./tools.cpp -o tools.o

linear_regression.o: ./algorithms/linear_regression.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/linear_regression.cpp -o linear_regression.o

naive_bayes_classifier.o: ./algorithms/naive_bayes_classifier.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/naive_bayes_classifier.cpp -o naive_bayes_classifier.o

logistic_regression.o: ./algorithms/logistic_regression.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/logistic_regression.cpp -o logistic_regression.o

support_vector_machine.o: ./algorithms/support_vector_machine.cpp
	$(CXX) $(CXXFLAGS) -c ./algorithms/support_vector_machine.cpp -o support_vector_machine.o

clean: 
	rm -f *.o main
