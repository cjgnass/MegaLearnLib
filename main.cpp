#include "./algorithms/linear_regression.h"
#include "./algorithms/logistic_regression.h"
#include "./algorithms/naive_bayes_classifier.h"
#include "./algorithms/support_vector_machine.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <set>
#include <vector>

int main() {
  Eigen::MatrixXd xs(8, 2);
  xs << 2, 2, 2, 3, 3, 2, 3, 3, -2, -2, -2, -3, -3, -2, -3, -3;

  Eigen::VectorXd ys(8);
  ys << 1, 1, 1, 1, -1, -1, -1, -1;
  SupportVectorMachine svm;
  svm.train(xs, ys, 1.0, 0.001, 1000, 4);

  Eigen::VectorXd x1(2);
  x1 << 6, 7;
  Eigen::VectorXd x2(2);
  x2 << -6, -7;
  
  std::cout << svm.predict(x1) << '\n';
  std::cout << svm.predict(x2) << '\n';
}

