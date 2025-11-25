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
  Eigen::MatrixXd X(6, 2);
  X << 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1, 2;

  Eigen::VectorXd y(6);
  y << 1, 3, 4, 6, 9, 11;

  LinearRegression lr;
  lr.train(X, y, 0.0001, 100000, 2);

  std::cout << "Weights: " << lr.getWeights().transpose() << '\n';
  std::cout << "Bias: " << lr.getBias() << '\n';

  // Test a new point
  Eigen::VectorXd xTest(2);
  xTest << 3, 2; // true y = 2*3 + 3*2 + 1 = 13
  std::cout << "Predicted: " << lr.predict(xTest) << '\n';
}
