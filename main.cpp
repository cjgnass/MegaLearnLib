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
  Eigen::MatrixXd X(6, 1);
  X << 9.9, 3.4, 2.4, 6.7, 5.5, 0.3;

  Eigen::VectorXd y(6);
  y << 1, 0, 0, 1, 0, 0;

  LogisticRegression lr;
  lr.train(X, y, 0.01, 100000, 3);

  std::cout << "Weights: " << lr.getWeights().transpose() << '\n';
  std::cout << "Bias: " << lr.getBias() << '\n';

  // Test a new point
  Eigen::VectorXd xTest(1);
  xTest << 5.6; // true y = 2*3 + 3*2 + 1 = 13
  std::cout << "Predicted: " << lr.predict(xTest) << '\n';
}
