#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random> 
#include <cassert>

class LinearRegression {
private:
  Eigen::VectorXd weights;

public:
  LinearRegression();
  Eigen::VectorXd getWeights();
  void train(Eigen::MatrixXd xs, Eigen::VectorXd ysTrue, double learningRate = 0.01, int epochs = 5000);
  double predict(Eigen::VectorXd x);
};
