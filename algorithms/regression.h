#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random> 
#include <cassert>

class Regression {
private:
  Eigen::VectorXd weights;

public:
  Regression();
  Eigen::VectorXd getWeights();
  void train(Eigen::MatrixXd, Eigen::VectorXd, double, int);
  double predict(Eigen::VectorXd x);
};
