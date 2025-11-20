#include "regression.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

Regression::Regression() : weights() {}

VectorXd Regression::getWeights() { return weights; }

void Regression::train(MatrixXd xs, VectorXd ysTrue,
                       double learningRate = 0.001, int epochs = 10000) {
  assert(xs.rows() == ysTrue.size());
  MatrixXd A = MatrixXd::Ones(xs.rows(), xs.cols() + 1);
  A.rightCols(xs.cols()) = xs;
  VectorXd w(A.cols());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-w.size(), w.size());
  for (int i = 0; i < w.size(); i++) {
    w(i) = dist(gen);
  }
  for (int epoch = 0; epoch < epochs; epoch++) {
    VectorXd ysPred = A * w;
    VectorXd error = ysPred - ysTrue;
    VectorXd gradient = (2.0 / ysPred.cols()) * (A.transpose() * error);
    w -= learningRate * gradient;
    double sse = 0;
    for (double e : error) {
      sse += pow(e, 2);
    }
    double mse = sse / error.size();
    std::cout << "Loss : " << mse << '\n';
  }
  weights = w;
}

double Regression::predict(VectorXd x) {
  VectorXd newX = VectorXd::Ones(x.size());
  newX.tail(x.size()) = x;
  return newX.dot(weights);
}
