#include "logistic_regression.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

LogisticRegression::LogisticRegression() {}
void LogisticRegression::train(MatrixXd &xs, VectorXd &ysTrue,
                               double learningRate, int epochs) {
  MatrixXd A = MatrixXd::Ones(xs.rows(), xs.cols() + 1);
  A.rightCols(xs.cols()) = xs;
  VectorXd w(A.cols());
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<double> dist(-w.size(), w.size());
  for (int i = 0; i < w.size(); i++) {
    w(i) = dist(gen);
  }
  for (int epoch = 0; epoch < epochs; epoch++) {
    VectorXd ysPred = A * w;
    VectorXd sigmoidOut =
        ysPred.unaryExpr([](double x) { return 1.0 / (1 + exp(-x)); });
    VectorXd gradient = A.transpose() * (sigmoidOut - ysTrue) / ysTrue.rows();
    w -= learningRate * gradient;
    double nll = 0.0;
    for (int samplei = 0; samplei < ysTrue.size(); samplei++) {
      double p = sigmoidOut(samplei);
      p = min(1.0 - 1e-12, max(1e-12, p));
      nll += -(ysTrue(samplei) * log(p) + (1.0 - ysTrue(samplei)) * log(1.0 - p));
    }
    cout << "Loss : " << nll << '\n';
  }
  weights = w;
}

int LogisticRegression::predict(VectorXd &x) { 
  VectorXd newX = VectorXd::Ones(x.size() + 1);
  newX.tail(x.size()) = x; 
  double sigmoidOut = 1.0 / (1 + exp(- newX.dot(weights)));
  if (sigmoidOut > .5) return 1;
  else return 0;
}

VectorXd LogisticRegression::getWeights() { 
  return weights;
}

