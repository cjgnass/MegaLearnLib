#include <Eigen/Dense>
#include <iostream>
using Eigen::VectorXd;
using Eigen::MatrixXd;

class LinearRegression {
private:
  VectorXd weights {};

public:
  LinearRegression() {}

  VectorXd getWeights() { 
    return weights;
  }
  
  void train(MatrixXd x, VectorXd yTrue, float learningRate = 0.00001, int epochs = 100000) { 
    MatrixXd A = MatrixXd::Ones(x.rows(), x.cols() + 1); 
    A.rightCols(x.cols()) = x;
    VectorXd w = VectorXd::Ones(A.cols());
    for (int epoch = 0; epoch < epochs; epoch++) { 
      VectorXd yPred = A * w; 
      VectorXd error = yPred - yTrue; 
      VectorXd gradient = (2 / yPred.cols()) * (A.transpose() * error);
      w -= learningRate * gradient;
    }
    weights = w;
  }

  double predict(VectorXd x) { 
    return 1.1;
  }

};
