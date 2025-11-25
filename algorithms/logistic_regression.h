// #include <Eigen/Dense>
#include <Eigen/Dense>
#include <iostream>
#include <random>

class LogisticRegression {
public:
  LogisticRegression();
  void train(Eigen::MatrixXd &xs, Eigen::VectorXd &ys,
             double learningRate = 0.001, int epochs = 50000);
  int predict(Eigen::VectorXd &x);
  Eigen::VectorXd getWeights();

private:
  Eigen::VectorXd weights;
};
