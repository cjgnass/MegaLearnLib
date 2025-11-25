#pragma once
#include <Eigen/Dense>

class SupportVectorMachine {
public:
  void train(Eigen::MatrixXd &X, Eigen::VectorXd &y, double regularization = 1,
             double learningRate = 0.01, int epochs = 5000, int batchSize = 0);
  int predict(Eigen::VectorXd &x);
  Eigen::VectorXd getWeights();
  double getBias();
  double getRegularization();
  double getLearningRate();
  int getEpochs();
  int getBatchSize();

private:
  Eigen::VectorXd weights;
  double bias;
  double regularization;
  double learningRate;
  int epochs;
  int batchSize;
};
