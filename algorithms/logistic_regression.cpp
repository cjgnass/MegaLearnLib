#include "logistic_regression.h"
#include "../tools.h"
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

void LogisticRegression::train(MatrixXd &X, VectorXd &y, double learningRate,
                               int epochs, int batchSize) {
  int nSamples = X.rows();
  int nFeatures = X.cols();
  VectorXd weights = VectorXd::Zero(nFeatures);
  double bias = 0;
  if (batchSize > nSamples || batchSize <= 0) {
    batchSize = nSamples;
  }
  for (int epoch = 0; epoch < epochs; epoch++) {
    vector<int> randomIdxs = getRandomIndices(nSamples);
    for (int batchStart = 0; batchStart < nSamples; batchStart += batchSize) {
      int currBatchSize = min(batchSize, nSamples - batchStart);
      MatrixXd batchX(currBatchSize, nFeatures);
      VectorXd batchy(currBatchSize);
      for (int bi = 0; bi < currBatchSize; bi++) {
        int i = batchStart + bi;
        batchX.row(bi) = X.row(randomIdxs[i]);
        batchy(bi) = y(randomIdxs[i]);
      }
      VectorXd z = batchX * weights;
      z.array() += bias;
      VectorXd sigmoidOutput =
          z.unaryExpr([](double x) { return 1.0 / (1 + exp(-x)); });
      VectorXd error = sigmoidOutput - batchy;
      VectorXd wGrad = (1.0 / currBatchSize) * batchX.transpose() * error;
      double bGrad = error.sum() / currBatchSize;
      weights -= learningRate * wGrad;
      bias -= learningRate * bGrad;
    }
    if (epoch % 10 == 0) {
      VectorXd z = X * weights; 
      z.array() += bias; 
      VectorXd sigmoidOutput =
          z.unaryExpr([](double x) { return 1.0 / (1 + exp(-x)); });
      double negativeLogLikelihood = 0.0;
      for (int samplei = 0; samplei < nSamples; samplei++) { 
        double p = sigmoidOutput(samplei);
        p = min(1.0 - 1e-12, max(1e-12, p));
        negativeLogLikelihood -= y(samplei) * log(p) + (1.0 - y(samplei)) * log(1.0 - p);
      }
      cout << "Epoch: " << epoch << " Loss: " << negativeLogLikelihood << '\n';
    }
  }
  this->weights = weights; 
  this->bias = bias; 
  this->learningRate = learningRate; 
  this->epochs = epochs; 
  this->batchSize = batchSize;
}


int LogisticRegression::predict(VectorXd &x) {
  double z = x.dot(weights);
  z += bias; 
  double sigmoidOut = 1.0 / (1 + exp(-z));
  if (sigmoidOut > .5)
    return 1;
  else
    return 0;
}

VectorXd LogisticRegression::getWeights() { return weights; }
double LogisticRegression::getBias() { return bias; }
double LogisticRegression::getLearningRate() { return learningRate; }
int LogisticRegression::getEpochs() { return epochs; }
int LogisticRegression::getBatchSize() { return batchSize; }
