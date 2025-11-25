#include "linear_regression.h"
#include "../tools.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void LinearRegression::train(MatrixXd &X, VectorXd &y, double learningRate,
                             int epochs, int batchSize) {
  int nSamples = X.rows();
  int nFeatures = X.cols();
  if (batchSize > nSamples || batchSize <= 0)
    batchSize = nSamples;
  VectorXd weights = VectorXd::Zero(nFeatures);
  double bias = 0;
  for (int epoch = 0; epoch < epochs; epoch++) {
    vector<int> randomIdxs = getRandomIndices(nSamples);
    for (int batchStart = 0; batchStart < nSamples; batchStart += batchSize) {
      int currBatchSize = min(nSamples - batchStart, batchSize);
      MatrixXd batchX(currBatchSize, nFeatures);
      VectorXd batchy(currBatchSize);
      for (int bi = 0; bi < currBatchSize; bi++) {
        int i = batchStart + bi;
        batchX.row(bi) = X.row(randomIdxs[i]);
        batchy(bi) = y(randomIdxs[i]);
      }

      VectorXd prediction = batchX * weights;
      prediction.array() += bias;
      VectorXd error = prediction - batchy;

      VectorXd wGrad = (2.0 / currBatchSize) * batchX.transpose() * error;
      double bGrad = (2.0 / currBatchSize) * error.sum();

      weights -= learningRate * wGrad;
      bias -= learningRate * bGrad;
    }
    if (epoch % 10 == 0) {
      VectorXd prediction = X * weights;
      prediction.array() += bias;
      VectorXd error = prediction - y;
      cout << "Epoch: " << epoch << " Loss: " << error.squaredNorm() << '\n';
    }
  }

  this->weights = weights;
  this->bias = bias;
  this->learningRate = learningRate;
  this->epochs = epochs;
  this->batchSize = batchSize;
}

double LinearRegression::predict(VectorXd &x) {
  return x.dot(weights) + bias;
}

VectorXd LinearRegression::getWeights() { return weights; }
double LinearRegression::getBias() { return bias; }
double LinearRegression::getLearningRate() { return learningRate; }
int LinearRegression::getEpochs() { return epochs; }
int LinearRegression::getBatchSize() { return batchSize; }
