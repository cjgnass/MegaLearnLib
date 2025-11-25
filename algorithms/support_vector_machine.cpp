#include "support_vector_machine.h"
#include "../tools.h"
#include <iostream>

using namespace std;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void SupportVectorMachine::train(MatrixXd &X, VectorXd &y,
                                 double regularization, double learningRate,
                                 int epochs, int batchSize) {
  int nSamples = X.rows();
  int nFeatures = X.cols();
  if (batchSize > nSamples || batchSize <= 0)
    batchSize = nSamples;
  double bias{0};
  VectorXd weights = VectorXd::Zero(nFeatures);
  for (int epoch = 0; epoch < epochs; epoch++) {
    vector<int> randomIdxs = getRandomIndices(nSamples);
    for (int batchStart = 0; batchStart < nSamples; batchStart += batchSize) {
      int currBatchSize =
          batchStart + batchSize > nSamples ? nSamples - batchStart : batchSize;
      MatrixXd batchX(currBatchSize, nFeatures);
      VectorXd batchy(currBatchSize);

      for (int i = batchStart; i < currBatchSize; i++) {
        int bi = i - batchStart;
        batchX.row(bi) = X.row(randomIdxs[i]);
        batchy(bi) = y(randomIdxs[i]);
      }
      VectorXd decisionScores = batchX * weights;
      decisionScores.array() += bias;
      ArrayXd margins = batchy.array() * decisionScores.array();
      ArrayXd active = (margins < 1.0).cast<double>();
      VectorXd yActive = (active * batchy.array()).matrix();
      VectorXd wGrad =
          weights - (regularization / static_cast<double>(batchSize)) *
                        (batchX.transpose() * yActive);
      double bGrad =
          (regularization / static_cast<double>(batchSize)) * (-yActive.sum());
      weights -= learningRate * wGrad;
      bias -= learningRate * bGrad;
    }
    VectorXd decisionScore = X * weights;
    decisionScore.array() += bias;
    ArrayXd margins = y.array() * decisionScore.array();
    if (epoch % 10 == 0) {
      ArrayXd hinge = (1.0 - margins).max(0.0);
      double loss = 0.5 * weights.squaredNorm() + regularization * hinge.mean();
      cout << "Epoch " << epoch << " Loss: " << loss << '\n';
    }
  }
  this->weights = weights;
  this->bias = bias;
  this->regularization = regularization;
  this->learningRate = learningRate;
  this->epochs = epochs;
  this->batchSize = batchSize;
}

int SupportVectorMachine::predict(VectorXd &x) { 
  double decisionScore = x.dot(weights);
  decisionScore += bias;
  return decisionScore > 0 ? 1 : -1;
}

VectorXd SupportVectorMachine::getWeights() { return weights; }

double SupportVectorMachine::getBias() { return bias; }

double SupportVectorMachine::getRegularization() { return regularization; }

double SupportVectorMachine::getLearningRate() { return learningRate; }

int SupportVectorMachine::getEpochs() { return epochs; }

int SupportVectorMachine::getBatchSize() { return batchSize; }
