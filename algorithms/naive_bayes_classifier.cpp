#include "naive_bayes_classifier.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace std;

void NaiveBayesClassifier::train(vector<vector<string>> &X, vector<string> &y) {
  int nSamples = X.size();

  set<string> classes{};
  for (int samplei = 0; samplei < nSamples; samplei++) {
    vector<string> xi = X[samplei];
    string yi = y[samplei];
    classes.insert(yi);
    if (classCount.find(yi) == classCount.end())
      classCount[yi] = 1;
    else
      classCount[yi]++;
    if (tokenOfClassCount.find(yi) == tokenOfClassCount.end())
      tokenOfClassCount[yi] = {};
    for (string token : xi) {
      tokenSet.insert(token);
      classTokensCount[yi]++;
      if (tokenOfClassCount[yi].find(token) == tokenOfClassCount[yi].end())
        tokenOfClassCount[yi][token] = 1;
      else
        tokenOfClassCount[yi][token]++;
    }
  }
  this->smoothing = smoothing;
  this->classes = classes;
}

unordered_map<string, double> NaiveBayesClassifier::predict(vector<string> &x,
                                                            double smoothing) {
  this->smoothing = smoothing;
  unordered_map<string, double> logOutputProbabilities{};
  unordered_map<string, double> outputProbabilities{};
  for (string token : tokenSet) {
    unordered_map<string, double> tokenProbabilities =
        getTokenProbabilities(token);
    for (auto &[yi, num] : classCount) {
      if (logOutputProbabilities.find(yi) == logOutputProbabilities.end())
        logOutputProbabilities[yi] = 0.0;
      if (find(x.begin(), x.end(), token) == x.end()) {
        logOutputProbabilities[yi] += log(1.0 - tokenProbabilities[yi]);

      } else {
        logOutputProbabilities[yi] += log(tokenProbabilities[yi]);
      }
    }
  }
  for (auto &[classi, logProbability] : logOutputProbabilities) {
    outputProbabilities[classi] = exp(logProbability);
  }
  return outputProbabilities;
}

unordered_map<string, double>
NaiveBayesClassifier::getTokenProbabilities(string &token) {
  cout << "Smoothing : " << smoothing << '\n';
  unordered_map<string, double> tokenProbabilities{};
  for (auto &[yi, num] : classTokensCount) {
    double numerator = tokenOfClassCount[yi][token] + smoothing;
    double denominator = num + smoothing * tokenSet.size();
    tokenProbabilities[yi] = numerator / denominator;
  }
  return tokenProbabilities;
}

set<string> NaiveBayesClassifier::getClasses() {
  for (string c : classes) {
    cout << c << '\n';
  }
  return classes;
}

double NaiveBayesClassifier::getSmoothing() { return smoothing; }
