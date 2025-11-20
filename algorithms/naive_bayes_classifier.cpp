#include "naive_bayes_classifier.h"
using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier()
    : k(0.1), classes(), classCount(), tokenSet(), tokenOfClassCount(),
      classTokensCount() {}

void NaiveBayesClassifier::train(vector<vector<string>> xs, vector<string> ys) {
  assert(xs.size() == ys.size());
  if (classes.empty()) {
    for (string y : ys) {
      classes.insert(y);
    }
  }
  int numOfSamples = xs.size();
  for (int i = 0; i < numOfSamples; i++) {
    vector<string> xi = xs[i];
    string yi = ys[i];
    if (classCount.find(yi) == classCount.end())
      classCount[yi] = 1;
    else
      classCount[yi]++;

    if (tokenOfClassCount.find(yi) == tokenOfClassCount.end())
      tokenOfClassCount[yi] = {};

    for (string token : xi) {
      tokenSet.insert(token);

      if (classTokensCount.find(yi) == classTokensCount.end())
        classTokensCount[yi] = 1;
      else
        classTokensCount[yi]++;

      if (tokenOfClassCount[yi].find(token) == tokenOfClassCount[yi].end())
        tokenOfClassCount[yi][token] = 1;
      else
        tokenOfClassCount[yi][token]++;
    }
  }
}

unordered_map<string, double> NaiveBayesClassifier::predict(vector<string> x) {
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
NaiveBayesClassifier::getTokenProbabilities(string token) {
  unordered_map<string, double> tokenProbabilities{};
  for (auto &[yi, num] : classTokensCount) {
    double numerator = tokenOfClassCount[yi][token] + k;
    double denominator = num + k * tokenSet.size();
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
