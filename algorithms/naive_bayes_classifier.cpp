#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>
using namespace std;

class NaiveBayesClassifier {
private:
  double k = 0.1;
  set<string> classes{};
  unordered_map<string, int> classCount{};
  set<string> tokenSet{};
  unordered_map<string, unordered_map<string, int>> tokenOfClassCount{};
  unordered_map<string, int> classTokensCount{};

public:
  NaiveBayesClassifier() {}
  NaiveBayesClassifier(set<string> classes, double k = 0.1) {
    this->classes = classes;
    this->k = k;
  }

  void train(vector<vector<string>> xs, vector<string> ys) {
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

  unordered_map<string, double> predict(vector<string> x) {
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

  unordered_map<string, double> getTokenProbabilities(string token) {
    unordered_map<string, double> tokenProbabilities{};
    for (auto &[yi, num] : classTokensCount) {
      double numerator = tokenOfClassCount[yi][token] + k;
      double denominator = num + k * tokenSet.size();
      tokenProbabilities[yi] = numerator / denominator;
    }
    return tokenProbabilities;
  }

  set<string> getClasses() {
    for (string c : classes) {
      cout << c << '\n';
    }
    return classes;
  }
};
