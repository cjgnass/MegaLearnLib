#pragma once 
#include <algorithm>
#include <set>
#include <vector>
#include <string> 
#include <unordered_map>

class NaiveBayesClassifier {
public:
  void train(std::vector<std::vector<std::string>> &X, std::vector<std::string> &y);
  std::unordered_map<std::string, double> predict(std::vector<std::string> &x, double smoothing = 0.5);
  std::unordered_map<std::string, double> getTokenProbabilities(std::string &token);
  std::set<std::string> getClasses();
  double getSmoothing();

private:
  double smoothing;
  std::set<std::string> classes;
  std::unordered_map<std::string, int> classCount;
  std::set<std::string> tokenSet;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> tokenOfClassCount;
  std::unordered_map<std::string, int> classTokensCount;
};
