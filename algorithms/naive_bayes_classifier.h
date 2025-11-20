#pragma once 
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

class NaiveBayesClassifier {
private:
  double k;
  std::set<std::string> classes;
  std::unordered_map<std::string, int> classCount;
  std::set<std::string> tokenSet;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> tokenOfClassCount;
  std::unordered_map<std::string, int> classTokensCount;

public:
  NaiveBayesClassifier();
  NaiveBayesClassifier(std::set<std::string>, double);
  void train(std::vector<std::vector<std::string>>, std::vector<std::string>);

  std::unordered_map<std::string, double> predict(std::vector<std::string>);
  std::unordered_map<std::string, double> getTokenProbabilities(std::string);
  std::set<std::string> getClasses();
};
