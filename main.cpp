#include "./algorithms/linear_regression.h"
#include "./algorithms/logistic_regression.h"
#include "./algorithms/naive_bayes_classifier.h"
#include "./algorithms/support_vector_machine.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <set>
#include <vector>

int main() {
  std::vector<std::vector<std::string>> X = {
      {"offer", "free", "win"}, {"free", "viagra"}, {"meeting", "schedule"},
      {"project", "deadline"},  {"win", "money"},   {"lunch", "meeting"}};

  std::vector<std::string> y = {"spam", "spam", "ham", "ham", "spam", "ham"};

  NaiveBayesClassifier nb;
  nb.train(X, y);

  std::vector<std::string> xTest = {"lunch", "break"};
  auto probs = nb.predict(xTest, 0.5); // smoothing = 1.0, e.g. Laplace

  std::cout << "Predicted class probabilities:\n";
  for (const auto &kv : probs) {
    std::cout << kv.first << ": " << kv.second << '\n';
  }
}
