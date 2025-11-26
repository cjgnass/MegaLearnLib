#pragma once
#include <vector>

enum class FeatureType : int { Numerical, Categorical };

struct Feature {
  FeatureType featureType = FeatureType::Numerical;
  double numericalFeature = 0.0;
  std::vector<int> categoricalFeature = {};
};

struct TreeNode {};

class DecisionTree {
public:
  void train(const std::vector<std::vector<Feature>> &X,
             const std::vector<int> &y, int maxDepth, int minSamplesSplit,
             int maxFeatures);

private:
};
