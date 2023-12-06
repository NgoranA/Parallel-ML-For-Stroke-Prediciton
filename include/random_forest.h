#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "../include/decision_tree.h"
#include <utility>
#include <vector>

class RandomForest {
public:
  RandomForest(int numTrees, int maxDepth, int numFeatures);
  ~RandomForest();

  void train(const std::vector<DataPoint> &trainingData);
  std::vector<int> predict(const std::vector<double> &features) const;
  static int majorityVoting(const std::vector<int> &predictions);

private:
  std::vector<DecisionTree *> trees;
  int numTrees;
  int maxDepth;
  int numFeatures; // New parameter to store the number of features
};

#endif // RANDOM_FOREST_H
