#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "../include/decision_tree.h"

class RandomForest {
public:
  RandomForest(int numTrees, int maxDepth, int numFeatures);
  ~RandomForest();

  void train(const std::vector<DataPoint> &trainingData);
  int predict(const std::vector<double> &features) const;

private:
  std::vector<DecisionTree *> trees;
  int numTrees;
  int maxDepth;
  int numFeatures; // New parameter to store the number of features
};

#endif // RANDOM_FOREST_H
