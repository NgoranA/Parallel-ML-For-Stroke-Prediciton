#include "../include/random_forest.h"
#include "../include/decision_tree.h"
#include <algorithm> // for std::count
#include <cstddef>
#include <random>
#include <vector>

// here we want to initialize the trees with an empty list of trees.
RandomForest::RandomForest(int numTrees, int maxDepth, int numFeatures)
    : numTrees(numTrees), maxDepth(maxDepth), numFeatures(numFeatures) {
  // Initialize the random forest with decision trees
  for (int i = 0; i < numTrees; ++i) {
    DecisionTree *tree = new DecisionTree(numFeatures, maxDepth);
    trees.push_back(tree);
  }
}

RandomForest::~RandomForest() {
  // Deallocate memory for decision trees
  for (DecisionTree *tree : trees) {
    delete tree;
  }
}

void RandomForest::train(const std::vector<DataPoint> &trainingData) {
  // Train each decision tree in the random forest
  for (DecisionTree *tree : trees) {
    // Create a bootstrap sample (sampling with replacement)
    std::vector<DataPoint> bootstrapSample;
    std::random_device random;
    std::mt19937 generate(random());
    std::uniform_int_distribution<size_t> distribute(0,
                                                     trainingData.size() - 1);
    for (size_t i = 0; i < trainingData.size(); ++i) {
      size_t index = distribute(generate);
      /* size_t index = static_cast<size_t>(std::rand() % trainingData.size());
       */
      bootstrapSample.push_back(trainingData[index]);
    }

    // Train the decision tree on the bootstrap sample
    tree->buildTree(bootstrapSample);
  }
}

int RandomForest::predict(const std::vector<double> &features) const {
  // Aggregate predictions from each decision tree (simple majority voting)
  std::vector<int> predictions(
      static_cast<std::vector<int>::size_type>(numTrees), 0);
  for (size_t i = 0; i < trees.size(); ++i) {
    predictions[i] = trees[i]->predict(features);
  }

  // Perform majority voting
  int majorityVote = 0;
  for (int label : predictions) {
    if (std::count(predictions.begin(), predictions.end(), label) >
        std::count(predictions.begin(), predictions.end(), majorityVote)) {
      majorityVote = label;
    }
  }

  return majorityVote;
}
