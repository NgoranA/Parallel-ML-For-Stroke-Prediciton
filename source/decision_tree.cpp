#include "../include/decision_tree.h"
#include "iostream"
#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>
#include <vector>

DecisionTree::DecisionTree(int numFeatures, int maxDepth)
    : numFeatures(numFeatures), maxDepth(maxDepth), root(nullptr) {
  // Constructor implementation
}

DecisionTree::~DecisionTree() { deleteTree(root); }

void DecisionTree::buildTree(const std::vector<DataPoint> &data) {
  recursiveBuildTree(root, data, 0);
}

// TODO:
int DecisionTree::predict(const std::vector<double> &features) const {
  // Implement prediction logic using the decision tree
  // This involves traversing the tree based on the features
  // and returning the predicted label at the leaf node.
  // This logic depends on the specific problem type

  // Start at the root of the tree
  TreeNode *currentNode = root;

  // Traverse the tree until a leaf node is reached
  while (currentNode != nullptr && currentNode->label == -1) {
    // Determine the direction to move (left or right) based on the feature
    // value
    if (features[static_cast<std::vector<DataPoint>::size_type>(
            currentNode->featureIndex)] <= currentNode->threshold) {
      currentNode = currentNode->left;
    } else {
      currentNode = currentNode->right;
    }
  }

  // Return the label of the leaf node
  return currentNode != nullptr ? currentNode->label
                                : -1; // Default to -1 if leaf node is not found
}

void DecisionTree::recursiveBuildTree(TreeNode *&node,
                                      const std::vector<DataPoint> &data,
                                      int depth) {

  // Implement recursive tree-building logic here
  // This involves finding the best split, creating child nodes, and recursively
  // building the tree. This logic depends on the specific problem type
  // (classification/regression). In our implementation the problem is a
  // classification problem.

  // Check termination conditions
  if (depth >= maxDepth || data.empty()) {
    // Create a leaf node and assign the most frequent label in the remaining
    // data
    node = new TreeNode{0, 0.0, getMostFrequentLabel(data), nullptr, nullptr};
    return;
  }

  // Find the best split based on Gini impurity
  int bestFeature;
  double bestThreshold;
  std::cout << "Best split start\n";
  findBestSplit(data, bestFeature, bestThreshold);
  std::cout << "Best split end\n";
  // Split the data into left and right subsets
  std::vector<DataPoint> leftSubset, rightSubset;
  for (const DataPoint &point : data) {
    if (point.features[static_cast<std::vector<DataPoint>::size_type>(
            bestFeature)] <= bestThreshold) {
      leftSubset.push_back(point);
    } else {
      rightSubset.push_back(point);
    }
  }

  // Create an internal node
  node = new TreeNode{bestFeature, bestThreshold, -1, nullptr, nullptr};

  // Recursively build left and right subtrees
  recursiveBuildTree(node->left, leftSubset, depth + 1);
  recursiveBuildTree(node->right, rightSubset, depth + 1);
}

int DecisionTree::getMostFrequentLabel(const std::vector<DataPoint> &data) {

  // Implement logic to determine the most frequent label in the dataset.

  // Count occurrences of each label
  std::unordered_map<int, int> labelCounts;

  for (const DataPoint &point : data) {
    int label = point.labels;
    labelCounts[label]++;
  }

  // Find the label with the maximum count
  int mostFrequentLabel = data[0].labels; // Default to the first label
  int maxCount = 0;

  for (const auto &pair : labelCounts) {
    if (pair.second > maxCount) {
      mostFrequentLabel = pair.first;
      maxCount = pair.second;
    }
  }

  return mostFrequentLabel;
}

double DecisionTree::calculateGini(const std::vector<DataPoint> &data) {
  // Calculate Gini impurity for a given set of data points
  std::cout << "Gini Calculation Start"
            << "\n";
  std::vector<DataPoint>::size_type totalDataPoints = data.size();
  if (totalDataPoints == 0) {
    return 0.0; // If the dataset is empty, Gini impurity is 0
  }

  // Count occurrences of each label
  std::vector<int> labelCounts; // Assuming labels are non-negative integers
  for (const DataPoint &point : data) {
    int label = point.labels;
    if (labelCounts.size() <= static_cast<size_t>(label)) {
      labelCounts.resize(static_cast<size_t>(label) + 1, 0);
    }
    labelCounts[static_cast<size_t>(label)]++;
  }

  // Calculate Gini impurity
  double giniImpurity = 1.0;
  for (int labelCount : labelCounts) {
    double labelProbability =
        static_cast<double>(labelCount) / static_cast<double>(totalDataPoints);
    giniImpurity -= labelProbability * labelProbability;
  }
  std::cout << "Gini calculation ends and returns"
            << "\n";
  std::cout << "Geni Impurity value: " << giniImpurity << "\n";
  return giniImpurity;
}

void DecisionTree::findBestSplit(const std::vector<DataPoint> &data,
                                 int &bestFeature, double &bestThreshold) {
  // Implement logic to find the best split for a given set of data points
  // This involves iterating over features and thresholds to minimize Gini
  // impurity.
  const std::vector<DataPoint>::size_type localNumFeatures =
      data[0].features.size();
  const std::vector<DataPoint>::size_type numDataPoints = data.size();
  double bestGini = 1.0; // Initialize with maximum impurity

  // Iterate over each feature
  for (std::vector<DataPoint>::size_type featureIndex = 0;
       featureIndex < localNumFeatures; ++featureIndex) {
    // Sort the data points based on the current feature
    std::vector<DataPoint> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end(),
              [featureIndex](const DataPoint &a, const DataPoint &b) {
                return a.features[featureIndex] < b.features[featureIndex];
              });

    // Iterate over possible thresholds
    for (int i = 1; i < static_cast<int>(numDataPoints); ++i) {
      // Calculate Gini impurity for the split at this threshold

      std::cout << "Gini calculation starts in best split function";
      double leftGini = calculateGini(
          {sortedData.begin(),
           sortedData.end() +
               static_cast<std::vector<DataPoint>::difference_type>(i)});
      double rightGini = calculateGini(
          {sortedData.begin() +
               static_cast<std::vector<DataPoint>::difference_type>(i),
           sortedData.end()});
      double weightedGini =
          (i * leftGini +
           (static_cast<double>(numDataPoints) - i) * rightGini) /
          static_cast<double>(numDataPoints);

      std::cout << "Gini calculation ends in best split function";
      // Update the best split if the current one is better
      if (weightedGini < bestGini) {
        bestGini = weightedGini;
        bestFeature = static_cast<int>(featureIndex);
        bestThreshold =
            (sortedData[static_cast<std::vector<DataPoint>::size_type>(i) - 1]
                 .features[featureIndex] +
             sortedData[static_cast<std::vector<DataPoint>::size_type>(i)]
                 .features[featureIndex]) /
            2.0;
      }
    }
  }
}

void DecisionTree::deleteTree(TreeNode *node) {
  // Implement logic to deallocate memory for the tree nodes.
  // This involves recursively deleting child nodes.
  // This logic depends on the specific structure of your tree.
  if (node == nullptr) {
    return; // Base case: Reached a null (empty) node
  }

  // Recursively delete the left and right subtrees
  deleteTree(node->left);
  deleteTree(node->right);

  // Delete the current node
  delete node;
}
