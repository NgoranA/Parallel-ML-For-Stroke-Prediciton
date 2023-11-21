#include "../include/decision_tree.h"
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

/* DecisionTree::DecisionTree() : root(nullptr){}; */
/**/
/* DecisionTree::~DecisionTree() { deleteTree(root); } */

// build the decision tree
void DecisionTree::buildTree(const std::vector<DataPoint> &data) {
  // build decisiontree recursively
  root = buildDecisionTree(data);
}

int DecisionTree::predict(const std::vector<double> &features) const {
  // Traverse the decision tree to make predictions
  TreeNode *current = root;
  // leaf nodes are expected to have non-negative labels.
  while (current->label == -1) {
    if (features[current->featureIndex] <= current->threshold) {
      current = current->left;
    } else {
      current = current->right;
    }
  }
  return current->label;
}

double DecisionTree::calculateGiniImpurity(const std::vector<DataPoint> &data) {
  // TODO : to be looked at again.
  size_t total_samples = data.size();
  if (total_samples == 0) {
    return 0.0; // avoid division by zero
  }

  // count the occurencies of each class in the dataset
  std::map<int, size_t> class_counts;
  for (const DataPoint &point : data) {
    class_counts[point.label]++;
  }
  double giniImpurity{1.0};
  for (const auto &pair : class_counts) {
    double proportion = static_cast<double>(pair.second) / total_samples;
    giniImpurity -= proportion * proportion;
  }
  return giniImpurity;
}

void DecisionTree::findBestSplit(const std::vector<DataPoint> &data,
                                 int &bestFeaure, double &bestThreshold) const {

  // TODO : to be be looked at again.
  double bestGini = std::numeric_limits<double>::infinity();

  for (int featureIndex = 0; featureIndex < data[0].features.size();
       ++featureIndex) {
    // sort data based on current feature
    std::vector<DataPoint> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end(),
              [featureIndex](const DataPoint &a, const DataPoint &b) {
                return a.features[featureIndex] < b.features[featureIndex];
              });
    for (int i = 1; i < sorted_data.size(); ++i) {
      double threshold = (sorted_data[i - 1].features[featureIndex] +
                          sorted_data[i].features[featureIndex]) /
                         2.0;
      std::vector<DataPoint> left_data, right_data;
      // split the data into left and right based on the current feature.
      for (const DataPoint &point : sorted_data) {
        if (point.features[featureIndex] <= threshold) {

          left_data.push_back(point);
        } else {
          right_data.push_back(point);
        }
      }
      // calculate Gini Impurity for the split
      double gini = (left_data.size() * calculateGiniImpurity(left_data) +
                     right_data.size() * calculateGiniImpurity(right_data)) /
                    data.size();

      // Update the best split of the current one is better.
      if (gini < bestGini) {
        bestGini = gini;
        bestFeaure = static_cast<int>(featureIndex);
        bestThreshold = threshold;
      }
    }
  }
}

TreeNode *
DecisionTree::buildDecisionTree(const std::vector<DataPoint> &data) const {
  TreeNode *node = new TreeNode;

  // check if all data points have the same label.
  std::set<int> unique_labels{};
  for (const DataPoint &point : data) {
    unique_labels.insert(point.label);
  }

  int unique_labels_count = unique_labels.size();
  if (unique_labels_count == 1) {
    // if all data points have the same label, create a leaf node.
    node->label = data[0].label;
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }

  // Find the best feature and threshold to split on/
  //
  int best_feature;
  double best_threshold;
  // TODO : insterad return the best values from the function.
  findBestSplit(data, best_feature, best_threshold);

  // split the data into lef and right based on the feature and the threshold.
  std::vector<DataPoint> left_data, right_data;
  for (const DataPoint &point : data) {
    if (point.features[best_feature] <= best_threshold) {
      left_data.push_back(point);
    } else {
      right_data.push_back(point);
    }
  }

  // recursively build the righ and left subtress.
  node->featureIndex = best_feature;
  node->threshold = best_threshold;
  node->label = -1;
  node->left = buildDecisionTree(left_data);
  node->right = buildDecisionTree(right_data);

  return node;
}

void DecisionTree::deleteTree(TreeNode *node) {
  if (node != nullptr) {
    deleteTree(node->left);
    deleteTree(node->right);
    deleteTree(node);
  }
}
