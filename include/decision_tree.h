#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "../include/dataset_handler.h"
#include <vector>
// Define the TreeNode struct
struct TreeNode {
  int featureIndex;
  double threshold;
  int label;
  TreeNode *left;
  TreeNode *right;
};

class DecisionTree {
public:
  DecisionTree(int numFeatures, int maxDepth);
  ~DecisionTree();

  void buildTree(const std::vector<DataPoint> &data);
  int predict(const std::vector<double> &features) const;

private:
  int numFeatures;
  int maxDepth;
  TreeNode *root;

  // Private methods for tree building
  void recursiveBuildTree(TreeNode *&node, const std::vector<DataPoint> &data,
                          int depth);
  static int getMostFrequentLabel(const std::vector<DataPoint> &data);
  static double calculateGini(const std::vector<DataPoint> &data);
  static void findBestSplit(const std::vector<DataPoint> &data,
                            int &bestFeature, double &bestThreshold);
  void deleteTree(TreeNode *node);
};

#endif // DECISION_TREE_H
