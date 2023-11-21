#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>

struct TreeNode {
  int featureIndex;
  double threshold;
  int label;
  TreeNode *left;
  TreeNode *right;
};

struct DataPoint {
  std::vector<double> features;
  int label;
};

class DecisionTree {

public:
  DecisionTree();
  ~DecisionTree();
  void buildTree(const std::vector<DataPoint> &data);

  int predict(const std::vector<double> &features) const;

private:
  double static calculateGiniImpurity(const std::vector<DataPoint> &data);
  void findBestSplit(const std::vector<DataPoint> &data, int &bestFeaure,
                     double &bestThreshold) const;
  TreeNode *buildDecisionTree(const std::vector<DataPoint> &data) const;
  void deleteTree(TreeNode *node);

private:
  TreeNode *root;
};
#endif // DECISION_TREE_H
