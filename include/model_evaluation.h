#ifndef MODEL_EVALUATION_H
#define MODEL_EVALUATION_H

#include "dataset_handler.h"
#include "random_forest.h"

#include <vector>

struct ConfusionMatrix {
  int truePositive;
  int trueNegative;
  int falsePositive;
  int falseNegative;

  ConfusionMatrix()
      : truePositive(0), trueNegative(0), falsePositive(0), falseNegative(0) {}
};
void evaluateModel(const std::vector<DataPoint> &validationSet,
                   const std::vector<int> &predictedLabels);

void calculateMetrics(const ConfusionMatrix &matrix, double &accuracy,
                      double &precision, double &recall, double &f1Score);
#endif // !MODEL_EVALUATION_H
