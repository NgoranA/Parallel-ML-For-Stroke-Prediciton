#include "../include/model_evaluation.h"

#include "iostream"
#include <limits>
#include <vector>

void evaluateModel(const std::vector<DataPoint> &validationSet,
                   const std::vector<int> &predictedLabels) {
  ConfusionMatrix confusionMatrix;

  // Assuming that the true labels are found in the validation dataset

  for (size_t i = 0; i < validationSet.size(); ++i) {

    // check if i is within the valid range
    if (i < validationSet.size()) {
      // verify each datapoint has the labels field
      if (validationSet[i].labels == 0) {
        // make prediction using the random forest
        int predictedLabel = predictedLabels[i];
        // compare the predictedLabel with the true labell
        int trueLabel = validationSet[i].labels;

        // update metrics based on the comparison
        if (trueLabel == 1) {
          if (predictedLabel == 1) {
            confusionMatrix.truePositive++;
          } else {
            confusionMatrix.trueNegative++;
          }
        } else {
          if (predictedLabel == 1) {
            confusionMatrix.falsePositive++;
          } else {
            confusionMatrix.falseNegative++;
          }
        }
      } else {
        std::cerr << "Error: Unexpected value in labels field."
                  << validationSet[i].labels << std::endl;
      }
    } else {
      std::cerr << "Error: Index 'i' is out of range." << std::endl;
    }
  }
  // calculate metrics
  double accuracy, precision, recall, f1Score;
  calculateMetrics(confusionMatrix, accuracy, precision, recall, f1Score);

  std::cout << "╔══════════════════╗" << std::endl;
  std::cout << "║ Model Evaluation ║" << std::endl;
  std::cout << "╚══════════════════╝" << std::endl;
  // Print or use the calculated metrics as needed
  std::cout << "Accuracy: " << accuracy << std::endl;
  std::cout << "Precision: " << precision << std::endl;
  std::cout << "Recall: " << recall << std::endl;
  std::cout << "F1 Score: " << f1Score << std::endl;
}

void calculateMetrics(const ConfusionMatrix &matrix, double &accuracy,
                      double &precision, double &recall, double &f1Score) {
  int total = matrix.truePositive + matrix.trueNegative + matrix.falsePositive +
              matrix.falseNegative;

  accuracy =
      static_cast<double>(matrix.truePositive + matrix.trueNegative) / total;

  // Handle the case where the denominator is zero to avoid division by zero
  double precisionDenominator = matrix.truePositive + matrix.falsePositive;
  precision =
      (std::abs(precisionDenominator) > std::numeric_limits<double>::epsilon())
          ? static_cast<double>(matrix.truePositive) / precisionDenominator
          : 0.0;

  double recallDenominator = matrix.truePositive + matrix.falseNegative;
  recall =
      (std::abs(recallDenominator) > std::numeric_limits<double>::epsilon())
          ? static_cast<double>(matrix.truePositive) / recallDenominator
          : 0.0;

  // Handle the case where both precision and recall are zero to avoid division
  // by zero
  f1Score =
      (std::abs(precision + recall) > std::numeric_limits<double>::epsilon())
          ? 2 * (precision * recall) / (precision + recall)
          : 0.0;
}
