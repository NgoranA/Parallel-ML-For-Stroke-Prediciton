#include "../include/cross_validation.h"
#include "../include/model_evaluation.h"
#include <algorithm> // For std::shuffle
#include <iostream>
#include <random> // For std::mt19937
#include <vector>

// Function to perform cross-validation
void crossValidation(const std::vector<DataPoint> &dataset, int numFolds) {
  // Shuffle the dataset
  std::vector<DataPoint> shuffledDataset = dataset;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffledDataset.begin(), shuffledDataset.end(), g);

  // Split the dataset into folds
  size_t foldSize = dataset.size() / static_cast<size_t>(numFolds);
  for (int fold = 0; fold < numFolds; ++fold) {
    // Split the dataset into training and validation folds
    std::vector<DataPoint> validationFold(
        shuffledDataset.begin() + fold * static_cast<int>(foldSize),
        shuffledDataset.begin() + (fold + 1) * static_cast<int>(foldSize));
    std::vector<DataPoint> trainingFolds;
    trainingFolds.insert(trainingFolds.end(), shuffledDataset.begin(),
                         shuffledDataset.begin() +
                             fold * static_cast<int>(foldSize));
    trainingFolds.insert(trainingFolds.end(),
                         shuffledDataset.begin() +
                             (fold + 1) * static_cast<int>(foldSize),
                         shuffledDataset.end());
    // Train your model on the training set
    RandomForest forest(200, 0, 2);
    forest.train(trainingFolds);

    std::cout << "╔══════════════════╗" << std::endl;
    std::cout << "║ Model Prediction ║" << std::endl;
    std::cout << "╚══════════════════╝" << std::endl;
    std::vector<int> predictions = forest.predict(validationFold[0].features);
    // this is the finalPrediction
    /* int finalPrediction = forest.majorityVoting(predictions); */
    // Evaluating our model
    std::cout << "╔══════════════════╗" << std::endl;
    std::cout << "║ Model Evaluation ║" << std::endl;
    std::cout << "╚══════════════════╝" << std::endl;
    evaluateModel(validationFold, predictions);
  }
}
