#include "../include/dataset_handler.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<DataPoint>
DatasetHandler::loadFeatureAndLabelCSV(const std::string &featuresFile,
                                       const std::string &labelsFile) {
  std::vector<std::vector<double>> features;
  std::vector<int> labels;
  std::vector<std::string> featuresColumns;
  std::vector<DataPoint> dataset;

  std::ifstream featureFileStream(featuresFile);
  if (!featureFileStream.is_open()) {
    throw std::runtime_error("Error opening files!");
  }

  // Load labels
  std::ifstream labelsFileStream(labelsFile);
  if (!labelsFileStream.is_open()) {
    throw std::runtime_error("Error opening files!");
  }

  // Read and discard the Headers
  std::string featuresHeaderLine;
  std::getline(featureFileStream, featuresHeaderLine);
  std::string targetHeaderLine;
  std::getline(labelsFileStream, targetHeaderLine);

  // Read the features and the labels and combine them into a single dataset.
  std::string featuresData;
  std::string labelsLine;
  while (std::getline(featureFileStream, featuresData) &&
         std::getline(labelsFileStream, labelsLine)) {
    std::istringstream ss(featuresData);
    std::string token;
    int label = std::stoi(labelsLine);
    std::vector<double> features_vector;
    while (std::getline(ss, token, ',')) {
      try {
        features_vector.push_back(std::stod(token));
      } catch (const std::invalid_argument &e) {
        std::cerr << "Error converting string to double in features file: "
                  << e.what() << std::endl;
        features_vector.push_back(0.0);
      }
    }
    // Validate that the number of features and labels match
    if (features.size() != labels.size()) {
      throw std::runtime_error("Mismatch in the number of features and labels");
    }
    DataPoint dataPoint{features_vector, label};
    dataset.push_back(dataPoint);
  }
  featureFileStream.close();
  labelsFileStream.close();

  return dataset;
}
