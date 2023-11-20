#include "../include/dataset_handler.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

DataContainer
DatasetHandler::loadFeatureAndLabelCSV(const std::string &featuresFile,
                                       const std::string &labelsFile) {
  std::vector<std::vector<double>> features;
  std::vector<int> labels;
  std::vector<std::string> featuresColumns;

  std::ifstream featureFileStream(featuresFile);
  if (!featureFileStream.is_open()) {
    throw std::runtime_error("Error opening files!");
  }

  // Load Headers
  std::string featuresHeaderLine;
  if (std::getline(featureFileStream, featuresHeaderLine)) {
    // Store the header values as strings
    std::istringstream headerStream(featuresHeaderLine);
    std::string columnName;
    /* std::vector<std::string> columnNames; */
    while (std::getline(headerStream, columnName, ',')) {
      featuresColumns.push_back(columnName);
    }
  }

  // Load features
  std::string featuresData;
  while (std::getline(featureFileStream, featuresData)) {
    std::istringstream ss(featuresData);
    std::string token;

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
    features.push_back(features_vector);
  }
  featureFileStream.close();

  // Load labels
  std::ifstream labelsFileStream(labelsFile);
  if (!labelsFileStream.is_open()) {
    throw std::runtime_error("Error opening files!");
  }

  std::string targetHeaderLine;
  if (std::getline(labelsFileStream, targetHeaderLine)) {
    // Store the header values as strings
    std::istringstream headerStream(targetHeaderLine);
    std::string columnName;
    std::vector<std::string> columnNames;
    while (std::getline(headerStream, columnName, ',')) {
      columnNames.push_back(columnName);
    }
  }

  std::string labelsLine;
  while (std::getline(labelsFileStream, labelsLine)) {
    try {
      labels.push_back(std::stoi(labelsLine));
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error converting string to double in labels file: "
                << e.what() << std::endl;
      labels.push_back(0);
    }
  }
  labelsFileStream.close();
  // Validate that the number of features and labels match
  if (features.size() != labels.size()) {
    throw std::runtime_error("Mismatch in the number of features and labels");
  }

  return {features, labels, featuresColumns};
}
