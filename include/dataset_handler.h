#pragma once
#include <string>
#include <utility>
#include <vector>
struct DataContainer {
  std::vector<std::vector<double>> features;
  std::vector<int> labels;
  std::vector<std::string> headers;
};

class DatasetHandler {
public:
  static void loadCSV(const std::string &featuresFile,
                      std::vector<std::vector<double>> &features,
                      std::vector<int> &labels);
  static DataContainer loadFeatureAndLabelCSV(const std::string &featuresFile,
                                              const std::string &labelsFile);
};
