#pragma once
#include <string>
#include <utility>
#include <vector>

struct DataPoint {
  std::vector<double> features;
  int labels;
};

class DatasetHandler {
public:
  static void loadCSV(const std::string &featuresFile,
                      std::vector<std::vector<double>> &features,
                      std::vector<int> &labels);
  static std::vector<DataPoint>
  loadFeatureAndLabelCSV(const std::string &featuresFile,
                         const std::string &labelsFile);
};
