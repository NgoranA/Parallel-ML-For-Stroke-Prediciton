#include "../include/cross_validation.h"
#include "../include/random_forest.h"
#include <iostream>
#include <ostream>
#include <vector>

int main() {

  std::cout << "╔═══════════╗" << std::endl;
  std::cout << "║ Main Menu ║" << std::endl;
  std::cout << "╚═══════════╝" << std::endl;
  std::cout << "Welcome to MLALgo Optimization!\n";
  std::cout << "Select the desired algorithm:\n";
  std::cout << "1. Random Forest\n";
  std::cout << "2. Support Vector Machine (SVM)\n";
  std::cout << "3. Parallel Random Forest\n";
  std::cout << "4. Parallel Support Vector Machine (SVM)\n";

  std::cout << "Enter the corresponding number: ";

  int algorithm_choice;
  std::cin >> algorithm_choice;

  switch (algorithm_choice) {
  case 1:

    try {
      std::vector<DataPoint> dataset = DatasetHandler::loadFeatureAndLabelCSV(
          "../data/X_train.csv", "../data/y_train.csv");
      RandomForest forest(200, 0, 2);
      forest.train(dataset);
      crossValidation(dataset, 10);

    } catch (const std::exception &e) {
      std::cerr << "Exception: " << e.what() << std::endl;
    }
    /* std::cout << "Normal Random Forest Selected!\n"; */
    break;
  case 2:
    std::cout << "Normal Support Vector Machine Selected!\n";
    break;
  case 3:
    std::cout << "Parallel Random Forest Selected\n";
    break;
  case 4:
    std::cout << "Parallel Support Vector Machine Selected\n";
    break;
  default:
    std::cout << "No algorithm was chosen with value: " << algorithm_choice
              << std::endl;
  }

  return 0;
}
