#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "dataset_handler.h"
void crossValidation(const std::vector<DataPoint> &dataset, int numFolds);

#endif // !CROSS_VALIDATION_H
