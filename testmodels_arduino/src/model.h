

//header for random forest

#pragma once
#include <vector>
int predict_leaf_index(std::vector<double> &pX);
//header for logistic regression
#pragma once
#include <vector>
std::vector<double> predict_lr(std::vector<int> &x);
//combining Random Forest and Logistic Regression
std::vector<double> predict(std::vector<double> features);
