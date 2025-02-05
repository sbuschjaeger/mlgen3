

//header for random forest

#pragma once
#include <vector>
#include <algorithm>
std::vector<float> predict_forest(std::vector<double> &pX);
std::vector<int> predict_leaf_indices(std::vector<double> &pX);
std::vector<float> predict_0(std::vector<double> &x);
int predict_0_leaf_index(std::vector<double> &x);
std::vector<float> predict_1(std::vector<double> &x);
int predict_1_leaf_index(std::vector<double> &x);

//header for logistic regression
#pragma once
#include <vector>
std::vector<double> predict(std::vector<int> &x);
//combining Random Forest and Logistic Regression
std::vector<double> predict(std::vector<double> features);
