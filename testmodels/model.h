
                #pragma once
                #include <vector>
                #include <algorithm>
                std::vector<double> predict(std::vector<double> &pX);
                std::vector<int> predict_leaf_indices(std::vector<double> &pX);
                		std::vector<double> predict_0(std::vector<double> &pX);
		std::vector<double> predict_1(std::vector<double> &pX);

            