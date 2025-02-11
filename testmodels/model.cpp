

//code for random forest

                #include "model.h"

			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};

			constexpr Node nodes_0[3] = {{ 2,0.5,false,false,1,2 },{ 3,0.5,true,true,0,1 },{ 2,7.5,true,true,2,3 }};

			int predict_0_leaf_index(std::vector<double> &x) {

				unsigned char i = 0;
				while(true) {
					if (x[nodes_0[i].feature] <= nodes_0[i].split){
						if (nodes_0[i].left_is_leaf) {
							i = nodes_0[i].left;
							break;
						} else {
							i = nodes_0[i].left;
						}
					} else {
						if (nodes_0[i].right_is_leaf) {
							i = nodes_0[i].right;
							break;
						} else {
							i = nodes_0[i].right;
						}
					}
				}
				return i;

			}



			constexpr Node nodes_1[3] = {{ 1,0.5,false,false,1,2 },{ 2,1.5,true,true,0,1 },{ 3,12.5,true,true,2,3 }};

			int predict_1_leaf_index(std::vector<double> &x) {

				unsigned char i = 0;
				while(true) {
					if (x[nodes_1[i].feature] <= nodes_1[i].split){
						if (nodes_1[i].left_is_leaf) {
							i = nodes_1[i].left;
							break;
						} else {
							i = nodes_1[i].left;
						}
					} else {
						if (nodes_1[i].right_is_leaf) {
							i = nodes_1[i].right;
							break;
						} else {
							i = nodes_1[i].right;
						}
					}
				}
				return i;

			}


                    std::vector<int> predict_leaf_indices(std::vector<double> &pX){
                        	std::vector<int> result_temp(2);
	result_temp[0] = predict_0_leaf_index(pX);
	result_temp[1] = predict_1_leaf_index(pX);
	std::vector<int> result = result_temp;

                        return result;
                    }

            //code for logistic regression



#include "model.h"
			constexpr float coef[10][4] = {{0.0, -0.49777128511510965, 0.0014357546267942355, 0.14865303666721558},{0.0, -0.3781664551644852, 0.016785793339943362, -0.09953115486614433},{0.0, 0.43539601720745547, 0.36078420246937254, -0.04241725294789687},{0.0, 0.4591102882209267, 0.24113648654068348, 0.05513638546319948},{0.0, -0.15594847756763947, -0.9448816577160603, -0.008373699895013673},{0.0, 0.38140554309377467, 0.43722792235283403, -0.1730594687614388},{0.0, -0.18288702948783123, -0.5597183280722707, 0.15989645064958466},{0.0, 0.14782198804756552, 0.08817914428426298, 0.07986440503428076},{0.0, 0.07751096089448564, 0.14571368608696236, -0.051844301046241284},{0.0, -0.28647155012915154, 0.21333699608751605, -0.06832440029756287}};
			constexpr float intercept[10] = {-1.209526370931553, 1.6063594010618316, -1.246645917305952, -1.465363782987438, 1.8946802718582654, -0.09338767912735471, 0.012906667526549688, -0.6679630658317331, 0.7149411193697806, 0.4539993563676861};

			std::vector<double> predict_lr(std::vector<int> &x) {
				std::vector<double> pred(10, 0);
				for (unsigned int j = 0; j < 10; ++j) {
					float sum = intercept[j]; 
					for (unsigned int i = 0; i < 4; ++i) {
						sum += coef[j][i] * x[i];
					}
					pred[j] += sum; 
				}
				return pred;
			}//combining Random Forest and Logistic Regression
std::vector<double> predict(std::vector<double> features) {
    std::vector<int> leaf_indices = predict_leaf_indices(features);
    return predict_lr(leaf_indices);
}
