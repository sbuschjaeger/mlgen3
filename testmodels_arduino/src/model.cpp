

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

			constexpr Node nodes[3] = {{ 1,0.5,false,false,1,2 },{ 2,0.5,true,true,0,1 },{ 2,14.5,true,true,2,3 }};

			int predict_leaf_index(std::vector<double> &x) {

				unsigned char i = 0;
				while(true) {
					if (x[nodes[i].feature] <= nodes[i].split){
						if (nodes[i].left_is_leaf) {
							i = nodes[i].left;
							break;
						} else {
							i = nodes[i].left;
						}
					} else {
						if (nodes[i].right_is_leaf) {
							i = nodes[i].right;
							break;
						} else {
							i = nodes[i].right;
						}
					}
				}
				return i;

			}//code for logistic regression



#include "model.h"
			constexpr float coef[10][4] = {{0.0, -0.7951751216171145, -0.013287243809268911, 0.12964652483487699},{0.0, -0.3838331090586107, -0.04228772906019765, -0.0865750395292451},{0.0, 0.39661282244954454, 0.29310376679564404, 0.008544088571896757},{0.0, 0.38895566035763557, 0.24371400410484584, 0.06617575079746116},{0.0, -0.18009041735924314, -0.6785315750838031, -0.055550497282745456},{0.0, 0.39737741622787337, 0.3923682048518242, -0.13124055871221318},{0.0, -0.21124971012219476, -0.5507934028958063, 0.13262239626955735},{0.0, 0.24822202552852315, 0.05830072924696316, 0.0869952155560012},{0.0, 0.17870251259290482, 0.09817360277877792, -0.04221049767673663},{0.0, -0.0395220789993277, 0.19923964307104974, -0.1084073828288571}};
			constexpr float intercept[10] = {-0.936162864400607, 1.7203482373796668, -1.469447012125338, -1.7221669220216613, 2.0637944388190577, -0.49480453431029764, 0.21224114723115872, -0.8018259094656276, 0.4872849060673596, 0.9407385128262833};

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
    std::vector<int> leaf_indices(1);
    leaf_indices[0] = predict_leaf_index(features);
    return predict_lr(leaf_indices);
}
