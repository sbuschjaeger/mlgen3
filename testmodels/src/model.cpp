#include "model.h"
                
			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};
			constexpr double predictions[2][10] = {{0.6288659793814433, 0.0, 0.020618556701030927, 0.0, 0.030927835051546393, 0.07216494845360824, 0.02577319587628866, 0.0, 0.005154639175257732, 0.21649484536082475},{0.0028222013170272815, 0.1213546566321731, 0.11288805268109126, 0.1288805268109125, 0.11853245531514581, 0.10348071495766697, 0.10630291627469426, 0.1213546566321731, 0.10442144873000941, 0.0799623706491063}};
			constexpr Node nodes[1] = {{ 36,0.5,true,true,0,1 }};

			std::vector<double> predict(std::vector<double> &x) {
				
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
				return std::vector<double>(predictions[i], predictions[i]+10);
			
			}