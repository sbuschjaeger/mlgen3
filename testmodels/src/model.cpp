#include "model.h"
                
			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};
			constexpr double predictions[4][10] = {{0.01818181818181818, 0.0, 0.01818181818181818, 0.01818181818181818, 0.0, 0.18181818181818182, 0.0, 0.0, 0.0, 0.7636363636363637},{0.8601398601398601, 0.0, 0.013986013986013986, 0.0, 0.04195804195804196, 0.04195804195804196, 0.027972027972027972, 0.0, 0.013986013986013986, 0.0},{0.0, 0.09404388714733543, 0.08463949843260188, 0.03134796238244514, 0.06896551724137931, 0.3385579937304075, 0.36363636363636365, 0.01567398119122257, 0.0, 0.003134796238244514},{0.005405405405405406, 0.12567567567567567, 0.12567567567567567, 0.17027027027027028, 0.12432432432432433, 0.012162162162162163, 0.005405405405405406, 0.15135135135135136, 0.17027027027027028, 0.10945945945945947}};
			constexpr Node nodes[3] = {{ 36,0.5,false,false,1,2 },{ 42,5.0,true,true,0,1 },{ 21,0.5,true,true,2,3 }};

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