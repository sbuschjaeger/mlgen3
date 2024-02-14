#include "model.h"
                
			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};
			constexpr double predictions[2][10] = {{0.6473684210526316, 0.0, 0.021052631578947368, 0.005263157894736842, 0.02631578947368421, 0.10526315789473684, 0.015789473684210527, 0.0, 0.005263157894736842, 0.1736842105263158},{0.0028116213683223993, 0.1105904404873477, 0.11715089034676664, 0.12089971883786317, 0.11527647610121837, 0.09840674789128398, 0.12277413308341144, 0.11715089034676664, 0.10871602624179943, 0.08622305529522024}};
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