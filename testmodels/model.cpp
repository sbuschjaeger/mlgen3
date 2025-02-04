#include "model.h"
                
			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};
			constexpr double predictions[4][10] = {{0.0, 0.0, 0.0, 0.01694915254237288, 0.0, 0.2542372881355932, 0.0, 0.0, 0.0, 0.7288135593220338},{0.8689655172413793, 0.0, 0.027586206896551724, 0.0, 0.020689655172413793, 0.034482758620689655, 0.027586206896551724, 0.0, 0.013793103448275862, 0.006896551724137931},{0.0012437810945273632, 0.13059701492537312, 0.1480099502487562, 0.15049751243781095, 0.013681592039800995, 0.1044776119402985, 0.0708955223880597, 0.1181592039800995, 0.15671641791044777, 0.10572139303482588},{0.004016064257028112, 0.060240963855421686, 0.004016064257028112, 0.0, 0.4979919678714859, 0.08032128514056225, 0.26104417670682734, 0.08032128514056225, 0.008032128514056224, 0.004016064257028112}};
			constexpr Node nodes[3] = {{ 36,0.5,false,false,1,2 },{ 42,3.5,true,true,0,1 },{ 33,3.5,true,true,2,3 }};

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
			
			}