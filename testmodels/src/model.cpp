#include "model.h"
                
			struct Node {
					unsigned char feature;
					double split;
					bool left_is_leaf;
					bool right_is_leaf;
					unsigned char left;
					unsigned char right;
				};
			constexpr double predictions[4][10] = {{0.0, 0.0, 0.041666666666666664, 0.020833333333333332, 0.0, 0.08333333333333333, 0.0, 0.0, 0.0, 0.8541666666666666},{0.8707482993197279, 0.0, 0.006802721088435374, 0.0, 0.027210884353741496, 0.047619047619047616, 0.02040816326530612, 0.0, 0.02040816326530612, 0.006802721088435374},{0.0, 0.09433962264150944, 0.10062893081761007, 0.025157232704402517, 0.059748427672955975, 0.31761006289308175, 0.3742138364779874, 0.018867924528301886, 0.009433962264150943, 0.0},{0.004032258064516129, 0.12365591397849462, 0.11693548387096774, 0.17204301075268819, 0.135752688172043, 0.012096774193548387, 0.002688172043010753, 0.15188172043010753, 0.15994623655913978, 0.12096774193548387}};
			constexpr Node nodes[3] = {{ 36,0.5,false,false,1,2 },{ 34,3.0,true,true,0,1 },{ 21,0.5,true,true,2,3 }};

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