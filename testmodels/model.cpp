
                #include "model.h"
                
                std::vector<double> predict_0(std::vector<double> &pX){
                    
                if (pX[54] <= 1.5){
                    
                if (pX[20] <= 15.5){
                    return std::vector<double>({0.054481546572934976,0.016695957820738138,0.0017574692442882249,0.028119507908611598,0.11335676625659051,0.10369068541300527,0.0008787346221441124,0.09314586994727592,0.06063268892794376,0.027240773286467488});
                } else {
                    return std::vector<double>({0.0,0.33203125,0.00390625,0.00390625,0.04296875,0.0078125,0.0,0.0625,0.0390625,0.0078125});
                }
            
                } else {
                    
                if (pX[29] <= 2.5){
                    return std::vector<double>({0.0,0.04591836734693878,0.11479591836734694,0.04591836734693878,0.0,0.01020408163265306,0.2602040816326531,0.0,0.02295918367346939,0.0});
                } else {
                    return std::vector<double>({0.09752747252747254,0.0315934065934066,0.09065934065934067,0.07829670329670331,0.0,0.023351648351648355,0.03846153846153847,0.0,0.045329670329670335,0.09478021978021979});
                }
            
                }
            
                }
                int predict_0_leaf_index(std::vector<double> &pX){
                    
                if (pX[54] <= 1.5){
                    
                if (pX[20] <= 15.5){
                    return 0;
                } else {
                    return 1;
                }
            
                } else {
                    
                if (pX[29] <= 2.5){
                    return 2;
                } else {
                    return 3;
                }
            
                }
            
                }
            
                std::vector<double> predict_1(std::vector<double> &pX){
                    
                if (pX[60] <= 7.5){
                    
                if (pX[20] <= 1.5){
                    return std::vector<double>({0.00625,0.0,0.0,0.00625,0.04375,0.25625,0.0,0.18125,0.00625,0.0});
                } else {
                    return std::vector<double>({0.0,0.014814814814814815,0.011111111111111112,0.014814814814814815,0.04814814814814815,0.040740740740740744,0.0,0.32592592592592595,0.022222222222222223,0.022222222222222223});
                }
            
                } else {
                    
                if (pX[38] <= 0.5){
                    return std::vector<double>({0.0038461538461538464,0.10865384615384616,0.11057692307692307,0.04326923076923077,0.008653846153846154,0.04903846153846154,0.05096153846153846,0.0,0.11153846153846154,0.013461538461538462});
                } else {
                    return std::vector<double>({0.09578544061302684,0.009578544061302683,0.0047892720306513415,0.05842911877394637,0.10632183908045978,0.04501915708812261,0.07950191570881228,0.010536398467432952,0.0067049808429118785,0.08333333333333334});
                }
            
                }
            
                }
                int predict_1_leaf_index(std::vector<double> &pX){
                    
                if (pX[60] <= 7.5){
                    
                if (pX[20] <= 1.5){
                    return 4;
                } else {
                    return 5;
                }
            
                } else {
                    
                if (pX[38] <= 0.5){
                    return 6;
                } else {
                    return 7;
                }
            
                }
            
                }
            
                std::vector<double> predict(std::vector<double> &pX){
                    	std::vector<double> result = predict_0(pX);
	std::vector<double> result_temp;
	result_temp = predict_1(pX);
	std::transform(result.begin(), result.end(), result_temp.begin(),result.begin(), std::plus<double>());

                    return result;
                }
                std::vector<int> predict_leaf_indices(std::vector<double> &pX){
                    	std::vector<int> result_temp(2);
	result_temp[0] = predict_0_leaf_index(pX);
	result_temp[1] = predict_1_leaf_index(pX);
	std::vector<int> result = result_temp;

                    return result;
                }
            