#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <assert.h>
#include <iomanip>
#include <random>


            #include "matquant_pt_vgg4_uniform_8bit.h"    
            typedef float OUTPUT_TYPE;
            typedef unsigned int LABEL_TYPE;
            typedef float FEATURE_TYPE;
            

auto read_csv(std::string &path) {
	std::vector<std::vector<FEATURE_TYPE>> X;
	std::vector<unsigned int> Y;

	std::ifstream file(path);
	std::string header;
	std::getline(file, header);

	unsigned int label_pos = 0;
	std::stringstream ss(header);
	std::string entry;
	while (std::getline(ss, entry, ',')) {
		if (entry == "label") {
			break;
		} else {
			label_pos++;
		}
	}

	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (line.size() > 0) {
				std::stringstream ss(line);
				entry = "";

				unsigned int i = 0;
				std::vector<FEATURE_TYPE> x;
				while (std::getline(ss, entry, ',')) {
					if (i == label_pos) {
						Y.push_back(static_cast<unsigned int>(std::stoi(entry)));
					} else {
						x.push_back(static_cast<FEATURE_TYPE>(std::stof(entry)));
					}
					++i;
				}
				X.push_back(x);
			}
		}
		file.close();
	}
	return std::make_tuple(X,Y);
}

auto benchmark(std::vector<std::vector<FEATURE_TYPE>> &X, std::vector<LABEL_TYPE> &Y, unsigned int repeat) {
	std::vector<OUTPUT_TYPE> output;
    
	unsigned int matches = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < repeat; ++k) {
    	matches = 0;
	    for (unsigned int i = 0; i < X.size(); ++i) {
	        unsigned int label = Y[i];

			output=predict(X[i]);

            OUTPUT_TYPE max = output[0];
            unsigned int argmax = 0;
            for (unsigned int j = 1; j < output.size(); j++) {
                if (output[j] > max) {
                    max = output[j];
                    argmax = j;
                }
            }

            if (argmax == label) {
                ++matches;
            }
	    }
    }

    
                auto end = std::chrono::high_resolution_clock::now();   
                auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (X.size() * repeat);
            float accuracy = static_cast<float>(matches) / X.size() * 100.f;
    return std::make_pair(accuracy, runtime);
}

void print_matquant_info() {
    std::cout << "MatQuant Model Information:" << std::endl;
    std::cout << "  Target Bit Width: " << TARGET_BITS << std::endl;
    
    if (IS_MIX_AND_MATCH) {
        std::cout << "  Mode: Mix-and-Match" << std::endl;
        std::cout << "  Layer Bit Configuration:" << std::endl;
        for (int i = 0; i < NUM_LAYERS; ++i) {
            std::cout << "    Layer " << i << ": " << LAYER_BITS[i] << "-bit" << std::endl;
        }
    } else {
        std::cout << "  Mode: Uniform Quantization" << std::endl;
        std::cout << "  All layers use " << TARGET_BITS << "-bit quantization" << std::endl;
    }
}

void set_random_seed(unsigned int seed) {
    std::srand(seed);
    std::cout << "Random seed set to: " << seed << std::endl;
}

int main (int argc, char *argv[]) {
	if (argc <= 2) {
		std::cout << "Please provide two arguments: path n_repetitions [seed]" << std::endl;
        return 1;
	}
	std::string path = std::string(argv[1]);
	unsigned int repeat = std::stoi(argv[2]);
	
	// Set random seed (default: 707)
	unsigned int seed = 707;
	if (argc > 3) {
	    seed = std::stoi(argv[3]);
	}
	set_random_seed(seed);

	auto data = read_csv(path);

	assert(std::get<0>(data).size() > 0);
	assert(std::get<0>(data).size() == std::get<1>(data).size());

    print_matquant_info();
    
    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
    auto results = benchmark(std::get<0>(data), std::get<1>(data), repeat);

    
                std::cout << "Accuracy: " << results.first << " %" << std::endl;
                std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
            
    
    return 0;
}
