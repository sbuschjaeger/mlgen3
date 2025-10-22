
            #pragma once
            #include <vector>
            #include <algorithm>
            #include <cmath>
            #include <limits>
            #include <fstream>
            #include <iostream>

            // Uncomment to enable debug output
            // #define DEBUG_MODE

            // Template function to load binary data
            template <typename T>
            bool load_binary_data(const std::string& filename, std::vector<T>& data, size_t expected_size) {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    std::cerr << "Error: Could not open file " << filename << std::endl;
                    return false;
                }
                
                // Get file size
                file.seekg(0, std::ios::end);
                size_t file_size = file.tellg();
                file.seekg(0, std::ios::beg);
                
                // Check if file size matches expected size
                if (file_size != expected_size * sizeof(T)) {
                    std::cerr << "Error: File size mismatch. Expected " << expected_size * sizeof(T) 
                              << " bytes, got " << file_size << " bytes." << std::endl;
                    return false;
                }
                
                // Resize vector and read data
                data.resize(expected_size);
                file.read(reinterpret_cast<char*>(data.data()), file_size);
                
                if (!file) {
                    std::cerr << "Error: Only " << file.gcount() << " bytes could be read" << std::endl;
                    return false;
                }
                
                return true;
            }

            // Function to load quantization parameters (scale, zero_point)
            inline bool load_quantization_params(const std::string& filename, float& scale, float& zero_point) {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    std::cerr << "Error: Could not open file " << filename << std::endl;
                    return false;
                }
                
                // Read scale and zero_point (2 floats = 8 bytes)
                file.read(reinterpret_cast<char*>(&scale), sizeof(float));
                file.read(reinterpret_cast<char*>(&zero_point), sizeof(float));
                
                if (!file) {
                    std::cerr << "Error: Failed to read quantization parameters" << std::endl;
                    return false;
                }
                
                return true;
            }

            // Function to perform bit slicing at runtime
            inline std::vector<int8_t> slice_bits(const std::vector<int8_t>& quantized, int original_bits, int target_bits) {
                std::vector<int8_t> sliced(quantized.size());
                int shift_bits = original_bits - target_bits;
                
                // Signed quantization
                int q_min = -(1 << (target_bits - 1));
                int q_max = (1 << (target_bits - 1)) - 1;
                
                for (size_t i = 0; i < quantized.size(); ++i) {
                    // Perform bit slicing with rounding
                    if (shift_bits > 0) {
                        // Get the bit at position target_bits+1 for rounding
                        int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
                        int floor_val = quantized[i] >> shift_bits;
                        sliced[i] = round_bit ? (floor_val + 1) : floor_val;
                        
                        // Clamp to ensure values are within the target bit-width range
                        sliced[i] = std::max(q_min, std::min(static_cast<int>(sliced[i]), q_max));
                        
                        // Scale back to original range
                        sliced[i] = sliced[i] << shift_bits;
                    } else {
                        sliced[i] = quantized[i];
                    }
                }
                
                return sliced;
            }

            // Function to dequantize values
            template <typename T>
            std::vector<float> dequantize(const std::vector<T>& quantized, float scale, float zero_point) {
                std::vector<float> dequantized(quantized.size());
                for (size_t i = 0; i < quantized.size(); ++i) {
                    dequantized[i] = (static_cast<float>(quantized[i]) - zero_point) * scale;
                }
                return dequantized;
            }

            // CNN utilities for reshaping and processing image inputs
            namespace cnn_utils {
                // Reshape 1D input to 3D tensor [channels][height][width]
                template <typename T>
                std::vector<std::vector<std::vector<T>>> reshape_input_to_3d(const std::vector<T>& input, 
                                                                           int channels, int height, int width) {
                    // Create 3D tensor with dimensions [channels][height][width]
                    std::vector<std::vector<std::vector<T>>> tensor(
                        channels, 
                        std::vector<std::vector<T>>(
                            height, 
                            std::vector<T>(width, 0.0f)
                        )
                    );
                    
                    // Fill the tensor with the input data
                    if (static_cast<int>(input.size()) >= channels * height * width) {
                        for (int c = 0; c < channels; c++) {
                            for (int h = 0; h < height; h++) {
                                for (int w = 0; w < width; w++) {
                                    int index = c * height * width + h * width + w;
                                    tensor[c][h][w] = input[index];
                                }
                            }
                        }
                    } else {
                        std::cerr << "Error: Input size mismatch. Expected at least " 
                                  << (channels * height * width) << " elements but got " 
                                  << input.size() << std::endl;
                    }
                    
                    return tensor;
                }
                
                // Flatten 3D tensor to 1D vector
                template <typename T>
                std::vector<T> flatten_3d_to_1d(const std::vector<std::vector<std::vector<T>>>& tensor) {
                    std::vector<T> flattened;
                    
                    for (const auto& channel : tensor) {
                        for (const auto& row : channel) {
                            for (const auto& val : row) {
                                flattened.push_back(val);
                            }
                        }
                    }
                    
                    return flattened;
                }
                
                // Apply ReLU to a 3D tensor in-place
                template <typename T>
                void apply_relu_3d(std::vector<std::vector<std::vector<T>>>& tensor) {
                    for (auto& channel : tensor) {
                        for (auto& row : channel) {
                            for (auto& val : row) {
                                val = std::max(T(0), val);
                            }
                        }
                    }
                }
                
                // Print statistics for 3D tensor (for debugging)
                template <typename T>
                void print_3d_tensor_stats(const std::vector<std::vector<std::vector<T>>>& tensor, const std::string& name) {
                    T min_val = std::numeric_limits<T>::max();
                    T max_val = std::numeric_limits<T>::lowest();
                    T sum = 0;
                    int count = 0;
                    
                    for (const auto& channel : tensor) {
                        for (const auto& row : channel) {
                            for (const auto& val : row) {
                                min_val = std::min(min_val, val);
                                max_val = std::max(max_val, val);
                                sum += val;
                                count++;
                            }
                        }
                    }
                    
                    std::cout << "  " << name << " - Min: " << min_val << ", Max: " << max_val 
                              << ", Mean: " << (count > 0 ? sum / count : 0) << std::endl;
                    
                    // Print first row of first channel
                    if (!tensor.empty() && !tensor[0].empty() && !tensor[0][0].empty()) {
                        std::cout << "  First channel, first row (first 5 values): ";
                        for (size_t i = 0; i < std::min(size_t(5), tensor[0][0].size()); i++) {
                            std::cout << tensor[0][0][i] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                
                // Print statistics for 1D tensor (for debugging)
                template <typename T>
                void print_1d_tensor_stats(const T* tensor, int size, const std::string& name) {
                    T min_val = std::numeric_limits<T>::max();
                    T max_val = std::numeric_limits<T>::lowest();
                    T sum = 0;
                    
                    for (int i = 0; i < size; i++) {
                        min_val = std::min(min_val, tensor[i]);
                        max_val = std::max(max_val, tensor[i]);
                        sum += tensor[i];
                    }
                    
                    std::cout << "  " << name << " - Min: " << min_val << ", Max: " << max_val 
                              << ", Mean: " << (size > 0 ? sum / size : 0) << std::endl;
                    
                    std::cout << "  First 10 values: ";
                    for (int i = 0; i < std::min(10, size); i++) {
                        std::cout << tensor[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            
            #define IS_MIX_AND_MATCH 0
            #define NUM_LAYERS 12
            constexpr int LAYER_BITS[NUM_LAYERS] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
            #define TARGET_BITS 8
            #define STORAGE_BITS 8
            #define QUANTIZE_SIGNED 1

            // CNN model parameters
            #define INPUT_HEIGHT 28
            #define INPUT_WIDTH 28
            #define INPUT_CHANNELS 1
            
            std::vector<float> predict(std::vector<float> &x);
        