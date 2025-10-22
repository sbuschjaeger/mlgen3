#pragma once
            #include <vector>
            #include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>

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
            
            #define IS_MIX_AND_MATCH 0
#define NUM_LAYERS 7
constexpr int LAYER_BITS[NUM_LAYERS] = {2, 2, 2, 2, 2, 2, 2};
#define TARGET_BITS 2
#define STORAGE_BITS 8
#define QUANTIZE_SIGNED 1

            std::vector<float> predict(std::vector<float> &x);