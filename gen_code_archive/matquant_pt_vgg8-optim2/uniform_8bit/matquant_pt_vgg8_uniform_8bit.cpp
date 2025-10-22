
            #include "matquant_pt_vgg8_uniform_8bit.h"
            #include <fstream>
            #include <iostream>
            #include <cmath>
            #include <algorithm>
            
            // Layer allocations
static float layer_21[8192];
static float layer_22[1024];
static float layer_23[1024];
static float layer_24[10];

            
            // Declare quantization arrays for all required layers
// conv layer 0
std::vector<int8_t> layer_0_weight_q8;
float layer_0_weight_scale = 1.0f;
float layer_0_weight_zero_point = 0.0f;
std::vector<float> layer_0_weights_dequant;
std::vector<int8_t> layer_0_bias_q8;
float layer_0_bias_scale = 1.0f;
float layer_0_bias_zero_point = 0.0f;
std::vector<float> layer_0_bias_dequant;

// batchnorm2d layer 2
std::vector<int8_t> layer_2_weight_q8;
float layer_2_weight_scale = 1.0f;
float layer_2_weight_zero_point = 0.0f;
std::vector<float> layer_2_weights_dequant;
std::vector<int8_t> layer_2_bias_q8;
float layer_2_bias_scale = 1.0f;
float layer_2_bias_zero_point = 0.0f;
std::vector<float> layer_2_bias_dequant;
std::vector<float> layer_2_running_mean;
std::vector<float> layer_2_running_var;
float layer_2_eps = 1e-5f;

// conv layer 4
std::vector<int8_t> layer_4_weight_q8;
float layer_4_weight_scale = 1.0f;
float layer_4_weight_zero_point = 0.0f;
std::vector<float> layer_4_weights_dequant;
std::vector<int8_t> layer_4_bias_q8;
float layer_4_bias_scale = 1.0f;
float layer_4_bias_zero_point = 0.0f;
std::vector<float> layer_4_bias_dequant;

// batchnorm2d layer 5
std::vector<int8_t> layer_5_weight_q8;
float layer_5_weight_scale = 1.0f;
float layer_5_weight_zero_point = 0.0f;
std::vector<float> layer_5_weights_dequant;
std::vector<int8_t> layer_5_bias_q8;
float layer_5_bias_scale = 1.0f;
float layer_5_bias_zero_point = 0.0f;
std::vector<float> layer_5_bias_dequant;
std::vector<float> layer_5_running_mean;
std::vector<float> layer_5_running_var;
float layer_5_eps = 1e-5f;

// conv layer 7
std::vector<int8_t> layer_7_weight_q8;
float layer_7_weight_scale = 1.0f;
float layer_7_weight_zero_point = 0.0f;
std::vector<float> layer_7_weights_dequant;
std::vector<int8_t> layer_7_bias_q8;
float layer_7_bias_scale = 1.0f;
float layer_7_bias_zero_point = 0.0f;
std::vector<float> layer_7_bias_dequant;

// batchnorm2d layer 9
std::vector<int8_t> layer_9_weight_q8;
float layer_9_weight_scale = 1.0f;
float layer_9_weight_zero_point = 0.0f;
std::vector<float> layer_9_weights_dequant;
std::vector<int8_t> layer_9_bias_q8;
float layer_9_bias_scale = 1.0f;
float layer_9_bias_zero_point = 0.0f;
std::vector<float> layer_9_bias_dequant;
std::vector<float> layer_9_running_mean;
std::vector<float> layer_9_running_var;
float layer_9_eps = 1e-5f;

// conv layer 11
std::vector<int8_t> layer_11_weight_q8;
float layer_11_weight_scale = 1.0f;
float layer_11_weight_zero_point = 0.0f;
std::vector<float> layer_11_weights_dequant;
std::vector<int8_t> layer_11_bias_q8;
float layer_11_bias_scale = 1.0f;
float layer_11_bias_zero_point = 0.0f;
std::vector<float> layer_11_bias_dequant;

// batchnorm2d layer 12
std::vector<int8_t> layer_12_weight_q8;
float layer_12_weight_scale = 1.0f;
float layer_12_weight_zero_point = 0.0f;
std::vector<float> layer_12_weights_dequant;
std::vector<int8_t> layer_12_bias_q8;
float layer_12_bias_scale = 1.0f;
float layer_12_bias_zero_point = 0.0f;
std::vector<float> layer_12_bias_dequant;
std::vector<float> layer_12_running_mean;
std::vector<float> layer_12_running_var;
float layer_12_eps = 1e-5f;

// conv layer 14
std::vector<int8_t> layer_14_weight_q8;
float layer_14_weight_scale = 1.0f;
float layer_14_weight_zero_point = 0.0f;
std::vector<float> layer_14_weights_dequant;
std::vector<int8_t> layer_14_bias_q8;
float layer_14_bias_scale = 1.0f;
float layer_14_bias_zero_point = 0.0f;
std::vector<float> layer_14_bias_dequant;

// batchnorm2d layer 16
std::vector<int8_t> layer_16_weight_q8;
float layer_16_weight_scale = 1.0f;
float layer_16_weight_zero_point = 0.0f;
std::vector<float> layer_16_weights_dequant;
std::vector<int8_t> layer_16_bias_q8;
float layer_16_bias_scale = 1.0f;
float layer_16_bias_zero_point = 0.0f;
std::vector<float> layer_16_bias_dequant;
std::vector<float> layer_16_running_mean;
std::vector<float> layer_16_running_var;
float layer_16_eps = 1e-5f;

// conv layer 18
std::vector<int8_t> layer_18_weight_q8;
float layer_18_weight_scale = 1.0f;
float layer_18_weight_zero_point = 0.0f;
std::vector<float> layer_18_weights_dequant;
std::vector<int8_t> layer_18_bias_q8;
float layer_18_bias_scale = 1.0f;
float layer_18_bias_zero_point = 0.0f;
std::vector<float> layer_18_bias_dequant;

// batchnorm2d layer 19
std::vector<int8_t> layer_19_weight_q8;
float layer_19_weight_scale = 1.0f;
float layer_19_weight_zero_point = 0.0f;
std::vector<float> layer_19_weights_dequant;
std::vector<int8_t> layer_19_bias_q8;
float layer_19_bias_scale = 1.0f;
float layer_19_bias_zero_point = 0.0f;
std::vector<float> layer_19_bias_dequant;
std::vector<float> layer_19_running_mean;
std::vector<float> layer_19_running_var;
float layer_19_eps = 1e-5f;

// linear layer 22
std::vector<int8_t> layer_22_weight_q8;
float layer_22_weight_scale = 1.0f;
float layer_22_weight_zero_point = 0.0f;
std::vector<float> layer_22_weights_dequant;
std::vector<int8_t> layer_22_bias_q8;
float layer_22_bias_scale = 1.0f;
float layer_22_bias_zero_point = 0.0f;
std::vector<float> layer_22_bias_dequant;

// linear layer 24
std::vector<int8_t> layer_24_weight_q8;
float layer_24_weight_scale = 1.0f;
float layer_24_weight_zero_point = 0.0f;
std::vector<float> layer_24_weights_dequant;
std::vector<int8_t> layer_24_bias_q8;
float layer_24_bias_scale = 1.0f;
float layer_24_bias_zero_point = 0.0f;
std::vector<float> layer_24_bias_dequant;

            
            std::vector<float> predict(std::vector<float> &x) {
                // Load binary files if not loaded
static bool files_loaded = false;
if (!files_loaded) {
    std::cout << std::endl << "Loading model parameters from binary files..." << std::endl;
    // Load weights for layer 0 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_0_weight.bin", layer_0_weight_q8, 3456);
    load_quantization_params("mq_pt_model_binary/layer_0_weight_qparams.bin", layer_0_weight_scale, layer_0_weight_zero_point);
    // Load bias for layer 0
    load_binary_data<int8_t>("mq_pt_model_binary/layer_0_bias.bin", layer_0_bias_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_0_bias_qparams.bin", layer_0_bias_scale, layer_0_bias_zero_point);
    // Load BatchNorm2d parameters for layer 2
    load_binary_data<int8_t>("mq_pt_model_binary/layer_2_weight.bin", layer_2_weight_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_2_weight_qparams.bin", layer_2_weight_scale, layer_2_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_2_bias.bin", layer_2_bias_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_2_bias_qparams.bin", layer_2_bias_scale, layer_2_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_2_running_mean.bin", layer_2_running_mean, 128);
    load_binary_data("mq_pt_model_binary/layer_2_running_var.bin", layer_2_running_var, 128);
    // Load eps parameter
    std::ifstream eps_file_2("mq_pt_model_binary/layer_2_eps.bin", std::ios::binary);
    if (eps_file_2.is_open()) {
        eps_file_2.read(reinterpret_cast<char*>(&layer_2_eps), sizeof(float));
        eps_file_2.close();
    }
    // Load weights for layer 4 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_4_weight.bin", layer_4_weight_q8, 147456);
    load_quantization_params("mq_pt_model_binary/layer_4_weight_qparams.bin", layer_4_weight_scale, layer_4_weight_zero_point);
    // Load bias for layer 4
    load_binary_data<int8_t>("mq_pt_model_binary/layer_4_bias.bin", layer_4_bias_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_4_bias_qparams.bin", layer_4_bias_scale, layer_4_bias_zero_point);
    // Load BatchNorm2d parameters for layer 5
    load_binary_data<int8_t>("mq_pt_model_binary/layer_5_weight.bin", layer_5_weight_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_5_weight_qparams.bin", layer_5_weight_scale, layer_5_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_5_bias.bin", layer_5_bias_q8, 128);
    load_quantization_params("mq_pt_model_binary/layer_5_bias_qparams.bin", layer_5_bias_scale, layer_5_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_5_running_mean.bin", layer_5_running_mean, 128);
    load_binary_data("mq_pt_model_binary/layer_5_running_var.bin", layer_5_running_var, 128);
    // Load eps parameter
    std::ifstream eps_file_5("mq_pt_model_binary/layer_5_eps.bin", std::ios::binary);
    if (eps_file_5.is_open()) {
        eps_file_5.read(reinterpret_cast<char*>(&layer_5_eps), sizeof(float));
        eps_file_5.close();
    }
    // Load weights for layer 7 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_7_weight.bin", layer_7_weight_q8, 294912);
    load_quantization_params("mq_pt_model_binary/layer_7_weight_qparams.bin", layer_7_weight_scale, layer_7_weight_zero_point);
    // Load bias for layer 7
    load_binary_data<int8_t>("mq_pt_model_binary/layer_7_bias.bin", layer_7_bias_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_7_bias_qparams.bin", layer_7_bias_scale, layer_7_bias_zero_point);
    // Load BatchNorm2d parameters for layer 9
    load_binary_data<int8_t>("mq_pt_model_binary/layer_9_weight.bin", layer_9_weight_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_9_weight_qparams.bin", layer_9_weight_scale, layer_9_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_9_bias.bin", layer_9_bias_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_9_bias_qparams.bin", layer_9_bias_scale, layer_9_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_9_running_mean.bin", layer_9_running_mean, 256);
    load_binary_data("mq_pt_model_binary/layer_9_running_var.bin", layer_9_running_var, 256);
    // Load eps parameter
    std::ifstream eps_file_9("mq_pt_model_binary/layer_9_eps.bin", std::ios::binary);
    if (eps_file_9.is_open()) {
        eps_file_9.read(reinterpret_cast<char*>(&layer_9_eps), sizeof(float));
        eps_file_9.close();
    }
    // Load weights for layer 11 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_11_weight.bin", layer_11_weight_q8, 589824);
    load_quantization_params("mq_pt_model_binary/layer_11_weight_qparams.bin", layer_11_weight_scale, layer_11_weight_zero_point);
    // Load bias for layer 11
    load_binary_data<int8_t>("mq_pt_model_binary/layer_11_bias.bin", layer_11_bias_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_11_bias_qparams.bin", layer_11_bias_scale, layer_11_bias_zero_point);
    // Load BatchNorm2d parameters for layer 12
    load_binary_data<int8_t>("mq_pt_model_binary/layer_12_weight.bin", layer_12_weight_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_12_weight_qparams.bin", layer_12_weight_scale, layer_12_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_12_bias.bin", layer_12_bias_q8, 256);
    load_quantization_params("mq_pt_model_binary/layer_12_bias_qparams.bin", layer_12_bias_scale, layer_12_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_12_running_mean.bin", layer_12_running_mean, 256);
    load_binary_data("mq_pt_model_binary/layer_12_running_var.bin", layer_12_running_var, 256);
    // Load eps parameter
    std::ifstream eps_file_12("mq_pt_model_binary/layer_12_eps.bin", std::ios::binary);
    if (eps_file_12.is_open()) {
        eps_file_12.read(reinterpret_cast<char*>(&layer_12_eps), sizeof(float));
        eps_file_12.close();
    }
    // Load weights for layer 14 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_14_weight.bin", layer_14_weight_q8, 1179648);
    load_quantization_params("mq_pt_model_binary/layer_14_weight_qparams.bin", layer_14_weight_scale, layer_14_weight_zero_point);
    // Load bias for layer 14
    load_binary_data<int8_t>("mq_pt_model_binary/layer_14_bias.bin", layer_14_bias_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_14_bias_qparams.bin", layer_14_bias_scale, layer_14_bias_zero_point);
    // Load BatchNorm2d parameters for layer 16
    load_binary_data<int8_t>("mq_pt_model_binary/layer_16_weight.bin", layer_16_weight_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_16_weight_qparams.bin", layer_16_weight_scale, layer_16_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_16_bias.bin", layer_16_bias_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_16_bias_qparams.bin", layer_16_bias_scale, layer_16_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_16_running_mean.bin", layer_16_running_mean, 512);
    load_binary_data("mq_pt_model_binary/layer_16_running_var.bin", layer_16_running_var, 512);
    // Load eps parameter
    std::ifstream eps_file_16("mq_pt_model_binary/layer_16_eps.bin", std::ios::binary);
    if (eps_file_16.is_open()) {
        eps_file_16.read(reinterpret_cast<char*>(&layer_16_eps), sizeof(float));
        eps_file_16.close();
    }
    // Load weights for layer 18 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_18_weight.bin", layer_18_weight_q8, 2359296);
    load_quantization_params("mq_pt_model_binary/layer_18_weight_qparams.bin", layer_18_weight_scale, layer_18_weight_zero_point);
    // Load bias for layer 18
    load_binary_data<int8_t>("mq_pt_model_binary/layer_18_bias.bin", layer_18_bias_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_18_bias_qparams.bin", layer_18_bias_scale, layer_18_bias_zero_point);
    // Load BatchNorm2d parameters for layer 19
    load_binary_data<int8_t>("mq_pt_model_binary/layer_19_weight.bin", layer_19_weight_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_19_weight_qparams.bin", layer_19_weight_scale, layer_19_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_19_bias.bin", layer_19_bias_q8, 512);
    load_quantization_params("mq_pt_model_binary/layer_19_bias_qparams.bin", layer_19_bias_scale, layer_19_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_19_running_mean.bin", layer_19_running_mean, 512);
    load_binary_data("mq_pt_model_binary/layer_19_running_var.bin", layer_19_running_var, 512);
    // Load eps parameter
    std::ifstream eps_file_19("mq_pt_model_binary/layer_19_eps.bin", std::ios::binary);
    if (eps_file_19.is_open()) {
        eps_file_19.read(reinterpret_cast<char*>(&layer_19_eps), sizeof(float));
        eps_file_19.close();
    }
    // Load weights for layer 22 (linear)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_22_weight.bin", layer_22_weight_q8, 8388608);
    load_quantization_params("mq_pt_model_binary/layer_22_weight_qparams.bin", layer_22_weight_scale, layer_22_weight_zero_point);
    // Load bias for layer 22
    load_binary_data<int8_t>("mq_pt_model_binary/layer_22_bias.bin", layer_22_bias_q8, 1024);
    load_quantization_params("mq_pt_model_binary/layer_22_bias_qparams.bin", layer_22_bias_scale, layer_22_bias_zero_point);
    // Load weights for layer 24 (linear)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_24_weight.bin", layer_24_weight_q8, 10240);
    load_quantization_params("mq_pt_model_binary/layer_24_weight_qparams.bin", layer_24_weight_scale, layer_24_weight_zero_point);
    // Load bias for layer 24
    load_binary_data<int8_t>("mq_pt_model_binary/layer_24_bias.bin", layer_24_bias_q8, 10);
    load_quantization_params("mq_pt_model_binary/layer_24_bias_qparams.bin", layer_24_bias_scale, layer_24_bias_zero_point);

    std::cout << "Precomputing dequantized weights and biases for faster inference..." << std::endl;
    // Precompute dequantized weights for Layer 0
    layer_0_weights_dequant.resize(3456);
    for (size_t i = 0; i < layer_0_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_0_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[0] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[0]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_0_weight_scale, layer_0_weight_zero_point);
        layer_0_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 0
    layer_0_bias_dequant.resize(128);
    for (size_t i = 0; i < layer_0_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_0_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[0] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[0]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_0_bias_scale, layer_0_bias_zero_point);
        layer_0_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 2
    layer_2_weights_dequant.resize(128);
    for (size_t i = 0; i < layer_2_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_2_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[2] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[2]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_2_weight_scale, layer_2_weight_zero_point);
        layer_2_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 2
    layer_2_bias_dequant.resize(128);
    for (size_t i = 0; i < layer_2_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_2_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[2] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[2]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_2_bias_scale, layer_2_bias_zero_point);
        layer_2_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 4
    layer_4_weights_dequant.resize(147456);
    for (size_t i = 0; i < layer_4_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_4_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[4] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[4]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_4_weight_scale, layer_4_weight_zero_point);
        layer_4_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 4
    layer_4_bias_dequant.resize(128);
    for (size_t i = 0; i < layer_4_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_4_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[4] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[4]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_4_bias_scale, layer_4_bias_zero_point);
        layer_4_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 5
    layer_5_weights_dequant.resize(128);
    for (size_t i = 0; i < layer_5_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_5_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[5] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[5]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_5_weight_scale, layer_5_weight_zero_point);
        layer_5_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 5
    layer_5_bias_dequant.resize(128);
    for (size_t i = 0; i < layer_5_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_5_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[5] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[5]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_5_bias_scale, layer_5_bias_zero_point);
        layer_5_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 7
    layer_7_weights_dequant.resize(294912);
    for (size_t i = 0; i < layer_7_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_7_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[7] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[7]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_7_weight_scale, layer_7_weight_zero_point);
        layer_7_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 7
    layer_7_bias_dequant.resize(256);
    for (size_t i = 0; i < layer_7_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_7_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[7] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[7]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_7_bias_scale, layer_7_bias_zero_point);
        layer_7_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 9
    layer_9_weights_dequant.resize(256);
    for (size_t i = 0; i < layer_9_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_9_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[9] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[9]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_9_weight_scale, layer_9_weight_zero_point);
        layer_9_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 9
    layer_9_bias_dequant.resize(256);
    for (size_t i = 0; i < layer_9_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_9_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[9] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[9]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_9_bias_scale, layer_9_bias_zero_point);
        layer_9_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 11
    layer_11_weights_dequant.resize(589824);
    for (size_t i = 0; i < layer_11_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_11_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[11] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[11]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_11_weight_scale, layer_11_weight_zero_point);
        layer_11_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 11
    layer_11_bias_dequant.resize(256);
    for (size_t i = 0; i < layer_11_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_11_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[11] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[11]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_11_bias_scale, layer_11_bias_zero_point);
        layer_11_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 12
    layer_12_weights_dequant.resize(256);
    for (size_t i = 0; i < layer_12_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_12_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[12] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[12]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_12_weight_scale, layer_12_weight_zero_point);
        layer_12_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 12
    layer_12_bias_dequant.resize(256);
    for (size_t i = 0; i < layer_12_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_12_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[12] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[12]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_12_bias_scale, layer_12_bias_zero_point);
        layer_12_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 14
    layer_14_weights_dequant.resize(1179648);
    for (size_t i = 0; i < layer_14_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_14_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[14] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[14]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_14_weight_scale, layer_14_weight_zero_point);
        layer_14_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 14
    layer_14_bias_dequant.resize(512);
    for (size_t i = 0; i < layer_14_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_14_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[14] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[14]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_14_bias_scale, layer_14_bias_zero_point);
        layer_14_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 16
    layer_16_weights_dequant.resize(512);
    for (size_t i = 0; i < layer_16_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_16_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[16] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[16]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_16_weight_scale, layer_16_weight_zero_point);
        layer_16_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 16
    layer_16_bias_dequant.resize(512);
    for (size_t i = 0; i < layer_16_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_16_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[16] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[16]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_16_bias_scale, layer_16_bias_zero_point);
        layer_16_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 18
    layer_18_weights_dequant.resize(2359296);
    for (size_t i = 0; i < layer_18_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_18_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[18] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[18]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_18_weight_scale, layer_18_weight_zero_point);
        layer_18_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 18
    layer_18_bias_dequant.resize(512);
    for (size_t i = 0; i < layer_18_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_18_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[18] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[18]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_18_bias_scale, layer_18_bias_zero_point);
        layer_18_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for BatchNorm Layer 19
    layer_19_weights_dequant.resize(512);
    for (size_t i = 0; i < layer_19_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_19_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[19] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[19]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_19_weight_scale, layer_19_weight_zero_point);
        layer_19_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 19
    layer_19_bias_dequant.resize(512);
    for (size_t i = 0; i < layer_19_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_19_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[19] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[19]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_19_bias_scale, layer_19_bias_zero_point);
        layer_19_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 22
    layer_22_weights_dequant.resize(8388608);
    for (size_t i = 0; i < layer_22_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_22_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[22] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[22]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_22_weight_scale, layer_22_weight_zero_point);
        layer_22_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 22
    layer_22_bias_dequant.resize(1024);
    for (size_t i = 0; i < layer_22_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_22_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[22] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[22]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_22_bias_scale, layer_22_bias_zero_point);
        layer_22_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 24
    layer_24_weights_dequant.resize(10240);
    for (size_t i = 0; i < layer_24_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_24_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[24] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[24]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_24_weight_scale, layer_24_weight_zero_point);
        layer_24_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for Layer 24
    layer_24_bias_dequant.resize(10);
    for (size_t i = 0; i < layer_24_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_24_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[24] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[24]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_24_bias_scale, layer_24_bias_zero_point);
        layer_24_bias_dequant[i] = dequant_bias[0];
    }
    files_loaded = true;
    std::cout << "Model parameters loaded successfully." << std::endl << std::endl;
}
                
                // Reshape input to 3D tensor (for convolution)
auto input_3d = cnn_utils::reshape_input_to_3d(x, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);

// Layer 0: Conv2d
std::vector<std::vector<std::vector<float>>> layer_0_3d(128, std::vector<std::vector<float>>(32, std::vector<float>(32, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 128; out_c++) {
    for (int h_out = 0; h_out < 32; h_out++) {
        for (int w_out = 0; w_out < 32; w_out++) {
            layer_0_3d[out_c][h_out][w_out] = layer_0_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 128; out_c++) {
    for (int in_c = 0; in_c < 3; in_c++) {
        for (int h_out = 0; h_out < 32; h_out++) {
            for (int w_out = 0; w_out < 32; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 32 || w_in < 0 || w_in >= 32) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 3 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_0_3d[out_c][h_out][w_out] += input_3d[in_c][h_in][w_in] * layer_0_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 1: MaxPool2d
std::vector<std::vector<std::vector<float>>> layer_1_3d(128, std::vector<std::vector<float>>(16, std::vector<float>(16, 0.0f)));

// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])
for (int c = 0; c < 128; c++) {
    for (int h_out = 0; h_out < 16; h_out++) {
        for (int w_out = 0; w_out < 16; w_out++) {
            // Calculate input region
            int h_start = h_out * 2 - 0;
            int w_start = w_out * 2 - 0;
            int h_end = std::min(h_start + 2, 32);
            int w_end = std::min(w_start + 2, 32);
            h_start = std::max(h_start, 0);
            w_start = std::max(w_start, 0);

            // Find max value in the kernel region
            float max_val = -std::numeric_limits<float>::infinity();
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    float val = layer_0_3d[c][h][w];
                    max_val = std::max(max_val, val);
                }
            }
            layer_1_3d[c][h_out][w_out] = max_val;
        }
    }
}
// Layer 2: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_2_3d(128, std::vector<std::vector<float>>(16, std::vector<float>(16, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 128; c++) {
    float inv_std = 1.0f / std::sqrt(layer_2_running_var[c] + layer_2_eps);
    for (int h = 0; h < 16; h++) {
        for (int w = 0; w < 16; w++) {
            float normalized = (layer_1_3d[c][h][w] - layer_2_running_mean[c]) * inv_std;
            layer_2_3d[c][h][w] = layer_2_weights_dequant[c] * normalized + layer_2_bias_dequant[c];
        }
    }
}
// Layer 3: ReLU
cnn_utils::apply_relu_3d(layer_2_3d);
std::vector<std::vector<std::vector<float>>> layer_3_3d = layer_2_3d;
// Layer 4: Conv2d
std::vector<std::vector<std::vector<float>>> layer_4_3d(128, std::vector<std::vector<float>>(16, std::vector<float>(16, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 128; out_c++) {
    for (int h_out = 0; h_out < 16; h_out++) {
        for (int w_out = 0; w_out < 16; w_out++) {
            layer_4_3d[out_c][h_out][w_out] = layer_4_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 128; out_c++) {
    for (int in_c = 0; in_c < 128; in_c++) {
        for (int h_out = 0; h_out < 16; h_out++) {
            for (int w_out = 0; w_out < 16; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 16 || w_in < 0 || w_in >= 16) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 128 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_4_3d[out_c][h_out][w_out] += layer_3_3d[in_c][h_in][w_in] * layer_4_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 5: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_5_3d(128, std::vector<std::vector<float>>(16, std::vector<float>(16, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 128; c++) {
    float inv_std = 1.0f / std::sqrt(layer_5_running_var[c] + layer_5_eps);
    for (int h = 0; h < 16; h++) {
        for (int w = 0; w < 16; w++) {
            float normalized = (layer_4_3d[c][h][w] - layer_5_running_mean[c]) * inv_std;
            layer_5_3d[c][h][w] = layer_5_weights_dequant[c] * normalized + layer_5_bias_dequant[c];
        }
    }
}
// Layer 6: ReLU
cnn_utils::apply_relu_3d(layer_5_3d);
std::vector<std::vector<std::vector<float>>> layer_6_3d = layer_5_3d;
// Layer 7: Conv2d
std::vector<std::vector<std::vector<float>>> layer_7_3d(256, std::vector<std::vector<float>>(16, std::vector<float>(16, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 256; out_c++) {
    for (int h_out = 0; h_out < 16; h_out++) {
        for (int w_out = 0; w_out < 16; w_out++) {
            layer_7_3d[out_c][h_out][w_out] = layer_7_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 256; out_c++) {
    for (int in_c = 0; in_c < 128; in_c++) {
        for (int h_out = 0; h_out < 16; h_out++) {
            for (int w_out = 0; w_out < 16; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 16 || w_in < 0 || w_in >= 16) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 128 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_7_3d[out_c][h_out][w_out] += layer_6_3d[in_c][h_in][w_in] * layer_7_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 8: MaxPool2d
std::vector<std::vector<std::vector<float>>> layer_8_3d(256, std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0f)));

// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])
for (int c = 0; c < 256; c++) {
    for (int h_out = 0; h_out < 8; h_out++) {
        for (int w_out = 0; w_out < 8; w_out++) {
            // Calculate input region
            int h_start = h_out * 2 - 0;
            int w_start = w_out * 2 - 0;
            int h_end = std::min(h_start + 2, 16);
            int w_end = std::min(w_start + 2, 16);
            h_start = std::max(h_start, 0);
            w_start = std::max(w_start, 0);

            // Find max value in the kernel region
            float max_val = -std::numeric_limits<float>::infinity();
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    float val = layer_7_3d[c][h][w];
                    max_val = std::max(max_val, val);
                }
            }
            layer_8_3d[c][h_out][w_out] = max_val;
        }
    }
}
// Layer 9: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_9_3d(256, std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 256; c++) {
    float inv_std = 1.0f / std::sqrt(layer_9_running_var[c] + layer_9_eps);
    for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
            float normalized = (layer_8_3d[c][h][w] - layer_9_running_mean[c]) * inv_std;
            layer_9_3d[c][h][w] = layer_9_weights_dequant[c] * normalized + layer_9_bias_dequant[c];
        }
    }
}
// Layer 10: ReLU
cnn_utils::apply_relu_3d(layer_9_3d);
std::vector<std::vector<std::vector<float>>> layer_10_3d = layer_9_3d;
// Layer 11: Conv2d
std::vector<std::vector<std::vector<float>>> layer_11_3d(256, std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 256; out_c++) {
    for (int h_out = 0; h_out < 8; h_out++) {
        for (int w_out = 0; w_out < 8; w_out++) {
            layer_11_3d[out_c][h_out][w_out] = layer_11_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 256; out_c++) {
    for (int in_c = 0; in_c < 256; in_c++) {
        for (int h_out = 0; h_out < 8; h_out++) {
            for (int w_out = 0; w_out < 8; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 8 || w_in < 0 || w_in >= 8) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 256 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_11_3d[out_c][h_out][w_out] += layer_10_3d[in_c][h_in][w_in] * layer_11_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 12: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_12_3d(256, std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 256; c++) {
    float inv_std = 1.0f / std::sqrt(layer_12_running_var[c] + layer_12_eps);
    for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
            float normalized = (layer_11_3d[c][h][w] - layer_12_running_mean[c]) * inv_std;
            layer_12_3d[c][h][w] = layer_12_weights_dequant[c] * normalized + layer_12_bias_dequant[c];
        }
    }
}
// Layer 13: ReLU
cnn_utils::apply_relu_3d(layer_12_3d);
std::vector<std::vector<std::vector<float>>> layer_13_3d = layer_12_3d;
// Layer 14: Conv2d
std::vector<std::vector<std::vector<float>>> layer_14_3d(512, std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 512; out_c++) {
    for (int h_out = 0; h_out < 8; h_out++) {
        for (int w_out = 0; w_out < 8; w_out++) {
            layer_14_3d[out_c][h_out][w_out] = layer_14_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 512; out_c++) {
    for (int in_c = 0; in_c < 256; in_c++) {
        for (int h_out = 0; h_out < 8; h_out++) {
            for (int w_out = 0; w_out < 8; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 8 || w_in < 0 || w_in >= 8) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 256 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_14_3d[out_c][h_out][w_out] += layer_13_3d[in_c][h_in][w_in] * layer_14_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 15: MaxPool2d
std::vector<std::vector<std::vector<float>>> layer_15_3d(512, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0f)));

// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])
for (int c = 0; c < 512; c++) {
    for (int h_out = 0; h_out < 4; h_out++) {
        for (int w_out = 0; w_out < 4; w_out++) {
            // Calculate input region
            int h_start = h_out * 2 - 0;
            int w_start = w_out * 2 - 0;
            int h_end = std::min(h_start + 2, 8);
            int w_end = std::min(w_start + 2, 8);
            h_start = std::max(h_start, 0);
            w_start = std::max(w_start, 0);

            // Find max value in the kernel region
            float max_val = -std::numeric_limits<float>::infinity();
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    float val = layer_14_3d[c][h][w];
                    max_val = std::max(max_val, val);
                }
            }
            layer_15_3d[c][h_out][w_out] = max_val;
        }
    }
}
// Layer 16: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_16_3d(512, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 512; c++) {
    float inv_std = 1.0f / std::sqrt(layer_16_running_var[c] + layer_16_eps);
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            float normalized = (layer_15_3d[c][h][w] - layer_16_running_mean[c]) * inv_std;
            layer_16_3d[c][h][w] = layer_16_weights_dequant[c] * normalized + layer_16_bias_dequant[c];
        }
    }
}
// Layer 17: ReLU
cnn_utils::apply_relu_3d(layer_16_3d);
std::vector<std::vector<std::vector<float>>> layer_17_3d = layer_16_3d;
// Layer 18: Conv2d
std::vector<std::vector<std::vector<float>>> layer_18_3d(512, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 512; out_c++) {
    for (int h_out = 0; h_out < 4; h_out++) {
        for (int w_out = 0; w_out < 4; w_out++) {
            layer_18_3d[out_c][h_out][w_out] = layer_18_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 512; out_c++) {
    for (int in_c = 0; in_c < 512; in_c++) {
        for (int h_out = 0; h_out < 4; h_out++) {
            for (int w_out = 0; w_out < 4; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 4 || w_in < 0 || w_in >= 4) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 512 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_18_3d[out_c][h_out][w_out] += layer_17_3d[in_c][h_in][w_in] * layer_18_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 19: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_19_3d(512, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 512; c++) {
    float inv_std = 1.0f / std::sqrt(layer_19_running_var[c] + layer_19_eps);
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            float normalized = (layer_18_3d[c][h][w] - layer_19_running_mean[c]) * inv_std;
            layer_19_3d[c][h][w] = layer_19_weights_dequant[c] * normalized + layer_19_bias_dequant[c];
        }
    }
}
// Layer 20: ReLU
cnn_utils::apply_relu_3d(layer_19_3d);
std::vector<std::vector<std::vector<float>>> layer_20_3d = layer_19_3d;
// Layer 21: Flatten
// Flatten 3D tensor to 1D (size: 8192)
int idx = 0;
for (int c = 0; c < 512; c++) {
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            layer_21[idx++] = layer_20_3d[c][h][w];
        }
    }
}
// Layer 22: Linear (8192 -> 1024)
// Inference formula: y = x * W^T + b
for (int o = 0; o < 1024; o++) {
    layer_22[o] = layer_22_bias_dequant[o];
}

for (int o = 0; o < 1024; o++) {
    for (int j = 0; j < 8192; j++) {
        layer_22[o] += layer_21[j] * layer_22_weights_dequant[o * 8192 + j];
    }
}
// Layer 23: ReLU
for (unsigned int j = 0; j < 1024; j++) {
    layer_23[j] = std::max(0.0f, layer_22[j]);
}
// Layer 24: Linear (1024 -> 10)
// Inference formula: y = x * W^T + b
for (int o = 0; o < 10; o++) {
    layer_24[o] = layer_24_bias_dequant[o];
}

for (int o = 0; o < 10; o++) {
    for (int j = 0; j < 1024; j++) {
        layer_24[o] += layer_23[j] * layer_24_weights_dequant[o * 1024 + j];
    }
}

// Return the output of the final layer
return std::vector<float>(layer_24, layer_24 + 10);
            }
        