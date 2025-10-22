
            #include "matquant_pt_vgg4_mix_08_08_44_44_92_92_112_112.h"
            #include <fstream>
            #include <iostream>
            #include <cmath>
            #include <algorithm>
            
            // Layer allocations
static float layer_8[3136];
static float layer_9[2048];
static float layer_10[2048];
static float layer_11[10];

            
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

// batchnorm2d layer 6
std::vector<int8_t> layer_6_weight_q8;
float layer_6_weight_scale = 1.0f;
float layer_6_weight_zero_point = 0.0f;
std::vector<float> layer_6_weights_dequant;
std::vector<int8_t> layer_6_bias_q8;
float layer_6_bias_scale = 1.0f;
float layer_6_bias_zero_point = 0.0f;
std::vector<float> layer_6_bias_dequant;
std::vector<float> layer_6_running_mean;
std::vector<float> layer_6_running_var;
float layer_6_eps = 1e-5f;

// linear layer 9
std::vector<int8_t> layer_9_weight_q8;
float layer_9_weight_scale = 1.0f;
float layer_9_weight_zero_point = 0.0f;
std::vector<float> layer_9_weights_dequant;
std::vector<int8_t> layer_9_bias_q8;
float layer_9_bias_scale = 1.0f;
float layer_9_bias_zero_point = 0.0f;
std::vector<float> layer_9_bias_dequant;

// linear layer 11
std::vector<int8_t> layer_11_weight_q8;
float layer_11_weight_scale = 1.0f;
float layer_11_weight_zero_point = 0.0f;
std::vector<float> layer_11_weights_dequant;
std::vector<int8_t> layer_11_bias_q8;
float layer_11_bias_scale = 1.0f;
float layer_11_bias_zero_point = 0.0f;
std::vector<float> layer_11_bias_dequant;

            
            std::vector<float> predict(std::vector<float> &x) {
                // Load binary files if not loaded
static bool files_loaded = false;
if (!files_loaded) {
    std::cout << std::endl << "Loading model parameters from binary files..." << std::endl;
    // Load weights for layer 0 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_0_weight.bin", layer_0_weight_q8, 576);
    load_quantization_params("mq_pt_model_binary/layer_0_weight_qparams.bin", layer_0_weight_scale, layer_0_weight_zero_point);
    // Load bias for layer 0
    load_binary_data<int8_t>("mq_pt_model_binary/layer_0_bias.bin", layer_0_bias_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_0_bias_qparams.bin", layer_0_bias_scale, layer_0_bias_zero_point);
    // Load BatchNorm2d parameters for layer 2
    load_binary_data<int8_t>("mq_pt_model_binary/layer_2_weight.bin", layer_2_weight_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_2_weight_qparams.bin", layer_2_weight_scale, layer_2_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_2_bias.bin", layer_2_bias_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_2_bias_qparams.bin", layer_2_bias_scale, layer_2_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_2_running_mean.bin", layer_2_running_mean, 64);
    load_binary_data("mq_pt_model_binary/layer_2_running_var.bin", layer_2_running_var, 64);
    // Load eps parameter
    std::ifstream eps_file_2("mq_pt_model_binary/layer_2_eps.bin", std::ios::binary);
    if (eps_file_2.is_open()) {
        eps_file_2.read(reinterpret_cast<char*>(&layer_2_eps), sizeof(float));
        eps_file_2.close();
    }
    // Load weights for layer 4 (conv)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_4_weight.bin", layer_4_weight_q8, 36864);
    load_quantization_params("mq_pt_model_binary/layer_4_weight_qparams.bin", layer_4_weight_scale, layer_4_weight_zero_point);
    // Load bias for layer 4
    load_binary_data<int8_t>("mq_pt_model_binary/layer_4_bias.bin", layer_4_bias_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_4_bias_qparams.bin", layer_4_bias_scale, layer_4_bias_zero_point);
    // Load BatchNorm2d parameters for layer 6
    load_binary_data<int8_t>("mq_pt_model_binary/layer_6_weight.bin", layer_6_weight_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_6_weight_qparams.bin", layer_6_weight_scale, layer_6_weight_zero_point);
    load_binary_data<int8_t>("mq_pt_model_binary/layer_6_bias.bin", layer_6_bias_q8, 64);
    load_quantization_params("mq_pt_model_binary/layer_6_bias_qparams.bin", layer_6_bias_scale, layer_6_bias_zero_point);
    load_binary_data("mq_pt_model_binary/layer_6_running_mean.bin", layer_6_running_mean, 64);
    load_binary_data("mq_pt_model_binary/layer_6_running_var.bin", layer_6_running_var, 64);
    // Load eps parameter
    std::ifstream eps_file_6("mq_pt_model_binary/layer_6_eps.bin", std::ios::binary);
    if (eps_file_6.is_open()) {
        eps_file_6.read(reinterpret_cast<char*>(&layer_6_eps), sizeof(float));
        eps_file_6.close();
    }
    // Load weights for layer 9 (linear)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_9_weight.bin", layer_9_weight_q8, 6422528);
    load_quantization_params("mq_pt_model_binary/layer_9_weight_qparams.bin", layer_9_weight_scale, layer_9_weight_zero_point);
    // Load bias for layer 9
    load_binary_data<int8_t>("mq_pt_model_binary/layer_9_bias.bin", layer_9_bias_q8, 2048);
    load_quantization_params("mq_pt_model_binary/layer_9_bias_qparams.bin", layer_9_bias_scale, layer_9_bias_zero_point);
    // Load weights for layer 11 (linear)
    load_binary_data<int8_t>("mq_pt_model_binary/layer_11_weight.bin", layer_11_weight_q8, 20480);
    load_quantization_params("mq_pt_model_binary/layer_11_weight_qparams.bin", layer_11_weight_scale, layer_11_weight_zero_point);
    // Load bias for layer 11
    load_binary_data<int8_t>("mq_pt_model_binary/layer_11_bias.bin", layer_11_bias_q8, 10);
    load_quantization_params("mq_pt_model_binary/layer_11_bias_qparams.bin", layer_11_bias_scale, layer_11_bias_zero_point);

    std::cout << "Precomputing dequantized weights and biases for faster inference..." << std::endl;
    // Precompute dequantized weights for Layer 0
    layer_0_weights_dequant.resize(576);
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
    layer_0_bias_dequant.resize(64);
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
    layer_2_weights_dequant.resize(64);
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
    layer_2_bias_dequant.resize(64);
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
    layer_4_weights_dequant.resize(36864);
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
    layer_4_bias_dequant.resize(64);
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
    // Precompute dequantized weights for BatchNorm Layer 6
    layer_6_weights_dequant.resize(64);
    for (size_t i = 0; i < layer_6_weight_q8.size(); i++) {
        std::vector<int8_t> weight_q8(1, layer_6_weight_q8[i]);
        std::vector<int8_t> sliced_weight;
        if (LAYER_BITS[6] < STORAGE_BITS) {
            sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[6]);
        } else {
            sliced_weight = weight_q8;
        }
        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_6_weight_scale, layer_6_weight_zero_point);
        layer_6_weights_dequant[i] = dequant_weight[0];
    }
    // Precompute dequantized biases for BatchNorm Layer 6
    layer_6_bias_dequant.resize(64);
    for (size_t i = 0; i < layer_6_bias_q8.size(); i++) {
        std::vector<int8_t> bias_q8(1, layer_6_bias_q8[i]);
        std::vector<int8_t> sliced_bias;
        if (LAYER_BITS[6] < STORAGE_BITS) {
            sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[6]);
        } else {
            sliced_bias = bias_q8;
        }
        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_6_bias_scale, layer_6_bias_zero_point);
        layer_6_bias_dequant[i] = dequant_bias[0];
    }
    // Precompute dequantized weights for Layer 9
    layer_9_weights_dequant.resize(6422528);
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
    // Precompute dequantized biases for Layer 9
    layer_9_bias_dequant.resize(2048);
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
    layer_11_weights_dequant.resize(20480);
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
    layer_11_bias_dequant.resize(10);
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
    files_loaded = true;
    std::cout << "Model parameters loaded successfully." << std::endl << std::endl;
}
                
                // Reshape input to 3D tensor (for convolution)
auto input_3d = cnn_utils::reshape_input_to_3d(x, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);

// Layer 0: Conv2d
std::vector<std::vector<std::vector<float>>> layer_0_3d(64, std::vector<std::vector<float>>(28, std::vector<float>(28, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 64; out_c++) {
    for (int h_out = 0; h_out < 28; h_out++) {
        for (int w_out = 0; w_out < 28; w_out++) {
            layer_0_3d[out_c][h_out][w_out] = layer_0_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 64; out_c++) {
    for (int in_c = 0; in_c < 1; in_c++) {
        for (int h_out = 0; h_out < 28; h_out++) {
            for (int w_out = 0; w_out < 28; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 28 || w_in < 0 || w_in >= 28) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 1 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_0_3d[out_c][h_out][w_out] += input_3d[in_c][h_in][w_in] * layer_0_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 1: MaxPool2d
std::vector<std::vector<std::vector<float>>> layer_1_3d(64, std::vector<std::vector<float>>(14, std::vector<float>(14, 0.0f)));

// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])
for (int c = 0; c < 64; c++) {
    for (int h_out = 0; h_out < 14; h_out++) {
        for (int w_out = 0; w_out < 14; w_out++) {
            // Calculate input region
            int h_start = h_out * 2 - 0;
            int w_start = w_out * 2 - 0;
            int h_end = std::min(h_start + 2, 28);
            int w_end = std::min(w_start + 2, 28);
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
std::vector<std::vector<std::vector<float>>> layer_2_3d(64, std::vector<std::vector<float>>(14, std::vector<float>(14, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 64; c++) {
    float inv_std = 1.0f / std::sqrt(layer_2_running_var[c] + layer_2_eps);
    for (int h = 0; h < 14; h++) {
        for (int w = 0; w < 14; w++) {
            float normalized = (layer_1_3d[c][h][w] - layer_2_running_mean[c]) * inv_std;
            layer_2_3d[c][h][w] = layer_2_weights_dequant[c] * normalized + layer_2_bias_dequant[c];
        }
    }
}
// Layer 3: ReLU
cnn_utils::apply_relu_3d(layer_2_3d);
std::vector<std::vector<std::vector<float>>> layer_3_3d = layer_2_3d;
// Layer 4: Conv2d
std::vector<std::vector<std::vector<float>>> layer_4_3d(64, std::vector<std::vector<float>>(14, std::vector<float>(14, 0.0f)));

// Initialize output with precomputed dequantized bias
for (int out_c = 0; out_c < 64; out_c++) {
    for (int h_out = 0; h_out < 14; h_out++) {
        for (int w_out = 0; w_out < 14; w_out++) {
            layer_4_3d[out_c][h_out][w_out] = layer_4_bias_dequant[out_c];
        }
    }
}

// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]
// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]
for (int out_c = 0; out_c < 64; out_c++) {
    for (int in_c = 0; in_c < 64; in_c++) {
        for (int h_out = 0; h_out < 14; h_out++) {
            for (int w_out = 0; w_out < 14; w_out++) {
                // Compute convolution with kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Calculate input position with padding
                        // padding=1, stride=1
                        int h_in = h_out * 1 + kh - 1;
                        int w_in = w_out * 1 + kw - 1;

                        // Skip if outside input boundaries (padding is implicit zero)
                        if (h_in < 0 || h_in >= 14 || w_in < 0 || w_in >= 14) {
                            continue;
                        }

                        // Calculate weight index: [out_c, in_c, kh, kw]
                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w
                        int weight_idx = ((out_c * 64 + in_c) * 3 + kh) * 3 + kw;

                        // Accumulate convolution result
                        layer_4_3d[out_c][h_out][w_out] += layer_3_3d[in_c][h_in][w_in] * layer_4_weights_dequant[weight_idx];
                    }
                }
            }
        }
    }
}
// Layer 5: MaxPool2d
std::vector<std::vector<std::vector<float>>> layer_5_3d(64, std::vector<std::vector<float>>(7, std::vector<float>(7, 0.0f)));

// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])
for (int c = 0; c < 64; c++) {
    for (int h_out = 0; h_out < 7; h_out++) {
        for (int w_out = 0; w_out < 7; w_out++) {
            // Calculate input region
            int h_start = h_out * 2 - 0;
            int w_start = w_out * 2 - 0;
            int h_end = std::min(h_start + 2, 14);
            int w_end = std::min(w_start + 2, 14);
            h_start = std::max(h_start, 0);
            w_start = std::max(w_start, 0);

            // Find max value in the kernel region
            float max_val = -std::numeric_limits<float>::infinity();
            for (int h = h_start; h < h_end; h++) {
                for (int w = w_start; w < w_end; w++) {
                    float val = layer_4_3d[c][h][w];
                    max_val = std::max(max_val, val);
                }
            }
            layer_5_3d[c][h_out][w_out] = max_val;
        }
    }
}
// Layer 6: BatchNorm2d
std::vector<std::vector<std::vector<float>>> layer_6_3d(64, std::vector<std::vector<float>>(7, std::vector<float>(7, 0.0f)));

// Apply BatchNorm inference formula:
// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
for (int c = 0; c < 64; c++) {
    float inv_std = 1.0f / std::sqrt(layer_6_running_var[c] + layer_6_eps);
    for (int h = 0; h < 7; h++) {
        for (int w = 0; w < 7; w++) {
            float normalized = (layer_5_3d[c][h][w] - layer_6_running_mean[c]) * inv_std;
            layer_6_3d[c][h][w] = layer_6_weights_dequant[c] * normalized + layer_6_bias_dequant[c];
        }
    }
}
// Layer 7: ReLU
cnn_utils::apply_relu_3d(layer_6_3d);
std::vector<std::vector<std::vector<float>>> layer_7_3d = layer_6_3d;
// Layer 8: Flatten
// Flatten 3D tensor to 1D (size: 3136)
int idx = 0;
for (int c = 0; c < 64; c++) {
    for (int h = 0; h < 7; h++) {
        for (int w = 0; w < 7; w++) {
            layer_8[idx++] = layer_7_3d[c][h][w];
        }
    }
}
// Layer 9: Linear (3136 -> 2048)
// Inference formula: y = x * W^T + b
for (int o = 0; o < 2048; o++) {
    layer_9[o] = layer_9_bias_dequant[o];
}

for (int o = 0; o < 2048; o++) {
    for (int j = 0; j < 3136; j++) {
        layer_9[o] += layer_8[j] * layer_9_weights_dequant[o * 3136 + j];
    }
}
// Layer 10: ReLU
for (unsigned int j = 0; j < 2048; j++) {
    layer_10[j] = std::max(0.0f, layer_9[j]);
}
// Layer 11: Linear (2048 -> 10)
// Inference formula: y = x * W^T + b
for (int o = 0; o < 10; o++) {
    layer_11[o] = layer_11_bias_dequant[o];
}

for (int o = 0; o < 10; o++) {
    for (int j = 0; j < 2048; j++) {
        layer_11[o] += layer_10[j] * layer_11_weights_dequant[o * 2048 + j];
    }
}

// Return the output of the final layer
return std::vector<float>(layer_11, layer_11 + 10);
            }
        