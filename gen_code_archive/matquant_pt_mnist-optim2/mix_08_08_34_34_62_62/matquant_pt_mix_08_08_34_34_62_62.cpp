
            #include "matquant_pt_mix_08_08_34_34_62_62.h"
            
            static float layer_0[512];
static int8_t layer_0_weight_q8[512][784];
static float layer_0_weight_scale;
static float layer_0_weight_zero_point;
static float layer_0_weights_dequant[512][784];
static int8_t layer_0_bias_q8[512];
static float layer_0_bias_scale;
static float layer_0_bias_zero_point;
static float layer_0_bias_dequant[512];
static float layer_1[512];
static int8_t layer_1_scale_q8[512];
static float layer_1_scale_scale;
static float layer_1_scale_zero_point;
static float layer_1_scale_dequant[512];
static int8_t layer_1_bias_q8[512];
static float layer_1_bias_scale;
static float layer_1_bias_zero_point;
static float layer_1_bias_dequant[512];
static float layer_2[512];
static float layer_3[512];
static int8_t layer_3_weight_q8[512][512];
static float layer_3_weight_scale;
static float layer_3_weight_zero_point;
static float layer_3_weights_dequant[512][512];
static int8_t layer_3_bias_q8[512];
static float layer_3_bias_scale;
static float layer_3_bias_zero_point;
static float layer_3_bias_dequant[512];
static float layer_4[512];
static int8_t layer_4_scale_q8[512];
static float layer_4_scale_scale;
static float layer_4_scale_zero_point;
static float layer_4_scale_dequant[512];
static int8_t layer_4_bias_q8[512];
static float layer_4_bias_scale;
static float layer_4_bias_zero_point;
static float layer_4_bias_dequant[512];
static float layer_5[512];
static float layer_6[10];
static int8_t layer_6_weight_q8[10][512];
static float layer_6_weight_scale;
static float layer_6_weight_zero_point;
static float layer_6_weights_dequant[10][512];
static int8_t layer_6_bias_q8[10];
static float layer_6_bias_scale;
static float layer_6_bias_zero_point;
static float layer_6_bias_dequant[10];

            
            // Function to load model parameters from binary files
            bool load_model_parameters() {
                bool success = true;
                
                
                    // Load and precompute weights for Linear layer 0
                    {
                        // Load quantized weights
                        std::vector<int8_t> layer_0_weight_data;
                        bool weight_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_0_weight.bin", 
                                                                      layer_0_weight_data, 
                                                                      512 * 784);
                        
                        // Load quantization parameters
                        bool qparam_success = load_quantization_params("mq_pt_model_binary/layer_0_weight_qparams.bin",
                                                                     layer_0_weight_scale, 
                                                                     layer_0_weight_zero_point);
                        
                        if (weight_success && qparam_success) {
                            // Copy to 2D array and precompute dequantized values
                            for (int i = 0; i < 512; ++i) {
                                for (int j = 0; j < 784; ++j) {
                                    int idx = i * 784 + j;
                                    layer_0_weight_q8[i][j] = layer_0_weight_data[idx];
                                    
                                    // Slice and dequantize
                                    std::vector<int8_t> weight_q8(1, layer_0_weight_q8[i][j]);
                                    std::vector<int8_t> sliced_weight;
                                    if (LAYER_BITS[0] < STORAGE_BITS) {
                                        sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[0]);
                                    } else {
                                        sliced_weight = weight_q8;
                                    }
                                    std::vector<float> dequant_weight = dequantize(sliced_weight, layer_0_weight_scale, layer_0_weight_zero_point);
                                    layer_0_weights_dequant[i][j] = dequant_weight[0];
                                }
                            }
                        } else {
                            success = false;
                            std::cerr << "Failed to load weights for layer 0" << std::endl;
                        }
                        
                        // Load and precompute biases
                        std::vector<int8_t> layer_0_bias_data;
                        bool bias_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_0_bias.bin", 
                                                                   layer_0_bias_data, 
                                                                   512);
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_0_bias_qparams.bin",
                                                                         layer_0_bias_scale, 
                                                                         layer_0_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_0_bias_q8[i] = layer_0_bias_data[i];
                                
                                // Slice and dequantize
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
                        } else {
                            success = false;
                            std::cerr << "Failed to load bias for layer 0" << std::endl;
                        }
                    }
                    // Load and precompute BatchNorm parameters for layer 1
                    {
                        // Load and precompute scale
                        std::vector<int8_t> layer_1_scale_data;
                        bool scale_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_1_scale.bin", 
                                                                    layer_1_scale_data, 
                                                                    512);
                        
                        bool scale_qparam_success = load_quantization_params("mq_pt_model_binary/layer_1_scale_qparams.bin",
                                                                         layer_1_scale_scale, 
                                                                         layer_1_scale_zero_point);
                        
                        if (scale_success && scale_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_1_scale_q8[i] = layer_1_scale_data[i];
                                
                                // Slice and dequantize
                                std::vector<int8_t> scale_q8(1, layer_1_scale_q8[i]);
                                std::vector<int8_t> sliced_scale;
                                if (LAYER_BITS[1] < STORAGE_BITS) {
                                    sliced_scale = slice_bits(scale_q8, STORAGE_BITS, LAYER_BITS[1]);
                                } else {
                                    sliced_scale = scale_q8;
                                }
                                std::vector<float> dequant_scale = dequantize(sliced_scale, layer_1_scale_scale, layer_1_scale_zero_point);
                                layer_1_scale_dequant[i] = dequant_scale[0];
                            }
                        } else {
                            success = false;
                        }
                        
                        // Load and precompute bias
                        std::vector<int8_t> layer_1_bias_data;
                        bool bias_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_1_bn_bias.bin", 
                                                                   layer_1_bias_data, 
                                                                   512);
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_1_bn_bias_qparams.bin",
                                                                         layer_1_bias_scale, 
                                                                         layer_1_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_1_bias_q8[i] = layer_1_bias_data[i];
                                
                                // Slice and dequantize
                                std::vector<int8_t> bias_q8(1, layer_1_bias_q8[i]);
                                std::vector<int8_t> sliced_bias;
                                if (LAYER_BITS[1] < STORAGE_BITS) {
                                    sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[1]);
                                } else {
                                    sliced_bias = bias_q8;
                                }
                                std::vector<float> dequant_bias = dequantize(sliced_bias, layer_1_bias_scale, layer_1_bias_zero_point);
                                layer_1_bias_dequant[i] = dequant_bias[0];
                            }
                        } else {
                            success = false;
                        }
                    }
                    // Load and precompute weights for Linear layer 3
                    {
                        // Load quantized weights
                        std::vector<int8_t> layer_3_weight_data;
                        bool weight_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_3_weight.bin", 
                                                                      layer_3_weight_data, 
                                                                      512 * 512);
                        
                        // Load quantization parameters
                        bool qparam_success = load_quantization_params("mq_pt_model_binary/layer_3_weight_qparams.bin",
                                                                     layer_3_weight_scale, 
                                                                     layer_3_weight_zero_point);
                        
                        if (weight_success && qparam_success) {
                            // Copy to 2D array and precompute dequantized values
                            for (int i = 0; i < 512; ++i) {
                                for (int j = 0; j < 512; ++j) {
                                    int idx = i * 512 + j;
                                    layer_3_weight_q8[i][j] = layer_3_weight_data[idx];
                                    
                                    // Slice and dequantize
                                    std::vector<int8_t> weight_q8(1, layer_3_weight_q8[i][j]);
                                    std::vector<int8_t> sliced_weight;
                                    if (LAYER_BITS[3] < STORAGE_BITS) {
                                        sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[3]);
                                    } else {
                                        sliced_weight = weight_q8;
                                    }
                                    std::vector<float> dequant_weight = dequantize(sliced_weight, layer_3_weight_scale, layer_3_weight_zero_point);
                                    layer_3_weights_dequant[i][j] = dequant_weight[0];
                                }
                            }
                        } else {
                            success = false;
                            std::cerr << "Failed to load weights for layer 3" << std::endl;
                        }
                        
                        // Load and precompute biases
                        std::vector<int8_t> layer_3_bias_data;
                        bool bias_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_3_bias.bin", 
                                                                   layer_3_bias_data, 
                                                                   512);
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_3_bias_qparams.bin",
                                                                         layer_3_bias_scale, 
                                                                         layer_3_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_3_bias_q8[i] = layer_3_bias_data[i];
                                
                                // Slice and dequantize
                                std::vector<int8_t> bias_q8(1, layer_3_bias_q8[i]);
                                std::vector<int8_t> sliced_bias;
                                if (LAYER_BITS[3] < STORAGE_BITS) {
                                    sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[3]);
                                } else {
                                    sliced_bias = bias_q8;
                                }
                                std::vector<float> dequant_bias = dequantize(sliced_bias, layer_3_bias_scale, layer_3_bias_zero_point);
                                layer_3_bias_dequant[i] = dequant_bias[0];
                            }
                        } else {
                            success = false;
                            std::cerr << "Failed to load bias for layer 3" << std::endl;
                        }
                    }
                    // Load and precompute BatchNorm parameters for layer 4
                    {
                        // Load and precompute scale
                        std::vector<int8_t> layer_4_scale_data;
                        bool scale_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_4_scale.bin", 
                                                                    layer_4_scale_data, 
                                                                    512);
                        
                        bool scale_qparam_success = load_quantization_params("mq_pt_model_binary/layer_4_scale_qparams.bin",
                                                                         layer_4_scale_scale, 
                                                                         layer_4_scale_zero_point);
                        
                        if (scale_success && scale_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_4_scale_q8[i] = layer_4_scale_data[i];
                                
                                // Slice and dequantize
                                std::vector<int8_t> scale_q8(1, layer_4_scale_q8[i]);
                                std::vector<int8_t> sliced_scale;
                                if (LAYER_BITS[4] < STORAGE_BITS) {
                                    sliced_scale = slice_bits(scale_q8, STORAGE_BITS, LAYER_BITS[4]);
                                } else {
                                    sliced_scale = scale_q8;
                                }
                                std::vector<float> dequant_scale = dequantize(sliced_scale, layer_4_scale_scale, layer_4_scale_zero_point);
                                layer_4_scale_dequant[i] = dequant_scale[0];
                            }
                        } else {
                            success = false;
                        }
                        
                        // Load and precompute bias
                        std::vector<int8_t> layer_4_bias_data;
                        bool bias_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_4_bn_bias.bin", 
                                                                   layer_4_bias_data, 
                                                                   512);
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_4_bn_bias_qparams.bin",
                                                                         layer_4_bias_scale, 
                                                                         layer_4_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {
                            for (int i = 0; i < 512; ++i) {
                                layer_4_bias_q8[i] = layer_4_bias_data[i];
                                
                                // Slice and dequantize
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
                        } else {
                            success = false;
                        }
                    }
                    // Load and precompute weights for Linear layer 6
                    {
                        // Load quantized weights
                        std::vector<int8_t> layer_6_weight_data;
                        bool weight_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_6_weight.bin", 
                                                                      layer_6_weight_data, 
                                                                      10 * 512);
                        
                        // Load quantization parameters
                        bool qparam_success = load_quantization_params("mq_pt_model_binary/layer_6_weight_qparams.bin",
                                                                     layer_6_weight_scale, 
                                                                     layer_6_weight_zero_point);
                        
                        if (weight_success && qparam_success) {
                            // Copy to 2D array and precompute dequantized values
                            for (int i = 0; i < 10; ++i) {
                                for (int j = 0; j < 512; ++j) {
                                    int idx = i * 512 + j;
                                    layer_6_weight_q8[i][j] = layer_6_weight_data[idx];
                                    
                                    // Slice and dequantize
                                    std::vector<int8_t> weight_q8(1, layer_6_weight_q8[i][j]);
                                    std::vector<int8_t> sliced_weight;
                                    if (LAYER_BITS[6] < STORAGE_BITS) {
                                        sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[6]);
                                    } else {
                                        sliced_weight = weight_q8;
                                    }
                                    std::vector<float> dequant_weight = dequantize(sliced_weight, layer_6_weight_scale, layer_6_weight_zero_point);
                                    layer_6_weights_dequant[i][j] = dequant_weight[0];
                                }
                            }
                        } else {
                            success = false;
                            std::cerr << "Failed to load weights for layer 6" << std::endl;
                        }
                        
                        // Load and precompute biases
                        std::vector<int8_t> layer_6_bias_data;
                        bool bias_success = load_binary_data<int8_t>("mq_pt_model_binary/layer_6_bias.bin", 
                                                                   layer_6_bias_data, 
                                                                   10);
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_6_bias_qparams.bin",
                                                                         layer_6_bias_scale, 
                                                                         layer_6_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {
                            for (int i = 0; i < 10; ++i) {
                                layer_6_bias_q8[i] = layer_6_bias_data[i];
                                
                                // Slice and dequantize
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
                        } else {
                            success = false;
                            std::cerr << "Failed to load bias for layer 6" << std::endl;
                        }
                    }
                
                return success;
            }
            
            
        std::vector<float> predict(std::vector<float> &x) {
            // Load model parameters if not loaded already
            static bool model_loaded = false;
            if (!model_loaded) {
                model_loaded = load_model_parameters();
                if (!model_loaded) {
                    std::cerr << "Warning: Failed to load some model parameters." << std::endl;
                }
            }
        
                // Linear layer 0 using precomputed dequantized weights
                for (int i = 0; i < 512; ++i) {
                    layer_0[i] = layer_0_bias_dequant[i];
                    for (int j = 0; j < 784; ++j) {
                        layer_0[i] += layer_0_weights_dequant[i][j] * x[j];
                    }
                }
                
                // BatchNorm layer 1 using precomputed dequantized parameters
                for (int i = 0; i < 512; ++i) {
                    layer_1[i] = layer_0[i] * layer_1_scale_dequant[i] + layer_1_bias_dequant[i];
                }
                
                // ReLU activation layer 2
                for (int i = 0; i < 512; ++i) {
                    layer_2[i] = std::max(0.0f, layer_1[i]);
                }
                
                // Linear layer 3 using precomputed dequantized weights
                for (int i = 0; i < 512; ++i) {
                    layer_3[i] = layer_3_bias_dequant[i];
                    for (int j = 0; j < 512; ++j) {
                        layer_3[i] += layer_3_weights_dequant[i][j] * layer_2[j];
                    }
                }
                
                // BatchNorm layer 4 using precomputed dequantized parameters
                for (int i = 0; i < 512; ++i) {
                    layer_4[i] = layer_3[i] * layer_4_scale_dequant[i] + layer_4_bias_dequant[i];
                }
                
                // ReLU activation layer 5
                for (int i = 0; i < 512; ++i) {
                    layer_5[i] = std::max(0.0f, layer_4[i]);
                }
                
                // Linear layer 6 using precomputed dequantized weights
                for (int i = 0; i < 10; ++i) {
                    layer_6[i] = layer_6_bias_dequant[i];
                    for (int j = 0; j < 512; ++j) {
                        layer_6[i] += layer_6_weights_dequant[i][j] * layer_5[j];
                    }
                }
                
            return std::vector<float>(layer_6, 
                                   layer_6 + 10);
        }
        
        