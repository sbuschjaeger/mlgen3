#include "matquant_pt_c.h"
#include <string.h>

/* Global arrays for model parameters */
static int8_t layer_0_lin_weight_q8[512][784];
static float layer_0_lin_weight_scale;
static float layer_0_lin_weight_zero_point;
static float layer_0_lin_output[512];
static int8_t layer_1_bn_scale_q8[512];
static float layer_1_bn_scale_scale;
static float layer_1_bn_scale_zero_point;
static int8_t layer_1_bn_bias_q8[512];
static float layer_1_bn_bias_scale;
static float layer_1_bn_bias_zero_point;
static float layer_1_bn_output[512];
static float layer_2_relu_output[512];
static int8_t layer_3_lin_weight_q8[512][512];
static float layer_3_lin_weight_scale;
static float layer_3_lin_weight_zero_point;
static float layer_3_lin_output[512];
static int8_t layer_4_bn_scale_q8[512];
static float layer_4_bn_scale_scale;
static float layer_4_bn_scale_zero_point;
static int8_t layer_4_bn_bias_q8[512];
static float layer_4_bn_bias_scale;
static float layer_4_bn_bias_zero_point;
static float layer_4_bn_output[512];
static float layer_5_relu_output[512];
static int8_t layer_6_lin_weight_q8[10][512];
static float layer_6_lin_weight_scale;
static float layer_6_lin_weight_zero_point;
static float layer_6_lin_output[10];


/* Load binary file helper */
static int load_binary_file(const char* filename, void* data, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 0;
    }
    
    size_t read_size = fread(data, 1, size, file);
    fclose(file);
    
    if (read_size != size) {
        fprintf(stderr, "Error: Expected %zu bytes, read %zu bytes\n", size, read_size);
        return 0;
    }
    
    return 1;
}

/* Load quantization parameters */
static int load_qparams(const char* filename, float* scale, float* zero_point) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 0;
    }
    
    size_t read_count = 0;
    read_count += fread(scale, sizeof(float), 1, file);
    read_count += fread(zero_point, sizeof(float), 1, file);
    fclose(file);
    
    if (read_count != 2) {
        fprintf(stderr, "Error: Failed to read quantization parameters from %s\n", filename);
        return 0;
    }
    
    return 1;
}

/* Dequantize function */
static inline float dequantize(int8_t value, float scale, float zero_point) {
    return ((float)value - zero_point) * scale;
}

/* Load all model parameters */
int load_model_parameters(void) {

    /* Load Linear layer 0 */
    if (!load_binary_file("mq_pt_model_binary/layer_0_lin_weight.bin", 
                         layer_0_lin_weight_q8, 
                         sizeof(layer_0_lin_weight_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_0_lin_weight_qparams.bin",
                     &layer_0_lin_weight_scale, 
                     &layer_0_lin_weight_zero_point)) return 0;
    

    /* Load BatchNorm layer 1 */
    if (!load_binary_file("mq_pt_model_binary/layer_1_bn_scale.bin",
                         layer_1_bn_scale_q8,
                         sizeof(layer_1_bn_scale_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_1_bn_scale_qparams.bin",
                     &layer_1_bn_scale_scale,
                     &layer_1_bn_scale_zero_point)) return 0;
    
    if (!load_binary_file("mq_pt_model_binary/layer_1_bn_bias.bin",
                         layer_1_bn_bias_q8,
                         sizeof(layer_1_bn_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_1_bn_bias_qparams.bin",
                     &layer_1_bn_bias_scale,
                     &layer_1_bn_bias_zero_point)) return 0;

    /* Load Linear layer 3 */
    if (!load_binary_file("mq_pt_model_binary/layer_3_lin_weight.bin", 
                         layer_3_lin_weight_q8, 
                         sizeof(layer_3_lin_weight_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_3_lin_weight_qparams.bin",
                     &layer_3_lin_weight_scale, 
                     &layer_3_lin_weight_zero_point)) return 0;
    

    /* Load BatchNorm layer 4 */
    if (!load_binary_file("mq_pt_model_binary/layer_4_bn_scale.bin",
                         layer_4_bn_scale_q8,
                         sizeof(layer_4_bn_scale_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_4_bn_scale_qparams.bin",
                     &layer_4_bn_scale_scale,
                     &layer_4_bn_scale_zero_point)) return 0;
    
    if (!load_binary_file("mq_pt_model_binary/layer_4_bn_bias.bin",
                         layer_4_bn_bias_q8,
                         sizeof(layer_4_bn_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_4_bn_bias_qparams.bin",
                     &layer_4_bn_bias_scale,
                     &layer_4_bn_bias_zero_point)) return 0;

    /* Load Linear layer 6 */
    if (!load_binary_file("mq_pt_model_binary/layer_6_lin_weight.bin", 
                         layer_6_lin_weight_q8, 
                         sizeof(layer_6_lin_weight_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_6_lin_weight_qparams.bin",
                     &layer_6_lin_weight_scale, 
                     &layer_6_lin_weight_zero_point)) return 0;
    

    return 1;
}


/* Predict function */
void predict(const float* input, float* output, int input_size, int output_size) {
    static int model_loaded = 0;
    if (!model_loaded) {
        if (!load_model_parameters()) {
            fprintf(stderr, "Error: Failed to load model parameters\n");
            return;
        }
        model_loaded = 1;
    }
    

    /* Linear layer 0 without bias */
    for (int i = 0; i < 512; i++) {
        float acc = 0.0f;
        for (int j = 0; j < 784; j++) {
            float weight_dequant = dequantize(layer_0_lin_weight_q8[i][j],
                                            layer_0_lin_weight_scale,
                                            layer_0_lin_weight_zero_point);
            acc += weight_dequant * input[j];
        }
        layer_0_lin_output[i] = acc;
    }

    /* BatchNorm layer 1 */
    for (int i = 0; i < 512; i++) {
        float scale_dequant = dequantize(layer_1_bn_scale_q8[i],
                                        layer_1_bn_scale_scale,
                                        layer_1_bn_scale_zero_point);
        float bias_dequant = dequantize(layer_1_bn_bias_q8[i],
                                       layer_1_bn_bias_scale,
                                       layer_1_bn_bias_zero_point);
        layer_1_bn_output[i] = layer_0_lin_output[i] * scale_dequant + bias_dequant;
    }

    /* ReLU layer 2 */
    for (int i = 0; i < 512; i++) {
        layer_2_relu_output[i] = layer_1_bn_output[i] > 0.0f ? layer_1_bn_output[i] : 0.0f;
    }

    /* Linear layer 3 without bias */
    for (int i = 0; i < 512; i++) {
        float acc = 0.0f;
        for (int j = 0; j < 512; j++) {
            float weight_dequant = dequantize(layer_3_lin_weight_q8[i][j],
                                            layer_3_lin_weight_scale,
                                            layer_3_lin_weight_zero_point);
            acc += weight_dequant * layer_2_relu_output[j];
        }
        layer_3_lin_output[i] = acc;
    }

    /* BatchNorm layer 4 */
    for (int i = 0; i < 512; i++) {
        float scale_dequant = dequantize(layer_4_bn_scale_q8[i],
                                        layer_4_bn_scale_scale,
                                        layer_4_bn_scale_zero_point);
        float bias_dequant = dequantize(layer_4_bn_bias_q8[i],
                                       layer_4_bn_bias_scale,
                                       layer_4_bn_bias_zero_point);
        layer_4_bn_output[i] = layer_3_lin_output[i] * scale_dequant + bias_dequant;
    }

    /* ReLU layer 5 */
    for (int i = 0; i < 512; i++) {
        layer_5_relu_output[i] = layer_4_bn_output[i] > 0.0f ? layer_4_bn_output[i] : 0.0f;
    }

    /* Linear layer 6 without bias */
    for (int i = 0; i < 10; i++) {
        float acc = 0.0f;
        for (int j = 0; j < 512; j++) {
            float weight_dequant = dequantize(layer_6_lin_weight_q8[i][j],
                                            layer_6_lin_weight_scale,
                                            layer_6_lin_weight_zero_point);
            acc += weight_dequant * layer_5_relu_output[j];
        }
        layer_6_lin_output[i] = acc;
    }

    /* Copy output */
    memcpy(output, layer_6_lin_output, output_size * sizeof(float));
}
