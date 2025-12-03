import numpy as np
import os
import struct
from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu

class MatQuantPT_C(Implementation):
    """
    Pure C implementation of MatQuant for MLP models.
    Generates integer-only arithmetic C code with 8-bit quantized weights.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float",
                 quantize_signed=True):
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.quantize_signed = quantize_signed
        self.filename = None
        self.model_binary_dir = None
        
    def set_filename(self, filename):
        """Set the filename for header inclusion."""
        self.filename = filename
        
    def set_model_binary_dir(self, model_dir):
        """Set directory for binary model files."""
        self.model_binary_dir = model_dir
        
    def _quantize_to_8bit(self, values):
        """Quantize values to 8-bit representation."""
        val_min = values.min()
        val_max = values.max()
        
        if self.quantize_signed:
            q_min, q_max = -128, 127
            scale = (val_max - val_min) / (q_max - q_min) if val_min != val_max else 1.0
            zero_point = -val_min / scale + q_min if scale != 0 else q_min
            quantized = np.clip(np.round(values / scale + zero_point), q_min, q_max).astype(np.int8)
        else:
            scale = (val_max - val_min) / 255.0 if val_min != val_max else 1.0
            zero_point = -val_min / scale if scale != 0 else 0.0
            quantized = np.clip(np.round(values / scale + zero_point), 0, 255).astype(np.uint8)
        
        return {"quantized_values": quantized, "scale": scale, "zero_point": zero_point}
    
    def extract_model_parameters(self):
        """Extract and save model parameters as 8-bit quantized binary files."""
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                # Quantize weights
                weight = layer.weight.astype(np.float32)
                qparams = self._quantize_to_8bit(weight)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_weight.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_weight_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Quantize bias
                bias = layer.bias.astype(np.float32)
                qparams = self._quantize_to_8bit(bias)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                    
            elif isinstance(layer, BatchNorm):
                # Quantize scale
                scale = layer.scale.astype(np.float32)
                qparams = self._quantize_to_8bit(scale)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_scale.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_scale_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Quantize bias
                bias = layer.bias.astype(np.float32)
                qparams = self._quantize_to_8bit(bias)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
    
    def implement(self):
        """Generate pure C code for the model."""
        if self.model_binary_dir:
            self.extract_model_parameters()
        
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        # Generate header
        self.header = self._generate_header(int_type)
        
        # Generate implementation
        self.code = self._generate_code(int_type)
    
    def _generate_header(self, int_type):
        """Generate C header file."""
        return f"""#ifndef {self.filename.upper()}_H
#define {self.filename.upper()}_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Model configuration */
#define NUM_LAYERS {len(self.model.layers)}
#define QUANTIZE_SIGNED {1 if self.quantize_signed else 0}

/* Function prototypes */
int load_model_parameters(void);
void predict(const float* input, float* output, int input_size, int output_size);

#endif /* {self.filename.upper()}_H */
"""
    
    def _generate_code(self, int_type):
        """Generate C implementation file."""
        code = f"""#include "{self.filename}.h"
#include <string.h>

/* Global arrays for model parameters */
"""
        
        # Declare global arrays for each layer
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                code += f"""static {int_type} layer_{lid}_weight_q8[{layer.output_shape}][{layer.input_shape}];
static float layer_{lid}_weight_scale;
static float layer_{lid}_weight_zero_point;
static {int_type} layer_{lid}_bias_q8[{layer.output_shape}];
static float layer_{lid}_bias_scale;
static float layer_{lid}_bias_zero_point;
static float layer_{lid}_output[{layer.output_shape}];
"""
            elif isinstance(layer, BatchNorm):
                code += f"""static {int_type} layer_{lid}_scale_q8[{layer.output_shape}];
static float layer_{lid}_scale_scale;
static float layer_{lid}_scale_zero_point;
static {int_type} layer_{lid}_bias_q8[{layer.output_shape}];
static float layer_{lid}_bias_scale;
static float layer_{lid}_bias_zero_point;
static float layer_{lid}_output[{layer.output_shape}];
"""
            elif isinstance(layer, Relu):
                code += f"""static float layer_{lid}_output[{layer.output_shape}];
"""
        
        # Add load_model_parameters function
        code += "\n" + self._generate_load_function(int_type)
        
        # Add predict function
        code += "\n" + self._generate_predict_function(int_type)
        
        return code
    
    def _generate_load_function(self, int_type):
        """Generate function to load model parameters from binary files."""
        code = """
/* Load binary file helper */
static int load_binary_file(const char* filename, void* data, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\\n", filename);
        return 0;
    }
    
    size_t read_size = fread(data, 1, size, file);
    fclose(file);
    
    if (read_size != size) {
        fprintf(stderr, "Error: Expected %zu bytes, read %zu bytes\\n", size, read_size);
        return 0;
    }
    
    return 1;
}

/* Load quantization parameters */
static int load_qparams(const char* filename, float* scale, float* zero_point) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\\n", filename);
        return 0;
    }
    
    size_t read_count = 0;
    read_count += fread(scale, sizeof(float), 1, file);
    read_count += fread(zero_point, sizeof(float), 1, file);
    fclose(file);
    
    if (read_count != 2) {
        fprintf(stderr, "Error: Failed to read quantization parameters from %s\\n", filename);
        return 0;
    }
    
    return 1;
}

/* Dequantize function */
static inline float dequantize(""" + int_type + """ value, float scale, float zero_point) {
    return ((float)value - zero_point) * scale;
}

/* Load all model parameters */
int load_model_parameters(void) {
"""
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                code += f"""
    /* Load Linear layer {lid} */
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_weight.bin", 
                         layer_{lid}_weight_q8, 
                         sizeof(layer_{lid}_weight_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_weight_qparams.bin",
                     &layer_{lid}_weight_scale, 
                     &layer_{lid}_weight_zero_point)) return 0;
    
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_bias.bin",
                         layer_{lid}_bias_q8,
                         sizeof(layer_{lid}_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_bias_qparams.bin",
                     &layer_{lid}_bias_scale,
                     &layer_{lid}_bias_zero_point)) return 0;
"""
            elif isinstance(layer, BatchNorm):
                code += f"""
    /* Load BatchNorm layer {lid} */
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_scale.bin",
                         layer_{lid}_scale_q8,
                         sizeof(layer_{lid}_scale_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_scale_qparams.bin",
                     &layer_{lid}_scale_scale,
                     &layer_{lid}_scale_zero_point)) return 0;
    
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_bn_bias.bin",
                         layer_{lid}_bias_q8,
                         sizeof(layer_{lid}_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_bn_bias_qparams.bin",
                     &layer_{lid}_bias_scale,
                     &layer_{lid}_bias_zero_point)) return 0;
"""
        
        code += """
    return 1;
}
"""
        return code
    
    def _generate_predict_function(self, int_type):
        """Generate prediction function with integer arithmetic."""
        code = """
/* Predict function */
void predict(const float* input, float* output, int input_size, int output_size) {
    static int model_loaded = 0;
    if (!model_loaded) {
        if (!load_model_parameters()) {
            fprintf(stderr, "Error: Failed to load model parameters\\n");
            return;
        }
        model_loaded = 1;
    }
    
"""
        
        for lid, layer in enumerate(self.model.layers):
            input_var = "input" if lid == 0 else f"layer_{lid-1}_output"
            output_var = f"layer_{lid}_output"
            
            if isinstance(layer, Linear):
                code += f"""
    /* Linear layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float acc = dequantize(layer_{lid}_bias_q8[i], 
                             layer_{lid}_bias_scale, 
                             layer_{lid}_bias_zero_point);
        for (int j = 0; j < {layer.input_shape}; j++) {{
            float weight_dequant = dequantize(layer_{lid}_weight_q8[i][j],
                                            layer_{lid}_weight_scale,
                                            layer_{lid}_weight_zero_point);
            acc += weight_dequant * {input_var}[j];
        }}
        {output_var}[i] = acc;
    }}
"""
            elif isinstance(layer, BatchNorm):
                code += f"""
    /* BatchNorm layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float scale_dequant = dequantize(layer_{lid}_scale_q8[i],
                                        layer_{lid}_scale_scale,
                                        layer_{lid}_scale_zero_point);
        float bias_dequant = dequantize(layer_{lid}_bias_q8[i],
                                       layer_{lid}_bias_scale,
                                       layer_{lid}_bias_zero_point);
        {output_var}[i] = {input_var}[i] * scale_dequant + bias_dequant;
    }}
"""
            elif isinstance(layer, Relu):
                code += f"""
    /* ReLU layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        {output_var}[i] = {input_var}[i] > 0.0f ? {input_var}[i] : 0.0f;
    }}
"""
        
        # Copy final output
        final_layer = len(self.model.layers) - 1
        code += f"""
    /* Copy output */
    memcpy(output, layer_{final_layer}_output, output_size * sizeof(float));
}}
"""
        
        return code
