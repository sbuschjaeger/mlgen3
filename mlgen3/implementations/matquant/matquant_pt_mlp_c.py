import numpy as np
import os
import struct
from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu
from mlgen3.utils.bitplane import pack_weights_2d_bitplane, pack_weights_bitplane

class MatQuantPT_C(Implementation):
    """
    Pure C implementation of MatQuant for MLP models.
    Generates integer-only arithmetic C code with 8-bit quantized weights.
    Supports both binary file loading and embedded header file weights.
    Weights are packed in bitplane-interleaved format for efficient bit-slicing.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float",
                 quantize_signed=True, use_bias=True, use_header_weights=True):
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.quantize_signed = quantize_signed
        self.use_bias = use_bias
        self.use_header_weights = use_header_weights
        self.filename = None
        self.model_binary_dir = None
        self.weights_header = None
        self.weights_debug_header = None
        
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
    
    def _format_array_1d(self, values, name, dtype_str="int8_t", items_per_line=16):
        """Format a 1D array as C code."""
        values = np.array(values).flatten()
        lines = []
        for i in range(0, len(values), items_per_line):
            chunk = values[i:min(i + items_per_line, len(values))]
            chunk_str = ", ".join(str(int(v)) for v in chunk)
            lines.append(f"    {chunk_str}")
        values_str = ",\n".join(lines)
        return f"static const {dtype_str} {name}[{len(values)}] = {{\n{values_str}\n}};\n"
    
    def _format_array_2d(self, values, name, rows, cols, dtype_str="int8_t", items_per_line=16):
        """Format a 2D array as C code."""
        values = np.array(values).reshape(rows, cols)
        lines = []
        for i in range(rows):
            row = values[i]
            row_parts = []
            for j in range(0, cols, items_per_line):
                chunk = row[j:min(j + items_per_line, cols)]
                chunk_str = ", ".join(str(int(v)) for v in chunk)
                row_parts.append(chunk_str)
            row_str = ", ".join(row_parts)
            lines.append(f"    {{{row_str}}}")
        values_str = ",\n".join(lines)
        return f"static const {dtype_str} {name}[{rows}][{cols}] = {{\n{values_str}\n}};\n"
    
    def extract_model_parameters_to_header(self):
        """Extract model parameters and generate C headers with packed and unpacked weights."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        # Header for PACKED weights (bitplane-interleaved)
        header_code = f"""/* Auto-generated model weights header - BITPLANE PACKED */
#ifndef {self.filename.upper()}_WEIGHTS_H
#define {self.filename.upper()}_WEIGHTS_H

#include <stdint.h>

/* 
 * Weights are packed in bitplane-interleaved format.
 * This allows extracting lower bit-width versions by reading fewer bytes:
 * - 2-bit: read bytes 0-3 per 16-value block
 * - 4-bit: read bytes 0-7 per 16-value block
 * - 8-bit: read all 16 bytes per block
 */

"""
        
        # Header for UNPACKED weights (original quantized values for debugging)
        debug_header_code = f"""/* Auto-generated DEBUG header with UNPACKED weights (original quantized values) */
#ifndef {self.filename.upper()}_WEIGHTS_DEBUG_H
#define {self.filename.upper()}_WEIGHTS_DEBUG_H

#include <stdint.h>

/* This file contains the original quantized int8 weights BEFORE bitplane packing.
 * Use this for debugging and verification purposes.
 */

"""
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                # Quantize weights
                weight = layer.weight.astype(np.float32)
                qparams = self._quantize_to_8bit(weight)
                quantized_weights = qparams["quantized_values"]
                
                # Pack weights using bitplane interleaving
                packed_weights = pack_weights_2d_bitplane(quantized_weights, signed=self.quantize_signed)
                
                # Add PACKED weights to main header
                header_code += f"/* Linear layer {lid} weights - BITPLANE PACKED */\n"
                header_code += self._format_array_2d(
                    packed_weights, 
                    f"layer_{lid}_lin_weight_q8",
                    layer.output_shape, layer.input_shape, int_type
                )
                header_code += f"static const float layer_{lid}_lin_weight_scale = {qparams['scale']:.10f}f;\n"
                header_code += f"static const float layer_{lid}_lin_weight_zero_point = {qparams['zero_point']:.10f}f;\n\n"
                
                # Add UNPACKED weights to debug header
                debug_header_code += f"/* Linear layer {lid} weights - UNPACKED (original quantized) */\n"
                debug_header_code += f"/* Shape: [{layer.output_shape}][{layer.input_shape}] */\n"
                debug_header_code += f"/* Scale: {qparams['scale']:.10f}, Zero Point: {qparams['zero_point']:.10f} */\n"
                debug_header_code += self._format_array_2d(
                    quantized_weights,
                    f"layer_{lid}_lin_weight_q8_unpacked",
                    layer.output_shape, layer.input_shape, int_type
                )
                debug_header_code += "\n"
                
                # Quantize bias if enabled AND bias exists
                if self.use_bias and layer.bias is not None:
                    bias = layer.bias.astype(np.float32)
                    bias_qparams = self._quantize_to_8bit(bias)
                    quantized_bias = bias_qparams["quantized_values"]
                    
                    # Pack bias using bitplane interleaving
                    packed_bias = pack_weights_bitplane(quantized_bias, signed=self.quantize_signed)
                    
                    # Add PACKED bias to main header
                    header_code += f"/* Linear layer {lid} bias - BITPLANE PACKED */\n"
                    header_code += self._format_array_1d(
                        packed_bias,
                        f"layer_{lid}_lin_bias_q8", int_type
                    )
                    header_code += f"static const float layer_{lid}_lin_bias_scale = {bias_qparams['scale']:.10f}f;\n"
                    header_code += f"static const float layer_{lid}_lin_bias_zero_point = {bias_qparams['zero_point']:.10f}f;\n\n"
                    
                    # Add UNPACKED bias to debug header
                    debug_header_code += f"/* Linear layer {lid} bias - UNPACKED (original quantized) */\n"
                    debug_header_code += f"/* Shape: [{layer.output_shape}] */\n"
                    debug_header_code += f"/* Scale: {bias_qparams['scale']:.10f}, Zero Point: {bias_qparams['zero_point']:.10f} */\n"
                    debug_header_code += self._format_array_1d(
                        quantized_bias,
                        f"layer_{lid}_lin_bias_q8_unpacked", int_type
                    )
                    debug_header_code += "\n"
                    
            elif isinstance(layer, BatchNorm):
                # Quantize scale
                scale = layer.scale.astype(np.float32)
                scale_qparams = self._quantize_to_8bit(scale)
                quantized_scale = scale_qparams["quantized_values"]
                packed_scale = pack_weights_bitplane(quantized_scale, signed=self.quantize_signed)
                
                header_code += f"/* BatchNorm layer {lid} scale - BITPLANE PACKED */\n"
                header_code += self._format_array_1d(
                    packed_scale,
                    f"layer_{lid}_bn_scale_q8", int_type
                )
                header_code += f"static const float layer_{lid}_bn_scale_scale = {scale_qparams['scale']:.10f}f;\n"
                header_code += f"static const float layer_{lid}_bn_scale_zero_point = {scale_qparams['zero_point']:.10f}f;\n\n"
                
                debug_header_code += f"/* BatchNorm layer {lid} scale - UNPACKED */\n"
                debug_header_code += f"/* Shape: [{layer.output_shape}] */\n"
                debug_header_code += self._format_array_1d(
                    quantized_scale,
                    f"layer_{lid}_bn_scale_q8_unpacked", int_type
                )
                debug_header_code += "\n"
                
                # Quantize bias
                bias = layer.bias.astype(np.float32)
                bias_qparams = self._quantize_to_8bit(bias)
                quantized_bias = bias_qparams["quantized_values"]
                packed_bias = pack_weights_bitplane(quantized_bias, signed=self.quantize_signed)
                
                header_code += f"/* BatchNorm layer {lid} bias - BITPLANE PACKED */\n"
                header_code += self._format_array_1d(
                    packed_bias,
                    f"layer_{lid}_bn_bias_q8", int_type
                )
                header_code += f"static const float layer_{lid}_bn_bias_scale = {bias_qparams['scale']:.10f}f;\n"
                header_code += f"static const float layer_{lid}_bn_bias_zero_point = {bias_qparams['zero_point']:.10f}f;\n\n"
                
                debug_header_code += f"/* BatchNorm layer {lid} bias - UNPACKED */\n"
                debug_header_code += f"/* Shape: [{layer.output_shape}] */\n"
                debug_header_code += self._format_array_1d(
                    quantized_bias,
                    f"layer_{lid}_bn_bias_q8_unpacked", int_type
                )
                debug_header_code += "\n"
        
        header_code += f"#endif /* {self.filename.upper()}_WEIGHTS_H */\n"
        debug_header_code += f"#endif /* {self.filename.upper()}_WEIGHTS_DEBUG_H */\n"
        
        self.weights_header = header_code
        self.weights_debug_header = debug_header_code
        return header_code
    
    def extract_model_parameters(self):
        """Extract and save model parameters as 8-bit quantized binary files."""
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                weight = layer.weight.astype(np.float32)
                qparams = self._quantize_to_8bit(weight)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_lin_weight.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_lin_weight_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Only save bias if it exists
                if self.use_bias and layer.bias is not None:
                    bias = layer.bias.astype(np.float32)
                    qparams = self._quantize_to_8bit(bias)
                    qparams["quantized_values"].tofile(
                        os.path.join(self.model_binary_dir, f"layer_{lid}_lin_bias.bin"))
                    with open(os.path.join(self.model_binary_dir, f"layer_{lid}_lin_bias_qparams.bin"), 'wb') as f:
                        f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                    
            elif isinstance(layer, BatchNorm):
                # Quantize scale
                scale = layer.scale.astype(np.float32)
                qparams = self._quantize_to_8bit(scale)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_bn_scale.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_scale_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                bias = layer.bias.astype(np.float32)
                qparams = self._quantize_to_8bit(bias)
                qparams["quantized_values"].tofile(
                    os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
    
    def implement(self):
        """Generate pure C code for the model."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        if self.use_header_weights:
            self.extract_model_parameters_to_header()
            self.header = self._generate_header_with_embedded_weights(int_type)
            self.code = self._generate_code_with_embedded_weights(int_type)
        else:
            if self.model_binary_dir:
                self.extract_model_parameters()
            self.header = self._generate_header(int_type)
            self.code = self._generate_code(int_type)
    
    def _generate_header_with_embedded_weights(self, int_type):
        """Generate C header file with embedded weights."""
        return f"""#ifndef {self.filename.upper()}_H
#define {self.filename.upper()}_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Model configuration */
#define NUM_LAYERS {len(self.model.layers)}
#define QUANTIZE_SIGNED {1 if self.quantize_signed else 0}
#define USE_BIAS {1 if self.use_bias else 0}

/* Function prototypes */
void predict(const float* input, float* output, int input_size, int output_size);

#endif /* {self.filename.upper()}_H */
"""

    def _generate_code_with_embedded_weights(self, int_type):
        """Generate C implementation file with embedded weights (no file loading)."""
        code = f"""#include "{self.filename}.h"
#include "{self.filename}_weights.h"
#include <string.h>

/* Output buffers */
"""
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                code += f"static float layer_{lid}_lin_output[{layer.output_shape}];\n"
            elif isinstance(layer, BatchNorm):
                code += f"static float layer_{lid}_bn_output[{layer.output_shape}];\n"
            elif isinstance(layer, Relu):
                code += f"static float layer_{lid}_relu_output[{layer.output_shape}];\n"
        
        code += f"""
/* Dequantize function */
static inline float dequantize({int_type} value, float scale, float zero_point) {{
    return ((float)value - zero_point) * scale;
}}

/* Predict function */
void predict(const float* input, float* output, int input_size, int output_size) {{
"""
        
        for lid, layer in enumerate(self.model.layers):
            if lid == 0:
                input_var = "input"
            else:
                prev_layer = self.model.layers[lid-1]
                if isinstance(prev_layer, Linear):
                    input_var = f"layer_{lid-1}_lin_output"
                elif isinstance(prev_layer, BatchNorm):
                    input_var = f"layer_{lid-1}_bn_output"
                elif isinstance(prev_layer, Relu):
                    input_var = f"layer_{lid-1}_relu_output"
                else:
                    input_var = f"layer_{lid-1}_output"
            
            if isinstance(layer, Linear):
                output_var = f"layer_{lid}_lin_output"
                
                # Check if this layer has bias (use_bias flag AND layer actually has bias)
                layer_has_bias = self.use_bias and layer.bias is not None
                
                if layer_has_bias:
                    code += f"""
    /* Linear layer {lid} with bias */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float acc = dequantize(layer_{lid}_lin_bias_q8[i], 
                             layer_{lid}_lin_bias_scale, 
                             layer_{lid}_lin_bias_zero_point);
        for (int j = 0; j < {layer.input_shape}; j++) {{
            float weight_dequant = dequantize(layer_{lid}_lin_weight_q8[i][j],
                                            layer_{lid}_lin_weight_scale,
                                            layer_{lid}_lin_weight_zero_point);
            acc += weight_dequant * {input_var}[j];
        }}
        {output_var}[i] = acc;
    }}
"""
                else:
                    code += f"""
    /* Linear layer {lid} without bias */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float acc = 0.0f;
        for (int j = 0; j < {layer.input_shape}; j++) {{
            float weight_dequant = dequantize(layer_{lid}_lin_weight_q8[i][j],
                                            layer_{lid}_lin_weight_scale,
                                            layer_{lid}_lin_weight_zero_point);
            acc += weight_dequant * {input_var}[j];
        }}
        {output_var}[i] = acc;
    }}
"""
            elif isinstance(layer, BatchNorm):
                output_var = f"layer_{lid}_bn_output"
                code += f"""
    /* BatchNorm layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float scale_dequant = dequantize(layer_{lid}_bn_scale_q8[i],
                                        layer_{lid}_bn_scale_scale,
                                        layer_{lid}_bn_scale_zero_point);
        float bias_dequant = dequantize(layer_{lid}_bn_bias_q8[i],
                                       layer_{lid}_bn_bias_scale,
                                       layer_{lid}_bn_bias_zero_point);
        {output_var}[i] = {input_var}[i] * scale_dequant + bias_dequant;
    }}
"""
            elif isinstance(layer, Relu):
                output_var = f"layer_{lid}_relu_output"
                code += f"""
    /* ReLU layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        {output_var}[i] = {input_var}[i] > 0.0f ? {input_var}[i] : 0.0f;
    }}
"""
        
        final_layer = self.model.layers[-1]
        if isinstance(final_layer, Linear):
            final_output_var = f"layer_{len(self.model.layers)-1}_lin_output"
        elif isinstance(final_layer, BatchNorm):
            final_output_var = f"layer_{len(self.model.layers)-1}_bn_output"
        elif isinstance(final_layer, Relu):
            final_output_var = f"layer_{len(self.model.layers)-1}_relu_output"
        else:
            final_output_var = f"layer_{len(self.model.layers)-1}_output"
            
        code += f"""
    /* Copy output */
    memcpy(output, {final_output_var}, output_size * sizeof(float));
}}
"""
        
        return code

    def _generate_header(self, int_type):
        """Generate C header file (for binary file mode)."""
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
        """Generate C implementation file (for binary file mode)."""
        code = f"""#include "{self.filename}.h"
#include <string.h>

/* Global arrays for model parameters */
"""
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                code += f"""static {int_type} layer_{lid}_lin_weight_q8[{layer.output_shape}][{layer.input_shape}];
static float layer_{lid}_lin_weight_scale;
static float layer_{lid}_lin_weight_zero_point;
"""
                # Only declare bias arrays if bias exists
                layer_has_bias = self.use_bias and layer.bias is not None
                if layer_has_bias:
                    code += f"""static {int_type} layer_{lid}_lin_bias_q8[{layer.output_shape}];
static float layer_{lid}_lin_bias_scale;
static float layer_{lid}_lin_bias_zero_point;
"""
                code += f"""static float layer_{lid}_lin_output[{layer.output_shape}];
"""
            elif isinstance(layer, BatchNorm):
                code += f"""static {int_type} layer_{lid}_bn_scale_q8[{layer.output_shape}];
static float layer_{lid}_bn_scale_scale;
static float layer_{lid}_bn_scale_zero_point;
static {int_type} layer_{lid}_bn_bias_q8[{layer.output_shape}];
static float layer_{lid}_bn_bias_scale;
static float layer_{lid}_bn_bias_zero_point;
static float layer_{lid}_bn_output[{layer.output_shape}];
"""
            elif isinstance(layer, Relu):
                code += f"""static float layer_{lid}_relu_output[{layer.output_shape}];
"""
        
        code += "\n" + self._generate_load_function(int_type)
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
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_lin_weight.bin", 
                         layer_{lid}_lin_weight_q8, 
                         sizeof(layer_{lid}_lin_weight_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_lin_weight_qparams.bin",
                     &layer_{lid}_lin_weight_scale, 
                     &layer_{lid}_lin_weight_zero_point)) return 0;
    
"""
                # Only load bias if it exists
                layer_has_bias = self.use_bias and layer.bias is not None
                if layer_has_bias:
                    code += f"""    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_lin_bias.bin",
                         layer_{lid}_lin_bias_q8,
                         sizeof(layer_{lid}_lin_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_lin_bias_qparams.bin",
                     &layer_{lid}_lin_bias_scale,
                     &layer_{lid}_lin_bias_zero_point)) return 0;
"""
            elif isinstance(layer, BatchNorm):
                code += f"""
    /* Load BatchNorm layer {lid} */
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_bn_scale.bin",
                         layer_{lid}_bn_scale_q8,
                         sizeof(layer_{lid}_bn_scale_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_bn_scale_qparams.bin",
                     &layer_{lid}_bn_scale_scale,
                     &layer_{lid}_bn_scale_zero_point)) return 0;
    
    if (!load_binary_file("mq_pt_model_binary/layer_{lid}_bn_bias.bin",
                         layer_{lid}_bn_bias_q8,
                         sizeof(layer_{lid}_bn_bias_q8))) return 0;
    if (!load_qparams("mq_pt_model_binary/layer_{lid}_bn_bias_qparams.bin",
                     &layer_{lid}_bn_bias_scale,
                     &layer_{lid}_bn_bias_zero_point)) return 0;
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
            if lid == 0:
                input_var = "input"
            else:
                prev_layer = self.model.layers[lid-1]
                if isinstance(prev_layer, Linear):
                    input_var = f"layer_{lid-1}_lin_output"
                elif isinstance(prev_layer, BatchNorm):
                    input_var = f"layer_{lid-1}_bn_output"
                elif isinstance(prev_layer, Relu):
                    input_var = f"layer_{lid-1}_relu_output"
                else:
                    input_var = f"layer_{lid-1}_output"
            
            if isinstance(layer, Linear):
                output_var = f"layer_{lid}_lin_output"
                
                # Check if this layer has bias
                layer_has_bias = self.use_bias and layer.bias is not None
                
                if layer_has_bias:
                    code += f"""
    /* Linear layer {lid} with bias */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float acc = dequantize(layer_{lid}_lin_bias_q8[i], 
                             layer_{lid}_lin_bias_scale, 
                             layer_{lid}_lin_bias_zero_point);
        for (int j = 0; j < {layer.input_shape}; j++) {{
            float weight_dequant = dequantize(layer_{lid}_lin_weight_q8[i][j],
                                            layer_{lid}_lin_weight_scale,
                                            layer_{lid}_lin_weight_zero_point);
            acc += weight_dequant * {input_var}[j];
        }}
        {output_var}[i] = acc;
    }}
"""
                else:
                    code += f"""
    /* Linear layer {lid} without bias */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float acc = 0.0f;
        for (int j = 0; j < {layer.input_shape}; j++) {{
            float weight_dequant = dequantize(layer_{lid}_lin_weight_q8[i][j],
                                            layer_{lid}_lin_weight_scale,
                                            layer_{lid}_lin_weight_zero_point);
            acc += weight_dequant * {input_var}[j];
        }}
        {output_var}[i] = acc;
    }}
"""
            elif isinstance(layer, BatchNorm):
                output_var = f"layer_{lid}_bn_output"
                code += f"""
    /* BatchNorm layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        float scale_dequant = dequantize(layer_{lid}_bn_scale_q8[i],
                                        layer_{lid}_bn_scale_scale,
                                        layer_{lid}_bn_scale_zero_point);
        float bias_dequant = dequantize(layer_{lid}_bn_bias_q8[i],
                                       layer_{lid}_bn_bias_scale,
                                       layer_{lid}_bn_bias_zero_point);
        {output_var}[i] = {input_var}[i] * scale_dequant + bias_dequant;
    }}
"""
            elif isinstance(layer, Relu):
                output_var = f"layer_{lid}_relu_output"
                code += f"""
    /* ReLU layer {lid} */
    for (int i = 0; i < {layer.output_shape}; i++) {{
        {output_var}[i] = {input_var}[i] > 0.0f ? {input_var}[i] : 0.0f;
    }}
"""
        
        final_layer = self.model.layers[-1]
        if isinstance(final_layer, Linear):
            final_output_var = f"layer_{len(self.model.layers)-1}_lin_output"
        elif isinstance(final_layer, BatchNorm):
            final_output_var = f"layer_{len(self.model.layers)-1}_bn_output"
        elif isinstance(final_layer, Relu):
            final_output_var = f"layer_{len(self.model.layers)-1}_relu_output"
        else:
            final_output_var = f"layer_{len(self.model.layers)-1}_output"
            
        code += f"""
    /* Copy output */
    memcpy(output, {final_output_var}, output_size * sizeof(float));
}}
"""
        
        return code
