import numpy as np
import os
import struct
from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class MatQuantPT(Implementation):
    """
    Implementation of Matryoshka Quantization for neural networks using PyTorch model files.
    This class generates C++ code for inference with MatQuant models loaded from binary files.
    All models are stored as 8-bit quantized values, with bit-slicing performed at runtime.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float", 
                 target_bits=8, mix_and_match_config=None, align=None, quantize_signed=False):
        """
        Initialize MatQuant implementation.
        
        Args:
            model: The MLGen3 neural network model
            feature_type: C++ type for input features
            label_type: C++ type for output labels
            internal_type: C++ type for internal computations
            target_bits: Target bit-width for quantization (default: 8)
            mix_and_match_config: Configuration for mix-and-match model (dict mapping layer names to bit-widths)
            align: Memory alignment for C++ arrays
            quantize_signed: Whether to use signed quantization (default: False)
        """
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.align = align
        self.target_bits = target_bits
        self.mix_and_match_config = mix_and_match_config
        self.max_bits = 8  # Max bits for MatQuant (fixed at 8)
        self.filename = None  # Will be set by the materializer
        self.model_binary_dir = None  # Will store the path to binary files
        self.quantize_signed = quantize_signed  # Explicitly passed parameter
    
    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
        
    def set_model_binary_dir(self, model_dir):
        """Set the directory where model binary files are stored."""
        self.model_binary_dir = model_dir
        
    def extract_model_parameters(self):
        """
        Extract model parameters and save them as binary files.
        All parameters are stored as 8-bit quantized values along with quantization parameters.
        Each layer's weights and biases will be saved separately, along with quantization information.
        """
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        # Extract parameters from each layer and save to binary files
        for lid, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weight'):
                # Get original weight values
                weight = layer.weight.astype(np.float32)
                
                # Quantize to 8-bit
                qparams = self._quantize_to_8bit(weight)
                weight_8bit = qparams["quantized_values"]
                
                # Save quantized weights
                weight_8bit.astype(np.uint8).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_weight.bin"))
                
                # Save quantization parameters (scale, zero_point)
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_weight_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
            if hasattr(layer, 'bias'):
                # Get original bias values
                bias = layer.bias.astype(np.float32)
                
                # Quantize to 8-bit
                qparams = self._quantize_to_8bit(bias)
                bias_8bit = qparams["quantized_values"]
                
                # Save quantized bias
                bias_8bit.astype(np.uint8).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                
                # Save quantization parameters (scale, zero_point)
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
            # Save BatchNorm parameters if applicable
            if isinstance(layer, BatchNorm):
                # Scale
                scale = layer.scale.astype(np.float32)
                qparams = self._quantize_to_8bit(scale)
                scale_8bit = qparams["quantized_values"]
                scale_8bit.astype(np.uint8).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_scale.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_scale_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Bias
                bias = layer.bias.astype(np.float32)
                qparams = self._quantize_to_8bit(bias)
                bias_8bit = qparams["quantized_values"]
                bias_8bit.astype(np.uint8).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
    def _quantize_to_8bit(self, values):
        """
        Quantize values to 8-bit representation.
        
        Args:
            values: NumPy array of values to quantize
            
        Returns:
            dict with quantized_values, scale, and zero_point
        """
        # Find min and max values
        val_min = values.min()
        val_max = values.max()
        
        # Calculate scaling factor and zero point
        if self.quantize_signed:
            # Signed quantization: range is [-128, 127]
            q_min = -128
            q_max = 127
            scale = (val_max - val_min) / (q_max - q_min) if val_min != val_max else 1.0
            zero_point = -val_min / scale + q_min if scale != 0 else q_min
        else:
            # Unsigned quantization: range is [0, 255]
            scale = (val_max - val_min) / 255.0 if val_min != val_max else 1.0
            zero_point = -val_min / scale if scale != 0 else 0.0
        
        # Quantize the values
        quantized = np.round(values / scale + zero_point)
        
        if self.quantize_signed:
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return {
            "quantized_values": quantized,
            "scale": scale,
            "zero_point": zero_point
        }
                
    def implement(self):
        """Implement the MatQuant model in C++ with binary file loading and runtime bit-slicing."""
        # Extract model parameters to binary files if model_binary_dir is set
        if self.model_binary_dir:
            self.extract_model_parameters()
        
        # Generate the layer declarations and allocations
        alloc = self._generate_layer_allocations()
        
        # Generate model loading function
        load_model_func = self._generate_load_model_function()
        
        # Generate prediction function
        predict_func = self._generate_predict_function()
        
        # Generate define statements for bit configurations
        define_statements = self._generate_define_statements()
        
        # Generate binary loading utilities
        binary_utils = self._generate_binary_utils()
        
        # Generate the header file
        self.header = f"""
            #pragma once
            #include <vector>
            #include <algorithm>
{binary_utils}
            
            {define_statements}
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
        """.strip()

        # Generate the implementation file
        self.code = f"""
            #include "{self.filename}.h"
            
            {alloc}
            
            // Function to load model parameters from binary files
            bool load_model_parameters() {{
                bool success = true;
                
                {load_model_func}
                
                return success;
            }}
            
            {predict_func}
        """

    def _generate_layer_allocations(self):
        """Generate C++ code for layer allocations."""
        alloc = ""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                # Allocate arrays for original values
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                
                # Allocate arrays for 8-bit quantized values and quantization parameters
                alloc += f"static {int_type} layer_{lid}_weight_q8[{layer.output_shape}][{layer.input_shape}];\n"
                alloc += f"static float layer_{lid}_weight_scale;\n"
                alloc += f"static float layer_{lid}_weight_zero_point;\n"
                # Add precomputed dequantized weights
                alloc += f"static float layer_{lid}_weights_dequant[{layer.output_shape}][{layer.input_shape}];\n"
                
                alloc += f"static {int_type} layer_{lid}_bias_q8[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_bias_scale;\n"
                alloc += f"static float layer_{lid}_bias_zero_point;\n"
                # Add precomputed dequantized biases
                alloc += f"static float layer_{lid}_bias_dequant[{layer.output_shape}];\n"
                
            elif isinstance(layer, BatchNorm):
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                
                # Allocate arrays for 8-bit quantized values and quantization parameters
                alloc += f"static {int_type} layer_{lid}_scale_q8[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_scale_scale;\n"
                alloc += f"static float layer_{lid}_scale_zero_point;\n"
                # Add precomputed dequantized scale
                alloc += f"static float layer_{lid}_scale_dequant[{layer.output_shape}];\n"
                
                alloc += f"static {int_type} layer_{lid}_bias_q8[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_bias_scale;\n"
                alloc += f"static float layer_{lid}_bias_zero_point;\n"
                # Add precomputed dequantized bias
                alloc += f"static float layer_{lid}_bias_dequant[{layer.output_shape}];\n"
                
            elif isinstance(layer, (Relu, Sigmoid, Sign, Step)):
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                
        return alloc

    def _generate_binary_utils(self):
        """Generate C++ code for binary file loading utilities and bit slicing."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        return f"""#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>

// Template function to load binary data
template <typename T>
bool load_binary_data(const std::string& filename, std::vector<T>& data, size_t expected_size) {{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {{
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }}
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Check if file size matches expected size
    if (file_size != expected_size * sizeof(T)) {{
        std::cerr << "Error: File size mismatch. Expected " << expected_size * sizeof(T) 
                  << " bytes, got " << file_size << " bytes." << std::endl;
        return false;
    }}
    
    // Resize vector and read data
    data.resize(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    if (!file) {{
        std::cerr << "Error: Only " << file.gcount() << " bytes could be read" << std::endl;
        return false;
    }}
    
    return true;
}}

// Function to load quantization parameters (scale, zero_point)
inline bool load_quantization_params(const std::string& filename, float& scale, float& zero_point) {{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {{
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }}
    
    // Read scale and zero_point (2 floats = 8 bytes)
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&zero_point), sizeof(float));
    
    if (!file) {{
        std::cerr << "Error: Failed to read quantization parameters" << std::endl;
        return false;
    }}
    
    return true;
}}

// Function to perform bit slicing at runtime
inline std::vector<{int_type}> slice_bits(const std::vector<{int_type}>& quantized, int original_bits, int target_bits) {{
    std::vector<{int_type}> sliced(quantized.size());
    int shift_bits = original_bits - target_bits;
    
    {"// Signed quantization" if self.quantize_signed else "// Unsigned quantization"}
    {"int q_min = -(1 << (target_bits - 1));" if self.quantize_signed else "int q_min = 0;"}
    {"int q_max = (1 << (target_bits - 1)) - 1;" if self.quantize_signed else "int q_max = (1 << target_bits) - 1;"}
    
    for (size_t i = 0; i < quantized.size(); ++i) {{
        // Perform bit slicing with rounding
        if (shift_bits > 0) {{
            // Get the bit at position target_bits+1 for rounding
            int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
            int floor_val = quantized[i] >> shift_bits;
            sliced[i] = round_bit ? (floor_val + 1) : floor_val;
            
            // Clamp to ensure values are within the target bit-width range
            sliced[i] = std::max(q_min, std::min(static_cast<int>(sliced[i]), q_max));
            
            // Scale back to original range
            sliced[i] = sliced[i] << shift_bits;
        }} else {{
            sliced[i] = quantized[i];
        }}
    }}
    
    return sliced;
}}

// Function to dequantize values
template <typename T>
std::vector<float> dequantize(const std::vector<T>& quantized, float scale, float zero_point) {{
    std::vector<float> dequantized(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i) {{
        dequantized[i] = (static_cast<float>(quantized[i]) - zero_point) * scale;
    }}
    return dequantized;
}}"""
    def _generate_define_statements(self):
        """Generate C++ define statements for model configuration."""
        define_statements = ""
        if self.mix_and_match_config:
            define_statements += "#define IS_MIX_AND_MATCH 1\n"
            define_statements += f"#define NUM_LAYERS {len(self.model.layers)}\n"
            
            # Create an array of bit-widths for each layer
            layer_bits = []
            for lid in range(len(self.model.layers)):
                # Try different possible layer name formats
                possible_layer_names = [
                    f"model.{lid}.weight",
                    f"layers.{lid}.weight", 
                    f"layer.{lid}.weight",
                    f"model.layers.{lid}.weight",
                    f"model.layer.{lid}.weight"
                ]
                
                # Use default bit width, then check if any of the possible names are in the config
                bits = self.target_bits
                for name in possible_layer_names:
                    if name in self.mix_and_match_config:
                        bits = self.mix_and_match_config[name]
                        break
                        
                layer_bits.append(bits)
            
            # Define LAYER_BITS as a proper C++ array
            define_statements += f"constexpr int LAYER_BITS[NUM_LAYERS] = {{{', '.join(str(b) for b in layer_bits)}}};\n"
        else:
            define_statements += "#define IS_MIX_AND_MATCH 0\n"
            define_statements += f"#define NUM_LAYERS {len(self.model.layers)}\n"
            # Define LAYER_BITS as a proper C++ array
            define_statements += f"constexpr int LAYER_BITS[NUM_LAYERS] = {{{', '.join([str(self.target_bits)] * len(self.model.layers))}}};\n"
            
        define_statements += f"#define TARGET_BITS {self.target_bits}\n"
        define_statements += "#define STORAGE_BITS 8\n"  # Always store as 8-bit
        define_statements += f"#define QUANTIZE_SIGNED {1 if self.quantize_signed else 0}\n"
        return define_statements

    def _generate_load_model_function(self):
        """Generate C++ code for loading model parameters from binary files with runtime bit slicing."""
        load_func = ""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                load_func += f"""
                    // Load and precompute weights for Linear layer {lid}
                    {{
                        // Load quantized weights
                        std::vector<{int_type}> layer_{lid}_weight_data;
                        bool weight_success = load_binary_data<{int_type}>("mq_pt_model_binary/layer_{lid}_weight.bin", 
                                                                      layer_{lid}_weight_data, 
                                                                      {layer.output_shape} * {layer.input_shape});
                        
                        // Load quantization parameters
                        bool qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_weight_qparams.bin",
                                                                     layer_{lid}_weight_scale, 
                                                                     layer_{lid}_weight_zero_point);
                        
                        if (weight_success && qparam_success) {{
                            // Copy to 2D array and precompute dequantized values
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                for (int j = 0; j < {layer.input_shape}; ++j) {{
                                    int idx = i * {layer.input_shape} + j;
                                    layer_{lid}_weight_q8[i][j] = layer_{lid}_weight_data[idx];
                                    
                                    // Slice and dequantize immediately
                                    std::vector<{int_type}> weight_q8(1, layer_{lid}_weight_q8[i][j]);
                                    std::vector<{int_type}> sliced_weight = slice_bits(weight_q8, STORAGE_BITS, LAYER_BITS[{lid}]);
                                    std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{lid}_weight_scale, layer_{lid}_weight_zero_point);
                                    layer_{lid}_weights_dequant[i][j] = dequant_weight[0];
                                }}
                            }}
                        }} else {{
                            success = false;
                            std::cerr << "Failed to load weights for layer {lid}" << std::endl;
                        }}
                        
                        // Load and precompute biases
                        std::vector<{int_type}> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<{int_type}>("mq_pt_model_binary/layer_{lid}_bias.bin", 
                                                                   layer_{lid}_bias_data, 
                                                                   {layer.output_shape});
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_bias_qparams.bin",
                                                                         layer_{lid}_bias_scale, 
                                                                         layer_{lid}_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias_q8[i] = layer_{lid}_bias_data[i];
                                
                                // Slice and dequantize immediately
                                std::vector<{int_type}> bias_q8(1, layer_{lid}_bias_q8[i]);
                                std::vector<{int_type}> sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[{lid}]);
                                std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{lid}_bias_scale, layer_{lid}_bias_zero_point);
                                layer_{lid}_bias_dequant[i] = dequant_bias[0];
                            }}
                        }} else {{
                            success = false;
                            std::cerr << "Failed to load bias for layer {lid}" << std::endl;
                        }}
                    }}"""
            elif isinstance(layer, BatchNorm):
                load_func += f"""
                    // Load and precompute BatchNorm parameters for layer {lid}
                    {{
                        // Load and precompute scale
                        std::vector<{int_type}> layer_{lid}_scale_data;
                        bool scale_success = load_binary_data<{int_type}>("mq_pt_model_binary/layer_{lid}_scale.bin", 
                                                                    layer_{lid}_scale_data, 
                                                                    {layer.output_shape});
                        
                        bool scale_qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_scale_qparams.bin",
                                                                         layer_{lid}_scale_scale, 
                                                                         layer_{lid}_scale_zero_point);
                        
                        if (scale_success && scale_qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_scale_q8[i] = layer_{lid}_scale_data[i];
                                
                                // Slice and dequantize immediately
                                std::vector<{int_type}> scale_q8(1, layer_{lid}_scale_q8[i]);
                                std::vector<{int_type}> sliced_scale = slice_bits(scale_q8, STORAGE_BITS, LAYER_BITS[{lid}]);
                                std::vector<float> dequant_scale = dequantize(sliced_scale, layer_{lid}_scale_scale, layer_{lid}_scale_zero_point);
                                layer_{lid}_scale_dequant[i] = dequant_scale[0];
                            }}
                        }} else {{
                            success = false;
                        }}
                        
                        // Load and precompute bias
                        std::vector<{int_type}> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<{int_type}>("mq_pt_model_binary/layer_{lid}_bn_bias.bin", 
                                                                   layer_{lid}_bias_data, 
                                                                   {layer.output_shape});
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_bn_bias_qparams.bin",
                                                                         layer_{lid}_bias_scale, 
                                                                         layer_{lid}_bias_zero_point);
                        
                        if (bias_success && bias_qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias_q8[i] = layer_{lid}_bias_data[i];
                                
                                // Slice and dequantize immediately
                                std::vector<{int_type}> bias_q8(1, layer_{lid}_bias_q8[i]);
                                std::vector<{int_type}> sliced_bias = slice_bits(bias_q8, STORAGE_BITS, LAYER_BITS[{lid}]);
                                std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{lid}_bias_scale, layer_{lid}_bias_zero_point);
                                layer_{lid}_bias_dequant[i] = dequant_bias[0];
                            }}
                        }} else {{
                            success = false;
                        }}
                    }}"""
        
        return load_func
    
    def _generate_predict_function(self):
        """Generate the C++ prediction function with precomputed dequantized weights."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        code = """
        std::vector<float> predict(std::vector<float> &x) {
            // Load model parameters if not loaded already
            static bool model_loaded = false;
            if (!model_loaded) {
                model_loaded = load_model_parameters();
                if (!model_loaded) {
                    std::cerr << "Warning: Failed to load some model parameters." << std::endl;
                }
            }
        """
        
        # Process each layer
        for lid, layer in enumerate(self.model.layers):
            if lid == 0:
                input_var = "x"
            else:
                input_var = f"layer_{lid-1}"
                
            if isinstance(layer, Linear):
                code += f"""
                // Linear layer {lid} using precomputed dequantized weights
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = layer_{lid}_bias_dequant[i];
                    for (int j = 0; j < {layer.input_shape}; ++j) {{
                        layer_{lid}[i] += layer_{lid}_weights_dequant[i][j] * {input_var}[j];
                    }}
                }}
                """
            elif isinstance(layer, BatchNorm):
                code += f"""
                // BatchNorm layer {lid} using precomputed dequantized parameters
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = {input_var}[i] * layer_{lid}_scale_dequant[i] + layer_{lid}_bias_dequant[i];
                }}
                """
            elif isinstance(layer, Relu):
                code += f"""
                // ReLU activation layer {lid}
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = std::max(0.0f, {input_var}[i]);
                }}
                """
            elif isinstance(layer, Sigmoid):
                code += f"""
                // Sigmoid activation layer {lid}
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = 1.0f / (1.0f + std::exp(-{input_var}[i]));
                }}
                """
            elif isinstance(layer, Sign):
                code += f"""
                // Sign activation layer {lid}
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    if ({input_var}[i] > 0) {{
                        layer_{lid}[i] = 1.0f;
                    }} else if ({input_var}[i] < 0) {{
                        layer_{lid}[i] = -1.0f;
                    }} else {{
                        layer_{lid}[i] = 0.0f;
                    }}
                }}
                """
            elif isinstance(layer, Step):
                if isinstance(layer.threshold, (list, np.ndarray)):
                    comp = ">=" if layer.threshold_is_high else ">"
                    code += f"""
                    // Step activation layer {lid} with array threshold
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        layer_{lid}[i] = {input_var}[i] {comp} 0.0f ? {layer.high} : {layer.low};
                    }}
                    """
                else:
                    comp = ">=" if layer.threshold_is_high else ">"
                    code += f"""
                    // Step activation layer {lid} with scalar threshold
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        layer_{lid}[i] = {input_var}[i] {comp} {layer.threshold} ? {layer.high} : {layer.low};
                    }}
                    """
        
        # Return the output of the last layer
        code += f"""
            return std::vector<float>(layer_{len(self.model.layers)-1}, 
                                   layer_{len(self.model.layers)-1} + {self.model.layers[-1].output_shape});
        }}
        """
        
        return code
