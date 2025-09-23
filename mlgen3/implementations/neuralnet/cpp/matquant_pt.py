#

# 1. C++ Independence from PyTorch Runtime
# The C++ implementation of MatQuant needs to run independently of the PyTorch ecosystem. While PyTorch offers a C++ API (libtorch), it:
# Adds significant overhead (100+ MB) to the deployment
# Introduces complex dependencies
# May not support custom operations like bit slicing efficiently

# 2. Direct Memory Access for Bit Slicing
# The MatQuant algorithm requires direct bit manipulation for its core slicing operation:
# Loading raw binary files gives us direct control over the memory representation needed for these operations.

# 3. Parameter-Specific Quantization
# Each parameter (weight matrix, bias vector) can have different quantization characteristics:
# Different scaling factors
# Different zero points
# Different optimal bit-width configurations
# Individual binary files allow the C++ code to load and process each parameter with its specific quantization requirements.

# 4. Simplified Implementation
# Reading binary files in C++ is straightforward and requires no external libraries:
# This approach is significantly simpler than parsing complex PyTorch model structures.

# 5. Cross-Platform Compatibility
# Binary files use standard formats that work across different platforms and architectures, making deployment more reliable and predictable.

#


import numpy as np
import os
from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class MatQuantPT(Implementation):
    """
    Implementation of Matryoshka Quantization for neural networks using PyTorch model files.
    This class generates C++ code for inference with MatQuant models loaded from binary files.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float", 
                 target_bits=8, mix_and_match_config=None, align=None):
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
        """
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.align = align
        self.target_bits = target_bits
        self.mix_and_match_config = mix_and_match_config
        self.max_bits = 8  # Max bits for MatQuant (fixed at 8)
        self.filename = None  # Will be set by the materializer
        self.model_binary_dir = None  # Will store the path to binary files
    
    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
        
    def set_model_binary_dir(self, model_dir):
        """Set the directory where model binary files are stored."""
        self.model_binary_dir = model_dir
        
    def extract_model_parameters(self):
        """
        Extract model parameters and save them as binary files.
        Each layer's weights and biases will be saved separately.
        """
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        # Extract parameters from each layer and save to binary files
        for lid, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weight'):
                # Save weight
                weight = layer.weight.astype(np.float32)
                weight.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_weight.bin"))
                
            if hasattr(layer, 'bias'):
                # Save bias
                bias = layer.bias.astype(np.float32)
                bias.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                
            # Save BatchNorm parameters if applicable
            if isinstance(layer, BatchNorm):
                layer.scale.astype(np.float32).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_scale.bin"))
                layer.bias.astype(np.float32).tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                
    def implement(self):
        """Implement the MatQuant model in C++ with binary file loading."""
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
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_weight[{layer.output_shape}][{layer.input_shape}];\n"
                alloc += f"static float layer_{lid}_bias[{layer.output_shape}];\n"
            elif isinstance(layer, BatchNorm):
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_scale[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_bias[{layer.output_shape}];\n"
            elif isinstance(layer, (Relu, Sigmoid, Sign, Step)):
                alloc += f"static float layer_{lid}[{layer.output_shape}];\n"
                
        return alloc
        
    def _generate_binary_utils(self):
        """Generate C++ code for binary file loading utilities."""
        return """#include <cmath>
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

// Function to quantize data (MinMax quantization)
template <typename T>
std::pair<std::vector<int>, std::pair<T, T>> quantize(const std::vector<T>& data, int bits) {
    // Find min and max values
    T data_min = std::numeric_limits<T>::max();
    T data_max = std::numeric_limits<T>::lowest();
    
    for (const auto& val : data) {
        data_min = std::min(data_min, val);
        data_max = std::max(data_max, val);
    }
    
    // Calculate scaling factor and zero point
    T scale = (data_max - data_min) / ((1 << bits) - 1);
    T zero_point = (scale != 0) ? (-data_min / scale) : 0;
    
    // Quantize the data
    std::vector<int> quantized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        int q = std::round(data[i] / scale + zero_point);
        quantized[i] = std::max(0, std::min(q, (1 << bits) - 1));
    }
    
    return {quantized, {scale, zero_point}};
}

// Function to dequantize data
template <typename T>
std::vector<T> dequantize(const std::vector<int>& quantized, T scale, T zero_point) {
    std::vector<T> dequantized(quantized.size());
    for (size_t i = 0; i < quantized.size(); ++i) {
        dequantized[i] = (quantized[i] - zero_point) * scale;
    }
    return dequantized;
}

// Function to slice bits for MatQuant
inline std::vector<int> slice_bits(const std::vector<int>& quantized, int original_bits, int target_bits, bool rounding = true) {
    std::vector<int> sliced(quantized.size());
    int shift_bits = original_bits - target_bits;
    
    for (size_t i = 0; i < quantized.size(); ++i) {
        if (rounding && shift_bits > 0) {
            // Get the bit at position target_bits+1 for rounding
            int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
            int floor_val = quantized[i] >> shift_bits;
            sliced[i] = round_bit ? (floor_val + 1) : floor_val;
        } else {
            sliced[i] = quantized[i] >> shift_bits;
        }
        
        // Clamp to ensure values are within the target bit-width range
        sliced[i] = std::max(0, std::min(sliced[i], (1 << target_bits) - 1));
        
        // Scale back to original range
        sliced[i] = sliced[i] << shift_bits;
    }
    
    return sliced;
}"""

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
        return define_statements

    def _generate_load_model_function(self):
        """Generate C++ code for loading model parameters from binary files."""
        load_func = ""
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                load_func += f"""
                    // Load weights and bias for Linear layer {lid}
                    {{
                        std::vector<float> layer_{lid}_weight_data;
                        bool weight_success = load_binary_data<float>("mq_pt_model_binary/layer_{lid}_weight.bin", 
                                                                    layer_{lid}_weight_data, 
                                                                    {layer.output_shape} * {layer.input_shape});
                        
                        // If weight loading failed, set default values
                        if (!weight_success) {{
                            success = false;
                            std::cerr << "Failed to load weights for layer {lid}" << std::endl;
                            // Set default values
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                for (int j = 0; j < {layer.input_shape}; ++j) {{
                                    layer_{lid}_weight[i][j] = 0.0f;
                                }}
                            }}
                        }} else {{
                            // Reshape flat vector into 2D array
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                for (int j = 0; j < {layer.input_shape}; ++j) {{
                                    layer_{lid}_weight[i][j] = layer_{lid}_weight_data[i * {layer.input_shape} + j];
                                }}
                            }}
                        }}
                        
                        // Load bias
                        std::vector<float> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<float>("mq_pt_model_binary/layer_{lid}_bias.bin", 
                                                                layer_{lid}_bias_data, 
                                                                {layer.output_shape});
                        
                        // If bias loading failed, set default values
                        if (!bias_success) {{
                            success = false;
                            std::cerr << "Failed to load bias for layer {lid}" << std::endl;
                            // Set default values
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias[i] = 0.0f;
                            }}
                        }} else {{
                            // Copy bias data
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias[i] = layer_{lid}_bias_data[i];
                            }}
                        }}
                    }}"""
            elif isinstance(layer, BatchNorm):
                load_func += f"""
                    // Load scale and bias for BatchNorm layer {lid}
                    {{
                        std::vector<float> layer_{lid}_scale_data;
                        bool scale_success = load_binary_data<float>("mq_pt_model_binary/layer_{lid}_scale.bin", 
                                                                layer_{lid}_scale_data, 
                                                                {layer.output_shape});
                        
                        // If scale loading failed, set default values
                        if (!scale_success) {{
                            success = false;
                            std::cerr << "Failed to load scale for BatchNorm layer {lid}" << std::endl;
                            // Set default values
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_scale[i] = 1.0f;
                            }}
                        }} else {{
                            // Copy scale data
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_scale[i] = layer_{lid}_scale_data[i];
                            }}
                        }}
                        
                        // Load bias
                        std::vector<float> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<float>("mq_pt_model_binary/layer_{lid}_bias.bin", 
                                                                layer_{lid}_bias_data, 
                                                                {layer.output_shape});
                        
                        // If bias loading failed, set default values
                        if (!bias_success) {{
                            success = false;
                            std::cerr << "Failed to load bias for BatchNorm layer {lid}" << std::endl;
                            // Set default values
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias[i] = 0.0f;
                            }}
                        }} else {{
                            // Copy bias data
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias[i] = layer_{lid}_bias_data[i];
                            }}
                        }}
                    }}"""
        
        return load_func
    
    def _generate_predict_function(self):
        """Generate the C++ prediction function with all layers."""
        code = ""
        
        # First, call the load model function at the beginning
        code += """
        std::vector<float> predict(std::vector<float> &x) {
            // Load model parameters if not loaded already
            static bool model_loaded = false;
            if (!model_loaded) {
                model_loaded = load_model_parameters();
                if (!model_loaded) {
                    std::cerr << "Warning: Failed to load some model parameters. Using default values." << std::endl;
                }
            }
        """
        
        # Process each layer
        for lid, layer in enumerate(self.model.layers):
            if lid == 0:
                input_var = "x"
            else:
                input_var = f"layer_{lid-1}"
                
            # Get the bit-width for this layer
            if self.mix_and_match_config:
                layer_name = f"model.{lid}.weight" if hasattr(layer, 'weight') else f"model.{lid}"
                layer_bits = self.target_bits
                if layer_name in self.mix_and_match_config:
                    layer_bits = self.mix_and_match_config[layer_name]
            else:
                layer_bits = self.target_bits
                
            if isinstance(layer, Linear):
                code += f"""
                // Linear layer {lid} with {layer_bits}-bit weights
                // Convert weights to vector for quantization
                std::vector<float> weight_vec_{lid};
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    for (int j = 0; j < {layer.input_shape}; ++j) {{
                        weight_vec_{lid}.push_back(layer_{lid}_weight[i][j]);
                    }}
                }}
                
                // Quantize weights to {self.max_bits} bits then slice to {layer_bits} bits
                auto [quantized_weights_{lid}, qparams_{lid}] = quantize(weight_vec_{lid}, {self.max_bits});
                auto sliced_weights_{lid} = slice_bits(quantized_weights_{lid}, {self.max_bits}, {layer_bits});
                auto dequant_weights_{lid} = dequantize(sliced_weights_{lid}, qparams_{lid}.first, qparams_{lid}.second);
                
                // Quantize bias
                std::vector<float> bias_vec_{lid}({layer.output_shape});
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    bias_vec_{lid}[i] = layer_{lid}_bias[i];
                }}
                auto [quantized_bias_{lid}, bparams_{lid}] = quantize(bias_vec_{lid}, {self.max_bits});
                auto sliced_bias_{lid} = slice_bits(quantized_bias_{lid}, {self.max_bits}, {layer_bits});
                auto dequant_bias_{lid} = dequantize(sliced_bias_{lid}, bparams_{lid}.first, bparams_{lid}.second);
                
                // Apply linear transformation
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = dequant_bias_{lid}[i];
                    for (int j = 0; j < {layer.input_shape}; ++j) {{
                        layer_{lid}[i] += dequant_weights_{lid}[i * {layer.input_shape} + j] * {input_var}[j];
                    }}
                }}
                """
            elif isinstance(layer, BatchNorm):
                code += f"""
                // BatchNorm layer {lid}
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}[i] = {input_var}[i] * layer_{lid}_scale[i] + layer_{lid}_bias[i];
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
                    if layer.threshold_is_high:
                        comp = ">="
                    else:
                        comp = ">"
                    code += f"""
                    // Step activation layer {lid} with array threshold
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        // Since threshold is loaded at runtime, we use a fixed comparison for simplicity
                        layer_{lid}[i] = {input_var}[i] {comp} 0.0f ? {layer.high} : {layer.low};
                    }}
                    """
                else:
                    if layer.threshold_is_high:
                        comp = ">="
                    else:
                        comp = ">"
                    code += f"""
                    // Step activation layer {lid} with scalar threshold {layer.threshold}
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
