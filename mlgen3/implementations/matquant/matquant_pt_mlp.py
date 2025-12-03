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
        
        # Track input scale for bias quantization
        # For MNIST inputs in [0, 1], use a scale that maps [0, 1] to int8 range
        input_scale = 1.0 / 127.0  # Maps [0, 1] to [0, 127] (or [-128, 127] if signed)
        
        # Extract parameters from each layer and save to binary files
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                # Get original weight values
                weight = layer.weight.astype(np.float32)
                
                # Quantize to 8-bit
                qparams = self._quantize_to_8bit(weight)
                weight_8bit = qparams["quantized_values"]
                weight_scale = qparams["scale"]
                
                # Save quantized weights
                weight_8bit.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_weight.bin"))
                
                # Save quantization parameters (scale, zero_point)
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_weight_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Get original bias values
                bias = layer.bias.astype(np.float32)
                
                # Bias quantization scale = weight_scale * input_scale (per paper Section 2.4)
                bias_scale = weight_scale * input_scale
                
                # Quantize bias: q_bias = round(bias / bias_scale)
                # With zero_point = 0 for bias
                bias_quantized = np.round(bias / bias_scale)
                bias_quantized = np.clip(bias_quantized, -(2**31), 2**31 - 1).astype(np.int32)
                
                # Save quantized bias (32-bit)
                bias_quantized.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                
                # Save bias quantization parameters
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', bias_scale, 0.0))
                
                # Update input_scale for next layer
                # The output will have scale = weight_scale * input_scale / output_scale
                # We use a fixed output scale to prevent accumulation
                input_scale = 1.0 / 127.0
                
            elif isinstance(layer, BatchNorm):
                # MLGen3 BatchNorm only stores scale and bias (already fused parameters)
                # The scale here is actually the fused scale: gamma / sqrt(var + eps)
                # The bias here is actually the fused bias: beta - mean * fused_scale
                
                # Get the already-fused parameters from MLGen3 BatchNorm
                bn_scale = layer.scale.astype(np.float32) if hasattr(layer, 'scale') else None
                bn_bias = layer.bias.astype(np.float32) if hasattr(layer, 'bias') else None
                
                if bn_scale is None or bn_bias is None:
                    raise AttributeError(f"BatchNorm layer {lid} is missing scale or bias. Available attributes: {dir(layer)}")
                
                # Quantize scale to 8-bit
                qparams = self._quantize_to_8bit(bn_scale)
                scale_8bit = qparams["quantized_values"]
                scale_8bit.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_scale.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_scale_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', qparams["scale"], qparams["zero_point"]))
                
                # Quantize bias with scale = input_scale (since BN bias is added after scaling)
                bn_bias_scale = input_scale
                bias_quantized = np.round(bn_bias / bn_bias_scale)
                bias_quantized = np.clip(bias_quantized, -(2**31), 2**31 - 1).astype(np.int32)
                
                bias_quantized.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias.bin"))
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', bn_bias_scale, 0.0))
                
                # Update input_scale after BatchNorm
                # Output scale = input_scale (BatchNorm doesn't change the scale in our integer representation)
                # We keep input_scale the same
                pass

    
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
    
    def _quantize_to_32bit(self, values):
        """
        Quantize bias values to 32-bit representation with zero point = 0.
        This follows the paper's recommendation in Section 2.4.
        
        Args:
            values: NumPy array of values to quantize
            
        Returns:
            dict with quantized_values, scale, and zero_point (always 0)
        """
        # Find min and max values
        val_min = values.min()
        val_max = values.max()
        
        # Calculate scaling factor (zero point is 0 for bias)
        if self.quantize_signed:
            # Signed 32-bit: range is [-2^31, 2^31 - 1]
            q_min = -(2**31)
            q_max = 2**31 - 1
            scale = max(abs(val_min), abs(val_max)) / (2**31 - 1) if val_min != val_max else 1.0
        else:
            # For bias, signed is more appropriate but keep flexibility
            q_min = 0
            q_max = 2**32 - 1
            scale = (val_max - val_min) / (2**32 - 1) if val_min != val_max else 1.0
        
        zero_point = 0.0  # Always 0 for bias as per paper
        
        # Quantize the values
        quantized = np.round(values / scale + zero_point)
        quantized = np.clip(quantized, q_min, q_max).astype(np.int32)
        
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
                # Allocate arrays for integer values (for integer arithmetic)
                alloc += f"static {int_type} layer_{lid}_int[{layer.output_shape}];\n"
                
                # Allocate arrays for 8-bit quantized weights
                alloc += f"static {int_type} layer_{lid}_weight_q8[{layer.output_shape}][{layer.input_shape}];\n"
                alloc += f"static float layer_{lid}_weight_scale;\n"
                alloc += f"static float layer_{lid}_weight_zero_point;\n"
                
                # Allocate arrays for 32-bit quantized bias
                alloc += f"static int32_t layer_{lid}_bias_q32[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_bias_scale;\n"
                
                # Output scale and zero point for this layer
                alloc += f"static float layer_{lid}_output_scale;\n"
                alloc += f"static {int_type} layer_{lid}_output_zero_point;\n"
                
            elif isinstance(layer, BatchNorm):
                alloc += f"static {int_type} layer_{lid}_int[{layer.output_shape}];\n"
                
                # Allocate arrays for 8-bit quantized values
                alloc += f"static {int_type} layer_{lid}_scale_q8[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_scale_scale;\n"
                alloc += f"static float layer_{lid}_scale_zero_point;\n"
                
                alloc += f"static int32_t layer_{lid}_bias_q32[{layer.output_shape}];\n"
                alloc += f"static float layer_{lid}_bias_scale;\n"
                
                alloc += f"static float layer_{lid}_output_scale;\n"
                alloc += f"static {int_type} layer_{lid}_output_zero_point;\n"
                
            elif isinstance(layer, (Relu, Sigmoid, Sign, Step)):
                alloc += f"static {int_type} layer_{lid}_int[{layer.output_shape}];\n"
                
        return alloc

    def _generate_binary_utils(self):
        """Generate C++ code for binary file loading utilities and integer-arithmetic operations."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        return f"""#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <cstdint>

// Template function to load binary data
template <typename T>
bool load_binary_data(const std::string& filename, std::vector<T>& data, size_t expected_size) {{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {{
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }}
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size != expected_size * sizeof(T)) {{
        std::cerr << "Error: File size mismatch. Expected " << expected_size * sizeof(T) 
                  << " bytes, got " << file_size << " bytes." << std::endl;
        return false;
    }}
    
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
    
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&zero_point), sizeof(float));
    
    if (!file) {{
        std::cerr << "Error: Failed to read quantization parameters" << std::endl;
        return false;
    }}
    
    return true;
}}

// Integer-only bit slicing (Section 2.2 of paper)
inline std::vector<{int_type}> slice_bits_int(const std::vector<{int_type}>& quantized, int original_bits, int target_bits) {{
    std::vector<{int_type}> sliced(quantized.size());
    int shift_bits = original_bits - target_bits;
    
    {"int q_min = -(1 << (target_bits - 1));" if self.quantize_signed else "int q_min = 0;"}
    {"int q_max = (1 << (target_bits - 1)) - 1;" if self.quantize_signed else "int q_max = (1 << target_bits) - 1;"}
    
    for (size_t i = 0; i < quantized.size(); ++i) {{
        if (shift_bits > 0) {{
            // Rounding: use bit at position target_bits+1
            int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
            int floor_val = quantized[i] >> shift_bits;
            int rounded = round_bit ? (floor_val + 1) : floor_val;
            
            // Clamp and scale back
            sliced[i] = std::max(q_min, std::min(rounded, q_max)) << shift_bits;
        }} else {{
            sliced[i] = quantized[i];
        }}
    }}
    
    return sliced;
}}

// Fixed-point multiplication (Section 2.2 of paper)
// Implements M = 2^(-n) * M0 where M0 is in [0.5, 1)
// M = (S1 * S2) / S3
struct FixedPointMultiplier {{
    int32_t M0;  // Normalized multiplier as int32
    int n;       // Exponent for 2^(-n)
    
    FixedPointMultiplier(float M) {{
        // Find n such that M = 2^(-n) * M0 with M0 in [0.5, 1)
        if (M == 0) {{
            M0 = 0;
            n = 0;
            return;
        }}
        
        n = 0;
        float M_normalized = M;
        while (M_normalized < 0.5f) {{
            M_normalized *= 2.0f;
            n++;
        }}
        while (M_normalized >= 1.0f) {{
            M_normalized *= 0.5f;
            n--;
        }}
        
        // Convert M0 to int32 (M0 * 2^31)
        M0 = static_cast<int32_t>(M_normalized * (1LL << 31));
    }}
    
    // Apply multiplication with rounding (Appendix B of paper)
    int32_t apply(int32_t value) const {{
        // Fixed-point multiply: (value * M0) >> 31
        int64_t product = static_cast<int64_t>(value) * static_cast<int64_t>(M0);
        int32_t result = static_cast<int32_t>((product + (1LL << 30)) >> 31);  // Round to nearest
        
        // Apply 2^(-n) with rounding
        if (n > 0) {{
            // Add rounding constant before shift
            int32_t round_const = 1 << (n - 1);
            result = (result + round_const) >> n;
        }} else if (n < 0) {{
            result = result << (-n);
        }}
        
        return result;
    }}
}};

// Integer-only matrix multiplication with bias (Section 2.3-2.4 of paper)
// Implements: q3(i,k) = Z3 + M * (N*Z1*Z2 - Z1*a2(k) - Z2*a1(i) + sum(q1(i,j)*q2(j,k)))
template<int ROWS, int COLS, int INNER>
void matmul_int_only(
    const {int_type} weights[ROWS][INNER],  // q1 (weights)
    const {int_type}* input,                 // q2 (activations)
    int32_t* output,                         // q3 (output accumulator)
    const int32_t* bias,                     // bias vector (32-bit)
    {int_type} Z1,                          // weight zero point
    {int_type} Z2,                          // input zero point
    const FixedPointMultiplier& multiplier
) {{
    // Precompute a1(i) = sum_j q1(i,j) for each row (Section 2.3)
    int32_t a1[ROWS];
    for (int i = 0; i < ROWS; ++i) {{
        int32_t sum = 0;
        for (int j = 0; j < INNER; ++j) {{
            sum += static_cast<int32_t>(weights[i][j]);
        }}
        a1[i] = sum;
    }}
    
    // Precompute a2 = sum_j q2(j) for input vector (Section 2.3)
    int32_t a2 = 0;
    for (int j = 0; j < INNER; ++j) {{
        a2 += static_cast<int32_t>(input[j]);
    }}
    
    // Main computation (Equation 7 from paper)
    for (int i = 0; i < ROWS; ++i) {{
        // Compute sum_j (q1(i,j) * q2(j))
        int32_t acc = 0;
        for (int j = 0; j < INNER; ++j) {{
            acc += static_cast<int32_t>(weights[i][j]) * static_cast<int32_t>(input[j]);
        }}
        
        // Apply formula: N*Z1*Z2 - Z1*a2 - Z2*a1(i) + acc
        int32_t offset = INNER * static_cast<int32_t>(Z1) * static_cast<int32_t>(Z2)
                       - static_cast<int32_t>(Z1) * a2
                       - static_cast<int32_t>(Z2) * a1[i]
                       + acc;
        
        // Apply multiplier M and add bias
        output[i] = multiplier.apply(offset) + bias[i];
    }}
}}

// Quantize float to integer with given parameters
inline {int_type} quantize_to_int(float value, float scale, {int_type} zero_point) {{
    {"int q_min = -128;" if self.quantize_signed else "int q_min = 0;"}
    {"int q_max = 127;" if self.quantize_signed else "int q_max = 255;"}
    
    int quantized = static_cast<int>(std::round(value / scale)) + static_cast<int>(zero_point);
    return static_cast<{int_type}>(std::max(q_min, std::min(quantized, q_max)));
}}

// Dequantize integer to float (for final output)
inline float dequantize_to_float({int_type} value, float scale, {int_type} zero_point) {{
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
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
        """Generate C++ code for loading model parameters from binary files."""
        load_func = ""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                load_func += f"""
                    // Load weights and bias for Linear layer {lid}
                    {{
                        std::vector<{int_type}> layer_{lid}_weight_data;
                        bool weight_success = load_binary_data<{int_type}>("mq_pt_model_binary/layer_{lid}_weight.bin", 
                                                                      layer_{lid}_weight_data, 
                                                                      {layer.output_shape} * {layer.input_shape});
                        
                        bool qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_weight_qparams.bin",
                                                                     layer_{lid}_weight_scale, 
                                                                     layer_{lid}_weight_zero_point);
                        
                        if (weight_success && qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                for (int j = 0; j < {layer.input_shape}; ++j) {{
                                    layer_{lid}_weight_q8[i][j] = layer_{lid}_weight_data[i * {layer.input_shape} + j];
                                }}
                            }}
                        }} else {{
                            success = false;
                        }}
                        
                        // Load bias (32-bit)
                        std::vector<int32_t> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<int32_t>("mq_pt_model_binary/layer_{lid}_bias.bin", 
                                                                   layer_{lid}_bias_data, 
                                                                   {layer.output_shape});
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_bias_qparams.bin",
                                                                         layer_{lid}_bias_scale, 
                                                                         layer_{lid}_weight_zero_point);  // Reuse for now
                        
                        if (bias_success && bias_qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias_q32[i] = layer_{lid}_bias_data[i];
                            }}
                        }} else {{
                            success = false;
                        }}
                        
                        // Initialize output quantization parameters (will be set dynamically)
                        layer_{lid}_output_scale = layer_{lid}_weight_scale;  // Placeholder
                        layer_{lid}_output_zero_point = 0;
                    }}"""
            elif isinstance(layer, BatchNorm):
                load_func += f"""
                    // Load BatchNorm parameters for layer {lid}
                    {{
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
                            }}
                        }} else {{
                            success = false;
                        }}
                        
                        // Load bias (32-bit)
                        std::vector<int32_t> layer_{lid}_bias_data;
                        bool bias_success = load_binary_data<int32_t>("mq_pt_model_binary/layer_{lid}_bn_bias.bin", 
                                                                   layer_{lid}_bias_data, 
                                                                   {layer.output_shape});
                        
                        bool bias_qparam_success = load_quantization_params("mq_pt_model_binary/layer_{lid}_bn_bias_qparams.bin",
                                                                         layer_{lid}_bias_scale, 
                                                                         layer_{lid}_scale_zero_point);
                        
                        if (bias_success && bias_qparam_success) {{
                            for (int i = 0; i < {layer.output_shape}; ++i) {{
                                layer_{lid}_bias_q32[i] = layer_{lid}_bias_data[i];
                            }}
                        }} else {{
                            success = false;
                        }}
                        
                        layer_{lid}_output_scale = layer_{lid}_scale_scale;
                        layer_{lid}_output_zero_point = 0;
                    }}"""
        
        return load_func
    
    def _generate_predict_function(self):
        """Generate the C++ prediction function with integer-only arithmetic."""
        int_type = "int8_t" if self.quantize_signed else "uint8_t"
        
        code = """
        std::vector<float> predict(std::vector<float> &x) {
            static bool model_loaded = false;
            if (!model_loaded) {
                model_loaded = load_model_parameters();
                if (!model_loaded) {
                    std::cerr << "Warning: Failed to load some model parameters." << std::endl;
                }
            }
            
            // Input quantization parameters
            // Map [0, 1] to int8 range
            float input_scale = 1.0f / 127.0f;
            """
        
        code += f"{int_type} input_zero_point = 0;\n"
        
        # Quantize input
        code += f"""
            // Quantize input to {int_type}
            {int_type} x_quantized[{self.model.layers[0].input_shape}];
            for (int i = 0; i < {self.model.layers[0].input_shape}; ++i) {{
                x_quantized[i] = quantize_to_int(x[i], input_scale, input_zero_point);
            }}
        """
        
        prev_layer_scale = "input_scale"
        prev_layer_zero_point = "input_zero_point"
        
        # Process each layer with integer arithmetic
        for lid, layer in enumerate(self.model.layers):
            if lid == 0:
                input_var = "x_quantized"
            else:
                input_var = f"layer_{lid-1}_int"
                
            if isinstance(layer, Linear):
                code += f"""
                // Linear layer {lid} using integer-only arithmetic
                {{
                    // Output scale: fixed to prevent saturation
                    layer_{lid}_output_scale = 1.0f / 127.0f;
                    layer_{lid}_output_zero_point = 0;
                    
                    // Compute multiplier M = (S_weight * S_input) / S_output
                    float M_value = (layer_{lid}_weight_scale * {prev_layer_scale}) / layer_{lid}_output_scale;
                    FixedPointMultiplier multiplier(M_value);
                    
                    int32_t temp_output[{layer.output_shape}];
                    
                    // Slice weights if needed
                    std::vector<{int_type}> weights_flat({layer.output_shape} * {layer.input_shape});
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        for (int j = 0; j < {layer.input_shape}; ++j) {{
                            weights_flat[i * {layer.input_shape} + j] = layer_{lid}_weight_q8[i][j];
                        }}
                    }}
                    
                    if (LAYER_BITS[{lid}] < STORAGE_BITS) {{
                        weights_flat = slice_bits_int(weights_flat, STORAGE_BITS, LAYER_BITS[{lid}]);
                    }}
                    
                    // Copy back to 2D array
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        for (int j = 0; j < {layer.input_shape}; ++j) {{
                            layer_{lid}_weight_q8[i][j] = weights_flat[i * {layer.input_shape} + j];
                        }}
                    }}
                    
                    // Integer-only matrix multiplication
                    matmul_int_only<{layer.output_shape}, 1, {layer.input_shape}>(
                        layer_{lid}_weight_q8,
                        {input_var},
                        temp_output,
                        layer_{lid}_bias_q32,
                        static_cast<{int_type}>(layer_{lid}_weight_zero_point),
                        {prev_layer_zero_point},
                        multiplier
                    );
                    
                    // Requantize output to {int_type} with proper clamping
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        // Clamp to int8 range (avoid -128 per paper recommendation)
                        {"int val = std::max(-127, std::min(127, static_cast<int>(temp_output[i])));" if self.quantize_signed else "int val = std::max(0, std::min(255, static_cast<int>(temp_output[i])));"}
                        layer_{lid}_int[i] = static_cast<{int_type}>(val);
                    }}
                }}
                """
                prev_layer_scale = f"layer_{lid}_output_scale"
                prev_layer_zero_point = f"layer_{lid}_output_zero_point"
                
            elif isinstance(layer, BatchNorm):
                # Simplified integer-only BatchNorm
                # Since we've precomputed fused scale and bias, we just need to apply:
                # out_int = round((scale_quantized * input_int + bias_quantized) / scale_factor)
                code += f"""
                // BatchNorm layer {lid} (integer-only with fused parameters)
                {{
                    // BatchNorm output uses same scale as input
                    layer_{lid}_output_scale = {prev_layer_scale};
                    layer_{lid}_output_zero_point = {prev_layer_zero_point};
                    
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        // Dequantize to apply BatchNorm in float (simplified approach)
                        // In a fully integer implementation, this would be done with integer scaling
                        float val = dequantize_to_float({input_var}[i], {prev_layer_scale}, {prev_layer_zero_point});
                        float scale_val = dequantize_to_float(layer_{lid}_scale_q8[i], layer_{lid}_scale_scale, 
                                                             static_cast<{int_type}>(layer_{lid}_scale_zero_point));
                        float bias_val = static_cast<float>(layer_{lid}_bias_q32[i]) * layer_{lid}_bias_scale;
                        
                        // Apply BatchNorm: out = scale * x + bias
                        val = val * scale_val + bias_val;
                        
                        // Requantize with same output scale
                        layer_{lid}_int[i] = quantize_to_int(val, layer_{lid}_output_scale, layer_{lid}_output_zero_point);
                    }}
                }}
                """
                # Scale remains the same after BatchNorm
                prev_layer_scale = f"layer_{lid}_output_scale"
                prev_layer_zero_point = f"layer_{lid}_output_zero_point"
                
            elif isinstance(layer, Relu):
                code += f"""
                // ReLU activation layer {lid} (integer-only)
                {{
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        // ReLU in integer domain: max(zero_point, x)
                        layer_{lid}_int[i] = {input_var}[i] > {prev_layer_zero_point} ? 
                                           {input_var}[i] : {prev_layer_zero_point};
                    }}
                }}
                """
            elif isinstance(layer, (Sigmoid, Sign, Step)):
                # These require more complex handling
                code += f"""
                // Activation layer {lid} (simplified float implementation)
                {{
                    for (int i = 0; i < {layer.output_shape}; ++i) {{
                        float val = dequantize_to_float({input_var}[i], {prev_layer_scale}, {prev_layer_zero_point});
                        // Apply activation (simplified - should implement in fixed-point)
                        layer_{lid}_int[i] = quantize_to_int(val, {prev_layer_scale}, {prev_layer_zero_point});
                    }}
                }}
                """
        
        # Dequantize final output
        final_layer = len(self.model.layers) - 1
        code += f"""
            // Dequantize final output
            std::vector<float> output({self.model.layers[-1].output_shape});
            for (int i = 0; i < {self.model.layers[-1].output_shape}; ++i) {{
                output[i] = dequantize_to_float(layer_{final_layer}_int[i], 
                                              layer_{final_layer}_output_scale, 
                                              layer_{final_layer}_output_zero_point);
            }}
            
            return output;
        }}
        """
        
        return code
