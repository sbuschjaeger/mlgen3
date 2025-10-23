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
    This class generates C++ code for integer-only inference with MatQuant models.
    All models use int8_t for weights/activations and int32_t for accumulators.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="int8_t", 
                 target_bits=8, mix_and_match_config=None, align=None, quantize_signed=True):
        """
        Initialize MatQuant implementation.
        
        Args:
            model: The MLGen3 neural network model
            feature_type: C++ type for input features (float for input conversion)
            label_type: C++ type for output labels (float for output conversion)
            internal_type: C++ type for internal computations (int8_t for integer inference)
            target_bits: Target bit-width for quantization (default: 8)
            mix_and_match_config: Configuration for mix-and-match model
            align: Memory alignment for C++ arrays
            quantize_signed: Whether to use signed quantization (default: True for int8)
        """
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.align = align
        self.target_bits = target_bits
        self.mix_and_match_config = mix_and_match_config
        self.max_bits = 8
        self.filename = None
        self.model_binary_dir = None
        self.quantize_signed = quantize_signed
        
        # Store quantization parameters for each layer
        self.layer_qparams = {}
    
    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
        
    def set_model_binary_dir(self, model_dir):
        """Set the directory where model binary files are stored."""
        self.model_binary_dir = model_dir
        
    def _compute_requantization_params(self, scale_in, scale_weight, scale_out):
        """
        Compute fixed-point multiplier and shift for requantization.
        Based on Section 2.3 of "Quantization and Training of Neural Networks 
        for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2017).
        
        The real-valued multiplier M = (S_input × S_weight) / S_output
        is approximated as M = 2^(-n) × M_0, where M_0 is in the range [0.5, 1).
        We then represent M_0 as a fixed-point number with 31 fractional bits.
        
        Args:
            scale_in: Input scale factor (S_input)
            scale_weight: Weight scale factor (S_weight)
            scale_out: Output scale factor (S_output)
            
        Returns:
            (multiplier, shift): int32_t multiplier (M_0 × 2^31) and shift amount (n)
        """
        # Compute the real multiplier M
        real_multiplier = (scale_in * scale_weight) / scale_out
        
        # Handle edge case where multiplier is very small
        if real_multiplier == 0.0:
            return 0, 0
        
        # Find the exponent such that real_multiplier = 2^(-shift) * significand
        # where significand is in [0.5, 1.0)
        
        shift = 0
        significand = real_multiplier
        
        # Normalize to [0.5, 1.0) range
        while significand < 0.5:
            significand *= 2.0
            shift += 1
        
        while significand >= 1.0:
            significand /= 2.0
            shift -= 1
        
        # Convert significand to a 32-bit fixed-point number with 31 fractional bits
        # significand is in [0.5, 1), so significand × 2^31 is in [2^30, 2^31)
        multiplier_int64 = int(round(significand * (1 << 31)))
        
        # Handle the case where rounding pushes us to 2^31
        if multiplier_int64 == (1 << 31):
            multiplier_int64 //= 2
            shift -= 1
        
        # Ensure multiplier fits in int32 range [0, 2^31)
        assert 0 <= multiplier_int64 < (1 << 31), f"Multiplier {multiplier_int64} out of range"
        
        return int(multiplier_int64), shift
    
    def extract_model_parameters(self):
        """
        Extract model parameters and save them as binary files.
        Store int8_t quantized values and compute requantization parameters.
        """
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        # For input layer quantization of normalized [0, 1] data to int8 range
        if self.quantize_signed:
            # Map [0, 1] to [-128, 127] using affine quantization
            # Use the full range: 0.0 -> -128, 1.0 -> 127

            input_scale = 1.0 / 255.0  # (1.0 - 0.0) / (127 - (-128))
            # input_scale = 1.0 / 127.0
            input_zero_point = -128.0
            # input_zero_point = 0.0
        else:
            # Map [0, 1] to [0, 255]
            input_scale = 1.0 / 255.0
            input_zero_point = 0.0
    
        # Save input quantization parameters
        with open(os.path.join(self.model_binary_dir, "input_qparams.bin"), 'wb') as f:
            f.write(struct.pack('ff', input_scale, input_zero_point))
    
        prev_output_scale = input_scale
        prev_output_zero_point = input_zero_point
    
        for lid, layer in enumerate(self.model.layers):
            layer_params = {}
            
            if isinstance(layer, Linear):
                # Quantize weights
                weight = layer.weight.astype(np.float32)
                weight_qparams = self._quantize_to_8bit(weight)
                weight_8bit = weight_qparams["quantized_values"]
                
                # Save quantized weights
                weight_8bit.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_weight.bin"))
                
                # Save scale and zero_point
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_weight_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', weight_qparams["scale"], weight_qparams["zero_point"]))
                
                layer_params['weight_scale'] = weight_qparams["scale"]
                layer_params['weight_zero_point'] = weight_qparams["zero_point"]
                
                # Quantize bias as int32
                # Per the paper (Section 2.4): S_bias = S_input × S_weight, Z_bias = 0
                bias = layer.bias.astype(np.float32)
                bias_scale = prev_output_scale * weight_qparams["scale"]
                
                # Quantize bias to int32 using zero_point = 0
                bias_quantized = np.round(bias / bias_scale).astype(np.int32)
                
                # Adjust bias to account for input zero point (Equation 7 in paper):
                # The term with zero points: NZ1Z2 - Z1*a2(k) - Z2*a1(i)
                # For bias, we need to subtract: Z_input * sum(q_weight[i])
                # IMPORTANT: Only adjust if prev_output_zero_point is actually non-zero
                if abs(prev_output_zero_point) > 1e-6:  # Use small epsilon to handle floating point comparison
                    weight_sums = weight_8bit.sum(axis=1)  # Sum across input dimension
                    bias_adjustment = np.round(prev_output_zero_point * weight_sums).astype(np.int32)
                    bias_quantized = bias_quantized - bias_adjustment
                    print(f"Layer {lid}: Adjusting bias by zero_point {prev_output_zero_point:.4f}, adjustment range: [{bias_adjustment.min()}, {bias_adjustment.max()}]")
                
                bias_quantized.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bias.bin"))
                
                layer_params['bias_scale'] = bias_scale
                
                # For output scale: According to the paper, we should maintain the same scale
                # throughout the network for intermediate layers. Only the final layer may
                # need a different scale for the logits.
                is_final_layer = (lid == len(self.model.layers) - 1) or \
                                 (lid == len(self.model.layers) - 2 and isinstance(self.model.layers[-1], (Relu, Sigmoid, Sign, Step)))
                
                if is_final_layer:
                    # For the final layer, use a scale that can represent the typical logit range
                    # Logits typically range from -10 to 10 for classification tasks
                    # Map this to int8 range: [-128, 127] represents approximately [-10, 10]
                    output_scale = 20.0 / 255.0  # (10 - (-10)) / (127 - (-128))
                else:
                    # For intermediate layers, keep using the same scale as input
                    # This is key to maintaining accuracy through the network
                    output_scale = prev_output_scale
                
                # After linear layer, the output zero_point is 0 (no offset)
                output_zero_point = 0.0
                
                layer_params['output_scale'] = output_scale
                layer_params['output_zero_point'] = output_zero_point
                
                # Compute requantization parameters using the paper's method
                multiplier, shift = self._compute_requantization_params(
                    prev_output_scale, weight_qparams["scale"], output_scale
                )
                layer_params['multiplier'] = multiplier
                layer_params['shift'] = shift

                # Save requantization parameters
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_requant_params.bin"), 'wb') as f:
                    f.write(struct.pack('ii', multiplier, shift))
                
                # Update scale for next layer
                prev_output_scale = output_scale
                prev_output_zero_point = output_zero_point
                
            elif isinstance(layer, BatchNorm):
                # BatchNorm: y = scale * (x - mean) / sqrt(var + eps) + bias
                # Fuse into: y = bn_scale * x + bn_bias
                
                scale = layer.scale.astype(np.float32) if hasattr(layer, 'scale') else np.ones(layer.output_shape, dtype=np.float32)
                bias = layer.bias.astype(np.float32) if hasattr(layer, 'bias') else np.zeros(layer.output_shape, dtype=np.float32)
                
                if hasattr(layer, 'mean'):
                    mean = layer.mean.astype(np.float32)
                elif hasattr(layer, 'running_mean'):
                    mean = layer.running_mean.astype(np.float32)
                else:
                    mean = np.zeros(layer.output_shape, dtype=np.float32)
                
                if hasattr(layer, 'var'):
                    var = layer.var.astype(np.float32)
                elif hasattr(layer, 'variance'):
                    var = layer.variance.astype(np.float32)
                elif hasattr(layer, 'running_var'):
                    var = layer.running_var.astype(np.float32)
                else:
                    var = np.ones(layer.output_shape, dtype=np.float32)
                
                eps = layer.eps if hasattr(layer, 'eps') else 1e-5
                
                # Compute fused parameters
                bn_scale = scale / np.sqrt(var + eps)
                bn_bias = bias - scale * mean / np.sqrt(var + eps)
                
                # Quantize scale as int8
                scale_qparams = self._quantize_to_8bit(bn_scale)
                scale_8bit = scale_qparams["quantized_values"]
                scale_8bit.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_scale.bin"))
                
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_scale_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('ff', scale_qparams["scale"], scale_qparams["zero_point"]))
                
                # Quantize bias with scale S_bias = S_input × S_scale
                bias_scale = prev_output_scale * scale_qparams["scale"]
                bias_int32 = np.round(bn_bias / bias_scale).astype(np.int32)
                
                # Adjust for input zero point
                # Note: After Linear layer, zero_point should be 0, but check to be safe
                if abs(prev_output_zero_point) > 1e-6:
                    scale_sums = scale_8bit
                    bias_adjustment = np.round(prev_output_zero_point * scale_sums).astype(np.int32)
                    bias_int32 = bias_int32 - bias_adjustment
                    print(f"Layer {lid} (BatchNorm): Adjusting bias by zero_point {prev_output_zero_point:.4f}, adjustment range: [{bias_adjustment.min()}, {bias_adjustment.max()}]")
                
                bias_int32.tofile(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias.bin"))
                
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_bn_bias_qparams.bin"), 'wb') as f:
                    f.write(struct.pack('f', bias_scale))
                
                layer_params['scale_scale'] = scale_qparams["scale"]
                layer_params['scale_zero_point'] = scale_qparams["zero_point"]
                layer_params['bias_scale'] = bias_scale
                layer_params['output_scale'] = prev_output_scale
                layer_params['output_zero_point'] = 0.0
                
                # Compute requantization parameters
                multiplier, shift = self._compute_requantization_params(
                    prev_output_scale, scale_qparams["scale"], prev_output_scale
                )
                layer_params['multiplier'] = multiplier
                layer_params['shift'] = shift
                
                # Save requantization parameters
                with open(os.path.join(self.model_binary_dir, f"layer_{lid}_requant_params.bin"), 'wb') as f:
                    f.write(struct.pack('ii', multiplier, shift))
                
                # BatchNorm output has zero_point = 0
                prev_output_zero_point = 0.0
                
            elif isinstance(layer, Relu):
                # ReLU preserves the scale and zero_point
                layer_params['output_scale'] = prev_output_scale
                layer_params['output_zero_point'] = prev_output_zero_point
            
            self.layer_qparams[lid] = layer_params
    
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
        """Generate C++ code for layer allocations using integer types."""
        alloc = ""
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                # Allocate int8_t arrays for activations and weights
                alloc += f"static int8_t layer_{lid}_output[{layer.output_shape}];\n"
                alloc += f"static int8_t layer_{lid}_weight[{layer.output_shape}][{layer.input_shape}];\n"
                alloc += f"static int32_t layer_{lid}_bias[{layer.output_shape}];\n"
                
                # Store quantization parameters
                alloc += f"static float layer_{lid}_weight_scale;\n"
                alloc += f"static float layer_{lid}_weight_zero_point;\n"
                alloc += f"static int32_t layer_{lid}_multiplier;\n"
                alloc += f"static int layer_{lid}_shift;\n"
                
            elif isinstance(layer, BatchNorm):
                # Allocate arrays for BatchNorm parameters
                alloc += f"static int8_t layer_{lid}_output[{layer.output_shape}];\n"
                alloc += f"static int8_t layer_{lid}_scale[{layer.output_shape}];\n"
                alloc += f"static int32_t layer_{lid}_bn_bias[{layer.output_shape}];\n"
                
                # Store quantization parameters
                alloc += f"static float layer_{lid}_scale_scale;\n"
                alloc += f"static float layer_{lid}_scale_zero_point;\n"
                alloc += f"static float layer_{lid}_bias_scale;\n"
                alloc += f"static int32_t layer_{lid}_multiplier;\n"
                alloc += f"static int layer_{lid}_shift;\n"
                
            elif isinstance(layer, (Relu, Sigmoid, Sign, Step)):
                # Only allocate output for activation layers
                alloc += f"static int8_t layer_{lid}_output[{layer.output_shape}];\n"
        
        # Add input quantization scale - use symmetric quantization for [0, 1] normalized data
        if self.quantize_signed:
            # alloc += "static float input_scale = 1.0f / 127.0f;  // Map [0, 1] to [-127, 127]\n"
            alloc += "static float input_scale = 1.0f / 255.0f;  // Map [0, 1] to [-127, 127]\n"
            # alloc += "static float input_zero_point = 0.0f;  // Symmetric quantization\n"
            alloc += "static float input_zero_point = -128.0f;  // Symmetric quantization\n"
        else:
            alloc += "static float input_scale = 1.0f / 255.0f;  // Map [0, 1] to [0, 255]\n"
            alloc += "static float input_zero_point = 0.0f;\n"
        
        return alloc

    def _generate_binary_utils(self):
        """Generate C++ code for binary file loading utilities."""
        
        return """#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <cstdint>

// INT8_MIN, INT8_MAX, INT32_MIN, INT32_MAX are defined in <cstdint>

// Load binary data for int8_t
inline bool load_int8_data(const std::string& filename, int8_t* data, size_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size != expected_size) {
        std::cerr << "Error: File size mismatch for " << filename << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(data), file_size);
    return file.good();
}

// Load binary data for int32_t
inline bool load_int32_data(const std::string& filename, int32_t* data, size_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size != expected_size * sizeof(int32_t)) {
        std::cerr << "Error: File size mismatch for " << filename << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(data), file_size);
    return file.good();
}

// Load quantization parameters
inline bool load_quantization_params(const std::string& filename, float& scale, float& zero_point) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&zero_point), sizeof(float));
    
    return file.good();
}

// Saturating rounding doubling high multiply
// Implements the operation described in Section 2.3 of the paper:
// "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
// Returns ((int64_t)a * (int64_t)b) / 2^31, rounded to nearest and saturated to int32 range
inline int32_t saturating_rounding_doubling_high_mul(int32_t a, int32_t b) {
    // Compute the 64-bit product
    int64_t product = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    
    // Nudge for rounding: add 2^30 to round to nearest when dividing by 2^31
    int64_t nudge = (product >= 0) ? (1ll << 30) : -(1ll << 30);
    // int64_t nudge = product >= 0 ? (1ll << 30) : (1ll - (1ll << 30));
    
    // Divide by 2^31 with rounding
    int64_t result = (product + nudge) >> 31;
    // int64_t result = (product + nudge) / (1ll << 31);
    
    // Saturate to int32 range
    if (result > INT32_MAX) {
        return INT32_MAX;
    } else if (result < INT32_MIN) {
        return INT32_MIN;
    }
    return static_cast<int32_t>(result);
}

// Rounding divide by power of two
// Implements right shift with rounding to nearest
// Modified to handle both positive and negative exponents
inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent == 0) {
        return x;
    }

    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

// Integer-only requantization using fixed-point arithmetic
// Based on Algorithm 1 in Section 2.3 of the paper
inline int8_t requantize_int32_to_int8(int32_t value, int32_t multiplier, int shift) {
    // Apply the fixed-point multiplier using saturating rounding doubling high multiply
    int32_t result = saturating_rounding_doubling_high_mul(value, multiplier);
    
    // Apply the right shift with rounding
    // For negative shifts (multiply instead of divide), we need to left shift
    if (shift < 0) {
        result = result * (1 << (-shift));
    } else {
        result = rounding_divide_by_pot(result, shift);
    }
    
    // Clamp to int8_t range
    if (result > INT8_MAX) {
        return INT8_MAX;
    } else if (result < INT8_MIN) {
        return INT8_MIN;
    }
    return static_cast<int8_t>(result);
}

// Quantize float input to int8
inline int8_t quantize_float_to_int8(float value, float scale, float zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale + zero_point));
    if (quantized > INT8_MAX) return INT8_MAX;
    if (quantized < INT8_MIN) return INT8_MIN;
    return static_cast<int8_t>(quantized);
}

// Dequantize int8 to float (for final output)
inline float dequantize_int8_to_float(int8_t value, float scale, float zero_point) {
    return (static_cast<float>(value) - zero_point) * scale;
}"""
    
    def _generate_load_model_function(self):
        """Generate C++ code for loading model parameters."""
        load_func = """
                // Load input quantization parameters
                if (!load_quantization_params("mq_pt_model_binary/input_qparams.bin",
                                             input_scale,
                                             input_zero_point)) {
                    std::cerr << "Failed to load input qparams" << std::endl;
                    success = false;
                }
                """
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                load_func += f"""
                    // Load Linear layer {lid} parameters
                    {{
                        // Load int8 weights
                        if (!load_int8_data("mq_pt_model_binary/layer_{lid}_weight.bin",
                                           &layer_{lid}_weight[0][0],
                                           {layer.output_shape * layer.input_shape})) {{
                            std::cerr << "Failed to load weights for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load int32 bias (already adjusted for input zero point)
                        if (!load_int32_data("mq_pt_model_binary/layer_{lid}_bias.bin",
                                            layer_{lid}_bias,
                                            {layer.output_shape})) {{
                            std::cerr << "Failed to load bias for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load quantization parameters
                        if (!load_quantization_params("mq_pt_model_binary/layer_{lid}_weight_qparams.bin",
                                                     layer_{lid}_weight_scale,
                                                     layer_{lid}_weight_zero_point)) {{
                            std::cerr << "Failed to load qparams for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load requantization parameters from file
                        std::ifstream requant_file("mq_pt_model_binary/layer_{lid}_requant_params.bin", std::ios::binary);
                        if (requant_file.is_open()) {{
                            requant_file.read(reinterpret_cast<char*>(&layer_{lid}_multiplier), sizeof(int32_t));
                            requant_file.read(reinterpret_cast<char*>(&layer_{lid}_shift), sizeof(int32_t));
                            requant_file.close();
                        }} else {{
                            std::cerr << "Failed to load requant params for layer {lid}" << std::endl;
                            success = false;
                        }}
                    }}"""
                    
            elif isinstance(layer, BatchNorm):
                load_func += f"""
                    // Load BatchNorm layer {lid} parameters
                    {{
                        // Load int8 scale
                        if (!load_int8_data("mq_pt_model_binary/layer_{lid}_scale.bin",
                                           layer_{lid}_scale,
                                           {layer.output_shape})) {{
                            std::cerr << "Failed to load scale for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load int32 bias (already adjusted for input zero point)
                        if (!load_int32_data("mq_pt_model_binary/layer_{lid}_bn_bias.bin",
                                            layer_{lid}_bn_bias,
                                            {layer.output_shape})) {{
                            std::cerr << "Failed to load bias for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load scale quantization parameters
                        if (!load_quantization_params("mq_pt_model_binary/layer_{lid}_scale_qparams.bin",
                                                     layer_{lid}_scale_scale,
                                                     layer_{lid}_scale_zero_point)) {{
                            std::cerr << "Failed to load scale qparams for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load bias scale
                        std::ifstream bias_scale_file("mq_pt_model_binary/layer_{lid}_bn_bias_qparams.bin", std::ios::binary);
                        if (bias_scale_file.is_open()) {{
                            bias_scale_file.read(reinterpret_cast<char*>(&layer_{lid}_bias_scale), sizeof(float));
                            bias_scale_file.close();
                        }} else {{
                            std::cerr << "Failed to load bias scale for layer {lid}" << std::endl;
                            success = false;
                        }}
                        
                        // Load requantization parameters from file
                        std::ifstream requant_file("mq_pt_model_binary/layer_{lid}_requant_params.bin", std::ios::binary);
                        if (requant_file.is_open()) {{
                            requant_file.read(reinterpret_cast<char*>(&layer_{lid}_multiplier), sizeof(int32_t));
                            requant_file.read(reinterpret_cast<char*>(&layer_{lid}_shift), sizeof(int32_t));
                            requant_file.close();
                        }} else {{
                            std::cerr << "Failed to load requant params for layer {lid}" << std::endl;
                            success = false;
                        }}
                    }}"""
        
        return load_func
    
    def _generate_predict_function(self):
        """Generate C++ prediction function with integer-only arithmetic."""
        code = """
        std::vector<float> predict(std::vector<float> &x) {
            static bool model_loaded = false;
            if (!model_loaded) {
                model_loaded = load_model_parameters();
                if (!model_loaded) {
                    std::cerr << "Warning: Failed to load model parameters." << std::endl;
                }
            }
            
            // Quantize input from float to int8
            int8_t input_quantized[""" + str(self.model.layers[0].input_shape) + """];
            for (int i = 0; i < """ + str(self.model.layers[0].input_shape) + """; ++i) {
                input_quantized[i] = quantize_float_to_int8(x[i], input_scale, input_zero_point);
            }
        """
        
        prev_output = "input_quantized"
        
        for lid, layer in enumerate(self.model.layers):
            if isinstance(layer, Linear):
                code += f"""
                // Linear layer {lid} - Integer matrix multiply
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    int32_t acc = layer_{lid}_bias[i];
                    for (int j = 0; j < {layer.input_shape}; ++j) {{
                        acc += static_cast<int32_t>({prev_output}[j]) * 
                               static_cast<int32_t>(layer_{lid}_weight[i][j]);
                    }}
                    // Requantize int32 accumulator to int8 output
                    // No zero_point addition needed - it's absorbed in the bias
                    layer_{lid}_output[i] = requantize_int32_to_int8(acc, 
                                                                     layer_{lid}_multiplier,
                                                                     layer_{lid}_shift);
                }}
                """
                prev_output = f"layer_{lid}_output"
                
            elif isinstance(layer, BatchNorm):
                code += f"""
                // BatchNorm layer {lid} - Integer scale and shift
                // y = scale * x + bias
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    // Multiply by scale (both int8)
                    int32_t acc = static_cast<int32_t>({prev_output}[i]) * 
                                  static_cast<int32_t>(layer_{lid}_scale[i]);

                    // Add bias (int32)
                    acc += layer_{lid}_bn_bias[i];
                    
                    // Requantize int32 accumulator to int8 output
                    // No zero_point addition needed - it's absorbed in the bias
                    layer_{lid}_output[i] = requantize_int32_to_int8(acc, 
                                                                     layer_{lid}_multiplier,
                                                                     layer_{lid}_shift);
                }}
                """
                prev_output = f"layer_{lid}_output"
                
            elif isinstance(layer, Relu):
                code += f"""
                // ReLU layer {lid} - Integer clamp to zero
                for (int i = 0; i < {layer.output_shape}; ++i) {{
                    layer_{lid}_output[i] = ({prev_output}[i] < 0) ? 0 : {prev_output}[i];
                }}
                """
                prev_output = f"layer_{lid}_output"
        
        # Dequantize final output to float
        final_layer_idx = len(self.model.layers) - 1
        final_output_size = self.model.layers[-1].output_shape
        
        # Find the last Linear layer for dequantization parameters
        last_linear_idx = final_layer_idx
        for i in range(len(self.model.layers) - 1, -1, -1):
            if isinstance(self.model.layers[i], Linear):
                last_linear_idx = i
                break
        
        # Get the output scale for the final layer from layer_qparams
        # We need to save this during code generation
        final_output_scale = self.layer_qparams[last_linear_idx]['output_scale']
        
        code += f"""
            // // Dequantize output to float using the final layer's output scale
            // std::vector<float> output({final_output_size});
            // for (int i = 0; i < {final_output_size}; ++i) {{
            //     // Use output scale instead of weight scale for dequantization
            //     output[i] = static_cast<float>({prev_output}[i]) * {final_output_scale}f;
            // }}

            // Dequantize output to float (zero_point is 0 for output)
            std::vector<float> output(10);
            for (int i = 0; i < 10; ++i) {{
                output[i] = dequantize_int8_to_float({prev_output}[i], 
                                                     {final_output_scale},
                                                     input_zero_point);
            }}

            return output;
        }}
        """
        
        return code
    
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
