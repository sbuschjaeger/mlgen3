import numpy as np
from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class MatQuant(Implementation):
    """
    Implementation of Matryoshka Quantization for neural networks.
    This class generates C++ code for inference with MatQuant models.
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

    def implement(self):
        """
        Implement the MatQuant model in C++.
        """
        alloc = ""
        code = ""
        header = "#include <algorithm>\n#include <cmath>\n#include <limits>\n"
        
        # Generate quantization utility functions
        quant_funcs = self._generate_quantization_functions()
        
        # Process each layer
        for lid, l in enumerate(self.model.layers):
            if lid == 0:
                input = "x"
            else:
                input = f"layer_{lid-1}"
            
            # Determine bit-width for this layer
            layer_bits = self.target_bits
            if self.mix_and_match_config is not None:
                # For mix-and-match, check if this layer has a specific bit-width
                layer_name = f"model.{lid}.weight" if hasattr(l, 'weight') else f"model.{lid}"
                if layer_name in self.mix_and_match_config:
                    layer_bits = self.mix_and_match_config[layer_name]
                    
            if isinstance(l, Linear):
                alloc_tmp, code_tmp = self._implement_linear(l, lid, input, layer_bits)
            elif isinstance(l, BatchNorm):
                alloc_tmp, code_tmp = self._implement_batchnorm(l, lid, input)
            elif isinstance(l, Relu):
                alloc_tmp, code_tmp = self._implement_relu(l, lid, input)
            elif isinstance(l, Sigmoid):
                alloc_tmp, code_tmp = self._implement_sigmoid(l, lid, input)
            elif isinstance(l, Step):
                alloc_tmp, code_tmp = self._implement_step(l, lid, input)
            elif isinstance(l, Sign):
                alloc_tmp, code_tmp = self._implement_sign(l, lid, input)
            else:
                raise ValueError(f"Layer type {type(l)} not supported by MatQuant implementation")
                
            alloc += alloc_tmp
            code += code_tmp
        
        # Generate define statements for bit configurations
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
            
            # # Print debug information
            # print(f"MatQuant Mix-and-Match Configuration:")
            # for lid, bits in enumerate(layer_bits):
            #     print(f"  Layer {lid}: {bits}-bit")
    
        else:
            define_statements += "#define IS_MIX_AND_MATCH 0\n"
            define_statements += f"#define NUM_LAYERS {len(self.model.layers)}\n"
            # Define LAYER_BITS as a proper C++ array
            define_statements += f"constexpr int LAYER_BITS[NUM_LAYERS] = {{{', '.join([str(self.target_bits)] * len(self.model.layers))}}};\n"
            
        define_statements += f"#define TARGET_BITS {self.target_bits}\n"
        
        # Generate the header and implementation files
        # Move all template function definitions to the header file
        self.header = f"""
            #pragma once
            #include <vector>
            {header}
            
            {define_statements}
            
            // MatQuant quantization functions
            template <typename T>
            std::pair<std::vector<int>, std::pair<T, T>> quantize(const std::vector<T>& weights, int bits) {{
                // Find min and max values
                T w_min = std::numeric_limits<T>::max();
                T w_max = std::numeric_limits<T>::lowest();
                
                for (const auto& w : weights) {{
                    w_min = std::min(w_min, w);
                    w_max = std::max(w_max, w);
                }}
                
                // Calculate scaling factor and zero point
                T scale = (w_max - w_min) / ((1 << bits) - 1);
                T zero_point = (scale != 0) ? (-w_min / scale) : 0;
                
                // Quantize the weights
                std::vector<int> quantized(weights.size());
                for (size_t i = 0; i < weights.size(); ++i) {{
                    int q = std::round(weights[i] / scale + zero_point);
                    quantized[i] = std::max(0, std::min(q, (1 << bits) - 1));
                }}
                
                return {{quantized, {{scale, zero_point}}}};
            }}
            
            template <typename T>
            std::vector<T> dequantize(const std::vector<int>& quantized, T scale, T zero_point) {{
                std::vector<T> dequantized(quantized.size());
                for (size_t i = 0; i < quantized.size(); ++i) {{
                    dequantized[i] = (quantized[i] - zero_point) * scale;
                }}
                return dequantized;
            }}
            
            // Move slice_bits function definition to header file
            inline std::vector<int> slice_bits(const std::vector<int>& quantized, int original_bits, int target_bits, bool rounding = true) {{
                std::vector<int> sliced(quantized.size());
                int shift_bits = original_bits - target_bits;
                
                for (size_t i = 0; i < quantized.size(); ++i) {{
                    if (rounding && shift_bits > 0) {{
                        // Get the bit at position target_bits+1 for rounding
                        int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
                        int floor_val = quantized[i] >> shift_bits;
                        sliced[i] = round_bit ? (floor_val + 1) : floor_val;
                    }} else {{
                        sliced[i] = quantized[i] >> shift_bits;
                    }}
                    
                    // Clamp to ensure values are within the target bit-width range
                    sliced[i] = std::max(0, std::min(sliced[i], (1 << target_bits) - 1));
                    
                    // Scale back to original range
                    sliced[i] = sliced[i] << shift_bits;
                }}
                
                return sliced;
            }}
            
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
        """.strip()

        self.code = f"""
            #include "{self.filename}.h"
            
            {alloc}
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
                {code}
                return std::vector<{self.label_type}>(layer_{len(self.model.layers)-1}, layer_{len(self.model.layers)-1}+{self.model.layers[-1].output_shape});
            }}
        """

    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
        
        # If we've already implemented, regenerate the code with the correct filename
        if hasattr(self, 'code') and hasattr(self, 'header'):
            # Only regenerate the code parts that use the filename
            self.code = self.code.replace('#include "None.h"', f'#include "{self.filename}.h"')
            if '#include "model.h"' in self.code:
                self.code = self.code.replace('#include "model.h"', f'#include "{self.filename}.h"')

    def _generate_quantization_functions(self):
        """Generate C++ functions for quantization, dequantization, and bit slicing."""
        return """
        // MatQuant quantization functions
        template <typename T>
        std::pair<std::vector<int>, std::pair<T, T>> quantize(const std::vector<T>& weights, int bits) {
            // Find min and max values
            T w_min = std::numeric_limits<T>::max();
            T w_max = std::numeric_limits<T>::lowest();
            
            for (const auto& w : weights) {
                w_min = std::min(w_min, w);
                w_max = std::max(w_max, w);
            }
            
            // Calculate scaling factor and zero point
            T scale = (w_max - w_min) / ((1 << bits) - 1);
            T zero_point = (scale != 0) ? (-w_min / scale) : 0;
            
            // Quantize the weights
            std::vector<int> quantized(weights.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                int q = std::round(weights[i] / scale + zero_point);
                quantized[i] = std::max(0, std::min(q, (1 << bits) - 1));
            }
            
            return {quantized, {scale, zero_point}};
        }
        
        template <typename T>
        std::vector<T> dequantize(const std::vector<int>& quantized, T scale, T zero_point) {
            std::vector<T> dequantized(quantized.size());
            for (size_t i = 0; i < quantized.size(); ++i) {
                dequantized[i] = (quantized[i] - zero_point) * scale;
            }
            return dequantized;
        }
        
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
        }
        """
        
    def _implement_linear(self, layer, lid, input, bits):
        """Implement a quantized Linear layer."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        # Convert NumPy arrays to Python lists for C++ compatibility
        weight_list = layer.weight.tolist()
        bias_list = layer.bias.tolist()
        
        # Generate C++ arrays for weights and biases
        weight_str = []
        for row in weight_list:
            row_str = "{" + ", ".join(str(val) for val in row) + "}"
            weight_str.append(row_str)
        weight_str = "{" + ", ".join(weight_str) + "}"
        
        bias_str = "{" + ", ".join(str(val) for val in bias_list) + "}"
        
        # Define weights, biases, and quantization parameters
        alloc += f"constexpr {self.internal_type} layer_{lid}_weight[{len(layer.weight)}][{len(layer.weight[0])}] = {weight_str};\n"
        alloc += f"constexpr {self.internal_type} layer_{lid}_bias[{len(layer.bias)}] = {bias_str};\n"
        
        # MinMax quantization params for this layer (pre-calculated during inference)
        alloc += f"constexpr {self.internal_type} layer_{lid}_scale = {1.0};\n"
        alloc += f"constexpr {self.internal_type} layer_{lid}_zero_point = {0.0};\n"
        
        # Generate code for quantized matrix multiplication
        code = f"""
            // Matryoshka Quantized Linear Layer (target: {bits}-bit)
            // First, flatten weight and input into vectors for quantization
            std::vector<{self.internal_type}> weight_vec_{lid};
            std::vector<{self.internal_type}> bias_vec_{lid}({layer.output_shape});
            
            for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                bias_vec_{lid}[d] = layer_{lid}_bias[d];
                for (unsigned int i = 0; i < {layer.input_shape}; i++) {{
                    weight_vec_{lid}.push_back(layer_{lid}_weight[d][i]);
                }}
            }}
            
            // Quantize weights to {self.max_bits} bits then slice to {bits} bits
            auto [quantized_weights_{lid}, qparams_{lid}] = quantize(weight_vec_{lid}, {self.max_bits});
            auto sliced_weights_{lid} = slice_bits(quantized_weights_{lid}, {self.max_bits}, {bits});
            auto dequant_weights_{lid} = dequantize(sliced_weights_{lid}, qparams_{lid}.first, qparams_{lid}.second);
            
            // Quantize bias if needed
            auto [quantized_bias_{lid}, bparams_{lid}] = quantize(bias_vec_{lid}, {self.max_bits});
            auto sliced_bias_{lid} = slice_bits(quantized_bias_{lid}, {self.max_bits}, {bits});
            auto dequant_bias_{lid} = dequantize(sliced_bias_{lid}, bparams_{lid}.first, bparams_{lid}.second);
            
            // Perform matrix multiplication with quantized weights
            for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                layer_{lid}[d] = dequant_bias_{lid}[d];
                for (unsigned int i = 0; i < {layer.input_shape}; i++) {{
                    layer_{lid}[d] += dequant_weights_{lid}[d * {layer.input_shape} + i] * {input}[i];
                }}
            }}
        """
        
        return alloc, code
        
    def _implement_batchnorm(self, layer, lid, input):
        """Implement BatchNorm layer."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        # Convert NumPy arrays to Python lists for C++ compatibility
        scale_str = "{" + ", ".join(str(val) for val in layer.scale.tolist()) + "}"
        bias_str = "{" + ", ".join(str(val) for val in layer.bias.tolist()) + "}"
        
        alloc += f"constexpr {self.internal_type} layer_{lid}_scale[{len(layer.scale)}] = {scale_str};\n"
        alloc += f"constexpr {self.internal_type} layer_{lid}_bias[{len(layer.bias)}] = {bias_str};\n"
        
        code = f"""
            // BatchNorm layer
            for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                layer_{lid}[d] = {input}[d] * layer_{lid}_scale[d] + layer_{lid}_bias[d];
            }}
        """
        
        return alloc, code
        
    def _implement_relu(self, layer, lid, input):
        """Implement ReLU activation."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        code = f"""
            // ReLU activation
            for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                layer_{lid}[i] = std::max({self.internal_type}(0), {input}[i]);
            }}
        """
        
        return alloc, code
        
    def _implement_sigmoid(self, layer, lid, input):
        """Implement Sigmoid activation."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        code = f"""
            // Sigmoid activation
            for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                layer_{lid}[i] = 1.0 / (1.0 + std::exp(-{input}[i]));
            }}
        """
        
        return alloc, code
        
    def _implement_step(self, layer, lid, input):
        """Implement Step activation."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        # Handle threshold value(s)
        if isinstance(layer.threshold, (list, np.ndarray)):
            threshold_str = "{" + ", ".join(str(val) for val in layer.threshold.tolist()) + "}"
            alloc += f"constexpr {self.internal_type} layer_{lid}_threshold[{len(layer.threshold)}] = {threshold_str};\n"
            
            if layer.threshold_is_high:
                comp = ">="
            else:
                comp = ">"
                
            code = f"""
                // Step activation with threshold array
                for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                    layer_{lid}[i] = {input}[i] {comp} layer_{lid}_threshold[i] ? {layer.high} : {layer.low};
                }}
            """
        else:
            if layer.threshold_is_high:
                comp = ">="
            else:
                comp = ">"
                
            code = f"""
                // Step activation with scalar threshold
                for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                    layer_{lid}[i] = {input}[i] {comp} {layer.threshold} ? {layer.high} : {layer.low};
                }}
            """
            
        return alloc, code
        
    def _implement_sign(self, layer, lid, input):
        """Implement Sign activation."""
        alloc = f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
        if self.align is not None and self.align > 0:
            alloc += f"__attribute__((aligned({self.align})));\n"
        else:
            alloc += ";\n"
            
        code = f"""
            // Sign activation
            for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                if ({input}[i] > 0) {{
                    layer_{lid}[i] = 1;
                }} else if ({input}[i] < 0) {{
                    layer_{lid}[i] = -1;
                }} else {{
                    layer_{lid}[i] = 0;
                }}
            }}
        """
        
        return alloc, code
        