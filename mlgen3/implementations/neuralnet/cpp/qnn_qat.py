import numpy as np

from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

# TODO generated c++ code stores weights as uint8_t, however the calculations are still done in floating point
# meaning that it still has to dequantize the value from the quantized value using zero_points and scaling_factors
# when deploying on the real hardware, we will want to change the generation into a separate adapter

class QNN(Implementation):
    """
    Quantized Neural Network implementation for MLGen3.
    This implementation converts weights to fixed-point representation using min-max quantization.
    Biases remain in floating point format for better precision.
    """
    def __init__(self, model, feature_type="int", label_type="int", internal_type="float", num_bits=8, align=None):
        """
        Initialize a QNN implementation.
        
        Args:
            model: MLGen3 model to implement
            feature_type: Type of input features (default: "int")
            label_type: Type of output labels (default: "int")
            internal_type: Type of internal calculations (default: "float")
            num_bits: Number of bits for weight quantization (default: 8)
            align: Memory alignment for performance optimization (default: None)
        """
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.num_bits = num_bits
        self.align = align
        
        # Validate number of bits
        if num_bits <= 0 or num_bits > 32:
            raise ValueError(f"Number of bits must be between 1 and 32, got {num_bits}")

    def quantize_weights(self, weights):
        """
        Quantize weights using min-max quantization to the specified number of bits.
        
        Args:
            weights: Numpy array of weights
            
        Returns:
            quantized_weights: Quantized weights as integers
            scale: Scale factor for dequantization
            zero_point: Zero point for dequantization
        """
        w_min = np.min(weights)
        w_max = np.max(weights)
        
        # Avoid division by zero
        if w_min == w_max:
            return np.zeros_like(weights, dtype=np.int32), 1.0, 0.0
            
        scale = (w_max - w_min) / (2**self.num_bits - 1)
        zero_point = -w_min / scale if scale != 0 else 0
        
        # Apply quantization formula
        quantized_weights = np.clip(np.round(weights / scale + zero_point), 0, 2**self.num_bits - 1).astype(np.int32)
        
        return quantized_weights, scale, zero_point
    
    def implement(self):
        """Generate C++ code for the quantized neural network model."""
        alloc = ""
        code = ""
        header = ""
        
        # Determine integer type based on num_bits - use unsigned types since our quantization outputs 0 to (2^bits-1)
        if self.num_bits <= 8:
            int_type = "uint8_t"
        elif self.num_bits <= 16:
            int_type = "uint16_t"
        else:
            int_type = "uint32_t"
        
        # Add required headers for quantized operations
        header += "#include <cmath>\n"
        header += "#include <cstdint>\n"
        
        # Process each layer in the model
        for lid, layer in enumerate(self.model.layers):
            # Code for allocating layer outputs
            alloc += f"static {self.internal_type} layer_{lid}[{layer.output_shape}]"
            if self.align is not None and self.align > 0:
                alloc += f"__attribute__((aligned({self.align})));\n"
            else:
                alloc += ";\n"
            
            # Determine input for this layer
            if lid == 0:
                input = "x"
            else:
                input = f"layer_{lid-1}"
            
            # Generate code based on layer type
            if isinstance(layer, Linear):
                # Quantize the weights
                quantized_weights, scale, zero_point = self.quantize_weights(layer.weight)
                
                # Create quantized weight array
                weight_list = quantized_weights.tolist()
                weight_str = []
                for row in weight_list:
                    row_str = "{" + ", ".join(str(val) for val in row) + "}"
                    weight_str.append(row_str)
                tmp_weight = ", ".join(weight_str)
                weight_array = f"constexpr {int_type} layer_{lid}_weight[{len(quantized_weights)}][{len(quantized_weights[0])}] = {{{tmp_weight}}};"
                
                # Keep biases as floating point
                bias_str = "{" + ", ".join(str(val) for val in layer.bias.tolist()) + "}"
                bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(layer.bias)}] = {bias_str};"
                
                # Scale factor and zero point for dequantization
                scale_array = f"constexpr {self.internal_type} layer_{lid}_scale = {scale};"
                zero_point_array = f"constexpr {self.internal_type} layer_{lid}_zero_point = {zero_point};"
                
                alloc += weight_array + "\n"
                alloc += bias_array + "\n"
                alloc += scale_array + "\n"
                alloc += zero_point_array + "\n"
                
                # Generate code to compute the layer output with quantized weights
                code += f"""
                    // Linear layer with quantized weights
                    for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                        layer_{lid}[d] = layer_{lid}_bias[d];
                    }}
                    for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                        for (unsigned int i = 0; i < {layer.input_shape}; i++) {{
                            // Dequantize weight on-the-fly: (quantized_weight - zero_point) * scale
                            {self.internal_type} dequantized_weight = (layer_{lid}_weight[d][i] - layer_{lid}_zero_point) * layer_{lid}_scale;
                            layer_{lid}[d] += dequantized_weight * {input}[i];
                        }}
                    }}
                """
            
            elif isinstance(layer, BatchNorm):
                # BatchNorm parameters remain as floating point
                scale_str = "{" + ", ".join(str(val) for val in layer.scale.tolist()) + "}"
                scale_array = f"constexpr {self.internal_type} layer_{lid}_scale[{len(layer.scale)}] = {scale_str};"
                
                bias_str = "{" + ", ".join(str(val) for val in layer.bias.tolist()) + "}"
                bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(layer.bias)}] = {bias_str};"
                
                alloc += scale_array + "\n"
                alloc += bias_array + "\n"
                
                code += f"""
                    // BatchNorm layer
                    for (unsigned int d = 0; d < {layer.output_shape}; d++) {{
                        layer_{lid}[d] = {input}[d] * layer_{lid}_scale[d] + layer_{lid}_bias[d];
                    }}
                """
            
            elif isinstance(layer, Relu):
                code += f"""
                    // ReLU activation
                    for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                        layer_{lid}[i] = {input}[i] >= 0 ? {input}[i] : 0;
                    }}
                """
            
            elif isinstance(layer, Sigmoid):
                code += f"""
                    // Sigmoid activation
                    for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                        layer_{lid}[i] = 1.0f / (1.0f + std::exp(-{input}[i]));
                    }}
                """
            
            elif isinstance(layer, Sign):
                code += f"""
                    // Sign activation
                    for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                        if ({input}[i] > 0) layer_{lid}[i] = 1;
                        else if ({input}[i] < 0) layer_{lid}[i] = -1;
                        else layer_{lid}[i] = 0;
                    }}
                """
            
            elif isinstance(layer, Step):
                if layer.threshold_is_high:
                    comp = ">="
                else:
                    comp = ">"
                
                if isinstance(layer.threshold, (list, np.ndarray)):
                    threshold_str = "{" + ", ".join(str(val) for val in layer.threshold.tolist()) + "}"
                    threshold_array = f"constexpr {self.internal_type} layer_{lid}_threshold[{len(layer.threshold)}] = {threshold_str};"
                    alloc += threshold_array + "\n"
                    threshold = f"layer_{lid}_threshold[i]"
                else:
                    threshold = layer.threshold
                
                code += f"""
                    // Step activation
                    for (unsigned int i = 0; i < {layer.output_shape}; i++) {{
                        layer_{lid}[i] = {input}[i] {comp} {threshold} ? {layer.high} : {layer.low};
                    }}
                """
            
            else:
                raise ValueError(f"Layer type {type(layer)} is not supported by QNN implementation")
        
        # Combine all the generated code
        self.code = f"""
            #include "model.h"
            {alloc}
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
                {code}
                return std::vector<{self.label_type}>(layer_{len(self.model.layers)-1}, layer_{len(self.model.layers)-1}+{self.model.layers[-1].output_shape});
            }}
        """
        
        self.header = f"""
            #pragma once
            #include <vector>
            {header}

            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
        """.strip()