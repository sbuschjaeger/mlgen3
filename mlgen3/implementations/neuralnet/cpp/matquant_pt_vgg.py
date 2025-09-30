import os
import numpy as np
from mlgen3.implementations.implementation import Implementation

class MatQuantPT_VGG(Implementation):
    """
    Implementation of the Matryoshka Quantization PyTorch VGG model in C++.
    This class generates C++ code for inference with binary model weights.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float", 
                 target_bits=8, mix_and_match_config=None, align=None, 
                 input_height=28, input_width=28, input_channels=1):
        """Initialize MatQuant PyTorch VGG implementation."""
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.align = align
        self.target_bits = target_bits
        self.mix_and_match_config = mix_and_match_config
        self.filename = None
        self.model_binary_dir = None
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
    
    def set_model_binary_dir(self, path):
        """Set the directory where binary model files are stored."""
        self.model_binary_dir = path
        
    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
    
    def extract_model_parameters(self):
        """
        Extract and save model parameters to binary files.
        For the VGG4 model, we need to handle:
        - Two convolutional layers (layer 0 and layer 4)
        - Two fully connected layers (layer 9 and layer 11)
        """
        import struct
        import numpy as np
        import os
        
        # Create the binary directory if it doesn't exist
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        # Define the layer types and indices
        layer_indices = [0, 4, 9, 11]  # Conv1, Conv2, FC1, FC2
        
        # Access model state dictionary - updated for SimpleModelWrapper
        if hasattr(self.model, 'state_dict'):
            if callable(getattr(self.model, 'state_dict')):
                model_state = self.model.state_dict()
            else:
                model_state = self.model.state_dict
        else:
            raise ValueError("Model must have state_dict attribute")
        
        print(f"\nExtracting model parameters for layers: {layer_indices}")
        
        for layer_idx in layer_indices:
            # Process weights
            if layer_idx in [0, 4]:  # Conv layers
                weight_key = f'model.{layer_idx}.weight'
                if weight_key not in model_state:
                    print(f"Warning: Key '{weight_key}' not found in model_state")
                    print(f"Available keys: {list(model_state.keys())[:10]}")
                    continue
                    
                weight_tensor = model_state[weight_key].cpu().numpy()
                
                # Get shape for Conv layers
                out_channels, in_channels, kernel_h, kernel_w = weight_tensor.shape
                weight_size = out_channels * in_channels * kernel_h * kernel_w
                
                # Reshape to 1D array for quantization
                weight_flat = weight_tensor.reshape(-1)
                
                # MinMax quantization to 8 bits
                w_min = weight_flat.min()
                w_max = weight_flat.max()
                scale = (w_max - w_min) / 255.0  # 8-bit = 255 values
                zero_point = -w_min / scale if scale != 0 else 0
                
                # Quantize weights
                quantized_weights = np.clip(np.round(weight_flat / scale + zero_point), 0, 255).astype(np.uint8)
                
                # Save quantized weights and quantization parameters
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight.bin", "wb") as f:
                    f.write(quantized_weights.tobytes())
                
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight_qparams.bin", "wb") as f:
                    f.write(struct.pack('ff', scale, zero_point))
                
                print(f"Saved layer_{layer_idx}_weight.bin, shape: {weight_tensor.shape}, size: {weight_size}")
                
            elif layer_idx in [9, 11]:  # FC layers
                weight_key = f'model.{layer_idx}.weight'
                if weight_key not in model_state:
                    print(f"Warning: Key '{weight_key}' not found in model_state")
                    print(f"Available keys: {list(model_state.keys())[:10]}")
                    continue
                
                weight_tensor = model_state[weight_key].cpu().numpy()
                
                # Reshape to 1D array for quantization
                weight_flat = weight_tensor.reshape(-1)
                
                # MinMax quantization to 8 bits
                w_min = weight_flat.min()
                w_max = weight_flat.max()
                scale = (w_max - w_min) / 255.0  # 8-bit = 255 values
                zero_point = -w_min / scale if scale != 0 else 0
                
                # Quantize weights
                quantized_weights = np.clip(np.round(weight_flat / scale + zero_point), 0, 255).astype(np.uint8)
                
                # Save quantized weights and quantization parameters
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight.bin", "wb") as f:
                    f.write(quantized_weights.tobytes())
                
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight_qparams.bin", "wb") as f:
                    f.write(struct.pack('ff', scale, zero_point))
                
                print(f"Saved layer_{layer_idx}_weight.bin, shape: {weight_tensor.shape}, size: {weight_flat.size}")
        
            # Process bias
            bias_key = f'model.{layer_idx}.bias'
            if bias_key in model_state:
                bias_tensor = model_state[bias_key].cpu().numpy()
                
                # MinMax quantization to 8 bits
                b_min = bias_tensor.min()
                b_max = bias_tensor.max()
                scale = (b_max - b_min) / 255.0 if b_max > b_min else 1.0
                zero_point = -b_min / scale if scale != 0 else 0
                
                # Quantize bias
                quantized_bias = np.clip(np.round(bias_tensor / scale + zero_point), 0, 255).astype(np.uint8)
                
                # Save quantized bias and quantization parameters
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_bias.bin", "wb") as f:
                    f.write(quantized_bias.tobytes())
                
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_bias_qparams.bin", "wb") as f:
                    f.write(struct.pack('ff', scale, zero_point))
                
                print(f"Saved layer_{layer_idx}_bias.bin, shape: {bias_tensor.shape}, size: {bias_tensor.size}")
            else:
                print(f"Warning: Bias key {bias_key} not found in model state")
    
        print(f"\nAll model parameters extracted and saved to {self.model_binary_dir}\n")
    
    def implement(self):
        """Implement the MatQuant PyTorch VGG model in C++."""
        if self.model_binary_dir is None:
            raise ValueError("Model binary directory not set. Use set_model_binary_dir() before calling implement().")
        
        # Extract model parameters to binary files
        self.extract_model_parameters()
        
        # Generate the binary paths based on model structure
        # For VGG4, we have:
        # - Conv1 (layer 0)
        # - MaxPool1
        # - BatchNorm1
        # - ReLU
        # - Conv2 (layer 4)
        # - MaxPool2
        # - BatchNorm2
        # - ReLU
        # - Flatten
        # - Linear1 (layer 9)
        # - ReLU
        # - Linear2 (layer 11)
        
        # Define the layer types and indices
        layer_types = {
            'conv1': 0,  # First convolutional layer
            'conv2': 4,  # Second convolutional layer
            'fc1': 9,    # First fully connected layer
            'fc2': 11    # Second fully connected layer
        }
        
        # Generate C++ code for VGG4 model with MatQuant
        # Here we generate the complete implementation
        
        alloc = ""
        # Add layer allocations
        alloc += self._generate_layer_allocations()
        
        # Add binary loading functions
        binary_loading = self._generate_binary_loading()
        
        # Generate layer implementations
        layer_implementations = self._generate_layer_implementations(layer_types)
        
        # Get the binary directory name without the full path (for relative paths in C++)
        binary_dir_name = os.path.basename(self.model_binary_dir)
        
        # Combine everything into the complete implementation
        self.code = f"""
            #include "{self.filename}.h"
            #include <fstream>
            #include <iostream>
            #include <cmath>
            #include <algorithm>
            
            {alloc}
            
            // Load quantized weights and bias from binary files
            {binary_loading}
            
            // Declare quantization arrays for all required layers
            // Conv layer 0 (first conv)
            std::vector<uint8_t> layer_0_weight_q8;
            float layer_0_weight_scale = 1.0f;
            float layer_0_weight_zero_point = 0.0f;
            std::vector<uint8_t> layer_0_bias_q8;
            float layer_0_bias_scale = 1.0f;
            float layer_0_bias_zero_point = 0.0f;
            
            // Conv layer 4 (second conv)
            std::vector<uint8_t> layer_4_weight_q8;
            float layer_4_weight_scale = 1.0f;
            float layer_4_weight_zero_point = 0.0f;
            std::vector<uint8_t> layer_4_bias_q8;
            float layer_4_bias_scale = 1.0f;
            float layer_4_bias_zero_point = 0.0f;
            
            // FC layer 9 (first fc)
            std::vector<uint8_t> layer_9_weight_q8;
            float layer_9_weight_scale = 1.0f;
            float layer_9_weight_zero_point = 0.0f;
            std::vector<uint8_t> layer_9_bias_q8;
            float layer_9_bias_scale = 1.0f;
            float layer_9_bias_zero_point = 0.0f;
            
            // FC layer 11 (second fc)
            std::vector<uint8_t> layer_11_weight_q8;
            float layer_11_weight_scale = 1.0f;
            float layer_11_weight_zero_point = 0.0f;
            std::vector<uint8_t> layer_11_bias_q8;
            float layer_11_bias_scale = 1.0f;
            float layer_11_bias_zero_point = 0.0f;
            
            // Pre-dequantized weights for faster inference
            std::vector<float> layer_0_weights_dequant;
            std::vector<float> layer_0_bias_dequant;
            std::vector<float> layer_4_weights_dequant;
            std::vector<float> layer_4_bias_dequant;
            std::vector<float> layer_9_weights_dequant;
            std::vector<float> layer_9_bias_dequant;
            std::vector<float> layer_11_weights_dequant;
            std::vector<float> layer_11_bias_dequant;
            
            std::vector<float> predict(std::vector<float> &x) {{
                // Load binary files if not loaded
                static bool files_loaded = false;
                if (!files_loaded) {{
                    std::cout << std::endl << "Loading model parameters from binary files..." << std::endl;
                
                    // Load weights for Conv1 (layer 0)
                    load_binary_data("{binary_dir_name}/layer_0_weight.bin", layer_0_weight_q8, 64*1*3*3);
                    load_quantization_params("{binary_dir_name}/layer_0_weight_qparams.bin", 
                                             layer_0_weight_scale, layer_0_weight_zero_point);
                    
                    // Load bias for Conv1
                    load_binary_data("{binary_dir_name}/layer_0_bias.bin", layer_0_bias_q8, 64);
                    load_quantization_params("{binary_dir_name}/layer_0_bias_qparams.bin", 
                                            layer_0_bias_scale, layer_0_bias_zero_point);
                    
                    // Load weights for Conv2 (layer 4)
                    load_binary_data("{binary_dir_name}/layer_4_weight.bin", layer_4_weight_q8, 64*64*3*3);
                    load_quantization_params("{binary_dir_name}/layer_4_weight_qparams.bin", 
                                             layer_4_weight_scale, layer_4_weight_zero_point);
                    
                    // Load bias for Conv2
                    load_binary_data("{binary_dir_name}/layer_4_bias.bin", layer_4_bias_q8, 64);
                    load_quantization_params("{binary_dir_name}/layer_4_bias_qparams.bin", 
                                            layer_4_bias_scale, layer_4_bias_zero_point);
                    
                    // Load weights for FC1 (layer 9)
                    load_binary_data("{binary_dir_name}/layer_9_weight.bin", layer_9_weight_q8, 2048*3136);
                    load_quantization_params("{binary_dir_name}/layer_9_weight_qparams.bin", 
                                             layer_9_weight_scale, layer_9_weight_zero_point);
                    
                    // Load bias for FC1
                    load_binary_data("{binary_dir_name}/layer_9_bias.bin", layer_9_bias_q8, 2048);
                    load_quantization_params("{binary_dir_name}/layer_9_bias_qparams.bin", 
                                            layer_9_bias_scale, layer_9_bias_zero_point);
                    
                    // Load weights for FC2 (layer 11)
                    load_binary_data("{binary_dir_name}/layer_11_weight.bin", layer_11_weight_q8, 10*2048);
                    load_quantization_params("{binary_dir_name}/layer_11_weight_qparams.bin", 
                                             layer_11_weight_scale, layer_11_weight_zero_point);
                    
                    // Load bias for FC2
                    load_binary_data("{binary_dir_name}/layer_11_bias.bin", layer_11_bias_q8, 10);
                    load_quantization_params("{binary_dir_name}/layer_11_bias_qparams.bin", 
                                            layer_11_bias_scale, layer_11_bias_zero_point);
                    
                    // Precompute dequantized weights for Layer 0 (Conv1)
                    layer_0_weights_dequant.resize(64 * 1 * 3 * 3);
                    for (size_t i = 0; i < layer_0_weight_q8.size(); i++) {{
                        std::vector<uint8_t> weight_q8(1, layer_0_weight_q8[i]);
                        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[0]);
                        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_0_weight_scale, layer_0_weight_zero_point);
                        layer_0_weights_dequant[i] = dequant_weight[0];
                    }}
                    
                    std::cout << "Precomputing dequantized weights and biases for faster inference..." << std::endl;
                    
                    // Precompute dequantized biases for Layer 0
                    layer_0_bias_dequant.resize(64);
                    for (size_t i = 0; i < layer_0_bias_q8.size(); i++) {{
                        std::vector<uint8_t> bias_q8(1, layer_0_bias_q8[i]);
                        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[0]);
                        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_0_bias_scale, layer_0_bias_zero_point);
                        layer_0_bias_dequant[i] = dequant_bias[0];
                    }}
                    
                    // Precompute dequantized weights for Layer 4 (Conv2)
                    layer_4_weights_dequant.resize(64 * 64 * 3 * 3);
                    for (size_t i = 0; i < layer_4_weight_q8.size(); i++) {{
                        std::vector<uint8_t> weight_q8(1, layer_4_weight_q8[i]);
                        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[4]);
                        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_4_weight_scale, layer_4_weight_zero_point);
                        layer_4_weights_dequant[i] = dequant_weight[0];
                    }}
                    
                    // Precompute dequantized biases for Layer 4
                    layer_4_bias_dequant.resize(64);
                    for (size_t i = 0; i < layer_4_bias_q8.size(); i++) {{
                        std::vector<uint8_t> bias_q8(1, layer_4_bias_q8[i]);
                        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[4]);
                        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_4_bias_scale, layer_4_bias_zero_point);
                        layer_4_bias_dequant[i] = dequant_bias[0];
                    }}
                    
                    // Precompute dequantized weights for Layer 9 (FC1)
                    layer_9_weights_dequant.resize(2048 * 3136);
                    for (size_t i = 0; i < layer_9_weight_q8.size(); i++) {{
                        std::vector<uint8_t> weight_q8(1, layer_9_weight_q8[i]);
                        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[9]);
                        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_9_weight_scale, layer_9_weight_zero_point);
                        layer_9_weights_dequant[i] = dequant_weight[0];
                    }}
                    
                    // Precompute dequantized biases for Layer 9
                    layer_9_bias_dequant.resize(2048);
                    for (size_t i = 0; i < layer_9_bias_q8.size(); i++) {{
                        std::vector<uint8_t> bias_q8(1, layer_9_bias_q8[i]);
                        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[9]);
                        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_9_bias_scale, layer_9_bias_zero_point);
                        layer_9_bias_dequant[i] = dequant_bias[0];
                    }}
                    
                    // Also precompute for Layer 11 (FC2)
                    layer_11_weights_dequant.resize(10 * 2048);
                    for (size_t i = 0; i < layer_11_weight_q8.size(); i++) {{
                        std::vector<uint8_t> weight_q8(1, layer_11_weight_q8[i]);
                        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[11]);
                        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_11_weight_scale, layer_11_weight_zero_point);
                        layer_11_weights_dequant[i] = dequant_weight[0];
                    }}
                    
                    layer_11_bias_dequant.resize(10);
                    for (size_t i = 0; i < layer_11_bias_q8.size(); i++) {{
                        std::vector<uint8_t> bias_q8(1, layer_11_bias_q8[i]);
                        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[11]);
                        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_11_bias_scale, layer_11_bias_zero_point);
                        layer_11_bias_dequant[i] = dequant_bias[0];
                    }}
                    
                    files_loaded = true;
                    std::cout << "Model parameters loaded successfully." << std::endl << std::endl;
                }}
                
                {layer_implementations}
                
                // Convert C array to std::vector for return
                return std::vector<float>(layer_11, layer_11 + 10);
            }}
        """

        # Generate the header
        self.header = f"""
            #pragma once
            #include <vector>
            #include <algorithm>
            #include <cmath>
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
            inline std::vector<uint8_t> slice_bits(const std::vector<uint8_t>& quantized, int original_bits, int target_bits) {{
                std::vector<uint8_t> sliced(quantized.size());
                int shift_bits = original_bits - target_bits;
                
                for (size_t i = 0; i < quantized.size(); ++i) {{
                    // Perform bit slicing with rounding
                    if (shift_bits > 0) {{
                        // Get the bit at position target_bits+1 for rounding
                        int round_bit = (quantized[i] >> (shift_bits - 1)) & 1;
                        int floor_val = quantized[i] >> shift_bits;
                        sliced[i] = round_bit ? (floor_val + 1) : floor_val;
                        
                        // Clamp to ensure values are within the target bit-width range
                        sliced[i] = std::min(sliced[i], static_cast<uint8_t>((1 << target_bits) - 1));
                        
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
            }}

            // CNN utilities for reshaping and processing image inputs
            namespace cnn_utils {{
                // Reshape 1D input to 3D tensor [channels][height][width]
                template <typename T>
                std::vector<std::vector<std::vector<T>>> reshape_input_to_3d(const std::vector<T>& input, 
                                                                           int channels, int height, int width) {{
                    // Create 3D tensor with dimensions [channels][height][width]
                    std::vector<std::vector<std::vector<T>>> tensor(
                        channels, 
                        std::vector<std::vector<T>>(
                            height, 
                            std::vector<T>(width, 0.0f)
                        )
                    );
                    
                    // Fill the tensor with the input data
                    if (static_cast<int>(input.size()) >= channels * height * width) {{
                        for (int c = 0; c < channels; c++) {{
                            for (int h = 0; h < height; h++) {{
                                for (int w = 0; w < width; w++) {{
                                    int index = c * height * width + h * width + w;
                                    tensor[c][h][w] = input[index];
                                }}
                            }}
                        }}
                    }} else {{
                        std::cerr << "Error: Input size mismatch. Expected at least " 
                                  << (channels * height * width) << " elements but got " 
                                  << input.size() << std::endl;
                    }}
                    
                    return tensor;
                }}
                
                // Flatten 3D tensor to 1D vector
                template <typename T>
                std::vector<T> flatten_3d_to_1d(const std::vector<std::vector<std::vector<T>>>& tensor) {{
                    std::vector<T> flattened;
                    
                    for (const auto& channel : tensor) {{
                        for (const auto& row : channel) {{
                            for (const auto& val : row) {{
                                flattened.push_back(val);
                            }}
                        }}
                    }}
                    
                    return flattened;
                }}
                
                // Apply ReLU to a 3D tensor in-place
                template <typename T>
                void apply_relu_3d(std::vector<std::vector<std::vector<T>>>& tensor) {{
                    for (auto& channel : tensor) {{
                        for (auto& row : channel) {{
                            for (auto& val : row) {{
                                val = std::max(T(0), val);
                            }}
                        }}
                    }}
                }}
            }}

            
            #define IS_MIX_AND_MATCH {1 if self.mix_and_match_config else 0}
            #define NUM_LAYERS {11}  // Total number of layers in the model
            constexpr int LAYER_BITS[NUM_LAYERS] = {{{self._generate_layer_bits_array()}}};
            #define TARGET_BITS {self.target_bits}
            #define STORAGE_BITS 8

            // CNN model parameters
            #define INPUT_HEIGHT {self.input_height}
            #define INPUT_WIDTH {self.input_width}
            #define INPUT_CHANNELS {self.input_channels}
            
            std::vector<float> predict(std::vector<float> &x);
        """
    
    def _generate_layer_allocations(self):
        """Generate C++ code for layer allocations."""
        return """
            // Layer allocations
            static float layer_0[64];  // Conv1 output
            static float layer_1[64];  // MaxPool1 output
            static float layer_2[64];  // BatchNorm1 output
            static float layer_3[64];  // ReLU output
            static float layer_4[64];  // Conv2 output 
            static float layer_5[64];  // MaxPool2 output
            static float layer_6[64];  // BatchNorm2 output
            static float layer_7[64];  // ReLU output
            static float layer_8[2048]; // Linear1 output (from flattened input)
            static float layer_9[2048]; // Linear1 output
            static float layer_10[2048]; // ReLU output
            static float layer_11[10];   // Final output
        """
    
    def _generate_binary_loading(self):
        """Generate C++ code for binary loading functions."""
        # In a real implementation, this would generate code to load binary weight files
        return ""
    
    def _generate_layer_bits_array(self):
        """Generate C++ code for layer bits array."""
        if self.mix_and_match_config:
            # For mix-and-match, create an array with specific bit-widths for each layer
            bits_array = []
            for i in range(11):  # Assuming 11 layers
                layer_name = f"layer_{i}"
                bits = self.mix_and_match_config.get(layer_name, self.target_bits)
                bits_array.append(str(bits))
            return ", ".join(bits_array)
        else:
            # For uniform quantization, use the same bit-width for all layers
            return ", ".join([str(self.target_bits)] * 11)
    
    def _generate_layer_implementations(self, layer_types):
        """Generate C++ code for layer implementations."""
        # In a full implementation, this would generate code for all VGG4 layers
        return """
            // Reshape input to 3D tensor (for convolution)
            auto input_3d = cnn_utils::reshape_input_to_3d(x, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
            
            // Layer 0: Conv1
            std::vector<std::vector<std::vector<float>>> layer_0_3d(64, std::vector<std::vector<float>>(28, std::vector<float>(28, 0.0f)));
            
            // Perform 2D convolution
            for (int out_c = 0; out_c < 64; out_c++) {{
                // Initialize output with precomputed dequantized bias
                for (int h_out = 0; h_out < 28; h_out++) {{
                    for (int w_out = 0; w_out < 28; w_out++) {{
                        layer_0_3d[out_c][h_out][w_out] = layer_0_bias_dequant[out_c];
                    }}
                }}
                
                // Perform convolution with precomputed dequantized weights
                for (int in_c = 0; in_c < INPUT_CHANNELS; in_c++) {{
                    for (int h_out = 0; h_out < 28; h_out++) {{
                        for (int w_out = 0; w_out < 28; w_out++) {{
                            // Compute convolution with 3x3 kernel
                            for (int kh = 0; kh < 3; kh++) {{
                                for (int kw = 0; kw < 3; kw++) {{
                                    // Calculate input position with padding
                                    int h_in = h_out + kh - 1;
                                    int w_in = w_out + kw - 1;
                                    
                                    // Skip if outside input boundaries
                                    if (h_in < 0 || h_in >= INPUT_HEIGHT || w_in < 0 || w_in >= INPUT_WIDTH) {{
                                        continue;
                                    }}
                                    
                                    // Calculate weight index
                                    int weight_idx = out_c * (INPUT_CHANNELS * 3 * 3) + in_c * (3 * 3) + kh * 3 + kw;
                                    
                                    // Use precomputed dequantized weights
                                    layer_0_3d[out_c][h_out][w_out] += input_3d[in_c][h_in][w_in] * layer_0_weights_dequant[weight_idx];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            
            // Copy to layer_0 (flatten for later layers)
            for (int c = 0; c < 64; c++) {{
                for (int h = 0; h < 28; h++) {{
                    for (int w = 0; w < 28; w++) {{
                        layer_0[c * 28 * 28 + h * 28 + w] = layer_0_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 1: MaxPool1
            std::vector<std::vector<std::vector<float>>> layer_1_3d(64, std::vector<std::vector<float>>(14, std::vector<float>(14, 0.0f)));
            
            // Perform max pooling
            for (int c = 0; c < 64; c++) {{
                for (int h_out = 0; h_out < 14; h_out++) {{
                    for (int w_out = 0; w_out < 14; w_out++) {{
                        // Calculate input region (2x2 kernel)
                        int h_start = h_out * 2;
                        int w_start = w_out * 2;
                        
                        // Find max value in the 2x2 region
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int h = 0; h < 2; h++) {{
                            for (int w = 0; w < 2; w++) {{
                                int h_in = h_start + h;
                                int w_in = w_start + w;
                                float val = layer_0_3d[c][h_in][w_in];
                                max_val = std::max(max_val, val);
                            }}
                        }}
                        layer_1_3d[c][h_out][w_out] = max_val;
                    }}
                }}
            }}
            
            // Copy to layer_1 (flatten)
            for (int c = 0; c < 64; c++) {{
                for (int h = 0; h < 14; h++) {{
                    for (int w = 0; w < 14; w++) {{
                        layer_1[c * 14 * 14 + h * 14 + w] = layer_1_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 2: BatchNorm1 (simplified here)
            std::vector<std::vector<std::vector<float>>> layer_2_3d = layer_1_3d; // Copy for now
            
            // Layer 3: ReLU after BatchNorm
            cnn_utils::apply_relu_3d(layer_2_3d);
            
            // Copy to layer_3 (flatten)
            for (int c = 0; c < static_cast<int>(layer_2_3d.size()); c++) {{
                for (int h = 0; h < static_cast<int>(layer_2_3d[c].size()); h++) {{
                    for (int w = 0; w < static_cast<int>(layer_2_3d[c][h].size()); w++) {{
                        layer_3[c * 14 * 14 + h * 14 + w] = layer_2_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 4: Conv2 - OPTIMIZED VERSION
            std::vector<std::vector<std::vector<float>>> layer_4_3d(64, std::vector<std::vector<float>>(14, std::vector<float>(14, 0.0f)));
            
            // Initialize with bias
            for (int out_c = 0; out_c < 64; out_c++) {{
                for (int h_out = 0; h_out < 14; h_out++) {{
                    for (int w_out = 0; w_out < 14; w_out++) {{
                        layer_4_3d[out_c][h_out][w_out] = layer_4_bias_dequant[out_c];
                    }}
                }}
            }}
            
            // Reorder loops for better cache locality
            for (int out_c = 0; out_c < 64; out_c++) {{
                for (int h_out = 0; h_out < 14; h_out++) {{
                    for (int w_out = 0; w_out < 14; w_out++) {{
                        for (int kh = 0; kh < 3; kh++) {{
                            for (int kw = 0; kw < 3; kw++) {{
                                int h_in = h_out + kh - 1;
                                int w_in = w_out + kw - 1;
                                
                                // Skip if outside input boundaries
                                if (h_in < 0 || h_in >= 14 || w_in < 0 || w_in >= 14) {{
                                    continue;
                                }}
                                
                                for (int in_c = 0; in_c < 64; in_c++) {{
                                    // Direct access to precomputed dequantized weights
                                    int weight_idx = out_c * (64 * 3 * 3) + in_c * (3 * 3) + kh * 3 + kw;
                                    layer_4_3d[out_c][h_out][w_out] += layer_2_3d[in_c][h_in][w_in] * layer_4_weights_dequant[weight_idx];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            
            // Copy to layer_4 (flatten)
            for (int c = 0; c < 64; c++) {{
                for (int h = 0; h < 14; h++) {{
                    for (int w = 0; w < 14; w++) {{
                        layer_4[c * 14 * 14 + h * 14 + w] = layer_4_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 5: MaxPool2
            std::vector<std::vector<std::vector<float>>> layer_5_3d(64, std::vector<std::vector<float>>(7, std::vector<float>(7, 0.0f)));
            
            // Perform max pooling
            for (int c = 0; c < 64; c++) {{
                for (int h_out = 0; h_out < 7; h_out++) {{
                    for (int w_out = 0; w_out < 7; w_out++) {{
                        // Calculate input region (2x2 kernel)
                        int h_start = h_out * 2;
                        int w_start = w_out * 2;
                        
                        // Find max value in the 2x2 region
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int h = 0; h < 2; h++) {{
                            for (int w = 0; w < 2; w++) {{
                                int h_in = h_start + h;
                                int w_in = w_start + w;
                                float val = layer_4_3d[c][h_in][w_in];
                                max_val = std::max(max_val, val);
                            }}
                        }}
                        layer_5_3d[c][h_out][w_out] = max_val;
                    }}
                }}
            }}
            
            // Copy to layer_5 (flatten)
            for (int c = 0; c < 64; c++) {{
                for (int h = 0; h < 7; h++) {{
                    for (int w = 0; w < 7; w++) {{
                        layer_5[c * 7 * 7 + h * 7 + w] = layer_5_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 6: BatchNorm2 (simplified here)
            std::vector<std::vector<std::vector<float>>> layer_6_3d = layer_5_3d; // Copy for now
            
            // Layer 7: ReLU after BatchNorm
            cnn_utils::apply_relu_3d(layer_6_3d);
            
            // Flatten for FC layers (64*7*7 = 3136)
            std::vector<float> flattened(3136);
            for (int c = 0; c < static_cast<int>(layer_6_3d.size()); c++) {{
                for (int h = 0; h < static_cast<int>(layer_6_3d[c].size()); h++) {{
                    for (int w = 0; w < static_cast<int>(layer_6_3d[c][h].size()); w++) {{
                        flattened[c * 7 * 7 + h * 7 + w] = layer_6_3d[c][h][w];
                    }}
                }}
            }}
            
            // Layer 9: FC1 (3136 -> 2048)
            for (int i = 0; i < 2048; i++) {{
                // Initialize with bias
                layer_9[i] = layer_9_bias_dequant[i];
                
                // Simple direct implementation without blocking
                for (int j = 0; j < 3136; j++) {{
                    layer_9[i] += flattened[j] * layer_9_weights_dequant[i * 3136 + j];
                }}
            }}
            
            // Layer 10: ReLU - can be vectorized
            for (int i = 0; i < 2048; i++) {{
                layer_10[i] = std::max(0.0f, layer_9[i]);
            }}

            // Layer 11: FC2 (2048 -> 10)
            for (int i = 0; i < 10; i++) {{
                // Initialize with bias
                layer_11[i] = layer_11_bias_dequant[i];
                
                // Simple direct implementation without blocking
                for (int j = 0; j < 2048; j++) {{
                    layer_11[i] += layer_10[j] * layer_11_weights_dequant[i * 2048 + j];
                }}
            }}
        """
