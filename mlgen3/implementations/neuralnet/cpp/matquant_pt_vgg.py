import os
import numpy as np
from mlgen3.implementations.implementation import Implementation
import torch
import torch.nn as nn

class MatQuantPT_VGG(Implementation):
    """
    Implementation of the Matryoshka Quantization PyTorch VGG model in C++.
    This class generates C++ code for inference with binary model weights.
    Supports any VGG model architecture with varying number of layers.
    """
    
    def __init__(self, model, feature_type="float", label_type="float", internal_type="float", 
                 target_bits=8, mix_and_match_config=None, align=None, 
                 input_height=28, input_width=28, input_channels=1, debug=False):
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
        self.debug = debug  # Add debug flag
        
        # Will be populated during model analysis
        self.layer_info = {}  # Store info about each layer
        self.quantizable_layers = []  # Indices of layers that need quantization
        self.layer_shapes = {}  # Store shapes of inputs/outputs for each layer

        # print("\n=============matquant pt vgg initialized===============")
        
    def set_model_binary_dir(self, path):
        """Set the directory where binary model files are stored."""
        self.model_binary_dir = path
        
    def set_filename(self, filename):
        """Set the filename to use for header inclusion."""
        self.filename = filename
    
    def analyze_model(self):
        """
        Analyze the model structure to extract layer information.
        This identifies quantizable layers (Conv2d, Conv1d, Linear) and computes input/output shapes.
        """
        if not hasattr(self.model, 'state_dict'):
            print("Model doesn't have state_dict attribute. Can't analyze structure.")
            return
            
        # Initialize structures to store layer information
        self.quantizable_layers = []
        self.layer_info = {}
        
        # Direct inspection of model layers if the model has a Sequential structure
        if hasattr(self.model, 'model'): # and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            
            for idx, layer in enumerate(layers):
                if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm2d)):
                    # Add to list of quantizable layers
                    self.quantizable_layers.append(idx)
                    
                    # Extract and store layer information based on type
                    if isinstance(layer, nn.Conv2d):
                        self.layer_info[idx] = {
                            'type': 'conv',
                            'out_channels': layer.out_channels,
                            'in_channels': layer.in_channels,
                            'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, tuple) 
                                          else (layer.kernel_size, layer.kernel_size),
                            'stride': layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride),
                            'padding': layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding),
                            'dilation': layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation),
                            'groups': layer.groups,
                            'has_bias': layer.bias is not None
                        }
                    elif isinstance(layer, nn.Conv1d):
                        self.layer_info[idx] = {
                            'type': 'conv1d',
                            'out_channels': layer.out_channels,
                            'in_channels': layer.in_channels,
                            'kernel_size': (layer.kernel_size[0], 1) if isinstance(layer.kernel_size, tuple)
                                          else (layer.kernel_size, 1),
                            'stride': layer.stride if isinstance(layer.stride, tuple) else (layer.stride,),
                            'padding': layer.padding if isinstance(layer.padding, tuple) else (layer.padding,),
                            'dilation': layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,),
                            'groups': layer.groups,
                            'has_bias': layer.bias is not None
                        }
                    elif isinstance(layer, nn.Linear):
                        self.layer_info[idx] = {
                            'type': 'linear',
                            'out_features': layer.out_features,
                            'in_features': layer.in_features,
                            'has_bias': layer.bias is not None
                        }
                    elif isinstance(layer, nn.BatchNorm2d):
                        self.layer_info[idx] = {
                            'type': 'batchnorm2d',
                            'num_features': layer.num_features,
                            'eps': layer.eps,
                            'affine': layer.affine
                        }
                    print(f"Found quantizable layer {idx}: {type(layer).__name__}")

        # Sort the quantizable layers by index
        self.quantizable_layers.sort()
        
        # Calculate input/output shapes for each layer
        self._calculate_layer_shapes()
        
        print(f"\nFound {len(self.quantizable_layers)} quantizable layers at indices: {self.quantizable_layers}")


    def _calculate_layer_shapes(self):
        """
        Calculate the input and output shapes for each layer based on model architecture.
        """

        # print("\n=============Calculate layer shapes...===============")

        # Create a simple list to track the current shape as we process each layer
        current_shape = (self.input_channels, self.input_height, self.input_width)
        
        # If we have the model structure as a Sequential, we can compute the exact shapes
        if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            layer_shapes = {}
            
            # For debugging
            # print("Calculating shapes for VGG model with structure:")
            # for i, layer in enumerate(layers):
            #     print(f"  {i}: {layer}") #.__class__.__name__})
            
            # Calculate shapes for each layer
            for i, layer in enumerate(layers):
                layer_shapes[i] = {'input_shape': current_shape}
                
                if isinstance(layer, nn.Conv2d):
                    # Extract parameters
                    out_channels = layer.out_channels
                    kernel_size = layer.kernel_size
                    stride = layer.stride
                    padding = layer.padding
                    
                    # Calculate output dimensions
                    h_in, w_in = current_shape[1], current_shape[2]
                    h_out = int((h_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
                    w_out = int((w_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
                    
                    current_shape = (out_channels, h_out, w_out)
                    
                elif isinstance(layer, nn.MaxPool2d):
                    # Extract parameters
                    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
                    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                    
                    # Calculate output dimensions
                    c_in, h_in, w_in = current_shape
                    h_out = int((h_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
                    w_out = int((w_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
                    
                    current_shape = (c_in, h_out, w_out)
                    
                elif isinstance(layer, nn.Flatten):
                    # Flatten the shape
                    current_shape = (np.prod(current_shape),)
                    
                elif isinstance(layer, nn.Linear):
                    # Linear layer changes the output dimension
                    current_shape = (layer.out_features,)
                
                # Other layers like BatchNorm, ReLU don't change shape
                layer_shapes[i]['output_shape'] = current_shape
            
            self.layer_shapes = layer_shapes
        else:
            print("Model is not Sequential. Shape calculation may be incomplete.")
    
    def extract_model_parameters(self):
        """
        Extract and save model parameters to binary files.
        Processes all quantizable layers identified during model analysis.
        """
        import struct
        import numpy as np
        import os
            
        # Create the binary directory if it doesn't exist
        os.makedirs(self.model_binary_dir, exist_ok=True)
        
        # Access model state dictionary
        if hasattr(self.model, 'state_dict'):
            if callable(getattr(self.model, 'state_dict')):
                model_state = self.model.state_dict()
            else:
                model_state = self.model.state_dict
        else:
            raise ValueError("Model must have state_dict attribute")
        
        print(f"\nExtracting model parameters for layers: {self.quantizable_layers}")
        
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            # Process weights for Conv and Linear layers
            if layer_type in ['conv', 'conv1d', 'linear']:
                weight_key = f'model.{layer_idx}.weight'
                if weight_key not in model_state:
                    print(f"Warning: Key '{weight_key}' not found in model_state")
                    print(f"Available keys: {list(model_state.keys())[:10]}")
                    continue
                    
                weight_tensor = model_state[weight_key].cpu().numpy()
                
                # Reshape to 1D array for quantization
                weight_flat = weight_tensor.reshape(-1)
                weight_size = weight_flat.size
                
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

                if layer_idx == 0:
                    print("=========================================================================")
                    print(f"Weight stats for layer {layer_idx}: min={w_min}, max={w_max}, scale={scale}, zero_point={zero_point}")
                    print(f"Sample weights: {weight_flat[:10]}")
                    print(f"Sample quantized weights: {quantized_weights[:10]}")
                    print("=========================================================================")
                
                print(f"Saved layer_{layer_idx}_weight.bin, shape: {weight_tensor.shape}, size: {weight_size}")
            
                # Process bias
                bias_key = f'model.{layer_idx}.bias'
                if bias_key in model_state and layer_info.get('has_bias', True):
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

                    if layer_idx == 0:
                        print("=========================================================================")
                        print(f"Bias stats for layer {layer_idx}: min={b_min}, max={b_max}, scale={scale}, zero_point={zero_point}")
                        print(f"Sample biases: {bias_tensor[:10]}")
                        print(f"Sample quantized bias: {quantized_bias[:10]}")
                        print("=========================================================================")
                    
                    print(f"Saved layer_{layer_idx}_bias.bin, shape: {bias_tensor.shape}, size: {bias_tensor.size}")
                else:
                    print(f"Warning: Bias key {bias_key} not found in model state or bias disabled for layer")
            
            # Process BatchNorm2d layers
            elif layer_type == 'batchnorm2d':
                # BatchNorm has weight (gamma), bias (beta), running_mean, running_var, and eps
                weight_key = f'model.{layer_idx}.weight'
                bias_key = f'model.{layer_idx}.bias'
                running_mean_key = f'model.{layer_idx}.running_mean'
                running_var_key = f'model.{layer_idx}.running_var'
                
                # Extract and quantize weight (gamma/scale)
                if weight_key in model_state and layer_info.get('affine', True):
                    weight_tensor = model_state[weight_key].cpu().numpy()
                    
                    # MinMax quantization to 8 bits
                    w_min = weight_tensor.min()
                    w_max = weight_tensor.max()
                    scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
                    zero_point = -w_min / scale if scale != 0 else 0
                    
                    # Quantize weights
                    quantized_weights = np.clip(np.round(weight_tensor / scale + zero_point), 0, 255).astype(np.uint8)
                    
                    # Save quantized weights and quantization parameters
                    with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight.bin", "wb") as f:
                        f.write(quantized_weights.tobytes())
                    
                    with open(f"{self.model_binary_dir}/layer_{layer_idx}_weight_qparams.bin", "wb") as f:
                        f.write(struct.pack('ff', scale, zero_point))
                    
                    print(f"Saved layer_{layer_idx}_weight.bin (BatchNorm), shape: {weight_tensor.shape}, size: {weight_tensor.size}")
                
                # Extract and quantize bias (beta/shift)
                if bias_key in model_state and layer_info.get('affine', True):
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
                    
                    print(f"Saved layer_{layer_idx}_bias.bin (BatchNorm), shape: {bias_tensor.shape}, size: {bias_tensor.size}")
                
                # Extract running_mean (no quantization needed for inference statistics)
                if running_mean_key in model_state:
                    running_mean = model_state[running_mean_key].cpu().numpy().astype(np.float32)
                    
                    with open(f"{self.model_binary_dir}/layer_{layer_idx}_running_mean.bin", "wb") as f:
                        f.write(running_mean.tobytes())
                    
                    print(f"Saved layer_{layer_idx}_running_mean.bin, shape: {running_mean.shape}, size: {running_mean.size}")
                
                # Extract running_var (no quantization needed for inference statistics)
                if running_var_key in model_state:
                    running_var = model_state[running_var_key].cpu().numpy().astype(np.float32)
                    
                    with open(f"{self.model_binary_dir}/layer_{layer_idx}_running_var.bin", "wb") as f:
                        f.write(running_var.tobytes())
                    
                    print(f"Saved layer_{layer_idx}_running_var.bin, shape: {running_var.shape}, size: {running_var.size}")
                
                # Save eps parameter
                eps = layer_info.get('eps', 1e-5)
                with open(f"{self.model_binary_dir}/layer_{layer_idx}_eps.bin", "wb") as f:
                    f.write(struct.pack('f', eps))
                
                print(f"Saved layer_{layer_idx}_eps.bin (value: {eps})")
    
        print(f"\nAll model parameters extracted and saved to {self.model_binary_dir}\n")
    
    def implement(self):
        """Implement the MatQuant PyTorch VGG model in C++."""


        # print("\n=============implementing...===============")

        if self.model_binary_dir is None:
            raise ValueError("Model binary directory not set. Use set_model_binary_dir() before calling implement().")
        
        # Analyze the model to extract information about layers
        if not self.quantizable_layers:
            self.analyze_model()
        
        # Extract model parameters to binary files
        self.extract_model_parameters()
        
        # Generate the binary paths based on model structure
        # Get the binary directory name without the full path (for relative paths in C++)
        binary_dir_name = os.path.basename(self.model_binary_dir)
        
        # Generate C++ code for VGG model with MatQuant
        self.code = self._generate_implementation_code(binary_dir_name)
        self.header = self._generate_header_code()
    
    def _generate_layer_allocations(self):

        # print("\n=============generating layer allocations...===============")
        """Generate C++ code for layer allocations based on the model structure."""
        alloc_code = "// Layer allocations\n"
        
        if not self.layer_shapes:
            return alloc_code + "// No layer shapes available\n"
        
        # Only allocate arrays for layers that actually need them
        # We'll allocate for: Conv outputs, MaxPool outputs, BatchNorm outputs, ReLU outputs,
        # Flatten output, and Linear outputs
        if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            
            for i, layer in enumerate(layers):
                if i not in self.layer_shapes:
                    continue
                
                output_shape = self.layer_shapes[i]['output_shape']
                
                # Calculate size based on shape
                if len(output_shape) == 1:
                    size = output_shape[0]
                elif len(output_shape) == 3:
                    c, h, w = output_shape
                    size = c * h * w
                else:
                    continue
                
                # Only allocate for specific layer types
                if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU, nn.Flatten, nn.Linear)):
                    alloc_code += f"static float layer_{i}[{size}];\n"
                else:
                    # If we don't have shape info, just create a placeholder
                    alloc_code += f"static float layer_{i}[1];  // Placeholder - shape unknown\n"
        else:
            # If we don't have layer shapes, make allocations based on quantizable layers
            for layer_idx in self.quantizable_layers:
                layer_info = self.layer_info.get(layer_idx, {})
                if layer_info.get('type') == 'conv':
                    out_channels = layer_info.get('out_channels', 64)
                    alloc_code += f"static float layer_{layer_idx}[{out_channels}];\n"
                elif layer_info.get('type') == 'linear':
                    out_features = layer_info.get('out_features', 10)
                    alloc_code += f"static float layer_{layer_idx}[{out_features}];\n"
                else:
                    # Default allocation if type unknown
                    alloc_code += f"static float layer_{layer_idx}[64];\n"
        
        return alloc_code
    
    def _generate_binary_loading_code(self, binary_dir_name):
        """Generate C++ code to load binary model parameters."""


        # print("\n=============generate binary loading code...===============")

        if not self.quantizable_layers:
            return "// No quantizable layers identified for binary loading"
            
        code = []
        code.append("// Load binary files if not loaded")
        code.append("static bool files_loaded = false;")
        code.append("if (!files_loaded) {")
        code.append("    std::cout << std::endl << \"Loading model parameters from binary files...\" << std::endl;")
        
        # Generate load statements for each quantizable layer
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            if layer_type == 'conv':
                in_channels = layer_info.get('in_channels', 1)
                out_channels = layer_info.get('out_channels', 64)
                kernel_h, kernel_w = layer_info.get('kernel_size', (3, 3))
                weight_size = out_channels * in_channels * kernel_h * kernel_w
                
                code.append(f"    // Load weights for layer {layer_idx} ({layer_type})")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {weight_size});")
                code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Load bias for layer {layer_idx}")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {out_channels});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                
            elif layer_type == 'linear':
                in_features = layer_info.get('in_features', 1024)
                out_features = layer_info.get('out_features', 10)
                weight_size = out_features * in_features
                
                code.append(f"    // Load weights for layer {layer_idx} ({layer_type})")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {weight_size});")
                code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Load bias for layer {layer_idx}")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {out_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
            
            elif layer_type == 'batchnorm2d':
                num_features = layer_info.get('num_features', 64)
                
                code.append(f"    // Load BatchNorm2d parameters for layer {layer_idx}")
                
                if layer_info.get('affine', True):
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {num_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {num_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                
                # Load running statistics for inference
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_running_mean.bin\", layer_{layer_idx}_running_mean, {num_features});")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_running_var.bin\", layer_{layer_idx}_running_var, {num_features});")
                code.append(f"    // Load eps parameter")
                code.append(f"    std::ifstream eps_file_{layer_idx}(\"{binary_dir_name}/layer_{layer_idx}_eps.bin\", std::ios::binary);")
                code.append(f"    if (eps_file_{layer_idx}.is_open()) {{")
                code.append(f"        eps_file_{layer_idx}.read(reinterpret_cast<char*>(&layer_{layer_idx}_eps), sizeof(float));")
                code.append(f"        eps_file_{layer_idx}.close();")
                code.append(f"    }}")
        
        # Add code to precompute dequantized weights for all layers
        code.append("\n    std::cout << \"Precomputing dequantized weights and biases for faster inference...\" << std::endl;")
        
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            if layer_type == 'conv':
                in_channels = layer_info.get('in_channels', 1)
                out_channels = layer_info.get('out_channels', 64)
                kernel_h, kernel_w = layer_info.get('kernel_size', (3, 3))
                weight_size = out_channels * in_channels * kernel_h * kernel_w
                
                code.append(f"    // Precompute dequantized weights for Layer {layer_idx}")
                code.append(f"    layer_{layer_idx}_weights_dequant.resize({weight_size});")
                code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                code.append("    }")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Precompute dequantized biases for Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({out_channels});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
                
            elif layer_type == 'linear':
                in_features = layer_info.get('in_features', 1024)
                out_features = layer_info.get('out_features', 10)
                weight_size = out_features * in_features
                
                code.append(f"    // Precompute dequantized weights for Layer {layer_idx}")
                code.append(f"    layer_{layer_idx}_weights_dequant.resize({weight_size});")
                code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                code.append("    }")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Precompute dequantized biases for Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({out_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
            
            elif layer_type == 'batchnorm2d':
                num_features = layer_info.get('num_features', 64)
                
                if layer_info.get('affine', True):
                    code.append(f"    // Precompute dequantized weights for BatchNorm Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_weights_dequant.resize({num_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                    code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                    code.append("    }")
                    
                    code.append(f"    // Precompute dequantized biases for BatchNorm Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({num_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
        
        code.append("    files_loaded = true;")
        code.append("    std::cout << \"Model parameters loaded successfully.\" << std::endl << std::endl;")
        code.append("}")
        
        return "\n".join(code)
    
    def _generate_layer_declarations(self):
        """Generate C++ code for quantization array declarations."""
        if not self.quantizable_layers:
            return "// No quantizable layers identified for declarations"
            
        declarations = []
        declarations.append("// Declare quantization arrays for all required layers")
        
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            declarations.append(f"// {layer_type} layer {layer_idx}")
            
            if layer_type in ['conv', 'conv1d', 'linear']:
                declarations.append(f"std::vector<uint8_t> layer_{layer_idx}_weight_q8;")
                declarations.append(f"float layer_{layer_idx}_weight_scale = 1.0f;")
                declarations.append(f"float layer_{layer_idx}_weight_zero_point = 0.0f;")
                declarations.append(f"std::vector<float> layer_{layer_idx}_weights_dequant;")
                
                if layer_info.get('has_bias', True):
                    declarations.append(f"std::vector<uint8_t> layer_{layer_idx}_bias_q8;")
                    declarations.append(f"float layer_{layer_idx}_bias_scale = 1.0f;")
                    declarations.append(f"float layer_{layer_idx}_bias_zero_point = 0.0f;")
                    declarations.append(f"std::vector<float> layer_{layer_idx}_bias_dequant;")
                    
            elif layer_type == 'batchnorm2d':
                if layer_info.get('affine', True):
                    declarations.append(f"std::vector<uint8_t> layer_{layer_idx}_weight_q8;")
                    declarations.append(f"float layer_{layer_idx}_weight_scale = 1.0f;")
                    declarations.append(f"float layer_{layer_idx}_weight_zero_point = 0.0f;")
                    declarations.append(f"std::vector<float> layer_{layer_idx}_weights_dequant;")
                    declarations.append(f"std::vector<uint8_t> layer_{layer_idx}_bias_q8;")
                    declarations.append(f"float layer_{layer_idx}_bias_scale = 1.0f;")
                    declarations.append(f"float layer_{layer_idx}_bias_zero_point = 0.0f;")
                    declarations.append(f"std::vector<float> layer_{layer_idx}_bias_dequant;")
                
                # Running statistics for inference
                declarations.append(f"std::vector<float> layer_{layer_idx}_running_mean;")
                declarations.append(f"std::vector<float> layer_{layer_idx}_running_var;")
                declarations.append(f"float layer_{layer_idx}_eps = 1e-5f;")
                
            declarations.append("")
            
        return "\n".join(declarations)
    
    def _generate_binary_loading_code(self, binary_dir_name):
        """Generate C++ code to load binary model parameters."""


        # print("\n=============generate binary loading code...===============")

        if not self.quantizable_layers:
            return "// No quantizable layers identified for binary loading"
            
        code = []
        code.append("// Load binary files if not loaded")
        code.append("static bool files_loaded = false;")
        code.append("if (!files_loaded) {")
        code.append("    std::cout << std::endl << \"Loading model parameters from binary files...\" << std::endl;")
        
        # Generate load statements for each quantizable layer
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            if layer_type == 'conv':
                in_channels = layer_info.get('in_channels', 1)
                out_channels = layer_info.get('out_channels', 64)
                kernel_h, kernel_w = layer_info.get('kernel_size', (3, 3))
                weight_size = out_channels * in_channels * kernel_h * kernel_w
                
                code.append(f"    // Load weights for layer {layer_idx} ({layer_type})")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {weight_size});")
                code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Load bias for layer {layer_idx}")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {out_channels});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                
            elif layer_type == 'linear':
                in_features = layer_info.get('in_features', 1024)
                out_features = layer_info.get('out_features', 10)
                weight_size = out_features * in_features
                
                code.append(f"    // Load weights for layer {layer_idx} ({layer_type})")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {weight_size});")
                code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Load bias for layer {layer_idx}")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {out_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
            
            elif layer_type == 'batchnorm2d':
                num_features = layer_info.get('num_features', 64)
                
                code.append(f"    // Load BatchNorm2d parameters for layer {layer_idx}")
                
                if layer_info.get('affine', True):
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_weight.bin\", layer_{layer_idx}_weight_q8, {num_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_weight_qparams.bin\", layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                    code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_bias.bin\", layer_{layer_idx}_bias_q8, {num_features});")
                    code.append(f"    load_quantization_params(\"{binary_dir_name}/layer_{layer_idx}_bias_qparams.bin\", layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                
                # Load running statistics for inference
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_running_mean.bin\", layer_{layer_idx}_running_mean, {num_features});")
                code.append(f"    load_binary_data(\"{binary_dir_name}/layer_{layer_idx}_running_var.bin\", layer_{layer_idx}_running_var, {num_features});")
                code.append(f"    // Load eps parameter")
                code.append(f"    std::ifstream eps_file_{layer_idx}(\"{binary_dir_name}/layer_{layer_idx}_eps.bin\", std::ios::binary);")
                code.append(f"    if (eps_file_{layer_idx}.is_open()) {{")
                code.append(f"        eps_file_{layer_idx}.read(reinterpret_cast<char*>(&layer_{layer_idx}_eps), sizeof(float));")
                code.append(f"        eps_file_{layer_idx}.close();")
                code.append(f"    }}")
        
        # Add code to precompute dequantized weights for all layers
        code.append("\n    std::cout << \"Precomputing dequantized weights and biases for faster inference...\" << std::endl;")
        
        for layer_idx in self.quantizable_layers:
            layer_info = self.layer_info.get(layer_idx, {})
            layer_type = layer_info.get('type', 'unknown')
            
            if layer_type == 'conv':
                in_channels = layer_info.get('in_channels', 1)
                out_channels = layer_info.get('out_channels', 64)
                kernel_h, kernel_w = layer_info.get('kernel_size', (3, 3))
                weight_size = out_channels * in_channels * kernel_h * kernel_w
                
                code.append(f"    // Precompute dequantized weights for Layer {layer_idx}")
                code.append(f"    layer_{layer_idx}_weights_dequant.resize({weight_size});")
                code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                code.append("    }")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Precompute dequantized biases for Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({out_channels});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
                
            elif layer_type == 'linear':
                in_features = layer_info.get('in_features', 1024)
                out_features = layer_info.get('out_features', 10)
                weight_size = out_features * in_features
                
                code.append(f"    // Precompute dequantized weights for Layer {layer_idx}")
                code.append(f"    layer_{layer_idx}_weights_dequant.resize({weight_size});")
                code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                code.append("    }")
                
                if layer_info.get('has_bias', True):
                    code.append(f"    // Precompute dequantized biases for Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({out_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
            
            elif layer_type == 'batchnorm2d':
                num_features = layer_info.get('num_features', 64)
                
                if layer_info.get('affine', True):
                    code.append(f"    // Precompute dequantized weights for BatchNorm Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_weights_dequant.resize({num_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_weight_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> weight_q8(1, layer_{layer_idx}_weight_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_weight = slice_bits(weight_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_weight = dequantize(sliced_weight, layer_{layer_idx}_weight_scale, layer_{layer_idx}_weight_zero_point);")
                    code.append(f"        layer_{layer_idx}_weights_dequant[i] = dequant_weight[0];")
                    code.append("    }")
                    
                    code.append(f"    // Precompute dequantized biases for BatchNorm Layer {layer_idx}")
                    code.append(f"    layer_{layer_idx}_bias_dequant.resize({num_features});")
                    code.append(f"    for (size_t i = 0; i < layer_{layer_idx}_bias_q8.size(); i++) {{")
                    code.append(f"        std::vector<uint8_t> bias_q8(1, layer_{layer_idx}_bias_q8[i]);")
                    code.append(f"        std::vector<uint8_t> sliced_bias = slice_bits(bias_q8, 8, LAYER_BITS[{layer_idx}]);")
                    code.append(f"        std::vector<float> dequant_bias = dequantize(sliced_bias, layer_{layer_idx}_bias_scale, layer_{layer_idx}_bias_zero_point);")
                    code.append(f"        layer_{layer_idx}_bias_dequant[i] = dequant_bias[0];")
                    code.append("    }")
        
        code.append("    files_loaded = true;")
        code.append("    std::cout << \"Model parameters loaded successfully.\" << std::endl << std::endl;")
        code.append("}")
        
        return "\n".join(code)
    
    def _generate_layer_implementations(self):
        """Generate C++ code for VGG model layer implementations."""
        if not hasattr(self.model, 'model') or not isinstance(self.model.model, nn.Sequential):
            return "// Model structure unknown, unable to generate layer implementations"
            
        layers = list(self.model.model)
        
        # Generate implementation for each layer in the model
        implementations = []
        implementations.append("// Reshape input to 3D tensor (for convolution)")
        implementations.append("auto input_3d = cnn_utils::reshape_input_to_3d(x, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);")
        
        if self.debug:
            implementations.append("")
            implementations.append("#ifdef DEBUG_MODE")
            implementations.append('std::cout << "\\nInput statistics:" << std::endl;')
            implementations.append('std::cout << "  Shape: [" << INPUT_CHANNELS << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << "]" << std::endl;')
            implementations.append('std::cout << "  First 10 values: ";')
            implementations.append('for (int i = 0; i < 10 && i < x.size(); i++) {')
            implementations.append('    std::cout << x[i] << " ";')
            implementations.append('}')
            implementations.append('std::cout << std::endl;')
            implementations.append("#endif")
        
        implementations.append("")
        
        # Track current tensor format and shape
        current_format = "3d"
        current_shape = (self.input_channels, self.input_height, self.input_width)
        
        for i, layer in enumerate(layers):
            # Skip if we don't have shape information for this layer
            if i not in self.layer_shapes:
                implementations.append(f"// Layer {i}: Shape information missing")
                continue
                
            # Get input and output shapes for this layer
            in_shape = self.layer_shapes[i]['input_shape']
            out_shape = self.layer_shapes[i]['output_shape']
            
            # Determine the input layer name
            if i == 0:
                input_name = "input_3d" if current_format == "3d" else "x"
            else:
                input_name = f"layer_{i-1}_3d" if current_format == "3d" else f"layer_{i-1}"
            
            # Process layer based on its type
            if isinstance(layer, nn.Conv2d):
                # Extract layer parameters
                out_channels, in_channels, kernel_h, kernel_w = layer.weight.shape
                stride = layer.stride
                padding = layer.padding
                
                # Determine output dimensions
                out_h, out_w = out_shape[1], out_shape[2] if len(out_shape) == 3 else (1, 1)
                
                # Generate convolution implementation with proper inference formula
                implementations.append(f"// Layer {i}: Conv2d")
                implementations.append(f"std::vector<std::vector<std::vector<float>>> layer_{i}_3d({out_channels}, std::vector<std::vector<float>>({out_h}, std::vector<float>({out_w}, 0.0f)));")
                implementations.append("")
                
                # Initialize output with bias if present
                if i in self.layer_info and self.layer_info[i].get('has_bias', True):
                    implementations.append("// Initialize output with precomputed dequantized bias")
                    implementations.append(f"for (int out_c = 0; out_c < {out_channels}; out_c++) {{")
                    implementations.append(f"    for (int h_out = 0; h_out < {out_h}; h_out++) {{")
                    implementations.append(f"        for (int w_out = 0; w_out < {out_w}; w_out++) {{")
                    implementations.append(f"            layer_{i}_3d[out_c][h_out][w_out] = layer_{i}_bias_dequant[out_c];")
                    implementations.append("        }")
                    implementations.append("    }")
                    implementations.append("}")
                else:
                    implementations.append("// No bias - initialize to zero")
                
                implementations.append("")
                implementations.append("// Perform convolution: y[out_c, h_out, w_out] = sum(x[in_c, h, w] * W[out_c, in_c, kh, kw]) + b[out_c]")
                implementations.append("// PyTorch Conv2d: weight shape is [out_channels, in_channels, kernel_h, kernel_w]")
                implementations.append(f"for (int out_c = 0; out_c < {out_channels}; out_c++) {{")
                implementations.append(f"    for (int in_c = 0; in_c < {in_channels}; in_c++) {{")
                implementations.append(f"        for (int h_out = 0; h_out < {out_h}; h_out++) {{")
                implementations.append(f"            for (int w_out = 0; w_out < {out_w}; w_out++) {{")
                implementations.append("                // Compute convolution with kernel")
                implementations.append(f"                for (int kh = 0; kh < {kernel_h}; kh++) {{")
                implementations.append(f"                    for (int kw = 0; kw < {kernel_w}; kw++) {{")
                implementations.append("                        // Calculate input position with padding")
                implementations.append(f"                        // padding={padding[0]}, stride={stride[0]}")
                implementations.append(f"                        int h_in = h_out * {stride[0]} + kh - {padding[0]};")
                implementations.append(f"                        int w_in = w_out * {stride[1]} + kw - {padding[1]};")
                implementations.append("")
                implementations.append("                        // Skip if outside input boundaries (padding is implicit zero)")
                implementations.append(f"                        if (h_in < 0 || h_in >= {in_shape[1]} || w_in < 0 || w_in >= {in_shape[2]}) {{")
                implementations.append("                            continue;")
                implementations.append("                        }")
                implementations.append("")
                implementations.append("                        // Calculate weight index: [out_c, in_c, kh, kw]")
                implementations.append("                        // Weight layout: out_channels * in_channels * kernel_h * kernel_w")
                implementations.append(f"                        int weight_idx = ((out_c * {in_channels} + in_c) * {kernel_h} + kh) * {kernel_w} + kw;")
                implementations.append("")
                implementations.append("                        // Accumulate convolution result")
                implementations.append(f"                        layer_{i}_3d[out_c][h_out][w_out] += {input_name}[in_c][h_in][w_in] * layer_{i}_weights_dequant[weight_idx];")
                implementations.append("                    }")
                implementations.append("                }")
                implementations.append("            }")
                implementations.append("        }")
                implementations.append("    }")
                implementations.append("}")
                
                if self.debug:
                    implementations.append("")
                    implementations.append("#ifdef DEBUG_MODE")
                    implementations.append(f'std::cout << "\\nLayer {i}: Conv2d" << std::endl;')
                    implementations.append(f'std::cout << "  Output shape: [{out_channels}, {out_h}, {out_w}]" << std::endl;')
                    implementations.append(f'cnn_utils::print_3d_tensor_stats(layer_{i}_3d, "Conv2d output");')
                    implementations.append("#endif")
                
                # Add flatten copy for Conv layers
                implementations.append("")
                implementations.append(f"// Copy to layer_{i} (flatten for later use)")
                implementations.append(f"for (int c = 0; c < {out_channels}; c++) {{")
                implementations.append(f"    for (int h = 0; h < {out_h}; h++) {{")
                implementations.append(f"        for (int w = 0; w < {out_w}; w++) {{")
                implementations.append(f"            layer_{i}[c * {out_h} * {out_w} + h * {out_w} + w] = layer_{i}_3d[c][h][w];")
                implementations.append("        }")
                implementations.append("    }")
                implementations.append("}")
                
                # Update current format and shape
                current_format = "3d"
                current_shape = out_shape
                
            elif isinstance(layer, nn.MaxPool2d):
                # Extract layer parameters
                kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
                padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
                
                # Determine input and output dimensions
                in_c, in_h, in_w = in_shape
                out_c, out_h, out_w = out_shape
                
                # Generate MaxPool implementation
                implementations.append(f"// Layer {i}: MaxPool2d")
                implementations.append(f"std::vector<std::vector<std::vector<float>>> layer_{i}_3d({out_c}, std::vector<std::vector<float>>({out_h}, std::vector<float>({out_w}, 0.0f)));")
                implementations.append("")
                implementations.append("// Perform max pooling: y[c, h_out, w_out] = max(x[c, h:h+kh, w:w+kw])")
                implementations.append(f"for (int c = 0; c < {out_c}; c++) {{")
                implementations.append(f"    for (int h_out = 0; h_out < {out_h}; h_out++) {{")
                implementations.append(f"        for (int w_out = 0; w_out < {out_w}; w_out++) {{")
                implementations.append("            // Calculate input region")
                implementations.append(f"            int h_start = h_out * {stride[0]} - {padding[0]};")
                implementations.append(f"            int w_start = w_out * {stride[1]} - {padding[1]};")
                implementations.append(f"            int h_end = std::min(h_start + {kernel_size[0]}, {in_h});")
                implementations.append(f"            int w_end = std::min(w_start + {kernel_size[1]}, {in_w});")
                implementations.append("            h_start = std::max(h_start, 0);")
                implementations.append("            w_start = std::max(w_start, 0);")
                implementations.append("")
                implementations.append("            // Find max value in the kernel region")
                implementations.append("            float max_val = -std::numeric_limits<float>::infinity();")
                implementations.append("            for (int h = h_start; h < h_end; h++) {")
                implementations.append("                for (int w = w_start; w < w_end; w++) {")
                implementations.append(f"                    float val = {input_name}[c][h][w];")
                implementations.append("                    max_val = std::max(max_val, val);")
                implementations.append("                }")
                implementations.append("            }")
                implementations.append(f"            layer_{i}_3d[c][h_out][w_out] = max_val;")
                implementations.append("        }")
                implementations.append("    }")
                implementations.append("}")
                
                if self.debug:
                    implementations.append("")
                    implementations.append("#ifdef DEBUG_MODE")
                    implementations.append(f'std::cout << "\\nLayer {i}: MaxPool2d" << std::endl;')
                    implementations.append(f'std::cout << "  Output shape: [{out_c}, {out_h}, {out_w}]" << std::endl;')
                    implementations.append(f'cnn_utils::print_3d_tensor_stats(layer_{i}_3d, "MaxPool2d output");')
                    implementations.append("#endif")
                
                # Add flatten copy for MaxPool layers
                implementations.append("")
                implementations.append(f"// Copy to layer_{i} (flatten for later use)")
                implementations.append(f"for (int c = 0; c < {out_c}; c++) {{")
                implementations.append(f"    for (int h = 0; h < {out_h}; h++) {{")
                implementations.append(f"        for (int w = 0; w < {out_w}; w++) {{")
                implementations.append(f"            layer_{i}[c * {out_h} * {out_w} + h * {out_w} + w] = layer_{i}_3d[c][h][w];")
                implementations.append("        }")
                implementations.append("    }")
                implementations.append("}")
                
                # Update current format and shape
                current_format = "3d"
                current_shape = out_shape
                
            elif isinstance(layer, nn.BatchNorm2d):
                # BatchNorm2d uses running statistics for inference
                # Formula: y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta
                if current_format == "3d":
                    c, h, w = out_shape
                    
                    implementations.append(f"// Layer {i}: BatchNorm2d")
                    implementations.append(f"std::vector<std::vector<std::vector<float>>> layer_{i}_3d({c}, std::vector<std::vector<float>>({h}, std::vector<float>({w}, 0.0f)));")
                    implementations.append("")
                    implementations.append("// Apply BatchNorm inference formula:")
                    implementations.append("// y = gamma * ((x - running_mean) / sqrt(running_var + eps)) + beta")
                    
                    layer_info = self.layer_info.get(i, {})
                    if layer_info.get('affine', True):
                        implementations.append(f"for (int c = 0; c < {c}; c++) {{")
                        implementations.append(f"    float inv_std = 1.0f / std::sqrt(layer_{i}_running_var[c] + layer_{i}_eps);")
                        implementations.append(f"    for (int h = 0; h < {h}; h++) {{")
                        implementations.append(f"        for (int w = 0; w < {w}; w++) {{")
                        implementations.append(f"            float normalized = ({input_name}[c][h][w] - layer_{i}_running_mean[c]) * inv_std;")
                        implementations.append(f"            layer_{i}_3d[c][h][w] = layer_{i}_weights_dequant[c] * normalized + layer_{i}_bias_dequant[c];")
                        implementations.append("        }")
                        implementations.append("    }")
                        implementations.append("}")
                    else:
                        # Non-affine BatchNorm (no learnable gamma/beta)
                        implementations.append(f"for (int c = 0; c < {c}; c++) {{")
                        implementations.append(f"    float inv_std = 1.0f / std::sqrt(layer_{i}_running_var[c] + layer_{i}_eps);")
                        implementations.append(f"    for (int h = 0; h < {h}; h++) {{")
                        implementations.append(f"        for (int w = 0; w < {w}; w++) {{")
                        implementations.append(f"            layer_{i}_3d[c][h][w] = ({input_name}[c][h][w] - layer_{i}_running_mean[c]) * inv_std;")
                        implementations.append("        }")
                        implementations.append("    }")
                        implementations.append("}")
                    
                    if self.debug:
                        implementations.append("")
                        implementations.append("#ifdef DEBUG_MODE")
                        implementations.append(f'std::cout << "\\nLayer {i}: BatchNorm2d" << std::endl;')
                        implementations.append(f'std::cout << "  Output shape: [{c}, {h}, {w}]" << std::endl;')
                        implementations.append(f'cnn_utils::print_3d_tensor_stats(layer_{i}_3d, "BatchNorm2d output");')
                        implementations.append("#endif")
                    
                    # Add flatten copy for BatchNorm layers
                    implementations.append("")
                    implementations.append(f"// Copy to layer_{i}")
                    implementations.append(f"for (int c = 0; c < {c}; c++) {{")
                    implementations.append(f"    for (int h = 0; h < {h}; h++) {{")
                    implementations.append(f"        for (int w = 0; w < {w}; w++) {{")
                    implementations.append(f"            layer_{i}[c * {h} * {w} + h * {w} + w] = layer_{i}_3d[c][h][w];")
                    implementations.append("        }")
                    implementations.append("    }")
                    implementations.append("}")
                    
                    current_format = "3d"
                else:
                    implementations.append(f"// Layer {i}: BatchNorm2d (skipped for non-3D tensor)")
                
            elif isinstance(layer, nn.ReLU):
                # Apply ReLU in-place
                if current_format == "3d":
                    implementations.append(f"// Layer {i}: ReLU")
                    implementations.append(f"cnn_utils::apply_relu_3d({input_name});")
                    implementations.append(f"std::vector<std::vector<std::vector<float>>> layer_{i}_3d = {input_name};")
                    
                    if self.debug:
                        implementations.append("")
                        implementations.append("#ifdef DEBUG_MODE")
                        implementations.append(f'std::cout << "\\nLayer {i}: ReLU" << std::endl;')
                        implementations.append(f'cnn_utils::print_3d_tensor_stats(layer_{i}_3d, "ReLU output");')
                        implementations.append("#endif")
                else:
                    implementations.append(f"// Layer {i}: ReLU")
                    implementations.append(f"for (unsigned int j = 0; j < {out_shape[0]}; j++) {{")
                    implementations.append(f"    layer_{i}[j] = std::max(0.0f, {input_name}[j]);")
                    implementations.append("}")
                    
                    if self.debug:
                        implementations.append("")
                        implementations.append("#ifdef DEBUG_MODE")
                        implementations.append(f'std::cout << "\\nLayer {i}: ReLU" << std::endl;')
                        implementations.append(f'cnn_utils::print_1d_tensor_stats(layer_{i}, {out_shape[0]}, "ReLU output");')
                        implementations.append("#endif")
                
            elif isinstance(layer, nn.Flatten):
                implementations.append(f"// Layer {i}: Flatten")
                
                if current_format == "3d":
                    # Get 3D dimensions
                    c, h, w = in_shape
                    flattened_size = c * h * w
                    
                    implementations.append(f"// Flatten 3D tensor to 1D (size: {flattened_size})")
                    implementations.append(f"std::vector<float> layer_{i}({flattened_size});")
                    implementations.append(f"int idx = 0;")
                    implementations.append(f"for (int c = 0; c < {c}; c++) {{")
                    implementations.append(f"    for (int h = 0; h < {h}; h++) {{")
                    implementations.append(f"        for (int w = 0; w < {w}; w++) {{")
                    implementations.append(f"            layer_{i}[idx++] = {input_name}[c][h][w];")
                    implementations.append("        }")
                    implementations.append("    }")
                    implementations.append("}")
                    
                    if self.debug:
                        implementations.append("")
                        implementations.append("#ifdef DEBUG_MODE")
                        implementations.append(f'std::cout << "\\nLayer {i}: Flatten" << std::endl;')
                        implementations.append(f'std::cout << "  Output shape: [{flattened_size}]" << std::endl;')
                        implementations.append(f'cnn_utils::print_1d_tensor_stats(layer_{i}.data(), {flattened_size}, "Flatten output");')
                        implementations.append("#endif")
                    
                    current_format = "1d"
                else:
                    implementations.append(f"// Already in 1D format, no need to flatten")
                    implementations.append(f"std::vector<float> layer_{i}({input_name}, {input_name} + {in_shape[0]});")
                
            elif isinstance(layer, nn.Linear):
                # Linear layer: y = x * W^T + b
                in_features = layer.in_features
                out_features = layer.out_features
                
                implementations.append(f"// Layer {i}: Linear ({in_features} -> {out_features})")
                implementations.append("// Inference formula: y = x * W^T + b")
                
                # Initialize output with bias if present
                layer_info = self.layer_info.get(i, {})
                if layer_info.get('has_bias', True):
                    implementations.append(f"for (int o = 0; o < {out_features}; o++) {{")
                    implementations.append(f"    layer_{i}[o] = layer_{i}_bias_dequant[o];")
                    implementations.append("}")
                else:
                    implementations.append(f"for (int o = 0; o < {out_features}; o++) {{")
                    implementations.append(f"    layer_{i}[o] = 0.0f;")
                    implementations.append("}")
                
                implementations.append("")
                
                # Matrix multiplication: output[o] += input[j] * weight[o][j]
                # PyTorch Linear: weight shape is [out_features, in_features]
                implementations.append(f"for (int o = 0; o < {out_features}; o++) {{")
                implementations.append(f"    for (int j = 0; j < {in_features}; j++) {{")
                implementations.append(f"        layer_{i}[o] += {input_name}[j] * layer_{i}_weights_dequant[o * {in_features} + j];")
                implementations.append("    }")
                implementations.append("}")
                
                if self.debug:
                    implementations.append("")
                    implementations.append("#ifdef DEBUG_MODE")
                    implementations.append(f'std::cout << "\\nLayer {i}: Linear" << std::endl;')
                    implementations.append(f'std::cout << "  Output shape: [{out_features}]" << std::endl;')
                    implementations.append(f'cnn_utils::print_1d_tensor_stats(layer_{i}, {out_features}, "Linear output");')
                    implementations.append("#endif")
                
                current_format = "1d"
                current_shape = out_shape
        
        # Add return statement for the final layer
        if len(layers) > 0:
            final_layer_idx = len(layers) - 1
            final_shape = self.layer_shapes.get(final_layer_idx, {}).get('output_shape', [10])
            final_size = final_shape[0] if len(final_shape) == 1 else np.prod(final_shape)
            
            implementations.append(f"\n// Return the output of the final layer")
            
            if self.debug:
                implementations.append("#ifdef DEBUG_MODE")
                implementations.append('std::cout << "\\nFinal output: ";')
                implementations.append(f'for (int i = 0; i < {final_size}; i++) {{')
                implementations.append(f'    std::cout << layer_{final_layer_idx}[i] << " ";')
                implementations.append('}')
                implementations.append('std::cout << std::endl;')
                implementations.append("#endif")
            
            implementations.append(f"return std::vector<float>(layer_{final_layer_idx}, layer_{final_layer_idx} + {final_size});")
        else:
            implementations.append("\n// No layers found in the model")
            implementations.append("return std::vector<float>();")
            
        return "\n".join(implementations)
    
    def _generate_layer_bits_array(self):
        """Generate C++ code for layer bits array."""
        if not hasattr(self.model, 'model') or not isinstance(self.model.model, nn.Sequential):
            return str(self.target_bits)
        
        layers = list(self.model.model)
        
        if self.mix_and_match_config:
            # For mix-and-match, create an array with specific bit-widths for each layer
            bits_array = []
            for i in range(len(layers)):
                # Try different possible layer name formats
                layer_name = f"model.{i}.weight"
                bits = self.mix_and_match_config.get(layer_name, self.target_bits)
                bits_array.append(str(bits))
            return ", ".join(bits_array)
        else:
            # For uniform quantization, use the same bit-width for all layers
            return ", ".join([str(self.target_bits)] * len(layers))
    
    def _generate_implementation_code(self, binary_dir_name):

        # print("\n=============generate implementation code...===============")
        """Generate the complete C++ implementation code."""
        # Generate the components of the implementation
        alloc = self._generate_layer_allocations()
        declarations = self._generate_layer_declarations()
        binary_loading = self._generate_binary_loading_code(binary_dir_name)
        layer_implementations = self._generate_layer_implementations()
        
        # Combine everything into the complete implementation
        code = f"""
            #include "{self.filename}.h"
            #include <fstream>
            #include <iostream>
            #include <cmath>
            #include <algorithm>
            
            {alloc}
            
            {declarations}
            
            std::vector<float> predict(std::vector<float> &x) {{
                {binary_loading}
                
                {layer_implementations}
            }}
        """
        
        return code
    
    def _generate_header_code(self):
        """Generate the complete C++ header code."""
        total_layers = 0
        if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Sequential):
            total_layers = len(self.model.model)
        else:
            # Estimate from the highest layer index in quantizable_layers
            total_layers = max(self.quantizable_layers) + 2 if self.quantizable_layers else 0
        
        header = f"""
            #pragma once
            #include <vector>
            #include <algorithm>
            #include <cmath>
            #include <limits>
            #include <fstream>
            #include <iostream>

            // Uncomment to enable debug output
            {'#define DEBUG_MODE' if self.debug else '// #define DEBUG_MODE'}

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
                
                // Print statistics for 3D tensor (for debugging)
                template <typename T>
                void print_3d_tensor_stats(const std::vector<std::vector<std::vector<T>>>& tensor, const std::string& name) {{
                    T min_val = std::numeric_limits<T>::max();
                    T max_val = std::numeric_limits<T>::lowest();
                    T sum = 0;
                    int count = 0;
                    
                    for (const auto& channel : tensor) {{
                        for (const auto& row : channel) {{
                            for (const auto& val : row) {{
                                min_val = std::min(min_val, val);
                                max_val = std::max(max_val, val);
                                sum += val;
                                count++;
                            }}
                        }}
                    }}
                    
                    std::cout << "  " << name << " - Min: " << min_val << ", Max: " << max_val 
                              << ", Mean: " << (count > 0 ? sum / count : 0) << std::endl;
                    
                    // Print first row of first channel
                    if (!tensor.empty() && !tensor[0].empty() && !tensor[0][0].empty()) {{
                        std::cout << "  First channel, first row (first 5 values): ";
                        for (size_t i = 0; i < std::min(size_t(5), tensor[0][0].size()); i++) {{
                            std::cout << tensor[0][0][i] << " ";
                        }}
                        std::cout << std::endl;
                    }}
                }}
                
                // Print statistics for 1D tensor (for debugging)
                template <typename T>
                void print_1d_tensor_stats(const T* tensor, int size, const std::string& name) {{
                    T min_val = std::numeric_limits<T>::max();
                    T max_val = std::numeric_limits<T>::lowest();
                    T sum = 0;
                    
                    for (int i = 0; i < size; i++) {{
                        min_val = std::min(min_val, tensor[i]);
                        max_val = std::max(max_val, tensor[i]);
                        sum += tensor[i];
                    }}
                    
                    std::cout << "  " << name << " - Min: " << min_val << ", Max: " << max_val 
                              << ", Mean: " << (size > 0 ? sum / size : 0) << std::endl;
                    
                    std::cout << "  First 10 values: ";
                    for (int i = 0; i < std::min(10, size); i++) {{
                        std::cout << tensor[i] << " ";
                    }}
                    std::cout << std::endl;
                }}
            }}

            
            #define IS_MIX_AND_MATCH {1 if self.mix_and_match_config else 0}
            #define NUM_LAYERS {total_layers}
            constexpr int LAYER_BITS[NUM_LAYERS] = {{{self._generate_layer_bits_array()}}};
            #define TARGET_BITS {self.target_bits}
            #define STORAGE_BITS 8

            // CNN model parameters
            #define INPUT_HEIGHT {self.input_height}
            #define INPUT_WIDTH {self.input_width}
            #define INPUT_CHANNELS {self.input_channels}
            
            std::vector<float> predict(std::vector<float> &x);
        """
        
        return header
