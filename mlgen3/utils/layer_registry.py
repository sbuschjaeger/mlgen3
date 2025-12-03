import torch.nn as nn

class LayerRegistry:
    def __init__(self, model, config):
        self.layers = {}  # Maps paths to layers
        self.flat_map = {}  # Maps flat indices to paths
        self.quantizable_params = []  # List of layer paths selected for quantization
        self.quantizable_activations = set()  # Set of layer paths with quantizable activations
        self.quantizable_weights = set()  # Set of layer paths with quantizable weights
        self.quantizable_biases = set()  # Set of layer paths with quantizable biases
        self.skip_connections = []  # Track skip connection addition points
        self.avgpool_layers = []  # Track global average pooling layers
        self.flat_index = 0

        self.quantized_params_list = config['quantization']['quantized_params_list']
        self.quantize_bias = config['quantization'].get('quantize_bias', True)
        self.quantize_weight = config['quantization'].get('quantize_weight', True)
        self.quantize_activation = config['quantization'].get('quantize_activation', True)

        self.model_type = self._detect_model_type(model)
        self.register_model(model)


    def _detect_model_type(self, model):
        """Detect if model is HuggingFace, TIMM, or custom."""
        model_class = model.__class__.__name__
        
        if 'ForImageClassification' in model_class:
            return 'huggingface'
        elif hasattr(model, 'forward_features'):  # TIMM models
            return 'timm'
        else:
            return 'custom'
    
    def register_model(self, model, parent_path=""):
        """Register a model and its layers for quantization."""
        if not isinstance(model, nn.Module):
            raise ValueError("The model must be an instance of nn.Module.")
        
        # Register the model recursively
        self.register_model_recursively(model, parent_path)
        
        # Empty array means no layers to quantize
        # Handle empty list or list with empty strings
        if len(self.quantized_params_list) == 0 or (len(self.quantized_params_list) == 1 and self.quantized_params_list[0] == ""):
            self.quantizable_params = []
        # If -1 is in the array, quantize all layers
        elif -1 in self.quantized_params_list or "all" in self.quantized_params_list:
            self.quantizable_params = self.get_layers_by_indices(None)
        # Handle both index-based and path-based specifications
        elif all(isinstance(x, int) for x in self.quantized_params_list):
            # Index-based selection
            self.quantizable_params = self.get_layers_by_indices(self.quantized_params_list)
        else:
            # Path-based selection
            self.quantizable_params = self.get_layers_by_paths(self.quantized_params_list)

        # Only call set_quantization if the model has this method (i.e., for quantized models)
        if hasattr(model, 'set_quantization'):
            model.set_quantization(self.quantizable_params, self.quantizable_activations)
    
    def register_model_recursively(self, model, parent_path=""):
        """Recursively register all layers in the model."""
        if isinstance(model, nn.Sequential):
            for i, layer in enumerate(model):
                current_path = f"{parent_path}.{i}" if parent_path else f"{i}"
                self._register_layer(layer, current_path)
                self.register_model_recursively(layer, current_path)
        elif hasattr(model, "_modules"):
            for name, layer in model._modules.items():
                if layer is not None:
                    current_path = f"{parent_path}.{name}" if parent_path else f"{name}"
                    self._register_layer(layer, current_path)
                    self.register_model_recursively(layer, current_path)
    
    def _register_layer(self, layer, path):
        """Register a single layer if it's quantizable."""

        # Register layers for activation quantization
        if self.quantize_activation:
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Hardtanh)):
                self.quantizable_activations.add(path)
            
            # Track Global Average Pooling layers
            # These operate on quantized INT8 activations and produce quantized outputs
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                self.avgpool_layers.append(path)
                # Also add to quantizable activations for tracking
                self.quantizable_activations.add(path)
            
            # Track skip connection points for HuggingFace models
            # These require special handling for scale matching
            layer_class = layer.__class__.__name__
            if 'BasicLayer' in layer_class or 'BottleneckLayer' in layer_class:
                self.skip_connections.append(path)
            
            # For TIMM models
            if 'BasicBlock' in layer_class or 'Bottleneck' in layer_class:
                self.skip_connections.append(path)
        
        # Register the layers for weight quantization
        if self.quantize_weight:
            weight_path = path + ".weight" if hasattr(layer, 'weight') else path
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self.layers[weight_path] = layer
                self.quantizable_weights.add(weight_path)
                self.flat_map[self.flat_index] = weight_path
                self.flat_index += 1

        # Register the layer bias (only if bias exists)
        if self.quantize_bias:
            # Check if layer has bias attribute and if it's not None
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_path = path + ".bias"
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    self.layers[bias_path] = layer
                    self.quantizable_biases.add(bias_path)
                    self.flat_map[self.flat_index] = bias_path
                    self.flat_index += 1
    
    def get_layers_by_indices(self, indices):
        """Convert flat indices to layer paths."""
        
        if indices is None or len(indices) == 0:
            return list(self.layers.keys())
        
        valid_paths = []
        for idx in indices:
            if idx in self.flat_map:
                valid_paths.append(self.flat_map[idx])
            else:
                self.print_layer_info()
                raise ValueError(f"Index {idx} does not correspond to a valid quantizable layer. Available indices: {sorted(self.flat_map.keys())}. See full list of quantizable layers above.")
        
        return valid_paths
    
    def get_layers_by_paths(self, paths):
        """Filter and validate layer paths."""

        # Filter out empty strings
        paths = [path for path in paths if path.strip()]
        if len(paths) == 0:
            return list(self.layers.keys())
        
        valid_paths = []
        for path in paths:
            # Check for exact match first
            if path in self.layers.keys():
                valid_paths.append(path)
            else:
                # Check for paths that end with the specified path (common prefix issue)
                matching_paths = [qpath for qpath in self.layers.keys() if qpath.endswith(path)]
                valid_paths.extend(matching_paths)
                if len(matching_paths) == 0:
                    self.print_layer_info()
                    raise ValueError(f"Layer '{path}' is not quantizable. Available layers: {self.layers.keys()}. See full list of quantizable layers above.")

        return valid_paths
    
    def print_layer_info(self):
        """Print information about registered layers for debugging."""

        print("\n--- Quantizable Layers ---")
        print("\n[id] [model.path]: [layer_type] [parameters] [size]")
        for idx, path in sorted(self.flat_map.items()):
            layer = self.layers[path]
            num_params = sum(p.numel() for p in layer.parameters())
            
            # Get size information
            size_info = "N/A"
            if hasattr(layer, 'weight') and layer.weight is not None:
                size_info = str(list(layer.weight.shape))
                
            print(f"[{idx}] {path}: {layer.__class__.__name__}, params: {num_params}, size: {size_info}")
        
        if self.skip_connections:
            print("\n--- Skip Connections ---")
            print("Layers with skip connections (require scale matching):")
            for path in self.skip_connections:
                print(f"  {path}")
        
        if self.avgpool_layers:
            print("\n--- Global Average Pooling ---")
            print("Layers performing global average pooling (operate on INT8, produce INT8):")
            for path in self.avgpool_layers:
                print(f"  {path}")
        
        print("\n--------------------------")
