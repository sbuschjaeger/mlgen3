import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

# from .activation_quantizer import MQ_ActivationQuantizer

# TODO model wrapper or layer wrapper or weight exchanger per layer?
# https://github.com/microsoft/BitBLAS/blob/9c9620678d2c292843a8d91506b573ff3d12eebc/integration/BitNet/modeling_bitnet.py#L1452
# https://github.com/microsoft/BitBLAS/blob/main/integration/BitNet/utils_quant#L104.py


class MQ_ActivationQuantizer(nn.Module):
    """Hook for quantizing activations in MatQuant."""
    def __init__(self, bits, device=None, rounding=True):
        super(MQ_ActivationQuantizer, self).__init__()
        # Store as single value if only one bit-width
        self.bits = bits if isinstance(bits, int) else (bits[0] if len(bits) == 1 else bits)
        self.max_bits = self.bits if isinstance(self.bits, int) else max(self.bits)
        self.device = device if device else torch.device('cpu')
        self.rounding = rounding
    
    def forward(self, x):
        # # Only quantize in training mode
        # if self.training:
        #     return self.quantize_activation(x)
        # return x
        return self.quantize_activation(x)
    
    def quantize_activation(self, x):
        x_min = x.min()
        x_max = x.max()
        
        # Avoid division by zero
        if x_min == x_max:
            return x
            
        # Quantize to max bits (which is self.bits for extracted models)
        scaling_factor = (x_max - x_min) / (2**self.max_bits - 1)
        zero_point = -x_min / scaling_factor if scaling_factor != 0 else 0
        
        # Apply MinMax quantization formula
        quantized_x = torch.clamp(torch.round(x / scaling_factor + zero_point), 0, 2**self.max_bits - 1)
        
        # For now, we only use the max bits version for activations
        # If MatQuant multi-bit activation is needed, we would slice and dequantize for each bit-width here
        
        # Dequantize
        dequantized_x = (quantized_x - zero_point) * scaling_factor
        
        # STE: Use quantized values for forward but gradients from original for backward
        return dequantized_x.detach() + (x - x.detach())


class MatQuant(nn.Module):
    def __init__(self, model, config):
        """
        Initialize Matryoshka Quantization.

        Args:
            model (nn.Module): PyTorch model to quantize.
            config (dict): Dictionary containing the following parameters:
                quantization:
                    - use_matquant (bool): Whether to use MatQuant (default required: True).
                    - use_codistillation (bool): (TODO implem) Whether to use co-distillation from higher to lower precisions (default: False).  
                    - use_qat (bool): Whether to use quantization aware training (default required: False).
                    - fx_mode (bool): Whether to use PyTorch FX Graph Mode for quantization (default required: False).      
                    - target_bits (List[int]): List of bit-widths to optimize for (default: [8, 4, 2]).
                    - loss_weights (Dict[int, float]): Dictionary mapping bit-widths to loss weights (default: {8: 0.4, 4: 0.4, 2: 1.0}).
                    - quantize_layers (List[str]): List of layer names to quantize (default: all FFN layers).
                    - quantize_target (str): What to quantize (weights_only, activations_only, weights_and_activations)
                    - quantize_bias (bool): Whether to also quantize bias terms (default: False)
        """

        super(MatQuant, self).__init__()
        self.model = model
        self.device = next(model.parameters()).device
        
        # Configuration parameters
        self.target_bits = sorted(config['quantization']['target_bits'], reverse=True)
        self.loss_weights = {k: v for k, v in sorted(config['quantization']['loss_weights'].items(), reverse=True)}
        self.quantize_target = config['quantization'].get('quantize_target', 'weights_only')
        self.quantize_bias = config['quantization'].get('quantize_bias', False)
        
        # Assert loss_weights keys are in target_bits
        invalid_bits = [bit for bit in self.loss_weights.keys() if bit not in self.target_bits]
        if invalid_bits:
            raise ValueError(f"Invalid bit-widths in loss_weights: {invalid_bits}. Must be one of {self.target_bits}")
        
        # Assert loss_weights values are between 0 and 1
        invalid_weights = [(bit, weight) for bit, weight in self.loss_weights.items() if not (0 <= weight <= 1)]
        if invalid_weights:
            raise ValueError(f"Invalid loss weights: {invalid_weights}. Values must be between 0 and 1")
        
        self.quantize_layers = sorted(config['quantization']['quantize_layers'])
        self.use_codistillation = config['quantization']['use_codistillation']

        # Set max bit-width (for training)
        self.max_train_bits = max(self.target_bits)

        # layer_paths and quantized_layers are set using set_quantized_layers after defining layer_registry
        self.layer_paths = None
        self.quantized_layers = None
        self.activation_hooks = []

    def reset_quantized_layers(self):
        """
        Reset the quantized layers to None.
        This is useful if you want to re-register the model with new layer paths.
        """
        self.layer_paths = None
        self.quantized_layers = None
        
        # Remove existing activation hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def set_quantized_layers(self, layer_paths, quantize_target=None):
        """
        Set the quantized layers for the model.

        Args:
            layer_paths (List[str]): List of layer paths to quantize.
            quantize_target (str, optional): What to quantize. Defaults to None (use instance value).
        """
        self.layer_paths = layer_paths
        if quantize_target:
            self.quantize_target = quantize_target
            
        self.quantized_layers = self.get_quantizable_params()
        
        # Setup activation quantization if needed
        if self.quantize_target in ['activations_only', 'weights_and_activations']:
            self._register_activation_hooks()

    def _register_activation_hooks(self):
        """Register hooks for activation quantization."""
        # Remove any existing hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        # Helper function to get layer from path, handling nested models correctly
        def get_layer_from_path(model, path):
            """Helper to get a layer from a path string."""
            parts = path.split('.')
            
            # Remove 'model.' prefix if it exists since we're already starting with self.model
            if parts[0] == 'model':
                parts = parts[1:]
            
            current = model
            
            # Handle both VGG-style (model.model.Sequential) and MLP-style (model.Sequential) structures
            if hasattr(current, 'model') and isinstance(current.model, nn.Sequential):
                current = current.model
            
            for part in parts:
                if part == "weight" or part == "bias":
                    # Skip weight/bias attributes as we're looking for the module
                    continue
                    
                if part.isdigit():
                    # Access by index in Sequential container
                    try:
                        current = current[int(part)]
                    except (TypeError, IndexError) as e:
                        raise AttributeError(f"Cannot access index {part} in {type(current)}: {e}")
                else:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        raise AttributeError(f"Cannot find attribute {part} in {type(current)}")
            
            return current
        
        # Register new hooks
        for path in self.layer_paths:
            # Get the base layer path (without .weight)
            layer_path = path.replace('.weight', '') if path.endswith('.weight') else path
            
            try:
                # Find the layer
                layer = get_layer_from_path(self.model, layer_path)
                
                # Only register hooks for layers that have activations (e.g., Conv2d, Linear)
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Hardtanh)):
                    # Register the activation quantizer
                    quantizer = MQ_ActivationQuantizer(self.target_bits, self.device)
                    hook = layer.register_forward_hook(lambda module, input_val, output: quantizer(output))
                    self.activation_hooks.append(hook)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register activation hook for layer {layer_path}: {e}")
                # Print detailed model structure to help debug
                if self.model is not None:
                    print(f"Model structure: {type(self.model).__name__}")
                    if hasattr(self.model, "model"):
                        print(f"Inner model structure: {type(self.model.model).__name__}")
                        # Print first few layers of the model for debugging
                        if isinstance(self.model.model, nn.Sequential):
                            for i, layer in enumerate(list(self.model.model)[:3]):
                                print(f"  Layer {i}: {type(layer).__name__}")

    def append_quantized_layers(self, layer_paths):
        """
        Append new quantized layers to the existing ones.

        Args:
            layer_paths (List[str]): List of additional layer paths to quantize.
        """
        if self.layer_paths is None:
            self.layer_paths = []
        
        self.layer_paths.extend(layer_paths)
        self.quantized_layers = self.get_quantizable_params()
        
        # Update activation hooks if needed
        if self.quantize_target in ['activations_only', 'weights_and_activations']:
            self._register_activation_hooks()

    def print_quantized_layers(self):
        """
        Print the currently set quantized layers.
        """
        print(f"Layer paths: {self.layer_paths}")
        print(f"Quantization target: {self.quantize_target}")

    def get_quantizable_params(self, model=None):
        """
        Get the quantizable parameters (weights and/or activations) from the given model or self.model
        The indices for the layers to be quantized can be specified before training in the config file.

        Args:
            model (torch.nn.Module): model to get parameters from (defaults to self.model).

        Returns:
            params (List[Tuple[str, torch.nn.Parameter]]): List of (name, parameter) pairs for quantizable parameters.
        """

        if model is None:
            model = self.model

        if self.layer_paths is None:
            raise ValueError("Layer paths are not set. Please set them using model.set_quantized_layers(layer_paths) before calling this method. " \
            " Note that the layer paths are set in the register_model function of the LayerRegistry class," \
            " therefore you need to call register_model(model) after creating the model." \
            " This should be handled automatically by the init_model() function.")

        # If only activations should be quantized, return empty list for weights
        if self.quantize_target == 'activations_only':
            return []

        params = []
        for name, param in model.named_parameters():
            # Add weight parameters if they match the layer paths
            if 'weight' in name:
                matching_paths = [path for path in self.layer_paths if name in path]
                if matching_paths:
                    params.append((name, param))
            
            # Add bias parameters if bias quantization is enabled
            elif self.quantize_bias and 'bias' in name:
                # Get the corresponding weight name to check if this bias should be quantized
                weight_name = name.replace('bias', 'weight')
                weight_matching_paths = [path for path in self.layer_paths if weight_name in path]
                
                if weight_matching_paths:
                    params.append((name, param))
            
        return params

    
    def quantize(self, w, c):
        """
        Quantize a tensor to the specified bit-width using MinMaxQuantization.

        Args:
            w (torch.Tensor): Weight tensor to quantize.
            c (int): Number of bits to quantize to.

        Returns:
            quantized_w (torch.Tensor): Quantized tensor in the original floating-point format.

            scaling_factor (float): Scaling factor (alpha) used for quantization.
            
            zero_point (float): Zero point (z) used for quantization.
        """

        # Determine scaling factor (alpha) based on min and max values (MinMax Quantization)
        w_min = w.min()
        w_max = w.max()
        scaling_factor = (w_max - w_min) / (2**c - 1)
        zero_point = -w_min / scaling_factor if scaling_factor != 0 else 0
        
        # Quantize the weights
        quantized_w = torch.clamp(torch.round(w / scaling_factor + zero_point), 0, 2**c - 1)

        # Dequantize separately, after slicing
                
        return quantized_w, scaling_factor, zero_point
    
    def straight_through_estimator(self, quantized_w, original_w):
        """
        STE: Use quantized weights for forward pass but original gradients for backward pass.

        - quantized_w.detach() creates a copy of quantized_w that is detached from the computational graph, 
        meaning gradients won't flow through it during backpropagation.
        - original_w is the original weight tensor (with gradient tracking).
        - original_w.detach() is the same tensor but without gradient tracking.
        => Their difference equals zero numerically, but maintains gradient flow from original_w.

        When we add these together, we get a tensor that:
        - Has the forward pass values of quantized_w (the quantized weights).
        - Has the backward pass gradients of original_w (the original weights).

        This is the STE trick - during forward pass we use the quantized (typically discrete) values, 
        but during backpropagation we pretend the function was an identity function, 
        allowing gradients to flow as if quantization didn't happen.

        This is necessary because quantization operations typically have zero gradients almost everywhere, 
        which would stop gradient-based training.
        
        Args:
            quantized_w (torch.Tensor): Quantized weights.
            original_w (torch.Tensor): Original weights.
            
        Returns:
            Modified weights with STE applied.
        """
        
        return quantized_w.detach() + (original_w - original_w.detach())
        # Equivalent to: original_w + (quantized_w - original_w).detach()?
    
    def slice_bits(self, x_int, original_bits, target_bits, rounding=True):
        """
        This function performs bit slicing to reduce the precision of integer tensors.
        When rounding is enabled, it uses the algorithm described in Appendix A of the MatQuant paper,
        where the bit at position target_bits+1 is used as the round bit.

        Args:
            x_int (torch.Tensor): Integer tensor quantized to original_bits.
            original_bits (int): Original bit-width of x_int.
            target_bits (int): Target bit-width to slice x_int to.
            rounding (bool): If True, apply rounding based on the bit at position target_bits+1;
                    otherwise, simply truncate the bits (default: True).
            prnt (bool): If True, print debug information during the slicing process (default: False).

        Returns:
            x_sliced (torch.Tensor): Sliced integer tensor scaled back to the original range.
        """
            
        # Perform right shift followed by left shift to extract MSBs
        shift_bits = original_bits - target_bits

        # rounding according to Appendix A Matquant paper
        if rounding:
            # Convert to integers for bit operations and extract round bit
            x_int_cast = x_int.long()
            # Calculate the position of the round bit (at position target_bits+1
            # -> the appropriate bit position is shift_bits-1 because of the subsequent operation in calculating round_bit)
            round_bit_pos = shift_bits - 1
            # Check if the bit at position round_bit_pos is set
            round_bit = ((x_int_cast // (2**round_bit_pos)) % 2).bool()

            # Use the round bit to decide between floor (round down) and ceil (round up)
            x_floor = torch.floor(x_int / (2**shift_bits))
            x_ceil = x_floor + 1
            x_sliced = torch.where(round_bit, x_ceil, x_floor)

        else:
            # Perform right shift followed by left shift to extract MSBs
            x_sliced = torch.floor(x_int / (2**shift_bits))

        # Clamp to ensure values are within the target bit-width range
        x_sliced = torch.clamp(x_sliced, 0, 2**target_bits - 1)

        # Scale back to original range (same as right shift)
        x_sliced = x_sliced * (2**shift_bits)
        
        return x_sliced
    
    def dequantize(self, quantized_w, scaling_factor, zero_point):
        """
        Dequantize an integer tensor back to floating point.

        Args:
            quantized_w (torch.Tensor): Quantized weights.
            scaling_factor (float): Scaling factor (alpha) used for quantization.
            zero_point (int): Zero point (z) used for quantization.

        Returns:
            Floating point tensor.
        """

        return (quantized_w - zero_point) * scaling_factor
    
    def forward_with_quant(self, x, target_bits=None, rounding=True):
        """
        Forward pass with quantization at the specified target bit-width
        Args:
            x (torch.Tensor): Input tensor.
            target_bits (int): Bit-width to quantize to (if None, use max_train_bits).
            rounding (bool): Whether to use rounding during bit slicing.
        Returns:
            output (torch.Tensor): Model output.
        """

        if target_bits is None:
            target_bits = self.max_train_bits
        
        # Handle weight quantization if needed
        if self.quantize_target in ['weights_only', 'weights_and_activations']:
            # Save original weights for restoration later
            original_weights = {}

            # Quantize weights for target layers
            for name, param in self.get_quantizable_params():
                # Save original weights
                original_weights[name] = param.data.clone()

                # Quantize to max_train_bits first
                quantized_w, scaling_factor, zero_point = self.quantize(param.data, self.max_train_bits)

                # If we need a lower precision, slice the bits
                if target_bits < self.max_train_bits:
                    sliced_w = self.slice_bits(quantized_w, self.max_train_bits, target_bits, rounding)
                    dequantized_w = self.dequantize(sliced_w, scaling_factor, zero_point)
                else:
                    dequantized_w = self.dequantize(quantized_w, scaling_factor, zero_point)

                # Replace the weight with (de)quantized version using STE
                param.data = self.straight_through_estimator(dequantized_w, param.data)

        # Note: Activation quantization is handled automatically through forward hooks
        # registered in _register_activation_hooks(). The hooks apply MQ_ActivationQuantizer
        # to layer outputs during the forward pass.
        
        # Forward pass with quantized weights and/or activations
        # Activation quantization is applied automatically by registered hooks
        output = self.model(x)

        # Restore original weights if we modified them
        if self.quantize_target in ['weights_only', 'weights_and_activations'] and original_weights:
            for name, param in self.get_quantizable_params():
                param.data = original_weights[name]

        return output
    
    def multi_precision_forward(self, x, rounding=True):
        """
        Forward pass with all target bit-widths for training
        Args:
            x: Input tensor
        Returns:
            Dictionary mapping bit-widths to model outputs trained on their respective bit-width
        """

        outputs = {}
        for bits in self.target_bits:   # already sorted in descending order, i.e. max_train_bits first
            output = self.forward_with_quant(x, bits, rounding)
            outputs[bits] = output

        return outputs
    
    def forward(self, x, rounding=True):
        """
        Standard forward pass.

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            output (torch.Tensor): Model output
        """

        output = self.forward_with_quant(x, self.max_train_bits, rounding)
        return output
    
    def matquant_loss(self, output, targets):
        """
        Calculate weighted loss across all bit-widths. loss_weights are defined in the respective config file.
        The loss is calculated for each bit-width and then weighted by the corresponding loss weight (λ_r).

        Args:
            output (torch.Tensor): (max_train_bits) 8-bit model outputs.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            total_loss (float): Weighted sum of losses.

            losses (Dict[int, float]): Dictionary mapping bit-widths to individual losses.
        """

        total_loss = 0
        losses = {}
        
        # Calculate loss for each bit-width
        for bit_width in self.target_bits:
            losses[bit_width] = F.cross_entropy(output[bit_width], targets)
        
        # Calculate total loss with loss weights for each bit-width (λ_r)
        for bit_width, loss in losses.items():
            total_loss += self.loss_weights.get(bit_width, 1.0) * loss
        
        
        # # Add co-distillation loss if enabled
        # if self.use_codistillation and len(self.target_bits) > 1:
        #     # Use outputs from highest precision as teacher for lower precisions
        #     highest_bits = max(self.target_bits)
        #     teacher_output = outputs[highest_bits]
            
        #     for bits in sorted(self.target_bits)[:-1]:  # Exclude the highest precision
        #         student_output = outputs[bits]
        #         distill_loss = F.kl_div(
        #             F.log_softmax(student_output / 2.0, dim=-1),
        #             F.softmax(teacher_output / 2.0, dim=-1),
        #             reduction='batchmean'
        #         ) * (2.0 ** 2)
                
        #         total_loss += distill_loss * self.loss_weights.get(bits, 1.0)
        
        return total_loss, losses

    
    def extract_model(self, target_bits, rounding=True):
        """
        Extract a model at the specified precision target_bits.
        
        Args:
            target_bits (int): Bit-width to extract the model at.
            rounding (bool): Whether to use rounding during bit slicing.
            
        Returns:
            extracted_model (nn.Module): Model quantized to target_bits precision.
        """
        # Create a copy of the model
        extracted_model = deepcopy(self.model)
        
        # Handle weight quantization if needed
        if self.quantize_target in ['weights_only', 'weights_and_activations']:
            # Quantize weights to max_train_bits first
            for name, param in self.get_quantizable_params(extracted_model):

                with torch.no_grad():
                    # Quantize to 8-bit
                    quantized_w, scaling_factor, zero_point = self.quantize(param.data, self.max_train_bits)

                    # Slice to target bits
                    if target_bits < self.max_train_bits:
                        quantized_w = self.slice_bits(quantized_w, self.max_train_bits, target_bits, rounding)
                    
                    # Dequantize
                    param.data = self.dequantize(quantized_w, scaling_factor, zero_point)

        return extracted_model
    
    def mix_and_match(self, bit_config, rounding=True):
        """
        Create a mix and match model with different precisions for different layers.
        This can be specified in the corresponding config file under evaluation:mix_and_match.

        Args:
            bit_config (dict): Dictionary mapping layer names to bit-widths.
            rounding (bool): If True, apply rounding based on the bit at position target_bits+1;
                    otherwise, simply truncate the bits (default: True).
                
        Returns:
            mixed_model (nn.Module): Model with mixed precision quantization.
        """

        # Create a copy of the model
        mixed_model = deepcopy(self.model)
        
        # Skip weight quantization if only activations should be quantized
        if self.quantize_target in ['weights_only', 'weights_and_activations']:
            quantizable_params = self.get_quantizable_params(mixed_model)

            # Check if all specified layers exist in the model
            available_layers = {name for name, _ in quantizable_params}
            invalid_layers = [layer for layer in bit_config if layer not in available_layers]
            if invalid_layers:
                raise ValueError(f"Invalid layer names specified: {invalid_layers}. "
                    f"Available layers: {list(available_layers)}")

            # Validate the bit_config and ensure all specified bit-widths are valid
            if not bit_config:
                raise ValueError("bit_config dictionary cannot be empty")
            
            # Apply different quantization to each layer based on config
            for name, param in quantizable_params:
                # For bias parameters, find the corresponding weight and use its bit-width
                if 'bias' in name and self.quantize_bias:
                    weight_name = name.replace('bias', 'weight')
                    bits = bit_config.get(weight_name, self.max_train_bits)
                else:
                    bits = bit_config.get(name, self.max_train_bits)  # Default to max bits if not specified
                
                # Quantize to max_config_bits first
                quantized_w, sf, zp = self.quantize(param.data, self.max_train_bits)
                
                # Slice to target bits
                if bits < self.max_train_bits:
                    quantized_w = self.slice_bits(quantized_w, self.max_train_bits, bits, rounding)
                    
                # Dequantize to get the final weights
                param.data = self.dequantize(quantized_w, sf, zp)
                
        return mixed_model

