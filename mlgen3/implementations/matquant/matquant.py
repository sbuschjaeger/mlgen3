import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

# from ..utils.metrics import print_tensor_binary
# from ..utils.batchnorm_folding import fold_model_batchnorm
from .activation_quantizer import MQ_ActivationQuantizer
from .input_quantizer import InputQuantizer


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
                    - quantized_params_list (List[str]): List of layer names to quantize (default: all FFN layers).
                    - quantize_bias (bool): Whether to also quantize bias terms (default: False)
                    - quantize_weight (bool): Whether to quantize weights (default: True)
                    - quantize_activation (bool): Whether to quantize activations (default: True)
                    - quantize_signed (bool): Whether to use signed quantization (default: True)
        """

        super(MatQuant, self).__init__()
        self.model = model
        self.device = next(model.parameters()).device
        
        # Configuration parameters
        self.target_bits = sorted(config['quantization']['target_bits'], reverse=True)
        self.loss_weights = {k: v for k, v in sorted(config['quantization']['loss_weights'].items(), reverse=True)}
        self.quantize_bias = config['quantization'].get('quantize_bias', True)
        self.quantize_weight = config['quantization'].get('quantize_weight', True)
        self.quantize_activation = config['quantization'].get('quantize_activation', True)
        self.quantize_signed = config['quantization'].get('quantize_signed', True)
        self.quantized_params_list = sorted(config['quantization']['quantized_params_list'])
        self.use_codistillation = config['quantization']['use_codistillation']

        # Add input quantization if enabled and not already in the base model
        self.use_input_quantization = config['quantization'].get('quantize_input', True)
        if self.use_input_quantization and not hasattr(self.model, 'use_input_quantization'):
            # Determine input range based on model type
            input_range = (-3.0, 3.0)  # Default range
            self.input_quantizer = InputQuantizer(input_range=input_range, signed=True)

        # Set max bit-width (for training)
        self.max_train_bits = max(self.target_bits)

        # These are set using set_quantization() after defining layer_registry
        self.quantizable_params = None
        self.quantizable_activations = None
        self.quantized_params = None
        self.activation_hooks = []
        self.skip_connection_paths = []  # Track skip connection paths for scale matching

        # Add flag for BatchNorm folding
        self.fold_bn_inference = config['quantization'].get('fold_bn_inference', True)
        
        # Add flag for dry iterations (full precision warm-up)
        self.dry_mode = False

    def set_dry_mode(self, dry_mode):
        """
        Set dry mode for MatQuant model.

        Args:
            dry_mode (bool): If True, the model will run in full precision mode without quantization.
        """
        self.dry_mode = dry_mode
        # Only set dry mode on underlying model if it has the method (custom models)
        if hasattr(self.model, 'set_dry_mode'):
            self.model.set_dry_mode(dry_mode)
        else:
            print("Warning: Underlying model does not have set_dry_mode method.")

    # Helper function to get layer from path, handling nested models correctly
    def get_layer_from_path(self, path, model=None):
        """Helper to get a layer from a path string."""

        if model is None:
            model = self.model

        parts = path.split('.')
        
        # Remove 'model.' prefix if it exists since we're already starting with self.model
        if parts[0] == 'model':
            parts = parts[1:]
        
        current = model
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def reset_quantization(self):
        """
        Reset the quantized layers to None.
        This is useful if you want to re-register the model with new layer paths.
        """
        self.quantizable_params = None
        self.quantizable_activations = None
        self.quantized_params = None
        
        # Remove existing activation hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def set_quantization(self, quantizable_params, quantizable_activations):
        """
        Set the quantized layers for the model.

        Args:
            quantizable_params (List[str]): List of layer paths to quantize.
            quantizable_activations (Set[str]): Set of layer paths with quantizable activations.
        """
        self.quantizable_params = quantizable_params
        self.quantizable_activations = quantizable_activations

        self.quantized_params = self.get_quantizable_params()
        
        # Setup activation quantization if needed
        if self.quantize_activation:
            self._register_activation_hooks()
    
    def _register_activation_hooks(self):
        """Register hooks for activation quantization."""

        # Remove any existing hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        # Register new hooks
        for path in self.quantizable_activations:
            try:
                # Find the layer
                layer = self.get_layer_from_path(path)
                
                # Register hook if layer is of a type that we want to quantize activations for
                # Global Average Pooling (AdaptiveAvgPool2d) operates on INT8 activations
                # and produces INT8 outputs (using INT32 accumulator internally)
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Hardtanh, nn.AdaptiveAvgPool2d)):
                    quantizer = MQ_ActivationQuantizer(self.target_bits, self.device, self.quantize_signed)
                    hook = layer.register_forward_hook(lambda module, input_val, output: quantizer(output))
                    self.activation_hooks.append(hook)
                
                # Track skip connection points for scale matching
                from ..models.resnet import BasicBlock, BottleneckBlock
                if isinstance(layer, (BasicBlock, BottleneckBlock)):
                    self.skip_connection_paths.append(path)
                    
            except (AttributeError, IndexError) as e:
                print(f"1-Warning: Could not register activation hook for layer {path}: {e}")
                # Print detailed model structure to help debug
                if self.model is not None:
                    print(f"Model structure: {type(self.model).__name__}")
                    if hasattr(self.model, "model"):
                        print(f"Inner model structure: {type(self.model.model).__name__}")
                        # Print first few layers of the model for debugging
                        if isinstance(self.model.model, nn.Sequential):
                            for i, layer in enumerate(list(self.model.model)[:3]):
                                print(f"  Layer {i}: {type(layer).__name__}")

    def print_quantization(self):
        """
        Print the currently set quantized layers.
        """
        print(f"Quantize weight: {self.quantize_weight}, bias: {self.quantize_bias}, activation: {self.quantize_activation}")
        print(f"Quantizable params: {self.quantizable_params}")
        print(f"Quantizable activations: {self.quantizable_activations}")

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

        if self.quantized_params is not None:
            return self.quantized_params

        if self.quantizable_params is None:
            raise ValueError("Layer paths are not set. Please set them using model.set_quantization(quantizable_params, quantizable_activations) before calling this method. " \
            " Note that the layer paths are set in the register_model function of the LayerRegistry class," \
            " therefore you need to call register_model(model) after creating the model." \
            " This should be handled automatically by the init_model() function.")

        # If only activations should be quantized, return empty list for weights
        if self.quantize_activation and not (self.quantize_weight or self.quantize_bias):
            return []

        params = []
        for name, param in model.named_parameters():
            matching_paths = [path for path in self.quantizable_params if name in path]
            if matching_paths:
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
        
        if self.quantize_signed:
            # print(f"    | signed quantization")
            # Signed quantization: range is [-2^(c-1), 2^(c-1) - 1]
            q_min = -(2**(c-1))
            q_max = 2**(c-1) - 1
        else:
            # print(f"    | unsigned quantization")
            # Unsigned quantization: range is [0, 2^c - 1]
            q_min = 0
            q_max = 2**c - 1

        scaling_factor = (w_max - w_min) / (q_max - q_min)
        zero_point = -w_min / scaling_factor + q_min if scaling_factor != 0 else q_min
        
        # Quantize the weights
        quantized_w = torch.clamp(torch.round(w / scaling_factor + zero_point), q_min, q_max)

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
    
    def slice_bits(self, x_int, original_bits, target_bits, prnt, rounding=True):
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

            if prnt:
                print("---- Bit Slicing with Rounding ----")
                print(f"Original bits: {original_bits}, Target bits: {target_bits}")
                print(f"Rounding: shift_bits={shift_bits}, round_bit_pos={round_bit_pos}")
                print(f"x_int sample: {x_int.view(-1)[:5].cpu().numpy()}")
                print(f"round bits: {round_bit.view(-1)[:5].cpu().numpy()}")

            # Use the round bit to decide between floor (round down) and ceil (round up)
            x_floor = torch.floor(x_int / (2**shift_bits))
            x_ceil = x_floor + 1
            x_sliced = torch.where(round_bit, x_ceil, x_floor)
            
            if prnt:
                print(f"x_floor sample: {x_floor.view(-1)[:5].cpu().numpy()}")
                print(f"x_ceil sample: {x_ceil.view(-1)[:5].cpu().numpy()}")
                print(f"x_sliced sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
                print("++++++++++++++++++++++++++++++++++++")

        else:
            # Perform right shift followed by left shift to extract MSBs
            x_sliced = torch.floor(x_int / (2**shift_bits))

        # Clamp to ensure values are within the target bit-width range
        if self.quantize_signed:
            q_min = -(2**(target_bits-1))
            q_max = 2**(target_bits-1) - 1
        else:
            q_min = 0
            q_max = 2**target_bits - 1
            
        if prnt:
            print(f"Clamping sliced values to range [{q_min}, {q_max}]")
            print(f"x_sliced before clamping sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
        x_sliced = torch.clamp(x_sliced, q_min, q_max)
        if prnt:
            print(f"x_sliced after clamping sample: {x_sliced.view(-1)[:5].cpu().numpy()}")

        # Scale back to original range (same as right shift)
        x_sliced = x_sliced * (2**shift_bits)
        if prnt:
            print(f"x_sliced after scaling back sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
            print("-----------------------------------")
        
        return x_sliced
    
    def slice_bits_lsb(self, x_int, original_bits, target_bits, prnt, rounding=True):
        """
        This function performs bit slicing to reduce the precision of integer tensors by
        keeping the least significant bits (LSBs) instead of the most significant bits.
        When rounding is enabled, it uses rounding based on the bit at the boundary.

        Args:
            x_int (torch.Tensor): Integer tensor quantized to original_bits.
            original_bits (int): Original bit-width of x_int.
            target_bits (int): Target bit-width to slice x_int to.
            rounding (bool): If True, apply rounding based on the boundary bit;
                    otherwise, simply truncate the bits (default: True).
            prnt (bool): If True, print debug information during the slicing process (default: False).

        Returns:
            x_sliced (torch.Tensor): Sliced integer tensor with LSBs extracted.
        """
            
        # # Number of bits to discard from the MSB side
        # shift_bits = original_bits - target_bits

        # Create mask to extract LSBs
        lsb_mask = (2**target_bits) - 1

        if prnt:
            print("---- Bit Slicing LSBs ----")
            print(f"Original bits: {original_bits}, Target bits: {target_bits}")
            print(f"LSB mask: {bin(lsb_mask)}")
            print(f"x_int sample: {x_int.view(-1)[:5].cpu().numpy()}")
            print(f"binary representation: {self.get_binary_representation(x_int.view(-1)[:5].cpu().numpy(), original_bits, 'msb')}")

        # Extract LSBs using bitwise AND
        x_int_cast = x_int.long()
        x_sliced = x_int_cast & lsb_mask

        if prnt:
            print(f"x_int_cast sample: {x_int_cast.view(-1)[:5].cpu().numpy()}")
            print(f"binary representation: {self.get_binary_representation(x_int_cast.view(-1)[:5].cpu().numpy(), original_bits, 'msb')}")
        
        if prnt:
            print(f"x_sliced1 (LSBs) sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
            print(f"binary representation: {self.get_binary_representation(x_sliced.view(-1)[:5].cpu().numpy(), target_bits, 'lsb')}")
        
        # Convert from unsigned to signed representation if needed
        if self.quantize_signed:
            # Check if MSB is set (indicating negative number in two's complement)
            sign_bit = 2**(target_bits - 1)
            # If MSB is set, convert to negative by subtracting 2^target_bits
            x_sliced = torch.where(x_sliced >= sign_bit, x_sliced - (2**target_bits), x_sliced)

            if prnt:
                print(f"x_sliced2 (LSBs) sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
                print(f"binary representation: {self.get_binary_representation(x_sliced.view(-1)[:5].cpu().numpy(), target_bits, 'lsb')}")
        
            q_min = -(2**(target_bits-1))
            q_max = 2**(target_bits-1) - 1
        else:
            q_min = 0
            q_max = 2**target_bits - 1
            
        if prnt:
            print(f"Clamping sliced values to range [{q_min}, {q_max}]")
            print(f"x_sliced before clamping sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
            print(f"binary representation: {self.get_binary_representation(x_sliced.view(-1)[:5].cpu().numpy(), target_bits, 'lsb')}")
        
        x_sliced = torch.clamp(x_sliced, q_min, q_max)
        
        if prnt:
            print(f"x_sliced after clamping sample: {x_sliced.view(-1)[:5].cpu().numpy()}")
            print(f"binary representation: {self.get_binary_representation(x_sliced.view(-1)[:5].cpu().numpy(), target_bits, 'lsb')}")
            print("-----------------------------------")
        
        return x_sliced #.float()
    
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
        Returns:
            output (torch.Tensor): Model output.
        """

        if target_bits is None:
            target_bits = self.max_train_bits

        # Quantize input if enabled and not already quantized by base model
        if self.use_input_quantization and not hasattr(self.model, 'use_input_quantization'):
            # Skip input quantization during dry iterations
            if not self.dry_mode:
                x = self.input_quantizer(x)

        bias_bits = 8

        print_bits = 20
        sb = 'msb'
        # sb = 'lsb'
        prnt = False   
        ext = False
        
        # if print_bits == target_bits:
        #     ext = True
        # if target_bits == 8:
        #     ext = True
            
        # Skip weight quantization if only activations should be quantized OR if in dry mode
        if (self.quantize_weight or self.quantize_bias) and not self.dry_mode:
            # Save original weights for restoration later
            original_params = {}

            # Quantize weights for target layers
            for name, param in self.quantized_params:

                # print(f"\nProcessing Q layer: {name}")

                if print_bits <= target_bits and 'stages.0.layers.0.layer.0' in name:
                    prnt = True
                else:
                    prnt = False

                # Save original weights
                original_params[name] = param.data.clone()

                if 'bias' in name and self.quantize_bias:
                    
                    if prnt:
                        print(f"\nQuantizing layer {name} to {bias_bits}-bit")
                        print(f"Original biases: {param.data.view(-1)[:5].cpu().numpy()}")

                    quantized_w, scaling_factor, zero_point = self.quantize(param.data, bias_bits) # Biases quantized to 32-bit as per papers

                    if prnt:
                        print(f"Quantized biases: {quantized_w.view(-1)[:5].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:5].cpu().numpy(), bias_bits, sb)}")

                    dequantized_w = self.dequantize(quantized_w, scaling_factor, zero_point)

                    if prnt:
                        print(f"Dequantized biases: {dequantized_w.view(-1)[:5].cpu().numpy()}")

                elif 'weight' in name and self.quantize_weight:

                    if prnt:
                        print(f"\nQuantizing layer {name} to {target_bits}-bit")
                        print(f"Original weights: {param.data.view(-1)[:5].cpu().numpy()}")

                    # Quantize to max_train_bits first
                    quantized_w, scaling_factor, zero_point = self.quantize(param.data, self.max_train_bits)

                    if prnt:
                        print(f"Quantized weights before slicing: {quantized_w.view(-1)[:5].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:5].cpu().numpy(), self.max_train_bits, sb)}")

                    # If we need a lower precision, slice the bits
                    if target_bits < self.max_train_bits:
                        if sb == 'msb':
                            sliced_w = self.slice_bits(quantized_w, self.max_train_bits, target_bits, prnt, rounding)
                        else:
                            sliced_w = self.slice_bits_lsb(quantized_w, self.max_train_bits, target_bits, prnt, rounding)

                        if prnt:
                            print(f"Sliced weights to {target_bits}-bit: {sliced_w.view(-1)[:5].cpu().numpy()}")
                            print(f"Binary representation: {self.get_binary_representation(sliced_w.view(-1)[:5].cpu().numpy(), target_bits, sb)}")

                        dequantized_w = self.dequantize(sliced_w, scaling_factor, zero_point)
                    else:
                        dequantized_w = self.dequantize(quantized_w, scaling_factor, zero_point)

                    if prnt:
                        print(f"Dequantized weights after slicing to {target_bits}-bit: {dequantized_w.view(-1)[:5].cpu().numpy()}")

                # Replace the weight with (de)quantized version
                param.data = self.straight_through_estimator(dequantized_w, param.data)


        if ext:
            exit(1)

        # Forward pass with quantized weights and/or activations
        # Activation quantization is handled by the hooks we registered

        output = self.model(x)

        # Restore original weights if we modified them
        if (self.quantize_weight or self.quantize_bias) and 'original_params' in locals():
            for name, param in self.quantized_params:
                param.data = original_params[name]

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

        self.target_bits = sorted(list(self.target_bits), reverse=True)

        if self.target_bits:
            self.max_train_bits = max(self.target_bits)
        
        # If in dry mode, only perform full precision forward pass
        if self.dry_mode:
            output = self.model(x)
            # Return output for all bit-widths (they'll all be the same during dry iterations)
            for bits in self.target_bits:
                outputs[bits] = output
            return outputs

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

        # print(f"\n=== MatQuant Forward Pass (dry_mode={self.dry_mode}) ===")

        # If in dry mode, skip quantization
        if self.dry_mode:
            return self.model(x)
        
        # If not in dry mode, apply quantization
        else:
            # print(f"input sample before: {x.view(-1)[:5].cpu().numpy()}")
            # Quantize input if enabled and not already quantized by base model
            if self.use_input_quantization and not hasattr(self.model, 'use_input_quantization'):
                x = self.input_quantizer(x)
                # print(f"input sample after quantization: {x.view(-1)[:5].cpu().numpy()}")

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

            # print(f"Loss at {bit_width}-bit: {losses[bit_width].item()}")
        
        # Calculate total loss with loss weights for each bit-width (λ_r)
        for bit_width, loss in losses.items():
            total_loss += self.loss_weights.get(bit_width, 1.0) * loss
            # print(f"at {bit_width}, l_r: {self.loss_weights.get(bit_width, 1.0)}, loss: {loss}, w_loss: {loss * self.loss_weights.get(bit_width, 1.0)}, t_loss: {total_loss}")
        
        
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
    
    def get_binary_representation(self, tensor, bits, sb):
        """
        Get the binary representation of values in a tensor, extracting either the MSB or LSB
        portion of an 8-bit two's-complement representation.

        Args:
            tensor (iterable): Iterable of numeric values (e.g. 1-D tensor or numpy array).
            bits (int): Number of bits to extract (<= 8).
            sb (str): Which side to extract from the full 8-bit value: 'msb' or 'lsb' (case-insensitive).

        Returns:
            List[str]: List of binary strings (length == bits) for each element.
        """
        if bits < 1 or bits > 32:
            raise ValueError("bits must be between 1 and 8")
        side = str(sb).lower()
        if side not in ("msb", "lsb"):
            raise ValueError("sb must be 'msb' or 'lsb'")

        # result = []
        # full_mask = (1 << 8) - 1  # mask for full 8-bit two's-complement
        # if side == "lsb":
        #     extract_mask = (1 << bits) - 1
        #     for val in tensor:
        #         int_val = int(val)
        #         masked = int_val & extract_mask  # keep lowest `bits`
        #         result.append(format(masked, f'0{bits}b'))
        # else:  # msb
        #     shift = 8 - bits
        #     for val in tensor:
        #         int_val = int(val)
        #         # normalize to 8-bit two's-complement, then shift to get MSBs
        #         masked_full = int_val & full_mask
        #         msb_part = masked_full >> shift
        #         result.append(format(msb_part, f'0{bits}b'))

        result = []
        for value in tensor:
            result.append(bin(value.astype(int)))

        return result

    
    def extract_model(self, target_bits, model=None, rounding=True):
        """
        Extract a model at the specified precision target_bits.

        Args:
            target_bits (int): Target bit-width to quantize the extracted model to.
            rounding (bool): If True, apply rounding based on the bit at position target_bits+1;
                    otherwise, simply truncate the bits (default: True).
        Returns:
            extracted_model (nn.Module): Copy of the model with weights quantized to target_bits.
        """

        bias_bits = 8

        print_bits = 20
        sb = 'msb'
        # sb = 'lsb'
        prnt = False
        ext = False

        # if print_bits == target_bits:
        #     ext = True
        # if target_bits == 8:
        #     ext = True

        if model is None:
            # Create a copy of the model
            extracted_model = deepcopy(self.model)
        else:
            extracted_model = model
        
        # Fold BatchNorm if enabled and not extracted_model.training
        if self.fold_bn_inference and not extracted_model.training:
            print(f"Folding BatchNorm layers into Conv/Linear layers for {target_bits}-bit model...")
            # fold_model_batchnorm returns a new model, so we need to reassign
            extracted_model = fold_model_batchnorm(extracted_model, inplace=True)
        
        if self.quantize_weight or self.quantize_bias:
            # Quantize weights of target layers
            # IMPORTANT: Get params from extracted_model, not self.model
            # IMPORTANT: After folding, we need to get a fresh list of parameters from the folded model
            # We can't use self.get_quantizable_params() because it's designed for self.model structure
            # Instead, we need to manually iterate through the extracted_model's parameters
            
            # Get the quantizable parameter paths from the original model
            quantizable_param_names = set()
            for path in self.quantizable_params:
                # Extract just the parameter name (e.g., 'model.2.weight' -> 'model.2.weight')
                quantizable_param_names.add(path)
            
            # Now iterate through extracted_model's named_parameters and quantize matching ones
            for name, param in extracted_model.named_parameters():
                # Check if this parameter matches any of our quantizable paths
                matching_paths = [path for path in quantizable_param_names if name in path or path in name]
                
                if not matching_paths:
                    continue  # Skip non-quantizable parameters

                if print_bits <= target_bits and 'model.2.weight' in name:
                    prnt = True
                else:
                    prnt = False

                if 'bias' in name and self.quantize_bias:
                    if prnt:
                        print(f"\nQuantizing layer {name} to {bias_bits}-bit")
                        print(f"Original biases: {param.data.view(-1)[:5].cpu().numpy()}")

                    quantized_w, sf, zp = self.quantize(param.data, bias_bits) # Biases quantized to 32-bit as per papers

                    if prnt:
                        print(f"Quantized biases: {quantized_w.view(-1)[:5].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:5].cpu().numpy(), bias_bits, sb)}")

                    dequantized_w = self.dequantize(quantized_w, sf, zp)

                    if prnt:
                        print(f"Dequantized biases: {dequantized_w.view(-1)[:5].cpu().numpy()}")
                
                elif 'weight' in name and self.quantize_weight:
                    if prnt:
                        print(f"\nQuantizing layer {name} to {target_bits}-bit")
                        print(f"Original weights: {param.data.view(-1)[:10].cpu().numpy()}")

                    # Quantize to max_extract_bits first
                    quantized_w, sf, zp = self.quantize(param.data, self.max_train_bits)
                    if prnt:
                        print(f"Quantized weights: {quantized_w.view(-1)[:10].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:10].cpu().numpy(), self.max_train_bits, sb)}")
                    
                    # If we need a lower precision, slice the bits
                    if target_bits < self.max_train_bits:
                        if sb == 'msb':
                            sliced_w = self.slice_bits(quantized_w, self.max_train_bits, target_bits, False, rounding)
                        else:
                            sliced_w = self.slice_bits_lsb(quantized_w, self.max_train_bits, target_bits, prnt, rounding)

                        if prnt:
                            print(f"Sliced weights to {target_bits}-bit: {sliced_w.view(-1)[:10].cpu().numpy()}")
                            print(f"Binary representation: {self.get_binary_representation(sliced_w.view(-1)[:10].cpu().numpy(), target_bits, sb)}")

                        dequantized_w = self.dequantize(sliced_w, sf, zp)
                    else:
                        dequantized_w = self.dequantize(quantized_w, sf, zp)

                    if prnt:
                        print(f"Dequantized weights after slicing to {target_bits}-bit: {dequantized_w.view(-1)[:10].cpu().numpy()}")

                param.data = dequantized_w

        # # Verify the weights were actually updated
        # for name, param in extracted_model.named_parameters(): #self.get_quantizable_params(extracted_model):
        #     if 'model.2.weight' in name:
        #         print(f"\nFinal weights in extracted model layer {name}: {param.data.view(-1)[:10].cpu().numpy()}")
        
        if ext:
            exit(1)
        
        # Activation hooks have already been set by set_quantization() during layer_registry

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
        
        # Fold BatchNorm if enabled and not mixed_model.training
        if self.fold_bn_inference and not mixed_model.training:
            print(f"Folding BatchNorm layers into Conv/Linear layers for mix-and-match model...")
            mixed_model = fold_model_batchnorm(mixed_model, inplace=True)

        if self.quantize_weight or self.quantize_bias:
            # Get the quantizable parameter paths from the original model
            quantizable_param_names = set()
            for path in self.quantizable_params:
                # Extract just the parameter name
                quantizable_param_names.add(path)

            # Check if all specified layers exist in the model
            available_layers = set()
            for name, _ in mixed_model.named_parameters():
                matching_paths = [path for path in quantizable_param_names if name in path or path in name]
                if matching_paths:
                    available_layers.add(name)
            
            invalid_layers = [layer for layer in bit_config if layer not in available_layers]
            if invalid_layers:
                raise ValueError(f"Invalid layer names specified: {invalid_layers}. "
                    f"Available layers: {list(available_layers)}")

            # Validate the bit_config and ensure all specified bit-widths are valid
            if not bit_config:
                raise ValueError("bit_config dictionary cannot be empty")
            
            bias_bits = 8

            print_bits = 20
            sb = 'msb'
            # sb = 'lsb'
            prnt = False
            ext = False
            
            # if print_bits == bits:
            #     ext = True
            # if bits == 8:
            #     ext = True

            # Apply different quantization to each layer based on config
            # Now iterate through mixed_model's named_parameters and quantize matching ones
            for name, param in mixed_model.named_parameters():
                # Check if this parameter matches any of our quantizable paths
                matching_paths = [path for path in quantizable_param_names if name in path or path in name]
                
                if not matching_paths:
                    continue  # Skip non-quantizable parameters

                bits = bit_config.get(name, self.max_train_bits)  # Default to max bits if not specified

                if print_bits <= bits and 'model.4.weight' in name:
                    prnt = True
                else:
                    prnt = False
                
                if 'bias' in name and self.quantize_bias:

                    if print_bits == bits:
                            print(f"\nQuantizing layer {name} to {bias_bits}-bit")
                            print(f"Original biases: {param.data.view(-1)[:5].cpu().numpy()}")

                    quantized_w, sf, zp = self.quantize(param.data, bias_bits) # Biases quantized to 32-bit as per papers
                    if print_bits == bits:
                        print(f"Quantized biases: {quantized_w.view(-1)[:5].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:5].cpu().numpy(), bias_bits, sb)}")

                    dequantized_w = self.dequantize(quantized_w, sf, zp)
                    if print_bits == bits:
                        print(f"Dequantized biases: {dequantized_w.view(-1)[:5].cpu().numpy()}")

                elif 'weight' in name and self.quantize_weight:

                    if print_bits == bits:
                        print(f"\nQuantizing layer {name} to {bits}-bit")
                        print(f"Original weights: {param.data.view(-1)[:5].cpu().numpy()}")

                    # Quantize to max_config_bits first
                    quantized_w, sf, zp = self.quantize(param.data, self.max_train_bits)

                    if print_bits == bits:
                        print(f"Quantized weights before slicing: {quantized_w.view(-1)[:5].cpu().numpy()}")
                        print(f"Binary representation: {self.get_binary_representation(quantized_w.view(-1)[:5].cpu().numpy(), self.max_train_bits, sb)}")

                    # Slice to target bits
                    if bits < self.max_train_bits:
                        if sb == 'msb':
                            sliced_w = self.slice_bits(quantized_w, self.max_train_bits, bits, prnt, rounding)
                        else:
                            sliced_w = self.slice_bits_lsb(quantized_w, self.max_train_bits, bits, prnt, rounding)

                        if print_bits == bits:
                            print(f"Sliced weights to {bits}-bit: {sliced_w.view(-1)[:5].cpu().numpy()}")
                            print(f"Binary representation: {self.get_binary_representation(sliced_w.view(-1)[:5].cpu().numpy(), bits, sb)}")

                        dequantized_w = self.dequantize(sliced_w, sf, zp)
                    else:
                        dequantized_w = self.dequantize(quantized_w, sf, zp)

                    if print_bits == bits:
                        print(f"Dequantized weights after slicing to {bits}-bit: {dequantized_w.view(-1)[:5].cpu().numpy()}")
                
                # Set the quantized weights
                param.data = dequantized_w

        if ext:
            exit(1)

        # Activation hooks have already been set by set_quantization() during layer_registry

        return mixed_model

