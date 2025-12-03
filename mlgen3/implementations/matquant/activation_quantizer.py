import torch
import torch.nn as nn

class QAT_ActivationQuantizer(nn.Module):
    """Hook for quantizing activations."""
    def __init__(self, bits, device=None, quantize_signed=True):
        super(QAT_ActivationQuantizer, self).__init__()
        self.bits = bits
        self.device = device if device else torch.device('cpu')
        self.quantize_signed = quantize_signed
        
        # Track scale and zero point for skip connection matching
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))

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
            
        if self.quantize_signed:
            # Signed quantization: range is [-2^(bits-1), 2^(bits-1) - 1]
            q_min = -(2**(self.bits-1))
            q_max = 2**(self.bits-1) - 1
        else:
            # Unsigned quantization: range is [0, 2^bits - 1]
            q_min = 0
            q_max = 2**self.bits - 1
            
        # Quantize to max bits (which is self.bits for extracted models)
        alpha = (x_max - x_min) / (q_max - q_min)
        z = q_min - x_min / alpha if alpha != 0 else q_min
        
        # Store scale and zero point for potential skip connection matching
        self.scale.data = alpha
        self.zero_point.data = z
        
        # Apply MinMax quantization formula with STE
        quantized_x = torch.clamp(torch.round(x / alpha + z), q_min, q_max)
        dequantized_x = (quantized_x - z) * alpha
        
        # STE: Use quantized values for forward but gradients from original for backward
        return dequantized_x.detach() + (x - x.detach())


class MQ_ActivationQuantizer(nn.Module):
    """
    Hook for quantizing activations in MatQuant.
    
    This quantizer handles:
    - Conv2d, Linear, ReLU, Hardtanh layer outputs
    - Global Average Pooling (AdaptiveAvgPool2d) outputs
      * Operates on INT8 inputs, produces INT8 outputs
      * Uses INT32 accumulator internally, then requantizes to INT8
      * No special handling needed - treat like other activations
    """
    def __init__(self, bits, device=None, rounding=True, quantize_signed=True):
        super(MQ_ActivationQuantizer, self).__init__()
        self.bits = bits if isinstance(bits, list) else [bits]
        self.q_bits = 8  # For now, using 8-bit for activations
        self.device = device if device else torch.device('cpu')
        self.rounding = rounding
        self.quantize_signed = quantize_signed
        
        # Track scale and zero point for skip connection matching
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
    
    def forward(self, x):

        # print("\n=== MQ Activation Quantization ===")
        # print("Quantizing activation with bits:", self.q_bits)
        # print("Activation before quantization:", x.flatten()[:5])

        x = self.quantize_activation(x)

        # print("Activation after quantization:", x.flatten()[:5])
        # print("=====================================\n")
        
        return x
    
    def quantize_activation(self, x):
        x_min = x.min()
        x_max = x.max()
        
        # Avoid division by zero
        if x_min == x_max:
            return x

        if self.quantize_signed:
            # Signed quantization: range is [-2^(q_bits-1), 2^(q_bits-1) - 1]
            q_min = -(2**(self.q_bits-1))
            q_max = 2**(self.q_bits-1) - 1
        else:
            # Unsigned quantization: range is [0, 2^q_bits - 1]
            q_min = 0
            q_max = 2**self.q_bits - 1
            
        # Quantize to max bits (which is self.bits for extracted models)
        scaling_factor = (x_max - x_min) / (q_max - q_min)
        zero_point = q_min - x_min / scaling_factor if scaling_factor != 0 else q_min
        
        # Store scale and zero point for potential skip connection matching
        self.scale.data = scaling_factor
        self.zero_point.data = zero_point
        
        # Apply MinMax quantization formula
        quantized_x = torch.clamp(torch.round(x / scaling_factor + zero_point), q_min, q_max)
        
        # For now, we only use the max bits version for activations
        # If MatQuant multi-bit activation is needed, we would slice and dequantize for each bit-width here
        
        # Dequantize
        dequantized_x = (quantized_x - zero_point) * scaling_factor
        
        # STE: Use quantized values for forward but gradients from original for backward
        return dequantized_x.detach() + (x - x.detach())