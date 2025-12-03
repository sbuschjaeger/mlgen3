import torch
import torch.nn as nn


class InputQuantizer(nn.Module):
    """
    Quantizes floating-point inputs to 8-bit integers for inference.
    During training, inputs remain in floating-point for data augmentation.
    """
    
    def __init__(self, input_range=(0.0, 1.0), signed=False):
        """
        Initialize the input quantizer.
        
        Args:
            input_range (tuple): Expected range of input values (min, max)
            signed (bool): Whether to use signed int8 quantization
        """
        super(InputQuantizer, self).__init__()
        self.input_range = input_range
        self.signed = signed
        
        if signed:
            self.q_min = -128
            self.q_max = 127
        else:
            self.q_min = 0
            self.q_max = 255
    
    def quantize_input(self, x):
        """
        Quantize floating-point input to 8-bit integer format.
        
        Args:
            x (torch.Tensor): Input tensor in floating-point
            
        Returns:
            torch.Tensor: Quantized and dequantized tensor
        """
        # Calculate scaling factor and zero point
        x_min, x_max = self.input_range
        scale = (x_max - x_min) / (self.q_max - self.q_min)
        zero_point = self.q_min - x_min / scale
        
        # Quantize
        x_int = torch.clamp(
            torch.round(x / scale + zero_point),
            self.q_min,
            self.q_max
        )
        
        # Dequantize back to float for computation
        x_dequant = (x_int - zero_point) * scale
        
        return x_dequant
    
    def forward(self, x):
        """
        Forward pass with conditional quantization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Quantized input (during inference) or original input (during training)
        """
        # Check if input is already in integer format (uint8)
        if x.dtype in [torch.uint8, torch.int8]:
            # Convert to float and normalize to expected range
            x = x.float()
            if x.max() > 1.0:  # Likely in [0, 255] range
                x = x / 255.0
        
        # During inference, quantize the input
        if not self.training:
            x = self.quantize_input(x)
        
        return x
