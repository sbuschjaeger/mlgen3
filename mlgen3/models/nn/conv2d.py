import numpy as np
from .layer import Layer

class Conv2D(Layer):
    """2D Convolutional layer

    Attributes:
        weight: The weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width]
        bias: The bias vector of shape [out_channels]
        kernel_size: The kernel size as (height, width) tuple
        stride: The stride as (height, width) tuple
        padding: The padding as (height, width) tuple
        input_shape: The input shape as (channels, height, width) tuple
        output_shape: The output shape as (channels, height, width) tuple
    """
    def __init__(self, weight, bias, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.weight = weight
        self.bias = bias
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # For Conv2D, input_shape is (in_channels, height, width)
        # and output_shape is (out_channels, out_height, out_width)
        in_channels = weight.shape[1]
        out_channels = weight.shape[0]
        
        # Will be set when input shape is known during forward pass
        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None
        
        super().__init__(in_channels, out_channels)
    
    def compute_output_shape(self, input_shape):
        """Compute output shape based on input shape"""
        _, in_height, in_width = input_shape
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return (self.weight.shape[0], out_height, out_width)
    
    def __call__(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Save dimensions for later use
        self.input_height = in_height
        self.input_width = in_width
        
        out_channels = self.weight.shape[0]
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        self.output_height = out_height
        self.output_width = out_width
        
        # Initialize output tensor
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # Apply padding if necessary
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded = np.pad(x, 
                            ((0, 0), (0, 0), 
                             (self.padding[0], self.padding[0]), 
                             (self.padding[1], self.padding[1])),
                            mode='constant')
        else:
            padded = x
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w_out * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract the current patch
                        patch = padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Apply convolution to this patch
                        output[b, c_out, h_out, w_out] = np.sum(patch * self.weight[c_out]) + self.bias[c_out]
        
        return output
