import numpy as np
from .layer import Layer

class MaxPool2D(Layer):
    """2D Max Pooling layer

    Attributes:
        kernel_size: The kernel size as (height, width) tuple
        stride: The stride as (height, width) tuple
        padding: The padding as (height, width) tuple
        input_shape: The input shape as (channels, height, width) tuple
        output_shape: The output shape as (channels, height, width) tuple
    """
    def __init__(self, kernel_size, stride=None, padding=(0, 0)):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Will be determined during the forward pass
        self.channels = None
        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None
        
        # Initialize with placeholder values; will be updated during first forward pass
        super().__init__(None, None)
    
    def compute_output_shape(self, input_shape):
        """Compute output shape based on input shape"""
        channels, in_height, in_width = input_shape
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return (channels, out_height, out_width)
    
    def __call__(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, channels, in_height, in_width = x.shape
        
        # Save dimensions for later use
        self.channels = channels
        self.input_height = in_height
        self.input_width = in_width
        self.input_shape = channels
        
        out_height = (in_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        self.output_height = out_height
        self.output_width = out_width
        self.output_shape = channels
        
        # Initialize output tensor
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Apply padding if necessary
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded = np.pad(x, 
                            ((0, 0), (0, 0), 
                             (self.padding[0], self.padding[0]), 
                             (self.padding[1], self.padding[1])),
                            mode='constant', constant_values=float('-inf'))  # Use -inf for max pooling padding
        else:
            padded = x
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride[0]
                        h_end = min(h_start + self.kernel_size[0], in_height + 2 * self.padding[0])
                        w_start = w_out * self.stride[1]
                        w_end = min(w_start + self.kernel_size[1], in_width + 2 * self.padding[1])
                        
                        # Extract the current patch
                        patch = padded[b, c, h_start:h_end, w_start:w_end]
                        
                        # Apply max pooling to this patch
                        output[b, c, h_out, w_out] = np.max(patch)
        
        return output
