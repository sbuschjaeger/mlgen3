import numpy as np
from .layer import Layer

class BatchNorm(Layer):
    """1D and 2D Batch normalization, depending on the given shape

    Scale and bias should already be transformed for inference according to https://arxiv.org/pdf/1502.03167.pdf

    Attributes:
        input_shape (tuple): The shape of the input tensor
        output_shape (tuple): The shape of the resulting output tensor, must match the input shape
        scale (float): The scale tensor
        bias (float): The bias tensor
    """
    def __init__(self, scale, bias, mean, var, epsilon):
        # Calculate scale and bias for inference according to the original paper
        # https://arxiv.org/abs/1502.03167
        self.bias = bias - scale * mean / np.sqrt(var + epsilon)
        self.scale = scale / np.sqrt(var + epsilon)
        super().__init__(self.bias.shape[0], self.bias.shape[0])
    
    def __call__(self, x):
        return x*self.scale + self.bias
