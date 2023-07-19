from .layer import Layer

class Linear(Layer):
    """General matrix multiplication

    For NHWC Layout the Fully Connected Layer is generally defined in ID Layout. The code generator
    uses DI regardless, because it should be faster in most cases (when I > D).

    The binary Gemm Layer only allows for weights and inputs -1 (False) and 1 (True). The weights and outputs
    are packed into ints of size binary_word_size. It works as follows:

        x = popcount(gemm_weight xnor previous_output) + gemm_bias
        x = 2 * x - binary_word_size

    The last step is necessary to revert the encoding of -1 as 0 (False).

    Attributes:
        input_shape = [N, I]: The dimension of the input tensor
        output_shape = [N, D]: The dimension of the resulting output tensor
        weight (D x I): The weights
        bias (D): The biases
    """
    def __init__(self, weight, bias):
        self.weight = weight 
        self.bias = bias 
        super().__init__(weight.shape[1], weight.shape[0])
    
    def __call__(self, x):
        return x @ self.weight.T + self.bias
