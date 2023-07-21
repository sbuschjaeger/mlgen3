import numpy as np

from .layer import Layer

class Sign(Layer):
    """Sign Layer

        f(x) =  1  for x > 0
        f(x) =  0  for x = 0
        f(x) = -1  for x < 0

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        output_shape = [N, C, H, W]: The shape of the resulting output tensor, must match the input shape
    """

    def __init__(self, input_shape):
        super().__init__(input_shape, input_shape)
    
    def __call__(self, x):
        x[x > 0] = 1
        x[x < 0] = -1
        return x

class Sigmoid(Layer):
    """Sigmoid activation

        f(x) = 1 / (1 + exp(-x))

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        output_shape = [N, C, H, W]: The shape of the resulting output tensor, must match the input shape
    """
    def __init__(self, input_shape):
        super().__init__(input_shape, input_shape)

    def __call__(self, x):
        return 1.0/(1.0 + np.exp(-x))
        
class Relu(Layer):
    """Rectified Linear Unit

        f(x) = max(0, x)

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        output_shape = [N, C, H, W]: The shape of the resulting output tensor, must match the input shape
    """

    def __init__(self, input_shape):
        super().__init__(input_shape, input_shape)
    
    def __call__(self, x):
        return np.maximum(0, x)

class Step(Layer):
    """Step Layer

        f(x) = high  for x > threshold
        f(x) = high  for x = threshold and threshold_is_high
        f(x) = low   for x = threshold and not threshold_is_high
        f(x) = low   for x < threshold

    This is the Activation Layer in a binary neural net as it has only two distinct outputs (in comparison
    to the three outputs of Sign Layers). There is no official support for Step Layers in ONNX.
    To generate a net with Step Layers, use the following ONNX structure:

        Greater + Where or
        Less + Where

    The code generator will convert this into a Step Layer if the binary argument is passed.

    Example in PyTorch:

        x = torch.where(x > 0, torch.tensor([1.0]), torch.tensor([-1.0]))

    When a BatchNormalization Layer follows directly afterwards, the scales and biases are embedded as thresholds
    of the Step Layer. The following holds since x is an integer:

        x * s - b > 0
        x > int(b / s)

    The output is directly packed into ints of size binary_word_size. This is done by setting each bit individually.
    The following sets the c'th leftmost bit to 1 or 0:

        output |= (1U << ((binary_word_size-1) - c % binary_word_size))
        output &= ~(1U << ((binary_word_size-1) - c % binary_word_size))

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        output_shape = [N, C, H, W]: The shape of the resulting output tensor, must match the input shape
        threshold: The threshold, can be scalar or numpy array
        low: Value selected at indices where x < threshold
        high: Value selected at indices where x > threshold
        threshold_is_high: Whether high value is selected where x = threshold
    """

    def __init__(self, input_shape, threshold=0, low=-1, high=1):
        super().__init__(input_shape, input_shape)

        self.threshold = threshold
        self.low = low 
        self.high = high
        self.threshold_is_high = True

    def __call__(self, x):

        if self.threshold_is_high:    
            x[x >= self.threshold] = self.high
            x[x < self.threshold] = self.low
        else:
            x[x > self.threshold] = self.high
            x[x <= self.threshold] = self.low
        
        return x

# class Softmax(Layer):
#     """Softmax (normalized exponential)

#     To combat numerical issues when doing softmax computation, a common trick is used that shifts
#     the input vector by subtracting the maximum element in it from all elements.

#         z = x - max(x)
#         numerator = np.exp(z)
#         denominator = np.sum(numerator)
#         softmax = numerator/denominator

#     Attributes:
#         output_shape = [N, D]: The dimension of the output tensor
#     """
    
#     def __init__(self, output_shape):
#         self.input_shape = self.output_shape = output_shape

#     def render(self, backend, **kwargs):
#         code_init = ''
#         code_alloc = super(Softmax, self).render('alloc', output_shape=self.output_shape, backend=backend, **kwargs)
#         code_predict = super(Softmax, self).render('softmax', output_size=self.output_shape[1], backend=backend,**kwargs)
#         return code_init, code_alloc, code_predict

#     def output_type(self, input_type, backend):
#         return 'float'


# class LogSoftmax(Layer):
#     """Log of Softmax

#     To combat numerical issues when doing softmax computation, a common trick is used that shifts
#     the input vector by subtracting the maximum element in it from all elements.

#         z = x - max(x)
#         numerator = np.exp(z)
#         denominator = np.sum(numerator)
#         softmax = numerator/denominator
#         logsoftmax = np.log(softmax)

#     Attributes:
#         output_shape = [N, D]: The dimension of the output tensor
#     """
#     def __init__(self, graph, node, input_shape):
#         super().__init__(input_shape, input_shape, "logsoftmax")