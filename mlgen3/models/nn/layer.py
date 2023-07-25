from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    @abstractmethod
    def __call__(self, x):
        pass