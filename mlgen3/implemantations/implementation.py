from abc import ABC,abstractmethod

class Implementation(ABC):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self):
        pass
    
    @abstractmethod
    def implement(self):
        pass
    