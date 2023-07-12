from abc import ABC,abstractmethod

class Implementation(ABC):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, feature_type, label_type):
        self.feature_type=feature_type
        self.label_type=label_type
    
    @abstractmethod
    def implement(self):
        pass
    