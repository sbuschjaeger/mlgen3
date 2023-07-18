from abc import ABC,abstractmethod

class Implementation(ABC):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, model, feature_type, label_type):
        if model is None:
            raise ValueError("Receive model that was None. Please provide a valid model")
        else:
            self.model = model
            
        self.feature_type = feature_type
        self.label_type = label_type
    
    @abstractmethod
    def implement(self):
        pass
    