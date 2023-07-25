from abc import ABC,abstractmethod

class Trainer(ABC):
    def __init__(self):
        self.model = None
    
    @abstractmethod
    def fit(self, model):
        pass

    def score(self):
        if self.model is not None:
            return self.model.score()
        else:
            raise ValueError(f"There is not model to score associated with this trainer. Did you call train?")
    
