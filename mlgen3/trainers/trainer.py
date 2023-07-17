from abc import ABC,abstractmethod

class Trainer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, model):
        pass

    @abstractmethod
    def score(self):
        pass
    
