from abc import ABC

class Trainer(ABC):

    def __init__(self):
    @abstractmethod
    def train(self, model, x,y):
        pass
    
