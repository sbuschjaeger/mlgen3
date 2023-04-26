from abc import ABC

class Ensemble(ABC, Implementation):

    def __init__(self):
    @abstractmethod
    def implement_member(self): #returned
        pass
    
    def implement(self):
        