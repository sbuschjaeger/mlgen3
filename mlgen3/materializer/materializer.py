from abc import ABC, abstractmethod

class Materializer(ABC):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation):
        self.implementation = implementation
        pass

    @abstractmethod
    def materialize(self, path):
        self.path = path
        pass

    @abstractmethod
    def deploy(self):
        pass

    @abstractmethod
    def run(self):
        pass