from abc import ABC

class Materializer(ABC):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self):
    @abstractmethod
    def materialize(self, path):
        self._path=path
    @abstractmethod
    def deploy(self):

    @abstractmethod
    def run(self):

    