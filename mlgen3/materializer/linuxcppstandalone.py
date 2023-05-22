from abc import ABC

class LinuxCPPStandalone(ABC, Materializer):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self):
    def materialize(self, path):
        self._path=path

    
    def generate_tests(self, path):
        