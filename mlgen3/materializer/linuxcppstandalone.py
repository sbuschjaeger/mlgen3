import os
from .materializer import Materializer

class LinuxCPPStandalone(Materializer):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, filename = None, run_test=False, measure_time=False, run_in_perf=False, code_ending="cpp", header_ending="h"):
        super().__init__(implementation)
        self.run_test = run_test
        self.measure_time = measure_time
        self.run_in_perf = run_in_perf
        self.filename = filename

    def materialize(self, path):
        super().materialize(path)

        if self.filename is None:
            name = "model"
        else:
            name = self.filename

        if not os.path.isdir(self._path):
            os.makedirs(self._path)

        with open(os.path.join(self._path, name + ".cpp"), 'w') as f:
            f.write(self._implementation._code)

        with open(os.path.join(self._path, name + ".h"), 'w') as f:
            f.write(self._implementation._header)

        # TODO add Makefile
    
    def generate_tests(self, path):
        pass
    
    def deploy(self):
        pass

    def run(self):
        pass