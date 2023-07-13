import os
import subprocess
import numpy as np
import pandas as pd
from importlib_resources import files

from .materializer import Materializer

class LinuxCPPStandalone(Materializer):

    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, filename = None, measure_accuracy=False, measure_time=False, measure_perf=False):
        super().__init__(implementation)
        self.measure_accuracy = measure_accuracy
        self.measure_time = measure_time
        self.measure_perf = measure_perf
        self.filename = "model" if filename is None else filename

        # TODO Implement perf performance tests
        assert measure_perf is False, "Perf performance tests are currently not implemented."
        # TODO Add scoring against reference implementation 

    def materialize(self, path):
        # I dont why we need to call this here. This does not really make sense? Why do we need to store the path? Simply for the deploy step? 
        super().materialize(path)

        if not os.path.isdir(self._path):
            os.makedirs(self._path)

        with open(os.path.join(self._path, self.filename + ".cpp"), 'w') as f:
            f.write(self._implementation._code)

        with open(os.path.join(self._path, self.filename + ".h"), 'w') as f:
            f.write(self._implementation._header)

    def generate_tests(self):
        main_str = files('mlgen3.materializer').joinpath('linuxcppstandalone_main.template').read_text()
        
        start_measurement = ""
        end_measurement = ""
        measure_results = ""
        print_measurements = ""

        if self.measure_time:
            start_measurement += "auto start = std::chrono::high_resolution_clock::now();"
            end_measurement += """
    auto end = std::chrono::high_resolution_clock::now();   
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (X.size() * repeat);
"""
        
        if self.measure_accuracy:
            end_measurement += "float accuracy = static_cast<float>(matches) / X.size() * 100.f;"
        
        if self.measure_time and not self.measure_accuracy:
            measure_results = "return runtime;"
            print_measurements = """
    std::cout << "Latency: " << results << " [ms/elem]" << std::endl;
    """
        elif not self.measure_time and self.measure_accuracy:
            measure_results = "return accuracy;"
            print_measurements = """
    std::cout << "Accuracy: " << accuracy << " %" << std::endl;
    """     
        elif self.measure_time and self.measure_accuracy:
            measure_results = "return std::make_pair(accuracy, runtime);"
            print_measurements = """
    std::cout << "Accuracy: " << results.first << " %" << std::endl;
    std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
"""

        # TODO this is currently hard-coded. Remove LABEL_TYPE 
        typedefinitions = f"""
#include "{self.filename}.h"    
typedef {self._implementation.label_type} OUTPUT_TYPE;
typedef unsigned int LABEL_TYPE;
typedef {self._implementation.feature_type} FEATURE_TYPE;
"""

        main_str = main_str.replace("{start_measurement}", start_measurement).replace("{end_measurement}", end_measurement).replace("{measure_results}", measure_results).replace("{print_measurements}", print_measurements).replace("{typedefinitions}", typedefinitions)
        
        return main_str
        
    
    def deploy(self):
        assert self.measure_perf or self.measure_accuracy or self.measure_time, "Cannot deploy model since no test code was generated for this implementation. Please set at-least on of the following arguments to true: measure_perf, measure_accuracy or measure_time"

        makefile_str = files('mlgen3.materializer').joinpath('linuxcppstandalone_makefile.template').read_text()
        makefile_str = makefile_str.replace("{filename}", self.filename)

        with open(os.path.join(self._path, "Makefile"), 'w') as f:
            f.write(makefile_str)

        if self.measure_time or self.measure_accuracy or self.measure_perf:
            with open(os.path.join(self._path, "main.cpp"), 'w') as f:
                f.write(self.generate_tests())
        
        # TODO This is a bit weird, refactor it?
        XTest = self._implementation.model.XTest.astype(np.float32)
        YTest = self._implementation.model.YTest
        dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTest[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
        dfTest.to_csv(os.path.join(self._path, "testing.csv"), header=True, index=False)
        
    def run(self, verbose = False):
        make_res = subprocess.run(f"cd {self._path} && make", capture_output=True, text=True, shell=True)
        if verbose:
            print(f"Running cd {self._path} && make")
            print(f"\tstdout: {make_res.stdout}")
            print(f"\tstderr: {make_res.stderr}")
        run_res = subprocess.run(f"cd {self._path} && ./{self.filename} testing.csv 2", capture_output=True, text=True, shell=True).stdout
        
        if verbose:
            print(f"cd {self._path} && ./{self.filename} testing.csv 2")
            print(f"\tstdout: {run_res.stdout}")
            print(f"\tstderr: {run_res.stderr}")

        metrics = {}
        lines = run_res.split("\n")
        for cur_line in lines:
            if len(cur_line) > 0:
                l = cur_line.split(":")
                metrics[l[0]] = l[1].split(" ")[1]
        
        return metrics