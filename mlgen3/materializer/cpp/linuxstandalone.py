import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from importlib_resources import files

from ..materializer import Materializer


class LinuxStandalone(Materializer):
    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, filename = None, measure_accuracy=False, measure_time=False, measure_perf=False, compiler="g++"):
        super().__init__(implementation)
        self.measure_accuracy = measure_accuracy
        self.measure_time = measure_time
        self.measure_perf = measure_perf
        self.filename = "model" if filename is None else filename
        self.compiler = compiler

        # TODO Implement perf performance tests
        assert measure_perf is False, "Perf performance tests are currently not implemented."
        # TODO Add scoring against reference implementation 

    def beautify(self, s):
        # TODO this seems to die sometimes, especially when the c++-code contains errors 
        try:
            from astyle_py import Astyle
            formatter = Astyle()
            formatter.set_options('--style=google --mode=c --delete-empty-lines')
            return formatter.format(s)
        except ImportError:
            return s

    def materialize(self, path):
        # I dont why we need to call this here. This does not really make sense? Why do we need to store the path? Simply for the deploy step? 
        super().materialize(path)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, self.filename + ".cpp"), 'w') as f:
            f.write(self.beautify(self.implementation.code))

        with open(os.path.join(self.path, self.filename + ".h"), 'w') as f:
            f.write(self.beautify(self.implementation.header))

    def generate_tests(self):
        main_str = files('mlgen3.materializer.cpp').joinpath('linuxstandalone_main.template').read_text()
        
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
            print_measurements = """std::cout << "Accuracy: " << accuracy << " %" << std::endl;"""
        elif self.measure_time and self.measure_accuracy:
            measure_results = "return std::make_pair(accuracy, runtime);"
            print_measurements = """
                std::cout << "Accuracy: " << results.first << " %" << std::endl;
                std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
            """

        # TODO this is currently hard-coded. Remove LABEL_TYPE 
        typedefinitions = f"""
            #include "{self.filename}.h"    
            typedef {self.implementation.label_type} OUTPUT_TYPE;
            typedef unsigned int LABEL_TYPE;
            typedef {self.implementation.feature_type} FEATURE_TYPE;
            """

        main_str = main_str.replace("{start_measurement}", start_measurement).replace("{end_measurement}", end_measurement).replace("{measure_results}", measure_results).replace("{print_measurements}", print_measurements).replace("{typedefinitions}", typedefinitions)
        
        return main_str
    
    def deploy(self):
        assert self.measure_perf or self.measure_accuracy or self.measure_time, "Cannot deploy model since no test code was generated for this implementation. Please set at-least on of the following arguments to true: measure_perf, measure_accuracy or measure_time"

        makefile_str = files('mlgen3.materializer.cpp').joinpath('linuxstandalone_makefile.template').read_text()
        makefile_str = makefile_str.replace("{filename}", self.filename).replace("{compiler}", self.compiler)

        with open(os.path.join(self.path, "Makefile"), 'w') as f:
            f.write(makefile_str)

        if self.measure_time or self.measure_accuracy or self.measure_perf:
            with open(os.path.join(self.path, "main.cpp"), 'w') as f:
                f.write(self.beautify(self.generate_tests()))
        
        # TODO This is a bit weird, refactor it?
        XTest = self.implementation.model.XTest.astype(np.float32)
        YTest = self.implementation.model.YTest
        dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTest[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
        dfTest.to_csv(os.path.join(self.path, "testing.csv"), header=True, index=False)
        
    def run(self, verbose = False):
        make_res = subprocess.run(f"cd {self.path} && make", capture_output=True, text=True, shell=True)
        if verbose:
            print(f"Running cd {self.path} && make")
            print(f"stdout: \n{make_res.stdout}")
            print(f"stderr: \n{make_res.stderr}")
        run_res = subprocess.run(f"cd {self.path} && ./{self.filename} testing.csv 2", capture_output=True, text=True, shell=True)
        
        if verbose:
            print(f"cd {self.path} && ./{self.filename} testing.csv 2")
            print(f"stdout: \n{run_res.stdout}")
            print(f"stderr: \n{run_res.stderr}")

        metrics = {}
        lines = run_res.stdout.split("\n")
        for cur_line in lines:
            if len(cur_line) > 0:
                l = cur_line.split(":")
                metrics[l[0]] = l[1].split(" ")[1]
        
        return metrics

    def clean(self):
        if self.path is not None and os.path.exists(self.path):
            shutil.rmtree(self.path)