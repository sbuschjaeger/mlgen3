import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from importlib_resources import files
import csv
import re
from sktime.datasets import write_dataframe_to_tsfile

from ..materializer import Materializer


class LinuxStandalone(Materializer):
    # Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(
        self,
        implementation,
        filename=None,
        measure_accuracy=False,
        measure_time=False,
        measure_perf=False,
        compiler="g++",
        use_onnx=False,
    ):
        super().__init__(implementation)
        self.measure_accuracy = measure_accuracy
        self.measure_time = measure_time
        self.measure_perf = measure_perf
        self.filename = "model" if filename is None else filename
        self.compiler = compiler
        self.use_onnx = use_onnx
        
        # Set filename for implementations that need it
        if hasattr(implementation, 'set_filename'):
            implementation.set_filename(self.filename)

        # TODO Implement perf performance tests
        assert (
            measure_perf is False
        ), "Perf performance tests are currently not implemented."
        # TODO Add scoring against reference implementation

    def beautify(self, s):
        # TODO this seems to die sometimes, especially when the c++-code contains errors
        try:
            from astyle_py import Astyle

            formatter = Astyle()
            formatter.set_options("--style=google --mode=c --delete-empty-lines")
            return formatter.format(s)
        except ImportError:
            return s

    def materialize(self, path):
        # I dont why we need to call this here. This does not really make sense? Why do we need to store the path? Simply for the deploy step?
        super().materialize(path)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # For MatQuant implementations, ensure filename is set before implementation
        if hasattr(self.implementation, 'set_filename'):
            self.implementation.set_filename(self.filename)
            
        # Re-implement if needed to ensure filename is used correctly
        if hasattr(self.implementation, 'implement'):
            self.implementation.implement()

        # For MatQuant implementations, use the custom template
        is_matquant = 'MatQuant' in self.implementation.__class__.__name__

        with open(os.path.join(self.path, self.filename + ".cpp"), "w") as f:
            f.write(self.beautify(self.implementation.code))

        with open(os.path.join(self.path, self.filename + ".h"), "w") as f:
            f.write(self.beautify(self.implementation.header))

    def generate_tests(self):
        main_str = ""
        # if self.implementation.model.timeseries_classification:
        #     main_str = (
        #     files("mlgen3.materializer.cpp")
        #     .joinpath("linuxstandalone_main_ts.template")
        #     .read_text()
        # )
        # else:
        main_str = (
            files("mlgen3.materializer.cpp")
            .joinpath("linuxstandalone_main.template")
            .read_text()
        )

        start_measurement = ""
        end_measurement = ""
        measure_results = ""
        print_measurements = ""

        if self.measure_time:
            start_measurement += (
                "auto start = std::chrono::high_resolution_clock::now();"
            )
            end_measurement += """
                auto end = std::chrono::high_resolution_clock::now();   
                auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (X.size() * repeat);
            """

        if self.measure_accuracy:
            end_measurement += (
                "float accuracy = static_cast<float>(matches) / X.size() * 100.f;"
            )

        if self.measure_time and not self.measure_accuracy:
            measure_results = "return runtime;"
            print_measurements = """
                std::cout << "Latency: " << results << " [ms/elem]" << std::endl;
            """
        elif not self.measure_time and self.measure_accuracy:
            measure_results = "return accuracy;"
            print_measurements = (
                """std::cout << "Accuracy: " << accuracy << " %" << std::endl;"""
            )
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
        label_position = ""
        # if self.implementation.model.timeseries_classification:
        #     label_position = f"unsigned int label_pos = {len(self.implementation.model.XTest.to_numpy()[0])};"

        main_str = (
            main_str.replace("{start_measurement}", start_measurement)
            .replace("{end_measurement}", end_measurement)
            .replace("{measure_results}", measure_results)
            .replace("{print_measurements}", print_measurements)
            .replace("{typedefinitions}", typedefinitions)
            .replace("{label_position}", label_position)
        )

        return main_str

    def deploy(self):
        assert (
            self.measure_perf or self.measure_accuracy or self.measure_time
        ), "Cannot deploy model since no test code was generated for this implementation. Please set at-least on of the following arguments to true: measure_perf, measure_accuracy or measure_time"

        # Select the appropriate makefile template based on implementation type
        if self.use_onnx:
            makefile_template = "linuxstandalone_makefile_onnx.template"
        elif 'MatQuantPT' in self.implementation.__class__.__name__:
            makefile_template = "linuxstandalone_makefile_mq_pt.template"
        elif 'MatQuant' in self.implementation.__class__.__name__:
            makefile_template = "linuxstandalone_makefile_mq.template"
        else:
            makefile_template = "linuxstandalone_makefile.template"
            
        makefile_str = (
            files("mlgen3.materializer.cpp")
            .joinpath(makefile_template)
            .read_text()
        )
        makefile_str = makefile_str.replace("{filename}", self.filename).replace(
            "{compiler}", self.compiler
        )

        with open(os.path.join(self.path, "Makefile"), "w") as f:
            f.write(makefile_str)

        if self.measure_time or self.measure_accuracy or self.measure_perf:
            # Select the appropriate main template
            if 'MatQuant' in self.implementation.__class__.__name__:
                main_template = "linuxstandalone_main_mq.template"
            else:
                main_template = "linuxstandalone_main.template"
                
            main_str = (
                files("mlgen3.materializer.cpp")
                .joinpath(main_template)
                .read_text()
            )
            
            with open(os.path.join(self.path, "main.cpp"), "w") as f:
                f.write(self.beautify(self._generate_main_code(main_str)))

        # Handle different data formats
        if type(self.implementation.model.XTest) == pd.core.frame.DataFrame:
            XTest = self.implementation.model.XTest.to_numpy()
        else:
            XTest = self.implementation.model.XTest
        
        # Check if we have a multi-dimensional input (like images) and flatten it
        if len(XTest.shape) > 2:
            print(f"Flattening input data from shape {XTest.shape}")
            # For CNN inputs - reshape from (batch_size, channels, height, width) to (batch_size, channels*height*width)
            XTest = XTest.reshape(XTest.shape[0], -1)
            print(f"New shape: {XTest.shape}")
            
        YTest = self.implementation.model.YTest

        # Create CSV file with flattened data
        XTest = XTest.astype(np.float32)
        dfTest = pd.concat(
            [
                pd.DataFrame(
                    XTest, columns=["f{}".format(i) for i in range(len(XTest[0]))]
                ),
                pd.DataFrame(YTest, columns=["label"]),
            ],
            axis=1,
        )
        dfTest.to_csv(os.path.join(self.path, "testing.csv"), header=True, index=False)

    def run(self, verbose=False):
        make_res = subprocess.run(
            f"cd {self.path} && make", capture_output=True, text=True, shell=True
        )
        if verbose:
            print(f"Running cd {self.path} && make")
            print(f"stdout: \n{make_res.stdout}")
            print(f"stderr: \n{make_res.stderr}")
        run_res = subprocess.run(
            f"cd {self.path} && ./{self.filename} testing.csv 2",
            capture_output=True,
            text=True,
            shell=True,
        )

        if verbose:
            print(f"cd {self.path} && ./{self.filename} testing.csv 2")
            print(f"stdout: \n{run_res.stdout}")
            print(f"stderr: \n{run_res.stderr}")

        metrics = {}
        lines = run_res.stdout.split("\n")
        for cur_line in lines:
            if len(cur_line) > 0:
                l = cur_line.split(":")
                if len(l) > 1:  # Check if the line contains a colon
                    try:
                        metrics[l[0]] = l[1].split(" ")[1]
                    except IndexError:
                        # If format is not as expected, store the whole value
                        metrics[l[0]] = l[1].strip()

        return metrics

    def clean(self):
        if self.path is not None and os.path.exists(self.path):
            shutil.rmtree(self.path)

    def _generate_main_code(self, main_template):
        """Generate the main.cpp code from the template."""
        start_measurement = ""
        end_measurement = ""
        measure_results = ""
        print_measurements = ""

        if self.measure_time:
            start_measurement += (
                "auto start = std::chrono::high_resolution_clock::now();"
            )
            end_measurement += """
                auto end = std::chrono::high_resolution_clock::now();   
                auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (X.size() * repeat);
            """

        if self.measure_accuracy:
            end_measurement += (
                "float accuracy = static_cast<float>(matches) / X.size() * 100.f;"
            )

        if self.measure_time and not self.measure_accuracy:
            measure_results = "return runtime;"
            print_measurements = """
                std::cout << "Latency: " << results << " [ms/elem]" << std::endl;
            """
        elif not self.measure_time and self.measure_accuracy:
            measure_results = "return accuracy;"
            print_measurements = (
                """std::cout << "Accuracy: " << accuracy << " %" << std::endl;"""
            )
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
        label_position = ""
        
        main_str = (
            main_template.replace("{start_measurement}", start_measurement)
            .replace("{end_measurement}", end_measurement)
            .replace("{measure_results}", measure_results)
            .replace("{print_measurements}", print_measurements)
            .replace("{typedefinitions}", typedefinitions)
            .replace("{label_position}", label_position)
        )

        return main_str
