import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from importlib_resources import files
import re
import serial 
import time

from ..materializer import Materializer


class Arduino(Materializer):
    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, filename = None, measure_accuracy=False, measure_time=False, measure_perf=False, compiler="g++"):
        super().__init__(implementation)
        self.measure_accuracy = measure_accuracy
        self.measure_time = measure_time
        self.measure_perf = measure_perf
        self.filename = "model" if (filename is None) or (filename == "main") else filename #main.cpp is hardcoded in the arduino materialize method
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

        if not os.path.isdir(self.path+"src"):
            os.makedirs(self.path+"src")

        with open(os.path.join(self.path+"src", self.filename + ".cpp"), 'w') as f: #creates model.cpp
            f.write(self.beautify(self.implementation.code))

        with open(os.path.join(self.path+"src", self.filename + ".h"), 'w') as f: #creates model.h
            f.write(self.beautify(self.implementation.header))

        with open(os.path.join(self.path+"src", "main.cpp"), 'w') as f: #creates main.cpp
            f.write(self.beautify(self.generate_maincpp()))



    
    def generate_maincpp(self):
        main_str = ""

        conversion_method = "" #due to different feature types, the Arduino needs a different conversion method from string to (int | float | double)

        if self.measure_accuracy:
            assert False, "Accuracy measurement is currently not supported for Arduino"
        

        if self.measure_time:
            main_str = files('mlgen3.materializer.cpp').joinpath('arduino_time_main.template').read_text()
        else:
            main_str = files('mlgen3.materializer.cpp').joinpath('arduino_main.template').read_text()


        # TODO this is currently hard-coded. Remove LABEL_TYPE 
        typedefinitions = f"""
            #include "{self.filename}.h"    
            typedef {self.implementation.label_type} OUTPUT_TYPE;
            typedef {self.implementation.label_type} LABEL_TYPE;
            typedef {self.implementation.feature_type} FEATURE_TYPE;
            """
        
        if self.implementation.feature_type == "float":
            conversion_method =  "toFloat()"
        else:
            if self.implementation.feature_type == "double":
                conversion_method = "toDouble()"
            else:
                if self.implementation.feature_type == "int":
                    conversion_method = "toInt()"
                else:
                    raise ValueError(f"Feature type {self.implementation.feature_type} is not supported by the Arduino materializer")

        main_str = main_str.replace("{typedefinitions}", typedefinitions).replace("{conversion_method}", conversion_method)
        

        return main_str


    def detect_connected_boards(self):
        output = subprocess.check_output(["platformio", "device", "list"]).decode("utf-8")
        print('a')
        first_line = output.splitlines()[0]
        print('b')
        return first_line
        
         # search for boards with regex
        pattern = r"^(.*?)\s+(.*?)\s+\w+"
        matches = re.findall(pattern, output, re.MULTILINE)
    
        connected_boards = []
    
        # extract connected boards
        for match in matches:
            board = match[1].strip()
            connected_boards.append(board)

        return connected_boards

    def pioini_generator(self, board):
        main_str = f"[env:{board}] \n platform = atmelavr \n board = {board} \n framework = arduino \n lib_deps = mike-matera/ArduinoSTL@^1.3.3"

        if self.measure_time:
            main_str += "\n     thomasfredericks/Chrono@^1.2.0"
        return main_str

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
    
    def deploy(self, board = None):
        #First we generate a new platformio project. later on, we just extend the platformio.ini file, to update the libraries
        #We didn't do that in the materialize method, because we need to know the connected boards. Otherwise, we would have to connect the board before materializing the code
        assert board is not None, "Please specify the board ID you want to deploy to. If you don't know your own board ID, please visit https://docs.platformio.org/en/latest/boards/index.html#atmel-avr and select your device."
        #if not os.path.isdir(os.path.join(self.path, "platformio.ini")):
        #    os.makedirs(os.path.join(self.path, "platformio.ini"))
        path = os.path.abspath(self.path)
        print(f"pio init --board {board} --project-dir {path} --project-option \" lib_deps= mike-matera/ArduinoSTL@^1.3.3 \"")
        subprocess.run(f"pio init --board {board} --project-dir {path}", shell=True)
        print("PlatformIO project created")
        time.sleep(5)
        
        print(os.path.join(path, "platformio.ini"))
        with open(os.path.join(path, "platformio.ini"), 'w') as f: #extends platformio.ini
            f.write(self.pioini_generator(board))
        print("run")
        process = subprocess.run(f"pio run -d {path} -t upload; pio device monitor", shell=True)
        time.sleep(5)
        

        #connect with Arduino for communication

        self.connect(process)
    
    def connect(self, process):
        
        #process.run(f"pio device monitor", shell=True)

        XTest = self.implementation.model.XTest.astype(np.float32)
        YTest = self.implementation.model.YTest
        
        try:
            for x,y in XTest,YTest:
                ser.write(str(x).encode()+b"\n")

                time.sleep(1)

                response = ser.readline()
                print("response:", response.decode().strip())
                print("label:", y)

        except KeyboardInterrupt:
            ser.close() #kills connection if program closed

        

      ############################################
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