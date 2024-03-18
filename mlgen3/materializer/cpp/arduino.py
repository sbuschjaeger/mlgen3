import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from importlib_resources import files
import re
import serial
import time
from sklearn import tree
import sklearn

from ..materializer import Materializer


class Arduino(Materializer):
    #Has a _code variable with <label_type> predict(<feature_type>[] pX);

    def __init__(self, implementation, amount_features, filename = None, measure_accuracy=False, measure_time=False, measure_perf=False, compiler="g++"):
        super().__init__(implementation)
        self.measure_accuracy = measure_accuracy
        self.measure_time = measure_time
        self.measure_perf = measure_perf
        self.filename = "model" if (filename is None) or (filename == "main") else filename #main.cpp is hardcoded in the arduino materialize method
        self.compiler = compiler
        self.amount_features = amount_features

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
            const int AMOUNT_FEATURES = {self.amount_features};
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



    def pioini_generator(self, board):
        main_str = f"[env:{board}] \n platform = atmelavr \n board = {board} \n framework = arduino \n lib_deps = mike-matera/ArduinoSTL@^1.3.3"

        if self.measure_time:
            main_str += "\n     thomasfredericks/Chrono@^1.2.0"
        return main_str

    
    def deploy(self, board = None):
        #First we generate a new platformio project. To update the libraries, we just extend the platformio.ini file
        #We didn't do that in the materialize method, because we need to know the ID of the connected board. Otherwise, we would have to connect the board before materializing the code
        assert board is not None, "Please specify the board ID you want to deploy to. If you don't know your own board ID, please visit https://docs.platformio.org/en/latest/boards/index.html#atmel-avr and select your device."

        path = os.path.abspath(self.path)
        #print(f"pio init --board {board} --project-dir {path} --project-option \" lib_deps= mike-matera/ArduinoSTL@^1.3.3 \"")
        subprocess.run(f"pio init --board {board} --project-dir {path}", shell=True)
        print("PlatformIO project created")
        time.sleep(5)
        
        #print(os.path.join(path, "platformio.ini"))
        with open(os.path.join(path, "platformio.ini"), 'w') as f: #extends platformio.ini
            f.write(self.pioini_generator(board))
        print("build and upload PlatformIO project")
        process = subprocess.run(f"pio run -d {path} -t upload; pio device monitor", shell=True)

    def run(self): #after deployment code already runs. so deploying and running the code cannot be split
        pass
        

    def clean(self):
        if self.path is not None and os.path.exists(self.path):
            shutil.rmtree(self.path)