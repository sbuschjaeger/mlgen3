from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import unittest
import os
import serial
import time

#from mlgen3.implementations.tree.cpp.native import Native
from mlgen3.implementations.tree.cpp.ifelse import IfElse
from mlgen3.implementations.tree.cpp.ensemble import Ensemble
from mlgen3.implementations.linear.cpp.native import Native
from mlgen3.materializer.cpp.arduino import Arduino
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.tree_ensemble import Tree
from mlgen3.models.tree_ensemble import Forest
from mlgen3.models.linear import Linear
from mlgen3.implementations.ssf import SSF as SSFImplementation
from mlgen3.models.ssf import SSF as SSFModel

class TestArduino(unittest.TestCase):

    materializer = 0
    
    #tests if the materializer can generate the c++ files
    def testMaterialize(self):
        materializer.materialize("./testmodels/")

        assert os.path.isdir("./testmodels/src")

        assert os.path.isfile("./testmodels/src/model.cpp")
        assert os.path.isfile("./testmodels/src/model.h")
        assert os.path.isfile("./testmodels/src/main.cpp")
    
    #tests if the materializer can deploy the c++ files to an Arduino board
    def testDeploy(self):
        materializer.deploy(board = "uno")

        #check if platformio.ini got modified. If yes, the platformio project also got initialized
        assert os.path.isfile("./testmodels/platformio.ini")
        assert os.path.isdir("./testmodels/.pio")
        

        with open("./testmodels/platformio.ini", "r") as file:
            contents = file.read()
            assert "lib_deps = mike-matera/ArduinoSTL@^1.3.3" in contents

            if materializer.measure_time:
                assert "thomasfredericks/Chrono@^1.2.0" in contents
            
        


        #please connect an Arduino Uno for testing. otherwise, please change the board ID to your own board ID in the main function
        # to look up your board ID, visit https://docs.platformio.org/en/latest/boards/index.html#atmel-avr
    def testAccuracy(self):
        assert materializer.implementation.model.XTest is not None and materializer.implementation.model.YTest is not None
        with serial.Serial('/dev/ttyACM0', 9600) as ser: #connects to the Arduino via serial port

            ser.flushInput()
            
            model = materializer.implementation.model

            XTest = materializer.implementation.model.XTest
            YTest = materializer.implementation.model.YTest

            i = 0

            for x,y in zip(XTest.values, YTest.values):
                input = (str(x)[1:-1]).replace("\n", "") #deletes newline breaks
                features = input.split(" ")
                for feature in features:
                    feature = feature.encode()
                    #print(input)
                    ser.write(feature+b"\n")
                
                time.sleep(1)
                
                response = str(ser.read(ser.in_waiting).strip())

                arduino_response = response[2] # "arduino response: b'3\r\n3'" . Take out the (first) prediction of the arduino. Not that clean, but due to hardcoded dataset with labels from 0-9, it works
                sklearn_response = str(model.predict(x.reshape(1, -1))[0])
                    
                print("arduino response:", arduino_response)
                print("sklearn response:", sklearn_response)

                #compare if arduino response equals sklearn response. We skip the first comparison because of the arduino startup print statement
                
                if i != 0:
                    assert arduino_response == sklearn_response, "arduino response not equal to sklearn response"
                if i > 9:
                    break
                
                i = i+1




if __name__ == "__main__":

    # Loads random data, to check if the prediction results on the arduino are equal to the prediction results of the sklearn model
    dataFrame = pd.read_csv("testing.csv")
    Y = dataFrame.pop("label")
    X = dataFrame

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)
    
    print("start generating testmodels")
    sktree = tree.DecisionTreeClassifier(max_depth=2)

    sktree.fit(X_train, Y_train)

    skforest = RandomForestClassifier(n_estimators=2, max_depth=2)
    skforest.fit(X_train, Y_train)



    #tree = Tree()
    #tree = Tree.from_sklearn(sktree)
    forest = Forest.from_sklearn(skforest)
    #tree.implement()

    #native = Native(tree, feature_type="double", label_type="double")
    #native.implement()

    #ifelse = IfElse(tree, feature_type="double", label_type="double")
    #ifelse.implement()

    #ensemble = IfElse(forest, feature_type="double", label_type="double")
    #ensemble.implement()

    sklog = LogisticRegression()
    sklog.fit(X_train, Y_train)
    log_reg = Linear.from_sklearn(sklog)
    implementation = Native(log_reg, feature_type="int", label_type="double")
    implementation.implement()

    ssf_model = SSFModel.from_sklearn(skforest, sklog)
    ssf_implementation = SSFImplementation(ssf_model, feature_type="double", label_type="double")
    ssf_implementation.implement()

    materializer = LinuxStandalone(ssf_implementation, measure_time=False)
    #materializer = Arduino(ssf_implementation, measure_time=False, amount_features=3)
    #log_reg.model.XTest = X_test
    #log_reg.model.YTest = Y_test

    ssf_model.XTest = X_test
    ssf_model.YTest = Y_test

    materializer.materialize("./testmodels/")

    #materializer.deploy(board = "megaatmega2560")
    
    unittest.main()


