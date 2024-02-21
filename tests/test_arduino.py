from sklearn import tree
import sklearn
import mlgen3
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from mlgen3.models.tree_ensemble.tree import Tree
from mlgen3.materializer.cpp.arduino import Arduino
from mlgen3.implemantations.tree.cpp.native import Native
import unittest
import os

class TestArduino(unittest.TestCase):
    
    def testMaterialize(self, materializer):
        materializer.materialize("./testmodels/")

        assert os.path.isdir("./testmodels/src")

        assert os.path.isfile("./testmodels/src/model.cpp")
        assert os.path.isfile("./testmodels/src/model.h")
        assert os.path.isfile("./testmodels/src/main.cpp")
    

    #for testing, please connect an Arduino Uno
    def testDeploy(self, materializer):
        materializer.deploy(board = "uno")

        #check if platformio.ini got modified. If yes, the platformio project also got initialized
        assert os.path.isfile("./testmodels/platformio.ini")
        assert os.path.isdir("./testmodels/.pio")
        

        with open("./testmodels/platformio.ini", "r") as file:
            contents = file.read()
            assert "lib_deps = mike-matera/ArduinoSTL@^1.3.3" in contents

            if materializer.measure_time:
                assert "thomasfredericks/Chrono@^1.2.0" in contents
            
        



    def testConnection(self):
        print("a")


    def testAccuracy(self):
        print("a")




if __name__ == "__main__":

    # Load data
    dataFrame = pd.read_csv("testing.csv");
    Y = dataFrame.pop("label")
    X = dataFrame

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)
    
    print("start generating testmodels")
    sktree = tree.DecisionTreeClassifier(max_depth=1)

    sktree.fit(X_train, Y_train)



    #tree = Tree()
    tree = Tree.from_sklearn(sktree)
    #tree.implement()

    native = Native(tree, feature_type="double", label_type="double")
    native.implement()
    
    materializer = Arduino(native, measure_time=True)

    materializer.materialize("./testmodels/")

    materializer.deploy(board = "uno")

