#!/usr/bin/env python3

import os
import tempfile
import unittest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.cpp.ifelse import IfElse
from mlgen3.materializer.linuxcppstandalone import LinuxCPPStandalone

from mlgen3.models.tree_ensemble.forest import Forest

class TestRandomForestClassifiers(unittest.TestCase):

    def test_from_scikitlearn(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        rf = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)
        rf.fit(X,y)

        forest = Forest(rf)
        scores = forest.score(X,y)
        forest_acc = scores["Accuracy"]
        rf_acc = accuracy_score(rf.predict(X), y)

        self.assertAlmostEqual(forest_acc, rf_acc, places=3)

    def test_ifelse_linuxstandalone(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0)
        rf.fit(X,y)

        forest = Forest(rf)
        scores = forest.score(X,y)
        forest_acc = scores["Accuracy"]
        rf_acc = accuracy_score(rf.predict(X), y)

        implementation = IfElse(forest, feature_type="float", label_type="float")
        implementation.implement()
        
        materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
        materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifier"))
        materializer.deploy() 
        output = materializer.run(True) 

        self.assertAlmostEqual(forest_acc, rf_acc, places=3)
        self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)

if __name__ == '__main__':
    unittest.main()