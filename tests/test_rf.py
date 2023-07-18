#!/usr/bin/env python3

import os
import shutil
import tempfile
import unittest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.cpp.ifelse import IfElse
from mlgen3.implemantations.cpp.native import Native
from mlgen3.materializer.linuxcppstandalone import LinuxCPPStandalone

from mlgen3.models.tree_ensemble.forest import Forest

class TestRandomForestClassifiers(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

        self.rfs = []
        for n in [1, 5]:
            for d in [1, 5]:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0)
                rf.fit(self.X,self.y)
                self.rfs.append(rf)

    def test_from_scikitlearn(self):
        for rf in self.rfs:
            msg = f"Running test_from_scikitlearn on RF with n_estimators={rf.n_estimators} and max_depth = {rf.max_depth}"
            with self.subTest(msg):
                forest = Forest(rf)
                scores = forest.score(self.X,self.y)
                forest_acc = scores["Accuracy"]
                rf_acc = accuracy_score(rf.predict(self.X), self.y)

                self.assertAlmostEqual(forest_acc, rf_acc, places=3)

    def test_ifelse_linuxstandalone(self):
        for rf in self.rfs:
            msg = f"Running test_ifelse_linuxstandalone on RF with n_estimators={rf.n_estimators} and max_depth = {rf.max_depth}"
            with self.subTest(msg):
                forest = Forest(rf)
                scores = forest.score(self.X,self.y)
                forest_acc = scores["Accuracy"]
                rf_acc = accuracy_score(rf.predict(self.X), self.y)

                implementation = IfElse(forest, feature_type="float", label_type="float")
                implementation.implement()
                
                materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierIfElse"))
                materializer.deploy() 
                output = materializer.run(True) 

                self.assertAlmostEqual(forest_acc, rf_acc, places=3)
                self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)

                if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierIfElse")):
                    shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierIfElse"))

    def test_native_linuxstandalone(self):
        for rf in self.rfs:
            forest = Forest(rf)
            scores = forest.score(self.X,self.y)
            forest_acc = scores["Accuracy"]
            rf_acc = accuracy_score(rf.predict(self.X), self.y)

            for it in [None, "int"]:
                for s in [1, 8]:
                    for fc in [True, False]:
                        msg = f"Running test_native_linuxstandalone on RF with n_estimators={rf.n_estimators},max_depth={rf.max_depth},int_type={it}, reorder_nodes=True, set_size={s}, force_cacheline={fc}"
                        with self.subTest(msg):
                            implementation = Native(forest, feature_type="float", label_type="float", int_type=it, reorder_nodes=True, set_size=s, force_cacheline=fc)
                            implementation.implement()
                            
                            materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                            materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))
                            materializer.deploy() 
                            output = materializer.run(True) 

                            self.assertAlmostEqual(forest_acc, rf_acc)
                            self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)
                            
                            if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative")):
                                shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))

                msg = f"Running test_native_linuxstandalone on RF with n_estimators={rf.n_estimators},max_depth={rf.max_depth},int_type={it}, reorder_nodes=False"
                with self.subTest(msg):
                    implementation = Native(forest, feature_type="float", label_type="float", reorder_nodes=False, int_type=it)
                    implementation.implement()
                    
                    materializer = LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                    materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))
                    materializer.deploy() 
                    output = materializer.run(True) 

                    self.assertAlmostEqual(forest_acc, rf_acc)
                    self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)
                    
                    if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative")):
                        shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))

    def test_ifelse_params(self):
        msg = f"Running test_ifelse_params on RF with model=None"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=None, feature_type="float", label_type="float")

        msg = f"Running test_ifelse_params on RF with model=5"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=5, feature_type="float", label_type="float")

    def test_native_params(self):
        msg = f"Running test_native_params on RF with model=None"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=None, feature_type="float", label_type="float")

        msg = f"Running test_native_params on RF with model=5"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=5, feature_type="float", label_type="float")

        rf = self.rfs[0]
        for s in [-1, None]:
            msg = f"Running test_native_params on RF with n_estimators={rf.n_estimators},max_depth={rf.max_depth},int_type=int, reorder_nodes=True, set_size={s}, force_cacheline=True"

            with self.subTest(msg):
                self.assertRaises(ValueError, Native, model=rf, feature_type="float", label_type="float", reorder_nodes=True, int_type="int", set_size=s, force_cacheline=True)

if __name__ == '__main__':
    unittest.main()