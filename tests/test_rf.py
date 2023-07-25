#!/usr/bin/env python3

import os
import shutil
import tempfile
import unittest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlgen3.implemantations.tree.cpp.ifelse import IfElse
from mlgen3.implemantations.tree.cpp.native import Native
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

from mlgen3.models.tree_ensemble.forest import Forest

class TestRandomForestClassifiers(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.XTrain, self.XTest, self.ytest, self.ytest = train_test_split(X, y, test_size=0.33, random_state=42)

        self.rfs = []
        for n in [1, 5]:
            for d in [1, 5]:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0)
                rf.fit(self.XTest,self.ytest)
                self.rfs.append(rf)

    def test_from_scikitlearn(self):
        for rf in self.rfs:
            msg = f"Running test_from_scikitlearn on RF with n_estimators={rf.n_estimators} and max_depth = {rf.max_depth}"
            with self.subTest(msg):
                forest = Forest.from_sklearn(rf)
                scores = forest.score(self.XTest,self.ytest)
                forest_acc = scores["Accuracy"]
                rf_acc = accuracy_score(rf.predict(self.XTest), self.ytest)

                self.assertAlmostEqual(forest_acc, rf_acc, places=3)
    
    def test_ifelse_linuxstandalone(self):
        for rf in self.rfs:
            msg = f"Running test_ifelse_linuxstandalone on RF with n_estimators={rf.n_estimators} and max_depth = {rf.max_depth}"
            with self.subTest(msg):
                forest = Forest.from_sklearn(rf)
                scores = forest.score(self.XTest,self.ytest)
                forest_acc = scores["Accuracy"]
                rf_acc = accuracy_score(rf.predict(self.XTest), self.ytest)

                implementation = IfElse(forest, feature_type="float", label_type="float")
                implementation.implement()
                
                materializer=LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierIfElse"))
                materializer.deploy() 
                output = materializer.run(True) 

                self.assertAlmostEqual(forest_acc, rf_acc, places=3)
                self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)

                materializer.clean()

    # TODO SOMETIMES THERE IS AN HERE
    @unittest.skip
    def test_native_linuxstandalone(self):
        for rf in self.rfs:
            forest = Forest.from_sklearn(rf)
            scores = forest.score(self.XTest,self.ytest)
            forest_acc = scores["Accuracy"]
            rf_acc = accuracy_score(rf.predict(self.XTest), self.ytest)

            for it in [None, "int"]:
                for s in [1, 8]:
                    for fc in [True, False]:
                        msg = f"Running test_native_linuxstandalone on RF with n_estimators={rf.n_estimators},max_depth={rf.max_depth},int_type={it}, reorder_nodes=True, set_size={s}, force_cacheline={fc}"
                        with self.subTest(msg):
                            implementation = Native(forest, feature_type="float", label_type="float", int_type=it, reorder_nodes=True, set_size=s, force_cacheline=fc)
                            implementation.implement()
                            
                            materializer=LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                            materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))
                            materializer.deploy() 
                            output = materializer.run(True) 

                            self.assertAlmostEqual(forest_acc, rf_acc)
                            self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)
                            
                            materializer.clean()

                msg = f"Running test_native_linuxstandalone on RF with n_estimators={rf.n_estimators},max_depth={rf.max_depth},int_type={it}, reorder_nodes=False"
                with self.subTest(msg):
                    implementation = Native(forest, feature_type="float", label_type="float", reorder_nodes=False, int_type=it)
                    implementation.implement()
                    
                    materializer = LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                    materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestRandomForestClassifierNative"))
                    materializer.deploy() 
                    output = materializer.run(True) 

                    self.assertAlmostEqual(forest_acc, rf_acc)
                    self.assertAlmostEqual(float(output["Accuracy"]), rf_acc*100.0, places=3)
                    
                    materializer.clean()

    def test_swap(self):
        for rf in self.rfs:
            forest = Forest.from_sklearn(rf)
            scores = forest.score(self.XTest,self.ytest)
            forest_acc_before = scores["Accuracy"]
            forest.swap_nodes()
            forest_acc_after = scores["Accuracy"]
            
            # TODO Enhance this test case and really check if we swapped something 
            self.assertAlmostEqual(forest_acc_before, forest_acc_after)

    def test_quantize(self):
        for rf in self.rfs:
            # TODO Enhance this test case and really check if we we do not break the forest
            for r in [None, 2**16]:
                forest = Forest.from_sklearn(rf)
                forest.quantize(quantize_leafs=r, quantize_splits=None)

                forest = Forest.from_sklearn(rf)
                forest.quantize(quantize_leafs=None, quantize_splits=r)

                forest = Forest.from_sklearn(rf)
                forest.quantize(quantize_leafs=r, quantize_splits=r)
            
            forest = Forest.from_sklearn(rf)
            forest.quantize(quantize_leafs=None, quantize_splits="rounding")

    # def test_pruning(self):
    #     for rf in self.rfs:
    #         if rf.n_estimators > 2:
    #             forest = Forest(rf)
    #             forest.prune(self.XTest, self.ytest, "reduced_error", n_estimators = 2)

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