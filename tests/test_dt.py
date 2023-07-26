#!/usr/bin/env python3

import os
import shutil
import tempfile
import numpy as np
import unittest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.tree.cpp.ifelse import IfElse
from mlgen3.implemantations.tree.cpp.native import Native
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

from mlgen3.models.tree_ensemble.tree import Tree
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.core.dataset import missing_value

class TestDecisionTreeClassifiers(unittest.TestCase):

    def setUp(self):
        self.X, self.Y = datasets.load_digits(return_X_y=True)

        self.dts = []
        for d in [1, 5, None]:
            dt = DecisionTreeClassifier(max_depth=d, random_state=0)
            dt.fit(self.X,self.Y)
            self.dts.append(dt)

    def test_from_scikitlearn(self):
        for dt in self.dts:
            tree = Tree.from_sklearn(dt)
            scores = tree.score(self.X,self.Y)
            tree_acc = scores["Accuracy"]
            dt_acc = accuracy_score(dt.predict(self.X), self.Y)

            self.assertAlmostEqual(dt_acc, tree_acc, places=3)

    def test_ifelse_linuxstandalone(self):
        for dt in self.dts:
            msg = f"Running test_ifelse_linuxstandalone on DT with max_depth = {dt.max_depth}"
            with self.subTest(msg):
                tree = Tree.from_sklearn(dt)
                scores = tree.score(self.X,self.Y)
                tree_acc = scores["Accuracy"]
                dt_acc = accuracy_score(dt.predict(self.X), self.Y)

                implementation = IfElse(tree, feature_type="float", label_type="float")
                implementation.implement()
                
                materializer=LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierIfElse"))
                materializer.deploy() 
                output = materializer.run() 

                self.assertAlmostEqual(tree_acc, dt_acc, places=3)
                self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)

                materializer.clean()
    
    def test_native_linuxstandalone(self):
        for dt in self.dts:
            tree = Tree.from_sklearn(dt)
            scores = tree.score(self.X,self.Y)
            tree_acc = scores["Accuracy"]
            dt_acc = accuracy_score(dt.predict(self.X), self.Y)

            for it in [None, "int"]:
                for s in [1, 8]:
                    for fc in [True, False]:
                        msg = f"Running test_native_linuxstandalone on DT with max_depth={dt.max_depth}, int_type={it}, reorder_nodes=True, set_size={s}, force_cacheline={fc}"
                        with self.subTest(msg):
                            implementation = Native(tree, feature_type="float", label_type="float", int_type=it, reorder_nodes=True, set_size=s, force_cacheline=fc)
                            implementation.implement()
                            
                            materializer=LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                            materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))
                            materializer.deploy() 
                            output = materializer.run() 

                            self.assertAlmostEqual(tree_acc, dt_acc)
                            self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)
                            
                            materializer.clean()

                msg = f"Running test_native_linuxstandalone on DT max_depth={dt.max_depth},int_type={it}, reorder_nodes=False"
                with self.subTest(msg):
                    implementation = Native(tree, feature_type="float", label_type="float", reorder_nodes=False, int_type=it)
                    implementation.implement()
                    
                    materializer = LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                    materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))
                    materializer.deploy() 
                    output = materializer.run() 

                    self.assertAlmostEqual(tree_acc, dt_acc)
                    self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)
                    
                    materializer.clean()

    def test_swap(self):
        for dt in self.dts:
            tree = Tree.from_sklearn(dt)
            scores = tree.score(self.X,self.Y)
            tree_acc_before = scores["Accuracy"]
            tree.swap_nodes()
            tree_acc_after = scores["Accuracy"]
            
            # TODO Enhance this test case and really check if we swapped something 
            self.assertAlmostEqual(tree_acc_before, tree_acc_after)

    def test_quantize(self):
        for dt in self.dts:
            # TODO Enhance this test case and really check if we we do not break the tree
            for r in [None, 2**16]:
                tree = Tree.from_sklearn(dt)
                tree.quantize(quantize_leafs=r, quantize_splits=None)

                tree = Tree.from_sklearn(dt)
                tree.quantize(quantize_leafs=None, quantize_splits=r)

                tree = Tree.from_sklearn(dt)
                tree.quantize(quantize_leafs=r, quantize_splits=r)
            
            tree = Tree.from_sklearn(dt)
            tree.quantize(quantize_leafs=None, quantize_splits="rounding")

    def test_ifelse_params(self):
        msg = f"Running test_ifelse_params on DT with model=None"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=None, feature_type="float", label_type="float")

        msg = f"Running test_ifelse_params on DT with model=5"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=5, feature_type="float", label_type="float")

    def test_native_params(self):
        msg = f"Running test_native_params on DT with model=None"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=None, feature_type="float", label_type="float")

        msg = f"Running test_native_params on DT with model=5"
        with self.subTest(msg):
            self.assertRaises(ValueError, Native, model=5, feature_type="float", label_type="float")

        dt = self.dts[0]
        for s in [-1, None]:
            msg = f"Running test_native_params on DT with max_depth={dt.max_depth},int_type=int, reorder_nodes=True, set_size={s}, force_cacheline=True"

            with self.subTest(msg):
                self.assertRaises(ValueError, Native, model=dt, feature_type="float", label_type="float", reorder_nodes=True, int_type="int", set_size=s, force_cacheline=True)

if __name__ == '__main__':
    unittest.main()