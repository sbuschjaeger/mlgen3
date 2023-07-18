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
from mlgen3.materializer.linuxcppstandalone import LinuxCPPStandalone

from mlgen3.models.tree_ensemble.tree import Tree
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.core.dataset import missing_value

class TestDecisionTreeClassifiers(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

        self.dts = []
        for d in [1, 5]:
            dt = DecisionTreeClassifier(max_depth=d, random_state=0)
            dt.fit(self.X,self.y)
            self.dts.append(dt)
 
    def test_from_weka(self):
        jvm.start()
        dataset = create_instances_from_matrices(self.X, self.y, name="Iris dataset")
        nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        nominal.inputformat(dataset)
        dataset = nominal.filter(dataset)
        dataset.class_is_last()
        dt = Classifier(classname="weka.classifiers.trees.J48", options=["-B"])
        dt.build_classifier(dataset)

        ypred = []
        for x in dataset:
            ypred.append(dt.classify_instance(x))
        
        dt_acc = accuracy_score(ypred, self.y)
        
        tree = Tree(dt)
        scores = tree.score(self.X,self.y)
        tree_acc = scores["Accuracy"]
        self.assertAlmostEqual(dt_acc, tree_acc, places=3)

        jvm.stop()

    def test_from_scikitlearn(self):
        for dt in self.dts:
            tree = Tree(dt)
            scores = tree.score(self.X,self.y)
            tree_acc = scores["Accuracy"]
            dt_acc = accuracy_score(dt.predict(self.X), self.y)

            self.assertAlmostEqual(dt_acc, tree_acc, places=3)

    def test_ifelse_linuxstandalone(self):
        for dt in self.dts:
            msg = f"Running test_ifelse_linuxstandalone on DT with max_depth = {dt.max_depth}"
            with self.subTest(msg):
                tree = Tree(dt)
                scores = tree.score(self.X,self.y)
                tree_acc = scores["Accuracy"]
                dt_acc = accuracy_score(dt.predict(self.X), self.y)

                implementation = IfElse(tree, feature_type="float", label_type="float")
                implementation.implement()
                
                materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierIfElse"))
                materializer.deploy() 
                output = materializer.run(True) 

                self.assertAlmostEqual(tree_acc, dt_acc, places=3)
                self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)

                if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierIfElse")):
                    shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierIfElse"))

    def test_native_linuxstandalone(self):
        for dt in self.dts:
            tree = Tree(dt)
            scores = tree.score(self.X,self.y)
            tree_acc = scores["Accuracy"]
            dt_acc = accuracy_score(dt.predict(self.X), self.y)

            for it in [None, "int"]:
                for s in [1, 8]:
                    for fc in [True, False]:
                        msg = f"Running test_native_linuxstandalone on DT with max_depth={dt.max_depth}, int_type={it}, reorder_nodes=True, set_size={s}, force_cacheline={fc}"
                        with self.subTest(msg):
                            implementation = Native(tree, feature_type="float", label_type="float", int_type=it, reorder_nodes=True, set_size=s, force_cacheline=fc)
                            implementation.implement()
                            
                            materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                            materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))
                            materializer.deploy() 
                            output = materializer.run(True) 

                            self.assertAlmostEqual(tree_acc, dt_acc)
                            self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)
                            
                            if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative")):
                                shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))

                msg = f"Running test_native_linuxstandalone on DT max_depth={dt.max_depth},int_type={it}, reorder_nodes=False"
                with self.subTest(msg):
                    implementation = Native(tree, feature_type="float", label_type="float", reorder_nodes=False, int_type=it)
                    implementation.implement()
                    
                    materializer = LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                    materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))
                    materializer.deploy() 
                    output = materializer.run(True) 

                    self.assertAlmostEqual(tree_acc, dt_acc)
                    self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)
                    
                    if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative")):
                        shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifierNative"))

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