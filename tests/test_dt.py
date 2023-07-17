#!/usr/bin/env python3

import os
import tempfile
import numpy as np
import unittest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.cpp.ifelse import IfElse
from mlgen3.materializer.linuxcppstandalone import LinuxCPPStandalone

from mlgen3.models.tree_ensemble.tree import Tree
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.core.dataset import missing_value

import math
from weka.core.dataset import Instances, Instance, Attribute
from weka.plot.graph import plot_graph

class TestDecisionTreeClassifiers(unittest.TestCase):

    def test_from_weka(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        jvm.start()
        dataset = create_instances_from_matrices(X, y, name="Iris dataset")
        nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        nominal.inputformat(dataset)
        dataset = nominal.filter(dataset)
        dataset.class_is_last()
        dt = Classifier(classname="weka.classifiers.trees.J48", options=["-B"])
        dt.build_classifier(dataset)

        ypred = []
        for x in dataset:
            ypred.append(dt.classify_instance(x))
        
        dt_acc = accuracy_score(ypred, y)
        
        tree = Tree(dt)
        scores = tree.score(X,y)
        tree_acc = scores["Accuracy"]
        self.assertAlmostEqual(tree_acc, dt_acc, places=3)

        jvm.stop()

    def test_from_scikitlearn(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        dt = DecisionTreeClassifier(random_state=0, max_depth=2)
        dt.fit(X,y)

        tree = Tree(dt)
        scores = tree.score(X,y)
        tree_acc = scores["Accuracy"]
        dt_acc = accuracy_score(dt.predict(X), y)

        self.assertAlmostEqual(tree_acc, dt_acc, places=3)

    def test_ifelse_linuxstandalone(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        dt = DecisionTreeClassifier(random_state=0, max_depth=2)
        dt.fit(X,y)

        tree = Tree(dt)
        scores = tree.score(X,y)
        tree_acc = scores["Accuracy"]
        dt_acc = accuracy_score(dt.predict(X), y)

        implementation = IfElse(tree, feature_type="float", label_type="float")
        implementation.implement()
        
        materializer=LinuxCPPStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
        materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestDecisionTreeClassifier"))
        materializer.deploy() 
        output = materializer.run(True) 

        self.assertAlmostEqual(tree_acc, dt_acc)
        self.assertAlmostEqual(float(output["Accuracy"]), dt_acc*100.0, places=3)

if __name__ == '__main__':
    unittest.main()