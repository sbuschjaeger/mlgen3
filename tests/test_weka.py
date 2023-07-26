#!/usr/bin/env python3
import unittest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from mlgen3.models.tree_ensemble.tree import Tree
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices
from weka.classifiers import Classifier
from weka.filters import Filter

class TestWeka(unittest.TestCase):

    def setUp(self):
        self.X, self.y = datasets.load_digits(return_X_y=True)

        self.dts = []
        for d in [1, 5, None]:
            dt = DecisionTreeClassifier(max_depth=d, random_state=0)
            dt.fit(self.X,self.y)
            self.dts.append(Tree.from_sklearn(dt))
 
    def test_dt_from_weka(self):
        jvm.start()
        dataset = create_instances_from_matrices(self.X, self.y, name="Digits dataset")
        nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        nominal.inputformat(dataset)
        dataset = nominal.filter(dataset)
        dataset.class_is_last()
        weka_dt = Classifier(classname="weka.classifiers.trees.J48", options=["-B"])
        weka_dt.build_classifier(dataset)

        ypred = []
        for x in dataset:
            ypred.append(weka_dt.classify_instance(x))
        
        weka_dt_acc = accuracy_score(ypred, self.y)
        
        tree = Tree.from_weka(weka_dt)
        scores = tree.score(self.X,self.y)
        tree_acc = scores["Accuracy"]
        self.assertAlmostEqual(weka_dt_acc, tree_acc, places=3)

        jvm.stop()

if __name__ == '__main__':
    unittest.main()