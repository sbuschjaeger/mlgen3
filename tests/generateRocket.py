#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath("../mlgen3_rocket/mlgen3"))
print(sys.path[-1])
import shutil
import tempfile
import numpy as np
import unittest
from sktime import datasets
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from mlgen3.implementations.linear.cpp.native import Native
from mlgen3.implementations.rocket import Rocket as RocketImplementation
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.linear import Linear
from mlgen3.models.rocket import Rocket
from sktime.transformations.panel.rocket import Rocket as RocketPanel
from sktime.datasets import load_from_tsfile


class GenerateRocket:
    

    def getData(self):
        path = os.path.join(os.path.dirname(__file__), "WalkingSittingStanding/")

        X_train, Y_train = load_from_tsfile(
            os.path.join(path, "WalkingSittingStanding_TRAIN.ts")
        )
        X_test, Y_test = load_from_tsfile(
            os.path.join(path, "WalkingSittingStanding_TEST.ts")
        )
        return (X_train, Y_train, X_test, Y_test)

    def setUp(self):
        # initialise dataset
        (X_train, Y_train, X_test, Y_test) = self.getData()
        
        self.X = X_train
        self.y = Y_train

        self.models = []

        # set up sktime model and transformation results for fitting
        sk_rocket: RocketPanel = RocketPanel(num_kernels=3, normalise=False)
        sk_rocket.fit(X_train)
        sk_transformation_results = sk_rocket.transform(X_train)

        # set up linear classifier
        sk_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

        # fit linear model to rocket features
        sk_classifier.fit(sk_transformation_results, Y_train)

        # create own model and add fitted linear model to rocket
        own_rocket = Rocket.from_sklearn(sk_rocket)
        own_rocket.addLinear(sk_classifier)

        self.models.append(own_rocket)

    def native_linuxstandalone(self):
        self.setUp()
        for model in self.models:
            for it in [None, "float"]:
                scores = model.score(self.X, self.y)
                implementation = RocketImplementation(
                    model,
                    feature_type="float",
                    label_type="float",
                    internal_type=it,
                )
                implementation.implement()

                materializer = LinuxStandalone(
                    implementation,
                    measure_accuracy=True,
                    measure_time=True,
                    measure_perf=False,
                )
                path = os.path.join(os.getcwd(), "mlgen3", "rocket_linuxstandalone")
                materializer.materialize(path)
                materializer.deploy()
                # output = materializer.run(True)
                # materializer.clean()


if __name__ == "__main__":
    test = GenerateRocket()
    test.native_linuxstandalone()
