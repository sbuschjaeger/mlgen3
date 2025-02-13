#!/usr/bin/env python3
import sys

sys.path.append("../mlgen3")
import os
import shutil
import tempfile
import numpy as np
import unittest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlgen3.implementations.linear.cpp.native import Native
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.linear import Linear


class TestLinearClassifiers(unittest.TestCase):

    def setUp(self):
        # TODO ADD BINARY DATASET
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

        self.models = []
        model = LogisticRegression(fit_intercept=True, random_state=0)
        model.fit(self.X, self.y)
        self.models.append(model)

        model = LogisticRegression(fit_intercept=False, random_state=0)
        model.fit(self.X, self.y)
        self.models.append(model)

    def test_from_scikitlearn(self):
        for m in self.models:
            msg = f"Running test_from_scikitlearn on Linear with fit_intercept={m.fit_intercept}"
            with self.subTest(msg):
                lin_model = Linear.from_sklearn(m)
                scores = lin_model.score(self.X, self.y)
                lin_model_acc = scores["Accuracy"]
                sk_acc = accuracy_score(m.predict(self.X), self.y)

                self.assertAlmostEqual(lin_model_acc, sk_acc, places=3)

    def test_native_linuxstandalone(self):
        for m in self.models:
            for it in [None, "float"]:
                msg = f"Running test_from_scikitlearn on Linear with fit_intercept={m.fit_intercept} and internal_type={it}"
                with self.subTest(msg):
                    lin_model = Linear.from_sklearn(m)
                    scores = lin_model.score(self.X, self.y)
                    lin_model_acc = scores["Accuracy"]
                    sk_acc = accuracy_score(m.predict(self.X), self.y)
                    implementation = Native(
                        lin_model,
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
                    materializer.materialize(
                        os.path.join(
                            tempfile.gettempdir(),
                            "mlgen3",
                            "TestLinearClassifierNative",
                        )
                    )
                    materializer.deploy()
                    output = materializer.run(True)

                    self.assertAlmostEqual(lin_model_acc, sk_acc, places=3)
                    self.assertAlmostEqual(
                        float(output["Accuracy"]), sk_acc * 100.0, places=3
                    )

                    materializer.clean()

    def test_native_params(self):
        msg = f"Running test_native_params on linear with model=None"
        with self.subTest(msg):
            self.assertRaises(
                ValueError,
                Native,
                model=None,
                feature_type="float",
                label_type="float",
                internal_type=None,
            )


if __name__ == "__main__":
    unittest.main()
