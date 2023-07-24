#!/usr/bin/env python3

import os
import shutil
import tempfile
import numpy as np
import unittest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.linear.cpp.native import Native
from mlgen3.materializer.cpp.linuxremote import LinuxRemote
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.linear import Linear

class TestLinearClassifiers(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

        self.model = LogisticRegression(fit_intercept=True, random_state=0)
        self.model.fit(self.X,self.y)

    def test_native_linuxstandalone(self):
        lin_model = Linear(self.model)
        scores = lin_model.score(self.X,self.y)
        lin_model_acc = scores["Accuracy"]
        sk_acc = accuracy_score(self.model.predict(self.X), self.y)
        implementation = Native(lin_model, feature_type="float", label_type="float",internal_type="float")
        implementation.implement()
        
        materializer = LinuxRemote(implementation, measure_accuracy=True, measure_time=True, measure_perf=False, hostname="lamarr-odroid1", remote_compile=True, ssh_config=None)
        materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative"))
        materializer.deploy(verbose=True) 
        output = materializer.run(True) 
        materializer.clean()
        
        self.assertAlmostEqual(lin_model_acc, sk_acc, places=3)
        self.assertAlmostEqual(float(output["Accuracy"]), sk_acc*100.0, places=3)

        # https://wiki.odroid.com/odroid-c4/software/building_kernel
        cross_materializer = LinuxRemote(implementation, measure_accuracy=True, measure_time=True, measure_perf=False, hostname="lamarr-odroid1", remote_compile=False, ssh_config=None, compiler="/opt/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")
        cross_materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative"))
        cross_materializer.deploy(verbose=True) 
        output = cross_materializer.run(True) 
        cross_materializer.clean()

        self.assertAlmostEqual(lin_model_acc, sk_acc, places=3)
        self.assertAlmostEqual(float(output["Accuracy"]), sk_acc*100.0, places=3)

        # TODO PROPER CLEANUP
        # if os.path.exists(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative")):
        #     shutil.rmtree(os.path.join(tempfile.gettempdir(), "mlgen3", "TestLinearClassifierNative"))

        # materializer=LinuxRemote(implementation, measure_accuracy=True, measure_time=True, measure_perf=False, hostname="ls8fpga3", remote_compile=False, ssh_config="TODO", compiler="TODO")

if __name__ == '__main__':
    unittest.main()