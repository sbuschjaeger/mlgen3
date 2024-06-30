import sys
import numpy as np
import pandas as pd

sys.path.append("../mlgen3_rocket/mlgen3")
import os
import unittest
from mlgen3.models.rocket import Rocket
from sktime.transformations.panel.rocket import Rocket as RocketPanel
from sktime.transformations.panel.rocket import MiniRocket as MiniRocketPanel
from sktime.datasets import load_from_tsfile
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV


class TestRocketClassifier(unittest.TestCase):

    def getData(self):

        path = os.path.join(os.path.dirname(__file__), "WalkingSittingStanding/")

        X_train, Y_train = load_from_tsfile(
            os.path.join(path, "WalkingSittingStanding_TRAIN.ts")
        )
        X_test, Y_test = load_from_tsfile(
            os.path.join(path, "WalkingSittingStanding_TEST.ts")
        )

        return (X_train, Y_train, X_test, Y_test)

    def testKernelConvolution(self):
        # initialise dataset
        (X_train, Y_train, X_test, Y_test) = self.getData()

        # set up sktime model and transformation results for comparison
        sk_rocket: RocketPanel = RocketPanel(num_kernels=2, normalise=False)
        sk_rocket.fit(X_train)
        sk_transformation_results = sk_rocket.transform(X_train)

        # set up own model and transformation results for comparison
        own_rocket = Rocket.from_sklearn(sk_rocket)
        own_transformation_results = own_rocket.transform(X_train)

        # if convolution is correct, own_x_train == self.X_train_transformed

        # if you wanna print the results:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        np.set_printoptions(threshold=np.inf)

        # compare both feature ouputs
        for i in range(np.shape(sk_transformation_results)[0]):
            for j in range(np.shape(sk_transformation_results)[1]):
                # results of sktime rocket got rounded on six decimal places, while our
                # rocket results got eight decimal places.

                a = sk_transformation_results.iloc[i, j]
                b = own_transformation_results[i, j]

                assert np.abs(a - b) <= 1e-05, (
                    "rocket results are not equal! a: ",
                    a,
                    " b: ",
                    b,
                    "difference: ",
                    abs(a - b),
                )

    def testRocketClassification_ridgeregression(self):
        # initialise dataset
        (X_train, Y_train, X_test, Y_test) = self.getData()

        # set up sktime model and transformation results for fitting
        sk_rocket: RocketPanel = RocketPanel(num_kernels=2, normalise=False)
        sk_rocket.fit(X_train)
        sk_transformation_results = sk_rocket.transform(X_train)

        # set up linear classifier
        sk_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

        # fit linear model to rocket features
        sk_classifier.fit(sk_transformation_results, Y_train)

        # create own model and add fitted linear model to rocket
        own_rocket = Rocket.from_sklearn(sk_rocket)
        own_rocket.addLinear(sk_classifier)

        # generate rocket results for comparison
        sk_rocket_results = sk_classifier.predict(sk_transformation_results)
        own_rocket_results = own_rocket.predict_proba(X_train)

        # compare both feature ouputs
        for i in range(len(sk_rocket_results)):
            a = sk_rocket_results[i]
            b = own_rocket_results[i]

            assert a == b, ("rocket results are not equal! a: ", a, " b: ", b)

    def testRocketClassification_logisticregression(self):
        # initialise dataset
        (X_train, Y_train, X_test, Y_test) = self.getData()

        # set up sktime model and transformation results for fitting
        sk_rocket: RocketPanel = RocketPanel(num_kernels=2, normalise=False)
        sk_rocket.fit(X_train)
        sk_transformation_results = sk_rocket.transform(X_train)

        # set up linear classifier
        sk_classifier = LogisticRegressionCV(max_iter=1000)

        # fit linear model to rocket features
        sk_classifier.fit(sk_transformation_results, Y_train)

        # create own model and add fitted linear model to rocket
        own_rocket = Rocket.from_sklearn(sk_rocket)
        own_rocket.addLinear(sk_classifier)

        # generate rocket results for comparison
        sk_rocket_results = sk_classifier.predict(sk_transformation_results)
        own_rocket_results = own_rocket.predict_proba(X_train)

        # compare both feature ouputs
        for i in range(len(sk_rocket_results)):
            a = sk_rocket_results[i]
            b = own_rocket_results[i]

            assert a == b, ("rocket results are not equal! a: ", a, " b: ", b)

    def testMiniRocketClassification_ridgeregression(self):
        # initialise dataset
        (X_train, Y_train, X_test, Y_test) = self.getData()

        # set up sktime model and transformation results for fitting
        sk_rocket: MiniRocketPanel = MiniRocketPanel(num_kernels=100)
        sk_rocket.fit(X_train)
        print(sk_rocket.__dict__)
        sk_transformation_results = sk_rocket.transform(X_train)

        # set up linear classifier
        sk_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

        # fit linear model to rocket features
        sk_classifier.fit(sk_transformation_results, Y_train)

        # create own model and add fitted linear model to rocket

        own_rocket = Rocket.from_sklearn(sk_rocket)
        own_rocket.addLinear(sk_classifier)

        # generate rocket results for comparison
        sk_rocket_results = sk_classifier.predict(sk_transformation_results)
        own_rocket_results = own_rocket.predict_proba(X_train)

        # compare both feature ouputs
        for i in range(len(sk_rocket_results)):
            a = sk_rocket_results[i]
            b = own_rocket_results[i]

            assert a == b, ("rocket results are not equal! a: ", a, " b: ", b)

        print("test passed")


if __name__ == "__main__":
    #test = TestRocketClassifier()
    #test.testMiniRocketClassification_ridgeregression()
    unittest.main()
