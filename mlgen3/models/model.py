from abc import ABC, abstractmethod

from enum import Enum

import numpy as np


class PredictionType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class Model(ABC):

    # Keep x,y, trainconfig, ... remembered
    # x_train, y_train, x_test, y_test

    def __init__(self, prediction_type):
        assert (
            prediction_type == PredictionType.CLASSIFICATION
        ), "Currently only classification is supported"
        self.prediction_type = prediction_type

        self.original_model = None
        self.XTrain = None
        self.XTest = None
        self.YTrain = None
        self.YTest = None

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X=None, y=None):
        if X is None or y is None:
            if self.XTest is None or self.YTest is None:
                raise ValueError(
                    f"Cannot score model, because there is no data to score it on. Make sure that either X and y are supplied for scoring or that XTest and YTest are set, i.e. by using a TestTrainSplit Trainer. X was {X} and Y was {y}. XTest was  {self.XTest} and YTest was {self.YTest}"
                )
            else:
                X = self.XTest
                y = self.YTest
        else:
            # TODO: Currently we have to call score() on a self-trained model _before_ we can properly implement it because implementation will access self.implementation.model.XTest
            self.XTest = X
            self.YTest = y

        if self.prediction_type == PredictionType.CLASSIFICATION:
            prediction = self.predict_proba(X)
            print(np.shape(prediction))
            accuracy = np.mean(prediction.argmax(axis=1) == y)

            # Compute some value
            return {"Accuracy": accuracy}
