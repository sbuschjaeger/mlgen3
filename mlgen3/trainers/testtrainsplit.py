from abc import ABC

from sklearn.model_selection import train_test_split
from .trainer import Trainer

class TestTrainSplit(Trainer):

    def __init__(self, X, y, test_size=0.25, random_state=None):
        # TODO train_test_split mit numpy impl.
        self.XTrain,self.XTest,self.YTrain,self.YTest = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self._model=None

    def train(self, model):
        self._model=model
        model.XTrain=self.XTrain
        model.XTest=self.XTest
        model.YTrain=self.YTrain
        model.YTest=self.YTest
        model.fit()
    
    def score(self):
        if self._model is not None:
            return self._model.score()
