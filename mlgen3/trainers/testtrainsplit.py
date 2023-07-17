from abc import ABC

from sklearn.model_selection import train_test_split
from .trainer import Trainer

class TestTrainSplit(Trainer):

    def __init__(self, X, y, test_size=0.25, random_state=None):
        self.X = X
        self.Y = y
        self.test_size = test_size
        self.random_state = random_state

    def train(self, model):
        model.XTrain,model.XTest,model.YTrain,model.YTest = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)
        model.fit()

        # TODO Do we really need this?
        self.model = model
    
    def score(self):
        if self.model is not None:
            return self.model.score()
        else:
            raise ValueError(f"There is not model to score associated with this trainer. Did you call train?")
