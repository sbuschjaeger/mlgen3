from abc import ABC

from sklearn.model_selection import train_test_split

from .trainer import Trainer
from mlgen3.models.linear import Linear
from mlgen3.models.tree_ensemble.forest import Forest
from mlgen3.models.tree_ensemble.tree import Tree

class SKTrainer(Trainer):

    def __init__(self, X, y, random_state = None, mode = "split", **kwargs):
        super().__init__()
        assert mode in ["split", "xval"], f"Mode must be one of {{split, xval}}, but you provided {mode}."

        self.X = X
        self.Y = y
        self.mode = mode
        self.split_params = kwargs
        self.random_state = random_state

    def fit(self, model):
        if self.mode == "split":
            XTrain,XTest,YTrain,YTest = train_test_split(self.X, self.Y, **self.split_params, random_state=self.random_state)
        else:
            raise ValueError("Xval is not yet implemented. Sorry!")

        model.fit(XTrain,YTrain)

        the_model = None
        # model is supposed to be a classifier from scikit-learn, but it would be a bit tedious to check for all supported scikit-learn estimators here. Hence, we assume that one of the model classes we support will be able to load this model. If not, then we can load the model anyway. Caveat: This makes debugging / reporting a bit more difficult.
        for c in [Tree, Forest, Linear]:
            try:
                the_model = c.from_sklearn(model)
                break
            except:
                pass
        
        if the_model is None:
            raise ValueError(f"Could not load the model of type {model}, tested {{Tree, Forest, Linear}}. Please make sure that the appropriate from_sklearn function is implemented and working as intended.")
        else:
            the_model.XTrain = XTrain
            the_model.XTest = XTest
            the_model.YTrain = YTrain
            the_model.YTest = YTest

        self.model = the_model
        return the_model
    
    def score(self):
        if self.model is not None:
            return self.model.score()
        else:
            raise ValueError(f"There is not model to score associated with this trainer. Did you call train?")
