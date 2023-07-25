from abc import ABC
import numpy as np

from weka.core.dataset import create_instances_from_matrices
from weka.filters import Filter

from .trainer import Trainer
from mlgen3.models.linear import Linear
from mlgen3.models.tree_ensemble.forest import Forest
from mlgen3.models.tree_ensemble.tree import Tree


class WekaTrainer(Trainer):

    def __init__(self, X, y, test_size):
        super().__init__()
        self.X = X
        self.Y = y
        self.test_size = test_size

    def fit(self, model):
        idx = np.random.shuffle(list(range(len(self.X))))
        n_test = int(len(self.X) * self.test_size)
        idx_test = idx[:n_test]
        idx_train = idx[n_test:]
        XTrain = self.X[idx_train,:]
        YTrain = self.Y[idx_train,:]
        XTest = self.X[idx_test,:]
        YTest = self.Y[idx_test,:]

        dataset = create_instances_from_matrices(XTrain, YTrain, name="MyDataset")
        nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        nominal.inputformat(dataset)
        dataset = nominal.filter(dataset)

        model.build_classifier(dataset)

        the_model = None
        # model is supposed to be a classifier from weka, but it would be a bit tedious to check for all supported scikit-learn estimators here. Hence, we assume that one of the model classes we support will be able to load this model. If not, then we can load the model anyway. Caveat: This makes debugging / reporting a bit more difficult.
        for c in [Tree, Forest, Linear]:
            try:
                the_model = c.from_weka(model)
                break
            except:
                pass
        
        if the_model is None:
            raise ValueError(f"Could not load the model of type {model}, tested {{Tree, Forest, Linear}}. Please make sure that the appropriate from_weka function is implemented and working as intended.")
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
