import os
import numpy as np
import json

from .tree_ensemble.tree import Tree
from .tree_ensemble.forest import Forest
from .linear import Linear
from .model import Model, PredictionType

#Takes input data and generates a feature vector for each sample by applying each tree in the forest to the sample and return the resulting leaf indices.
#The leaf indices are then used as the input feature vector for a logistic regression.
class SSF(Model):

    def __init__(self):
        super().__init__(PredictionType.CLASSIFICATION)
        self.forest = None
        self.lr = None

    @classmethod
    def from_sklearn(cls, forest_model, lr_model):
        model = SSF()
        model.forest = Forest.from_sklearn(forest_model)
        model.lr = Linear.from_sklearn(lr_model)
        return model
    
    def predict_proba(self, X):
        # Generate leaf indices for each sample
        X_leaves = np.array([self.forest.apply(_test) for _test in X])
        X_leaves = np.squeeze(X_leaves, axis=-1)
        # Apply logistic regression to leaf indices
        prediction = self.lr.predict_proba(X_leaves)
        prediction = np.squeeze(prediction, axis=-1)
        return 1.0/(1.0 + np.exp(-prediction)) #there is no activation function in the Linear model, so we apply sigmoid here