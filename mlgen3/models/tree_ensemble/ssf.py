import os
import numpy as np
import json

from .tree import Tree
from .forest import Forest
from ..linear import Linear
from ..model import Model, PredictionType

#Takes input data and generates a feature vector for each sample by applying each tree in the forest to the sample and return the resulting leaf indices.
#The leaf indices are then used as the input feature vector for a logistic regression.
class SSF(Model):

    def __init__(self):
        super().__init__(PredictionType.REGRESSION)
        self.forest = None
        self.lr = None

    @classmethod
    def from_sklearn(cls, forest_model, lr_model):
        model = SSF()
        model.forest = Forest.from_sklearn(forest_model)
        model.lr = Linear.from_sklearn(lr_model)
        return model