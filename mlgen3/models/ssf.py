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
        model.node_mapping = [
            {n.id: i for i, n in enumerate(e.get_leaf_nodes())} 
            for e in model.forest.trees
        ]

        return model
    
    def predict_proba(self, X):
        X_one_hot = []
        for i, (e, node_map) in enumerate(zip(self.forest.trees, self.node_mapping)):
            adjusted_idx = [node_map[idx] for idx in e.apply(X)] 

            one_hot_leaves = np.zeros( (X.shape[0], len(node_map)) )
            one_hot_leaves[np.arange(len(X)),adjusted_idx] = 1 
            X_one_hot.append(one_hot_leaves)

        X_one_hot = np.concatenate(X_one_hot,axis=1)

        # Apply logistic regression to leaf indices
        return self.lr.predict_proba(X_one_hot)