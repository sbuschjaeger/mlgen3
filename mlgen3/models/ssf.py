import copy
import os
import numpy as np
import json

from sklearn.base import check_is_fitted

from .tree_ensemble.tree import Node, Tree
from .tree_ensemble.forest import Forest
from .linear import Linear
from .model import Model, PredictionType

#Takes input data and generates a feature vector for each sample by applying each tree in the forest to the sample and return the resulting leaf indices.
#The leaf indices are then used as the input feature vector for a logistic regression.
class SSF(Model):

    def __init__(self, forest = None, lr = None):
        super().__init__(PredictionType.CLASSIFICATION)
        self.forest = forest
        self.lr = lr

    @classmethod
    def from_sklearn(cls, forest_model, lr_model, threshold=0.4):
        model = SSF()

        model.forest = Forest.from_sklearn(forest_model)
        stumps = SSF.get_stumps(model.forest, threshold)
        model.forest.trees = stumps
        model.forest.weights = [1 for _ in range(len(stumps))]

        model.lr = Linear.from_sklearn(lr_model)
        model.node_mapping = [
            {n.id: i for i, n in enumerate(e.get_leaf_nodes())} 
            for e in model.forest.trees
        ]

        return model

    @staticmethod
    def from_data(X_train, y_train, sk_rf, lr, threshold):
        model = SSF()

        try:
            check_is_fitted(sk_rf)
        except:
            sk_rf.fit(X_train, y_train)
        
        model.forest = Forest.from_sklearn(sk_rf)
        stumps = SSF.get_stumps(model.forest, threshold)
        model.forest.trees = stumps
        model.forest.weights = [1 for _ in range(len(stumps))]

        model.node_mapping = [
            {n.id: i for i, n in enumerate(e.get_leaf_nodes())} 
            for e in model.forest.trees
        ]

        X_one_hot = []
        for i, (e, node_map) in enumerate(zip(model.forest.trees, model.node_mapping)):
            adjusted_idx = [node_map[idx] for idx in e.apply(X_train)] 

            one_hot_leaves = np.zeros( (X_train.shape[0], len(node_map)) )
            one_hot_leaves[np.arange(len(X_train)),adjusted_idx] = 1 
            X_one_hot.append(one_hot_leaves)

        X_one_hot = np.concatenate(X_one_hot,axis=1)

        lr.fit(X_one_hot, y_train)
        model.lr = Linear.from_sklearn(lr)
        return model, sk_rf, lr

    @classmethod
    def get_stumps(cls, forest, threshold):
        def traverse(cls, node, threshold):
            trees = []
            if (node.probLeft != None and node.probRight != None):
                if (node.probLeft >= threshold or node.probRight >= threshold):
                    left = Node()
                    left.id = 1
                    left.prediction = [0, 1]
                    left.numSamples = node.leftChild.numSamples

                    right = Node()
                    right.id = 2
                    right.prediction = [1, 0]
                    right.numSamples = node.rightChild.numSamples

                    root = Node()
                    root.id = 0
                    root.numSamples = node.numSamples
                    root.probLeft = node.probLeft
                    root.probRight = node.probRight
                    root.feature = node.feature 
                    root.split = node.split 
                    root.prediction = None
                    root.leftChild = left
                    root.rightChild = right
                    
                    t = Tree()
                    t.nodes = [root, left, right]
                    t.head = root

                    trees.append(t)
                
                trees.extend( traverse(cls, node.leftChild, threshold) )
                trees.extend( traverse(cls, node.rightChild, threshold) )
            return trees
    
        stumps = []
        for tree in forest.trees:
            stumps.extend( traverse(cls, tree.head, threshold) )

        return stumps

    def predict_proba(self, X):
        
        X_one_hot = []
        for i, (e, node_map) in enumerate(zip(self.forest.trees, self.node_mapping)):
            adjusted_idx = [node_map[idx] for idx in e.apply(X)] 

            one_hot_leaves = np.zeros( (X.shape[0], len(node_map)) )
            one_hot_leaves[np.arange(len(X)),adjusted_idx] = 1 
            X_one_hot.append(one_hot_leaves)

        X_one_hot = np.concatenate(X_one_hot,axis=1)

        return self.lr.predict_proba(X_one_hot)