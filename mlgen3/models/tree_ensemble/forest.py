import os
import numpy as np
import json

from .tree import Tree
from ..model import Model, PredcitionType

class Forest(Model):
	def __init__(self):
		super().__init__(PredcitionType.CLASSIFICATION)
		self.trees = []
		self.weights = []

	@classmethod
	def from_sklearn(cls, sk_model):
		from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier

		"""Generates a new ensemble from an sklearn ensemble.

		Args:
			sk_model: A scikit-learn ensemble. Currently supported are :code:`{BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor}`

		Returns:
			Ensemble: The newly generated ensemble.
		"""

		model = Forest()
		 
		if isinstance(sk_model, (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)):
			#obj.category = "ensemble"

			#if isinstance(sk_model, AdaBoostClassifier):
				#obj.type = "AdaBoostClassifier_" + sk_model.algorithm #AdaBoost Type SAMME, SAMME.R

			num_models = len(sk_model.estimators_)
			if isinstance(sk_model, (AdaBoostClassifier)):
				model.weights = sk_model.estimator_weights_
			elif isinstance(sk_model, (GradientBoostingClassifier)):
				model.weights = [sk_model.learning_rate for _ in range(num_models*sk_model.n_classes_)] #weights are equal to the learning rate for GradientBoosting
				if sk_model.init_ != 'zero':
					raise ValueError("""'zero' is the only supported init classifier for gradient boosting models""")
					#TODO implement class prior classifier					 
			else:
				model.weights = [1.0/num_models for i in range(num_models)]

			model.trees = []
			for i, base in enumerate(sk_model.estimators_):
				model.trees.append(Tree.from_sklearn(base))
			
			# # Do not trust the number of classes of the ensemble, because a linear / quadratic function may have fewer classes
			# model.classes = model.models[0].classes

		else:
			raise ValueError("""
				Received an unrecognized sklearn model. Expected was one of: 
				BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
			""" % type(sk_model).__name__)
		return model

	@classmethod
	def from_dict(cls, data):
		"""Generates a new ensemble from the given dictionary. It is assumed that the ensemble has previously been stored with the :meth:`Ensemble.to_dict` method.

		Args:
			data (dict): The dictionary from which this ensemble should be generated. 

		Returns:
			Ensemble: The newly generated ensemble.
		"""
		model = Forest()
		#obj = super().from_dict(data)

		for entry in data["models"]:
			if "file" in entry:
				if os.path.isfile( entry["file"] ):
					with open(entry["file"]) as f:
						data = json.load(f)
					model.trees.append(Tree.from_dict(data))
			else:
				model.trees.append(Tree.from_dict(entry["model"]))
			model.weights.append(entry["weight"])

		return model

	def predict_proba(self,X):
		"""Applies this ensemble to the given data and provides the predicted probabilities for each example in X.

		Args:
			X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

		Returns:
			numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
		"""
		if len(X.shape) == 1:
			X = X.reshape(1,X.shape[0])
		
		all_proba = []
		for m, w in zip(self.trees, self.weights):
			iproba = w * m.predict_proba(X)
			all_proba.append(iproba)
		all_proba = np.array(all_proba)

		return all_proba.sum(axis=0)

	def swap_nodes(self):
		for t in self.trees:
			t.swap_nodes()

	def quantize(self, quantize_splits = None, quantize_leafs = None):
		for t in self.trees:
			t.quantize(quantize_splits=quantize_splits, quantize_leafs=quantize_leafs)

	def prune(self, x_prune, y_prune, pruning_method, **kwargs):
		"""Performs ensemble pruning on the given model given the data. Pruning is implemented via the PyPruning package (https://github.com/sbuschjaeger/PyPruning).
		
		The specific pruning method can be chosen via :code:`pruning_method` and all additional arguments are passed to this function. For more details see https://sbuschjaeger.github.io/PyPruning/html/papers.html
		"""
		from PyPruning.Papers import create_pruner

		pruner = create_pruner(pruning_method, **kwargs)
		n_classes = self.trees[0].n_classes
		pruner.prune(x_prune, y_prune, self.trees, n_classes = n_classes)

	def to_dict(self):
		"""Stores this ensemble as a dictionary which can be loaded with :meth:`Ensemble.from_dict`.

		Returns:
			dict: The dictionary representation of this ensemble.
		"""
		model_dict = super().to_dict()

		trees = []
		for m,w in zip(self.trees, self.weights):
			d = {}
			d["weight"] = w
			d["model"] = m.to_dict()
			trees.append(d)
		model_dict["trees"] = trees

		return model_dict