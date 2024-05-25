import numpy as np
from .model import Model, PredictionType

class Linear(Model):
	def __init__(self):
		super().__init__(PredictionType.CLASSIFICATION)
		self.coef = []
		self.intercept = []

	@classmethod
	def from_sklearn(cls, sk_model):
		"""Generates a new linear model from sklearn. 

		Args:
			sk_model (LinearModel): A LinearModel trained in sklearn (e.g. SGDClassifier, RidgeClassifier, Perceptron, etc.).
			name (str, optional): The name of this model. Defaults to "Model".
			accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.

		Returns:
			Linear: The newly generated linear model.
		"""
		model = Linear()
		model.intercept = sk_model.intercept_
		model.coef = sk_model.coef_.T
		model.original_model = sk_model

		return model
	
	@classmethod
	def from_dict(cls, data):
		"""Generates a new linear model from the given dictionary. It is assumed that a linear model has previously been stored with the :meth:`Linear.to_dict` method.

		Args:
			data (dict): The dictionary from which this linear model should be generated. 

		Returns:
			Tree: The newly generated linear model.
		"""
		model = Linear()
		model.intercept = np.array(data["intercept"])
		model.coef = np.array(data["coeff"])

		return model

	def predict_proba(self,X):
		"""Applies this linear model to the given data and provides the predicted probabilities for each example in X.

		Args:
			X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

		Returns:
			numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
		"""
		if len(X.shape) == 1:
			X = X.reshape(1,X.shape[0])
		
		# Somewhat stolen and modified from safe_sparse_dot in sklearn extmath.py
		if X.ndim > 2 or self.coef.ndim > 2:
			proba = np.dot(X, self.coef)
		else:
			proba = X @ self.coef

		# proba = []
		# for x in X:
		#     proba.append(np.inner(x, self.coef) + self.intercept)
		return np.array(proba) + self.intercept
		
	def to_dict(self):
		"""Stores this linear model as a dictionary which can be loaded with :meth:`Linear.from_dict`.

		Returns:
			dict: The dictionary representation of this linear model.
		"""
		return {
			"intercept":self.intercept,
			"coeff":self.coef
		}