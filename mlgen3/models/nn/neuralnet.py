from sklearn.metrics import accuracy_score
from math import ceil

import numpy as np
from mlgen3.models.nn.activations import Step

from mlgen3.models.nn.batchnorm import BatchNorm
from ..model import Model

class NeuralNet(Model):
	"""A (simplified) neural network model. This class currently supports feed-forward multi-layer perceptrons as well as feed-forward convnets. In detail the following operations are supported
		
		- Linear Layer
		- Convolutional Layer
		- Sigmoid Activation
		- ReLU Activation
		- LeakyRelu Activation
		- MaxPool
		- AveragePool
		- LogSoftmax
		- LogSoftmax
		- Multiplication with a constant (Mul)
		- Reshape
		- BatchNormalization
	
	All layers are stored in :code:`self.layer` which is already order for execution. Additionally, the original onnx_model is stored in :code:`self.onnx_model`.

	This class loads ONNX files to build the internal computation graph. This can sometimes become a little tricky since the ONNX exporter work differently for each framework / version. In PyToch we usually use 

	.. code-block:: python

		dummy_x = torch.randn(1, x_train.shape[1], requires_grad=False)
		torch.onnx.export(model, dummy_x, os.path.join(out_path,name), training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

	**Important**: This class automatically merges "Constant -> Greater -> Constant -> Constant -> Where" operations into a single step layer. This is specifically designed to parse Binarized Neural Networks, but might be wrong for some types of networks. 
	"""
	def __init__(self, model):
		"""Constructor of NeuralNet.

		Args:
			onnx_neural_net (str): Path to the onnx file.
			accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.
			name (str, optional): The name of this model. Defaults to "Model".
		"""
		super().__init__(model)
		
		self.layers = []

		# Check if the classifier is already fitted
		# TODO Do we really want / need this?
		if model is not None:
			self.init_from_fitted(model)


	def score_model(self, x, y):
		prediction = self.predict_proba(x).argmax(axis=1)
		accuracy = accuracy_score(y, prediction)

		#Compute some value
		return {"Accuracy": accuracy}

	def init_from_fitted(self, original_model):
		self.layers = original_model

	def merge_bn_and_step(self):
		"""Merges subsequent BatchNorm and Step layers into a new Step layer with adapted thresholds in a single pass. Currently there is no recursive merging applied.

		TODO: Perform merging recursively. 

		Args:
			model (NeuralNet): The NeuralNet model.

		Returns:
			NeuralNet: The NeuralNet model with merged layers.
		"""
		# Merge BN + Step layers for BNNs
		new_layers = []
		last_layer = None
		for layer_id, layer in enumerate(self.layers):
			if last_layer is not None:
				if isinstance(last_layer, BatchNorm) and isinstance(layer, Step):
					layer.threshold = layer.threshold -last_layer.bias / last_layer.scale
				else:
					new_layers.append(last_layer)
			last_layer = layer
			
		new_layers.append(last_layer)
		self.layers = new_layers


	def predict_proba(self,X):
		"""Applies this NeuralNet to the given data and provides the predicted probabilities for each example in X. This function internally calls :code:`onnxruntime.InferenceSession` for inference.. 

		Args:
			X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

		Returns:
			numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
		"""
		if len(X.shape) == 1:
			X = X.reshape(1,X.shape[0])
		
		for l in self.layers:
			X = l(X)

		return X