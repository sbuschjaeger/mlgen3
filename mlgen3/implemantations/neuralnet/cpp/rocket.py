#implementation of ROCKET https://arxiv.org/abs/1910.13051

import numpy as np

from mlgen3.implemantations.implementation import Implementation
from mlgen3.models.nn.linear import Linear
from sktime.transformations.panel.rocket import (Rocket, MiniRocket, MiniRocketMultivariate, MiniRocketMultivariateVariable)
from sktime.classification.kernel_based import RocketClassifier
from sktime.pipeline import Pipeline

class Rocket(Implementation):

	#constructor taking rocket panel
	def __init__(self, model, feature_type="int", label_type="int", internal_type = "float"):
		super().__init__(model,feature_type,label_type)
		self.internal_type = internal_type

		if isinstance(model, Rocket) or isinstance(model, MiniRocket) or isinstance(model, MiniRocketMultivariate) or isinstance(model, MiniRocketMultivariateVariable):
			self.dict = model.__dict__
			self.kernel = self.dict["kernel"] #kernel = tupel ([weights], [lengths], [biases], [dilations], [paddings], [num_channel_indices], [channel_indices])
			self.weights = self.kernel[0]
			self.length = self.kernel[1]
			self.biases = self.kernel[2]
			self.dilations = self.kernel[3]
			self.paddings = self.kernel[4]
			self.num_channels = self.kernel[5]
			self.channel_indices = self.kernel[6]

		elif isinstance(model, RocketClassifier): #complicated to extract the weights
			print("missing")
			#...
		elif isinstance(model, Pipeline):
			print("missing")
			#...
			#self.dict = model.steps[0].__dict__
			#self.kernel = self.dict["kernel"]
			#self.linear = model.steps[2] #model.steps[1] is StandardScaler
		else:
			raise ValueError("Model isn't part of Rocket")
				

		self.dict = model.__dict__
		self.kernel = self.dict["kernel"]


	#constructor taking rocket-panel AND linear-model
	def __init__(self, panel, linearmodel, feature_type="int", label_type="int", internal_type = "float"):
		super().__init__(panel,feature_type,label_type)
		self.internal_type = internal_type

		if isinstance(panel, Rocket) or isinstance(panel, MiniRocket) or isinstance(panel, MiniRocketMultivariate) or isinstance(panel, MiniRocketMultivariateVariable):
			self.dict = panel.__dict__
			self.kernel = self.dict["kernel"] #kernel = tupel ([weights], [lengths], [biases], [dilations], [paddings], [num_channel_indices], [channel_indices])
			self.weights = self.kernel[0]
			self.lengths = self.kernel[1]
			self.biases = self.kernel[2]
			self.dilations = self.kernel[3]
			self.paddings = self.kernel[4]
			self.num_channel_indices = self.kernel[5]
			self.channel_indices = self.kernel[6]

			self.linear = linearmodel

	
	#returns a list with every entry being a kernel. every kernel is described as a quintuple of (weights:matrix, channels:list, bias:float, dilation:int, padding:int)
	def getKernellist(self):
		weights = self.weights.tolist()
		assert len(weights) > 0, "weights are empty"
		lengths = self.lengths.tolist()
		channel_indices = self.channel_indices.tolist()
		biases = self.biases.tolist()
		dilations = self.dilations.tolist()
		paddings = self.paddings.tolist()
		num_channels = self.num_channels.tolist()
		num_channels.reverse()
		matrix = []
		kernel_list = []
		
		for c in num_channels:
			length = lengths.pop()
			channels = [channel_indices.pop() for i in range(0, c)] # due to multivariate rocket, each kernel can target multiple input channels. That's why the kernel-width could variate and is not always equal.
			channels.reverse() # We also have to remember, which input channels a kernel convolutes over. So we save a kernel with the associated channel_indices
			while c > 0:
				vector = [weights.pop() for i in range(0, length)]  # extract the weights for each vector of a kernel
				vector.reverse()
				c = c - 1
				matrix.append(vector)  # matrix consists of multiple vectors for multiple channels

			matrix.reverse()
			matrix = np.array(matrix)
			matrix = np.squeeze(matrix)  # if kernel got only one channel
			bias = biases.pop()
			dilation = dilations.pop()
			padding = paddings.pop()
			kernel_list.append((matrix, channels, bias, dilation, padding))
			matrix = []

		kernel_list.reverse()
		return kernel_list


		


		

	#If RocketImplementation gets a panel.rocket, no linear model is given. So it has to be set before implementing
	def setLinearModel(self, linear):
		self.linear = linear

	def implement(self):
		assert self.linear is not None, "Linear model has to be set before implementing"
		assert self.kernel is not None, "Kernels are missing"

		alloc = ""
		code = ""
		header = ""



		 





		