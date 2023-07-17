import json, os
import numpy as np

from functools import reduce


# TODO REMOVE THIS DEPENDENCY?
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree
from weka.classifiers import Classifier

from ..model import Model

class Node:
	"""
	A single node of a Decision Tree. There is nothing fancy going on here. It stores all the relevant attributes of a node. 
	"""
	def __init__(self):
		"""Generates a new node. All attributes are initialize to None.
		"""
		# The ID of this node. Makes addressing sometimes easier 
		self.id = None
		
		# The total number of samples seen at this node 
		self.numSamples = None
		
		# The probability to follow the left child
		self.probLeft = None
		
		# The probability to follow the right child
		self.probRight = None

		# An array of predictions, where each entry represents the class weight / probability
		# The specific meaning of each entry depends on the specific method (AdaBoost vs RandomForest)
		self.prediction = None

		# The 'way' of comparison. Usually this is "<=". However, for non-numeric 
		# features, one sometimes need equals comparisons "==". This is currently not 
		# implemented. 
		self.isCategorical = None

		# The index of the feature to compare against
		self.feature = None

		# The threashold the features gets compared against
		self.split = None

		# The right child of this node inside the tree
		self.rightChild = None

		# The left child of this node inside the tree
		self.leftChild = None

		# The probability of this node accumulated from the probabilities of previous
		# edges on the same path.
		# Note: This field is only used after calling getProbAllPaths once
		self.pathProb = None

class Tree(Model):
	"""
	A Decision Tree implementation. There is nothing fancy going on here. It stores all nodes in an array :code:`self.nodes` and has a pointer :code:`self.head` which points to the root node of the tree. Per construction it is safe to assume that :code:`self.head = self.nodes[0]`. 
	"""
	def __init__(self, model):
		super().__init__(model)
		# Array of all nodes
		self.nodes = []

		# Pointer to the root node of this tree
		self.head = None

		# Check if the classifier is already fitted
		# TODO Do we really want / need this?
		if model is not None:
			self.init_from_fitted(model)

	def init_from_fitted(self, original_model):
		if isinstance(original_model, DecisionTreeClassifier):
			if hasattr(original_model, "tree_"):
				tmp = Tree.from_sklearn(original_model)
				self.nodes = tmp.nodes
				self.head = tmp.head
		elif isinstance(original_model, Classifier) and original_model.classname == "weka.classifiers.trees.J48":
			# TODO Is there a better way to determine if this weka model has been build yet?
			if str(original_model) != "No classifier built":
				tmp = Tree.from_weka(original_model)
				self.nodes = tmp.nodes
				self.head = tmp.head
		else:
			raise ValueError("Unrecognizued model is passed to Tree.init_from_fitted(model). Currently implemented are sklearn.tree.DecisionTreeClassifier and weka.classifiers.trees.J48!")
		self.populate_path_probs()

	def predict_proba(self,X):
		"""Applies this tree to the given data and provides the predicted probabilities for each example in X.

		Args:
			X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

		Returns:
			numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
		"""
		if len(X.shape) == 1:
			X = X.reshape(1,X.shape[0])
		
		proba = []
		for x in X:
			node = self.head

			while(node.prediction is None):
				if (x[node.feature] <= node.split): 
					node = node.leftChild
				else:
					node = node.rightChild
			proba.append(node.prediction)

		return np.array(proba)

	def score_model(self, x, y):
		prediction = self.predict_proba(x).argmax(axis=1)
		accuracy = accuracy_score(y, prediction)

		#Compute some value
		return {"Accuracy": accuracy}

	@classmethod
	def from_weka(cls, weka_model):
		if "-B" not in weka_model.to_dict()["options"]:
			raise ValueError("The supplied weka tree was not trained with the '-B' option enabled. Currently we only support binary splits, so please supply '-B' before training your weka model.")

		tree = Tree(None)
		tree.model = weka_model
		tree.nodes = []
		weka_str = str(weka_model)

		def parse_node(subtree, tree):
			n = Node()
			n.id = len(tree.nodes)
			tree.nodes.append(n)

			if "!=" in subtree[0]:
				raise ValueError("Encountered a decision tree with nominal values. This is currently not supported. Please make sure the tree only contains numerical values.")

			if len(subtree) == 1:
				n.prediction = int(subtree[0].split("(")[0].split(":")[1])
				# See https://github.com/bnjmn/weka/blob/master/weka/src/main/java/weka/classifiers/trees/j48/ClassifierSplitModel.java#L180
				if "/" in subtree[0].split("(")[1]:
					n.numSamples = float(subtree[0].split("(")[1].split("/")[0])
				else:
					n.numSamples = float(subtree[0].split("(")[1].split(")")[0])

				return n
			else:
				l = subtree[0]
				# Weka starts to count with 1 instead of 0 ...
				n.feature = int(l.split("<=")[0].split("x")[1]) - 1
				n.split = float(l.split("<=")[1].split(":")[0])
				idx = np.where([f"x{n.feature+1} > {n.split}" in l for l in subtree])[0][0]
				
				# Check if the left node is a leaf, if not ignore it while parsing
				if "(" not in subtree[0]: 
					n.leftChild = parse_node(subtree[1:idx], tree)
				else:
					n.leftChild = parse_node(subtree[:idx], tree)

				# Check if the right node is a leaf, if not ignore it while parsing
				if "(" in subtree[idx]:
					n.rightChild = parse_node(subtree[idx:], tree)
				else:
					n.rightChild = parse_node(subtree[idx+1:], tree)
				return n
		
		tree.head = parse_node(weka_str.split("\n")[3:-5], tree)

		# TODO: Find a way to extract probablities from weka. So far, we are only extracting predictions
		n_classes = int(max([n.prediction for n in tree.nodes if n.prediction is not None])) + 1
		for n in tree.nodes:
			if n.prediction is not None:
				pred = np.zeros(n_classes)
				pred[n.prediction] = 1
				n.prediction = pred

		# Make sure that all other statistics of each tree is populated correctly
		def populate_no_samples(n):
			if n.numSamples is None:
				left = populate_no_samples(n.leftChild)
				right = populate_no_samples(n.rightChild)
				n.numSamples = left + right
				n.probLeft = left / n.numSamples
				n.probRight = right / n.numSamples

			return n.numSamples
		
		_ = populate_no_samples(tree.head)

		return tree
	
	@classmethod
	def from_sklearn(cls, sk_model, ensemble_type = None):
		"""Generates a new tree from an sklearn tree.

		Args:
			sk_model (DecisionTreeClassifier): A DecisionTreeClassifier trained in sklearn.
			ensemble_type (str, optional): Indicates from which sciki-learn ensemble (e.g. :code:`RandomForestClassifier`, :code:`AdaBoostClassifier_SAMME.R`, :code:`AdaBoostClassifier_SAMME`) this DecisionTreeClassifier has been trained, because the probabilities of the leaf-nodes are interpeted differently for each ensemble. If None is set, then a regular DecisionTreeClassifier is assumed. Defaults to None.

		Returns:
			Tree: The newly generated tree.
		"""
		tree = Tree(None)
		tree.model = sk_model
		
		tree.nodes = []
		tree.head = None

		sk_tree = sk_model.tree_

		node_ids = [0]
		tmp_nodes = [Node()]
		while(len(node_ids) > 0):
			cur_node = node_ids.pop(0)
			node = tmp_nodes.pop(0)

			node.numSamples = int(sk_tree.n_node_samples[cur_node])
			if sk_tree.children_left[cur_node] == _tree.TREE_LEAF and sk_tree.children_right[cur_node] == _tree.TREE_LEAF:
				# Get array of prediction probabilities for each class
				proba = sk_tree.value[cur_node][0, :]  
				#computations of prediction values for bagging and boosting models:
				if ensemble_type is None or ensemble_type == "RandomForestClassifier":
					proba = proba / sum(proba)
				elif ensemble_type == "AdaBoostClassifier_SAMME.R":
					proba = proba/sum(proba)
					np.clip(proba, np.finfo(proba.dtype).eps, None, out = proba)
					log_proba = np.log(proba)
					proba = (log_proba - (1. /sk_model.n_classes_) * log_proba.sum())
				elif ensemble_type == "AdaBoostClassifier_SAMME":
					proba = proba/sum(proba)
					proba = proba #*weights
				elif isinstance(sk_model, DecisionTreeRegressor):
					proba = proba #*weights

				node.prediction = proba #* weight
			else:
				node.feature = sk_tree.feature[cur_node]
				node.split = sk_tree.threshold[cur_node]

				node.isCategorical = False # Note: So far, sklearn does only support numrical features, see https://github.com/scikit-learn/scikit-learn/pull/4899
				samplesLeft = float(sk_tree.n_node_samples[sk_tree.children_left[cur_node]])
				samplesRight = float(sk_tree.n_node_samples[sk_tree.children_right[cur_node]])
				node.probLeft = samplesLeft / node.numSamples
				node.probRight = samplesRight / node.numSamples
			
			node.id = len(tree.nodes)
			if node.id == 0:
				tree.head = node

			tree.nodes.append(node)

			if node.prediction is None:
				leftChild = sk_tree.children_left[cur_node]
				node_ids.append(leftChild)
				node.leftChild = Node()
				tmp_nodes.append(node.leftChild)

				rightChild = sk_tree.children_right[cur_node]
				node_ids.append(rightChild)
				node.rightChild = Node()
				tmp_nodes.append(node.rightChild)

		return tree

	@classmethod
	def from_dict(cls, data):
		"""Generates a new tree from the given dictionary. It is assumed that a tree has previously been stored with the :meth:`Tree.to_dict` method.

		Args:
			data (dict): The dictionary from which this tree should be generated. 

		Returns:
			Tree: The newly generated tree.
		"""
		tree = Tree()
		tree.nodes = []

		nodes = [Node()]
		tree.head = nodes[0]
		dicts = [data["model"]]
		while(len(nodes) > 0):
			node = nodes.pop(0)
			entry = dicts.pop(0)
			node.id = len(tree.nodes) #entry["id"]
			node.numSamples = int(entry["numSamples"])
			node.pathProb = entry["pathProb"]

			if "prediction" in entry and entry["prediction"] is not None:
				node.prediction = entry["prediction"]
				tree.nodes.append(node)
			else:
				node.probLeft = float(entry["probLeft"])
				node.probRight = float(entry["probRight"])
				node.isCategorical = (entry["isCategorical"] == "True")
				node.feature = int(entry["feature"])
				node.split = entry["split"]
				#node.rightChild = entry["rightChild"]["id"]
				#node.leftChild = entry["leftChild"]["id"]

				tree.nodes.append(node)
				if node.prediction is None:
					node.rightChild = Node()
					nodes.append(node.rightChild)
					dicts.append(entry["rightChild"])

					node.leftChild = Node()
					nodes.append(node.leftChild)
					dicts.append(entry["leftChild"])
		
		return tree

	def _to_dict(self, node):
		model_dict = {}
		
		if node is None:
			node = self.head
		
		#model_dict["id"] = node.id
		model_dict["probLeft"] = node.probLeft
		model_dict["probRight"] = node.probRight
		model_dict["prediction"] = node.prediction
		model_dict["isCategorical"] = node.isCategorical
		model_dict["feature"] = node.feature
		model_dict["split"] = node.split
		model_dict["pathProb"] = node.pathProb
		model_dict["numSamples"] = node.numSamples

		if node.rightChild is not None:
			model_dict["rightChild"] = self._to_dict(node.rightChild)

		if node.leftChild is not None:
			model_dict["leftChild"] = self._to_dict(node.leftChild)

		return model_dict

	def to_dict(self):
		"""Stores this tree as a dictionary which can be loaded with :meth:`Tree.from_dict`.

		Returns:
			dict: The dictionary representation of this tree.
		"""
		model_dict = super().to_dict()
		model_dict["model"] = self._to_dict(self.head)
		
		return model_dict

	def populate_path_probs(self, node = None, curPath = None, allPaths = None, pathNodes = None, pathLabels = None):
		if node is None:
			node = self.head

		if curPath is None:
			curPath = []

		if allPaths is None:
			allPaths = []

		if pathNodes is None:
			pathNodes = []

		if pathLabels is None:
			pathLabels = []

		if node.prediction is not None:
			allPaths.append(curPath)
			pathLabels.append(pathNodes)
			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Leaf nodes "+str(node.id)+" : "+str(curProb))
		else:
			if len(pathNodes) == 0:
				curPath.append(1)

			pathNodes.append(node.id)

			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Root or Split nodes "+str(node.id)+ " : " +str(curProb))
			self.populate_path_probs(node.leftChild, curPath + [node.probLeft], allPaths, pathNodes + [node.leftChild.id], pathLabels)
			self.populate_path_probs(node.rightChild, curPath + [node.probRight], allPaths, pathNodes + [node.rightChild.id], pathLabels)

		#return allPaths, pathLabels

	## SOME STATISTICS FUNCTIONS ##

	# def getMaxDepth(self):
	#     paths = self.getAllPaths()
	#     return max([len(p) for p in paths])

	# def getAvgDepth(self):
	#     paths = self.getAllPaths()
	#     return sum([len(p) for p in paths]) / len(paths)

	# def getNumNodes(self):
	#     return len(self.nodes)
