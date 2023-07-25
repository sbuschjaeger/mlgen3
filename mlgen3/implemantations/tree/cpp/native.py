import heapq

from .ensemble import Ensemble

class Native(Ensemble):

	def __init__(self, model, feature_type="int", label_type="int", int_type=None, reorder_nodes = False, set_size = 8, force_cacheline = False):
		super().__init__(model,feature_type,label_type)
		self.reorder_nodes = reorder_nodes
		self.set_size = set_size
		self.force_cacheline = force_cacheline
		self.int_type = int_type

		if reorder_nodes and (set_size is None or set_size <= 0):
			raise ValueError("If you want to reorder nodes, please give a valid set size!")

	def reorder(self, model, set_size = 8, force_cacheline = False, **kwargs):
		"""Extracts a list of inner_nodes and leaf_nodes from the model while storing additional left_is_leaf / right_is_leaf / id fields in the inner_nodes for the code generation. The left_is_leaf/right_is_leaf fields indicate if the left/right child of an inner node is a leaf note, whereas the id field can be used to access the correct index in the array, e.g. by using node.leftChild.id. This method tries to place nodes in consecutive order which have a maximum probability to be executed together. This basically implements algorithm 2 from the given reference.

		Reference:
			Buschjäger, Sebastian, et al. "Realization of random forest for real-time evaluation through tree framing." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

		Args:
			model (Tree): A given tree model.
			set_size (int, optional): The cache set size. Defaults to 8.
			force_cacheline (bool, optional): If True then "padding" nodes are introduced to fill the entire cache line. Defaults to False.

		Returns:
			inner_nodes: The list of inner_nodes in the order given by the BFS
			leaf_nodes: The list of leaf_nodes in the order given by the BFS
		"""
		leaf_nodes = []
		inner_nodes = []
		to_expand = []
		# Per convention heappush uses the first element of a tuple for comparisons. We are using
		#   (pathProb, parent id, true/false if this node is a left subtree, node)
		# to manage the nodes. Note that heapq maintains a min-heap, whereas we require a max-heap. 
		# Hence we use the negative pathProb.
		heapq.heappush(to_expand, (-model.head.pathProb,-1, False, model.nodes[0]))

		while( len(to_expand) > 0 ):
			# Extract the current node with its meta information. The pathProb can be ignored 
			_, pid, is_left, n = heapq.heappop(to_expand) 
			cset_size = 0

			# Is the current set full already? 
			while (cset_size < set_size):
				if n.prediction is not None:
					# A leaf node is found and hence this path cannot be explored further.
					if pid >= 0:

						# Make sure the id of our parent node points to the correct index and set is_leaf correctly
						if is_left:
							inner_nodes[pid].leftChild.id = len(leaf_nodes)
							inner_nodes[pid].left_is_leaf = "true"
						else:
							inner_nodes[pid].rightChild.id = len(leaf_nodes)
							inner_nodes[pid].right_is_leaf = "true"

					if force_cacheline:
						# Fill in padding / dummy nodes if cset is not full yet
						for _ in range(cset_size - set_size):
							inner_nodes.append(model.head)
							cset_size += 1
					
					leaf_nodes.append(n)
					break
				else:
					# An inner node is found and hence we may explore this path. This node is added to the inner_nodes 
					# and hence cset_size increases by one
					cset_size += 1
					cid = len(inner_nodes)
					inner_nodes.append(n)

					# Just set the is_leaf fields to false for all nodes. If we reach an actual leaf node then we will
					# set it to "true" in the above code path
					n.left_is_leaf = "false"
					n.right_is_leaf = "false"
					
					if pid >= 0:
						# Make sure the id of our parent node points to the correct index
						if is_left:
							inner_nodes[pid].leftChild.id = cid
						else:
							inner_nodes[pid].rightChild.id = cid

					# Directly explore the left / right sub-tree without using the heap. 
					# Put the other sub-tree on the heap for later. 
					# Since heappush maintains the heap-invariant there is not need to call heapify
					if cset_size < set_size:
						if n.leftChild.pathProb > n.rightChild.pathProb:
							heapq.heappush(to_expand, (-n.rightChild.pathProb, cid, False, n.rightChild))
							pid, is_left, n = cid, True, n.leftChild
						else:
							heapq.heappush(to_expand, (-n.leftChild.pathProb, cid, True, n.leftChild))
							pid, is_left, n = cid, False, n.rightChild
					else:
						# If the set size is already full then continue normally by including both children into the heap
						heapq.heappush(to_expand, (-n.rightChild.pathProb, cid, False, n.rightChild))
						heapq.heappush(to_expand, (-n.leftChild.pathProb, cid, True, n.leftChild))

		return inner_nodes, leaf_nodes

	def get_nodes(self, model):
		"""Extracts a list of inner_nodes and leaf_nodes from the model while storing additional left_is_leaf / right_is_leaf / id fields in the inner_nodes for the code generation. The left_is_leaf/right_is_leaf fields indicate if the left/right child of an inner node is a leaf note, whereas the id field can be used to access the correct index in the array, e.g. by using node.leftChild.id. This method traverses the tree in BFS order and does not perform any optimizations on the order.

		Args:
			model (Tree): A given tree model.

		Returns:
			inner_nodes: The list of inner_nodes in the order given by the BFS
			leaf_nodes: The list of leaf_nodes in the order given by the BFS
		"""
		leaf_nodes = []
		inner_nodes = []
		to_expand = [(-1, False, model.nodes[0]) ]

		# Make sure that the nodes are correctly numbered given their current order
		# To do so, traverse the tree in BFS order and maintain a tuple (parent id, true/false if this is a left child, node)
		# We also split the inner nodes and the leaf nodes into two arrays inner_nodes and leaf_nodes
		# Last we make sure to set the left_is_leaf / right_is_leaf fields of the node which is then accessed during code generation
		while( len(to_expand) > 0 ):
			pid, is_left, n = to_expand.pop(0)

			if n.prediction is not None:
				if pid >= 0:
					# Make sure the id of our parent node points to the correct index and set is_leaf correctly
					if is_left:
						inner_nodes[pid].leftChild.id = len(leaf_nodes)
						inner_nodes[pid].left_is_leaf = "true"
					else:
						inner_nodes[pid].rightChild.id = len(leaf_nodes)
						inner_nodes[pid].right_is_leaf = "true"

				leaf_nodes.append(n)
			else:
				cid = len(inner_nodes)
				inner_nodes.append(n)

				# Just set the is_leaf fields to false for all nodes. If we reach an actual leaf node then we will
				# set it to "true" in the above code path
				n.left_is_leaf = "false"
				n.right_is_leaf = "false"
				
				if pid >= 0:
					# Make sure the id of our parent node points to the correct index
					if is_left:
						inner_nodes[pid].leftChild.id = cid
					else:
						inner_nodes[pid].rightChild.id = cid

				to_expand.append( (cid, True, n.leftChild) )
				to_expand.append( (cid, False, n.rightChild) )
		return inner_nodes, leaf_nodes

	def implement_member(self, number): 
		if number is None:
			tree = self.model.trees[0]
			header = f"std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x)"
		else:
			tree = self.model.trees[number]
			header = f"std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &x)"

		if self.reorder_nodes:
			tree.populate_path_probs()
			inner_nodes, leaf_nodes = self.reorder(tree, self.set_size, self.force_cacheline)
		else:
			inner_nodes, leaf_nodes = self.get_nodes(tree)

		if self.int_type is None:
			if len(inner_nodes) < 2**8:
				self.int_type = "unsigned char"
			elif len(inner_nodes) < 2**16:
				self.int_type = "unsigned short"
			elif len(inner_nodes) < 2**32:
				self.int_type = "unsigned int"
			else:
				self.int_type = "unsigned long"
		
		if number is None or number == 0:
			node_struct = f"""
				struct Node {{
					{self.int_type} feature;
					{self.feature_type} split;
					bool left_is_leaf;
					bool right_is_leaf;
					{self.int_type} left;
					{self.int_type} right;
				}};
			""".strip()
		else:
			node_struct = ""

		suffix = "_" + str(number) if number is not None else ""

		preds = ",".join([str(list(n.prediction)).replace("[", "{").replace("]","}") for n in leaf_nodes])
		pred_array = f"constexpr {self.label_type} predictions{suffix}[{len(leaf_nodes)}][{tree.n_classes}] = {{{preds}}};"

		if len(inner_nodes) > 0:
			nodes = ",".join([f"{{ {n.feature},{n.split},{n.left_is_leaf},{n.right_is_leaf},{n.leftChild.id},{n.rightChild.id} }}" for n in inner_nodes])
			nodes_array = f"constexpr Node nodes{suffix}[{len(inner_nodes)}] = {{{nodes}}};"
		else:
			nodes_array = ""

		if len(inner_nodes) > 0:
			core_loop = f"""
				{self.int_type} i = 0;
				while(true) {{
					if (x[nodes{suffix}[i].feature] <= nodes{suffix}[i].split){{
						if (nodes{suffix}[i].left_is_leaf) {{
							i = nodes{suffix}[i].left;
							break;
						}} else {{
							i = nodes{suffix}[i].left;
						}}
					}} else {{
						if (nodes{suffix}[i].right_is_leaf) {{
							i = nodes{suffix}[i].right;
							break;
						}} else {{
							i = nodes{suffix}[i].right;
						}}
					}}
				}}
				return std::vector<{self.label_type}>(predictions{suffix}[i], predictions{suffix}[i]+{tree.n_classes});
			"""
		else:
			core_loop = f"return std::vector<{self.label_type}>(predictions{suffix}[0]+{tree.n_classes});"

		code = f"""
			{node_struct}
			{pred_array}
			{nodes_array}

			{header} {{
				{core_loop}
			}}
		"""
		return header + ";", code