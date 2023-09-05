import numpy as np
from mlgen3.implemantations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

def get_step(l, lid, input_packed = False, output_packed = True, **kwargs):
	assert isinstance(l, Step), f"Provided layer of type {l}, but expected a nn.Step layer!"

	binary_word_size = kwargs.pop("binary_word_size", 32) 
	output_type = kwargs.pop("output_type", "int") 
	int_type = kwargs.pop("int_type", "int") 
	align = kwargs.pop("align", False) 
	internal_type = kwargs.pop("internal_type", "float") 

	alloc = ""
	code = ""

	if lid == 0:
		input = "x"
	else:
		input = f"layer_{lid-1}"

	if input_packed and output_packed:
		raise ValueError(f"input_packed = True and output_packed = True is currently not implemented for nn.Step.")
	elif input_packed and not output_packed:
		if l.threshold_is_high:
			comp = ">="
		else:
			comp = ">"

		if isinstance(l.threshold, (list, np.ndarray)):
			# TODO round to next bit for comparison
			tmp_threshold = str(l.threshold.tolist()).replace("[", "{").replace("]","}")
			threshold_array = f"constexpr {internal_type} layer_{lid}_threshold[{len(l.threshold)}] = {tmp_threshold};"
			alloc += threshold_array + "\n"

			threshold = f"layer_{lid}_threshold[i]"
		else:
			threshold = l.threshold

		code += f"""
			for (unsigned int i = 0; i < {l.output_shape}; i++) {{
				auto the_bit = ({input} >> i) & 1U;
				layer_{lid}[i] = the_bit {comp} {threshold} ? {l.high} : {l.low};
			}}
		"""
	elif not input_packed and output_packed:
		output_type = "unsigned int"
		alloc += f"static {output_type} layer_{lid}[{int(np.ceil(l.output_shape/binary_word_size))}]"
		if align is not None and align > 0:
			alloc += f"__attribute__((aligned({align})));\n"
		else:
			alloc += ";\n"

		if l.threshold_is_high:
			comp = ">="
		else:
			comp = ">"

		n_loops = int(np.ceil(l.output_shape / binary_word_size))
		upper = min(l.output_shape, binary_word_size)

		if isinstance(l.threshold, (list, np.ndarray)):
			# TODO round to next integer and use the int type 

			tmp_threshold = str(list(l.threshold)).replace("[", "{").replace("]","}")
			threshold_array = f"constexpr float layer_{lid}_threshold[{len(l.threshold)}] = {tmp_threshold};"
			alloc += threshold_array + "\n"

			threshold = f"layer_{lid}_threshold[idx]"
		else:
			threshold = l.threshold
		
		if binary_word_size >= 64:
			bit = "1ULL"
		else:
			bit = "1U"

		code += f"""
			// Step Layer
			for (unsigned int j = 0; j < {n_loops}; ++j) {{
				layer_{lid}[j] = 0;
				for (unsigned int i = 0; i < {upper}; ++i) {{
					auto idx = {upper}*j+i;
					if (idx < {l.input_shape} && {input}[idx] {comp} {threshold}) {{
						//layer_{lid}[i] |= {bit} << (32 - 1 - i);
						layer_{lid}[j] |= {bit} << (i);
					}} 
				}}
			}}
		"""	

		# for i in range(n_loops):
		# 	upper = min(l.output_shape, binary_word_size)

		# 	code += f"""
		# 		// Step Layer
		# 		layer_{lid}[{i}] = 0;
		# 		for (unsigned int i = 0; i < {upper}; i++) {{
		# 			if ({input}[i] {comp} {threshold}) {{
		# 				//layer_{lid}[{i}] |= {bit} << (32 - 1 - i);
		# 				layer_{lid}[{i}] |= {bit} << (i);
		# 			}} 
		# 		}}
		# 	"""	
	else:
		alloc += f"static {int_type} layer_{lid}[{l.output_shape}]"
		if align is not None and align > 0:
			alloc += f"__attribute__((aligned({align})));\n"
		else:
			alloc += ";\n"

		if l.threshold_is_high:
			comp = ">="
		else:
			comp = ">"

		if isinstance(l.threshold, (list, np.ndarray)):
			tmp_threshold = str(l.threshold.tolist()).replace("[", "{").replace("]","}")
			threshold_array = f"constexpr {internal_type} layer_{lid}_threshold[{len(l.threshold)}] = {tmp_threshold};"
			alloc += threshold_array + "\n"

			threshold = f"layer_{lid}_threshold[i]"
		else:
			threshold = l.threshold

		code += f"""
			// Regular Step Layer
			for (unsigned int i = 0; i < {l.output_shape}; i++) {{
				layer_{lid}[i] = {input}[i] {comp} {threshold} ? {l.high} : {l.low};
			}}
		"""
	
	return alloc, code

def get_linear(l, lid, input_packed = False, output_packed = True, **kwargs):
	assert isinstance(l, Linear), f"Provided layer of type {l}, but expected a nn.linear layer!"
	assert output_packed is False, "This implementation does not supported a packed output for linear layers. Please use a subsequent step layer to pack it."
	alloc = ""
	code = ""

	if lid == 0:
		input = "x"
	else:
		input = f"layer_{lid-1}"

	binary_word_size = kwargs.pop("binary_word_size", 32) 
	align = kwargs.pop("align", False) 
	uint_type = kwargs.pop("uint_type", "unsigned int") 
	int_type = kwargs.pop("int_type", "int") 
	popcount = kwargs.pop("popcount", "__builtin_popcount") 
	remaining_bits = kwargs.pop("remaining_bits", None)

	if not input_packed:
		# TODO fix types
		output_type = "float"

		alloc += f"static {output_type} layer_{lid}[{l.output_shape}]"
		if align is not None and align > 0:
			alloc += f"__attribute__((aligned({align})));\n"
		else:
			alloc += ";\n"

		tmp_weight = ",".join([str(list(c1)).replace("[", "{").replace("]","}") for c1 in l.weight])
		weight_array = f"constexpr {output_type} layer_{lid}_weight[{len(l.weight)}][{len(l.weight[0])}] = {{{tmp_weight}}};"

		tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
		bias_array = f"constexpr {output_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"

		alloc += weight_array + "\n"
		alloc += bias_array + "\n"

		code += f"""
			// Regular Linear Layer
			for (unsigned int d = 0; d < {l.output_shape}; d++) {{
				layer_{lid}[d] = layer_{lid}_bias[d];
			}}
			for (unsigned int d = 0; d < {l.output_shape}; d++) {{
				for (unsigned int i = 0; i < {l.input_shape}; i++) {{
					layer_{lid}[d] += layer_{lid}_weight[d][i] * {input}[i];
				}}
			}}
		"""
	else:
		# TODO fix types
		output_type = "int"

		alloc += f"static {output_type} layer_{lid}[{l.output_shape}]"
		if align is not None and align > 0:
			alloc += f"__attribute__((aligned({align})));\n"
		else:
			alloc += ";\n"
			
		weight_binary = (l.weight + 1) // 2

		# Fill with zeros to make the array size divisible by binary_word_size. This will will push the remainder weights
		# to the most significant bits in the last packed int which matches the behaviour of the Step Layer
		next_higher_divisible = int(np.ceil(l.weight.shape[-1] / binary_word_size)) * binary_word_size
		zeros = np.zeros(weight_binary.shape[:-1] + (next_higher_divisible - weight_binary.shape[-1],), dtype=weight_binary.dtype)
		weight_binary = np.append(weight_binary, zeros, axis=-1)
		weight_binary = weight_binary.astype(int)

		weight_hex = []
		for i in range(weight_binary.shape[0]):
			tmp = []
			for j in range(0, len(weight_binary[i]), binary_word_size):
				b = ''.join([str(s) for s in reversed(weight_binary[i][j:j+binary_word_size])])
				tmp.append(hex(int(b, 2)))
			weight_hex.append(tmp)
		weight_hex = np.array(weight_hex)
		# weight_binary = [
		# 	[
		# 		hex(int(b, 2)) for b in textwrap.wrap(''.join([str(w) for w in weight_binary[i]]),self.binary_word_size)
		# 	] for i in range(weight_binary.shape[0])
		# ]
		
		weight_hex_str = ",".join([str(list(c1)).replace("[", "{").replace("]","}").replace("'","") for c1 in weight_hex])
		#weight_hex_str = str(list(weight_hex)).replace("[", "{").replace("]","}").replace("'","")
		weight_array = f"constexpr {uint_type} layer_{lid}_weight[{weight_hex.shape[0]}][{weight_hex.shape[1]}] = {{{weight_hex_str}}};"

		tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
		bias_array = f"constexpr {int_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"
		# do the binary alloc
		alloc += weight_array + "\n"
		alloc += bias_array + "\n"

		if remaining_bits is None or remaining_bits == 0:
			code += f"""
				// Linear Layer
				for (unsigned int d = 0; d < {l.output_shape}; d++) {{
					layer_{lid}[d] = layer_{lid}_bias[d];
				}}
				for (unsigned int d = 0; d < {l.output_shape}; d++) {{
					for (int i = 0; i < {int(np.ceil(l.input_shape/binary_word_size))}; i++) {{
						layer_{lid}[d] += 2 * {popcount}(~(layer_{lid}_weight[d][i] ^ {input}[i])) - {binary_word_size};
					}}
				}}
			"""
		else:
			code += f"""
				// Linear Layer
				for (unsigned int d = 0; d < {l.output_shape}; d++) {{
					layer_{lid}[d] = layer_{lid}_bias[d];
				}}
				for (unsigned int d = 0; d < {l.output_shape}; d++) {{
					for (int i = 0; i < {int(np.ceil(l.input_shape/binary_word_size))}; i++) {{
						if (i == {int(np.ceil(l.input_shape/binary_word_size)) - 1}) {{
							// There are only {remaining_bits} bits in the last bitvector, so we have to ignore {binary_word_size - remaining_bits} bits (given a {binary_word_size} architecture)
							layer_{lid}[d] += 2 * ({popcount}(~(layer_{lid}_weight[d][i] ^ {input}[i])) - {binary_word_size - remaining_bits}) - {remaining_bits};
						}} else {{
							layer_{lid}[d] += 2 * {popcount}(~(layer_{lid}_weight[d][i] ^ {input}[i])) - {binary_word_size};
						}}
					}}
				}}
			"""

	return alloc, code

def get_bn(l, lid, **kwargs):
	assert isinstance(l, BatchNorm), f"Provided layer of type {l}, but expected a nn.linear layer!"
	alloc = ""
	code = ""

	internal_type = kwargs.pop("internal_type", "float") 
	align = kwargs.pop("align", False) 
	output_type = kwargs.pop("output_type", "float") 

	if lid == 0:
		input = "x"
	else:
		input = f"layer_{lid-1}"

	alloc += f"static {output_type} layer_{lid}[{l.output_shape}]"
	if align is not None and align > 0:
		alloc += f"__attribute__((aligned({align})));\n"
	else:
		alloc += ";\n"

	tmp_scale = str(l.scale.tolist()).replace("[", "{").replace("]","}")
	scale_array = f"constexpr {internal_type} layer_{lid}_scale[{len(l.scale)}] = {tmp_scale};"
	
	tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
	bias_array = f"constexpr {internal_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"

	alloc += scale_array + "\n"
	alloc += bias_array + "\n"

	code += f"""
		for (unsigned int d = 0; d < {l.output_shape}; d++) {{
			layer_{lid}[d] = {input}[d] * layer_{lid}_scale[d] + layer_{lid}_bias[d];
		}}
	"""

	return alloc, code

class BNN(Implementation):

	def __init__(self, model, feature_type="int", label_type="int"):
		super().__init__(model,feature_type,label_type)
		
		# TODO Improve arguments
		self.align = None
		self.binary_word_size = 32 
		if self.binary_word_size <= 32:
			self.popcount = "__builtin_popcount"
		else:
			self.popcount = "__builtin_popcountll"

		self.model.merge_bn_and_step()
		for (i, l) in enumerate(self.model.layers):
			if isinstance(l, Linear) and not np.all( (l.weight == 1) | (l.weight == -1) ) and not np.all( (l.bias == 1) | (l.bias == -1) ):
				raise ValueError("Encountered values other than {-1,+1} in the weight/bias of the linear layer. Check your model or consider using the cpp.nhwc implementation.")
			elif isinstance(l, (BatchNorm)) and i > 0 and i < len(self.model.layers) - 1:
				raise ValueError(f"Encountered layer of type {l} in one of the hidden layers of the network. Currently, only Linear and Step layers are supported. If you have a BatchNorm layer inside of your model, make sure it is followed by a Step layer so both can be merged. If this is not possible consider using the cpp.nhwc implementation.")

	def implement(self):
		alloc = ""
		code = ""

		is_packed = False
		binary_word_size = 32
		remaining_bits = None
		for lid, l in enumerate(self.model.layers):
			is_last = (lid == len(self.model.layers) - 1)

			if isinstance(l, Linear):
				tmp_alloc, tmp_code = get_linear(l, lid, input_packed=is_packed, output_packed=False, binary_word_size=binary_word_size,remaining_bits=remaining_bits)
				is_packed = False
			elif isinstance(l,Step):
				tmp_alloc, tmp_code = get_step(l, lid, input_packed=is_packed, output_packed=not is_last, output_type="int" if not is_last else "float", binary_word_size=binary_word_size)
				is_packed = True
				remaining_bits = l.output_shape % binary_word_size
			elif isinstance(l, BatchNorm):
				is_packed = False
				tmp_alloc, tmp_code = get_bn(l, lid)
			else:
				raise ValueError(f"Encountered layer {l} which is not supported.")

			alloc += tmp_alloc
			code += tmp_code
			
		self.code = f"""
			#include "model.h"
			{alloc}
			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
				{code}
				return std::vector<{self.label_type}>(layer_{len(self.model.layers)-1}, layer_{len(self.model.layers)-1}+{self.model.layers[-1].output_shape});
			}}
		"""

		self.header = f"""
			#pragma once
			#include <vector>
			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
		""".strip()