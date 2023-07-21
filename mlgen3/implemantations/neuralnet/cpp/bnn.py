import textwrap
import numpy as np
from mlgen3.implemantations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class BNN(Implementation):

	def __init__(self, model, feature_type="int", label_type="int"):
		super().__init__(model,feature_type,label_type)
		
		# TODO 
		self.align = None
		self.binary_word_size = 32 
		if self.binary_word_size <= 32:
			self.popcount = "__builtin_popcount"
		else:
			self.popcount = "__builtin_popcountll"

		self.model.merge_bn_and_step()
		for l in self.model.layers:
			if isinstance(l, Linear) and not np.all( (l.weight == 1) | (l.weight == -1) ) and not np.all( (l.bias == 1) | (l.bias == -1) ):
				raise ValueError("Encountered values other than {-1,+1} in the weight/bias of the linear layer. Check your model or consider using the cpp.nhwc implementation.")
			elif isinstance(l, (BatchNorm, Sigmoid, Relu, Sign)):
				raise ValueError(f"Encountered layer of type {l} which is not supported. Please make sure your model only contains {{Linear,BatchNorm,Step}} layers. If not, check your model or consider using the cpp.nhwc implementation.")

	def implement(self):
		alloc = ""
		code = ""

		for lid, l in enumerate(self.model.layers):
			# TODO We currently assume that the first layer is linear layer. This should be a bit more general
			# TODO This stuff only works for 1d at the moment
			if lid == 0:
				# TODO Properly handel output types
				# output_type = larger_datatype(input_type, ctype(layer.weight.dtype))
        		# output_type = larger_datatype(output_type, ctype(layer.bias.dtype))

				output_type = "float"
				if not isinstance(l, Linear):
					raise ValueError(f"For the cpp.bnn implementation, the first layer must be a Linear layer, but you provded {l}")

				alloc += f"static {output_type} layer_{lid}[{l.output_shape}]"
				if self.align is not None and self.align > 0:
					alloc += "__attribute__((aligned({self.align})));\n"
				else:
					alloc += ";\n"
				input = "x"

				tmp_weight = ",".join([str(list(c1)).replace("[", "{").replace("]","}") for c1 in l.weight])
				weight_array = f"constexpr {output_type} layer_{lid}_weight[{len(l.weight)}][{len(l.weight[0])}] = {{{tmp_weight}}};"

				tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
				bias_array = f"constexpr {output_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"

				alloc += weight_array + "\n"
				alloc += bias_array + "\n"

				code += f"""
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
				# TODO Properly handel output types
				output_type = "int"
				uint_type = "unsigned int"

				# static {{ output_type }} layer_{{ layer_id }}_output[{{ (output_shape[1] / binary_word_size)|round(method='ceil')|int }}]{% endif %}{% if align > 1 %} __attribute__((aligned({{ align }}))){% endif %};

				alloc += f"static {output_type} layer_{lid}[{int(np.ceil(l.output_shape/self.binary_word_size))}]"
				if self.align is not None and self.align > 0:
					alloc += "__attribute__((aligned({self.align})));\n"
				else:
					alloc += ";\n"

				if isinstance(l, Linear):
					# TODO OUTPUT SIZES AND BIAS ARE CURRENTLY NOT CORRECT! FIX THIS 

					weight_binary = (l.weight + 1) // 2

					# Fill with zeros to make the array size divisible by binary_word_size. This will will push the remainder weights
					# to the most significant bits in the last packed int which matches the behaviour of the Step Layer
					next_higher_divisible = int(np.ceil(l.weight.shape[-1] / self.binary_word_size)) * self.binary_word_size
					zeros = np.zeros(weight_binary.shape[:-1] + (next_higher_divisible - weight_binary.shape[-1],), dtype=weight_binary.dtype)
					weight_binary = np.append(weight_binary, zeros, axis=-1)
					weight_binary = weight_binary.astype(int)

					weight_hex = []
					for i in range(weight_binary.shape[0]):
						tmp = []
						for j in range(0, len(weight_binary[i]), self.binary_word_size):
							b = ''.join([str(s) for s in weight_binary[i][j:j+self.binary_word_size]])
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
					bias_array = f"constexpr {output_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"
					# do the binary alloc
					alloc += weight_array + "\n"
					alloc += bias_array + "\n"

					code += f"""
						for (unsigned int d = 0; d < {l.output_shape}; d++) {{
							layer_{lid}[d] = layer_{lid}_bias[d];
						}}
						for (unsigned int d = 0; d < {l.output_shape}; d++) {{
							for (int i = 0; i < {int(np.ceil(l.input_shape/self.binary_word_size))}; i++) {{
								layer_{lid}[d] += 2 * {self.popcount}(({uint_type})~({uint_type})(layer_{lid}_weight[d][i] ^ layer_{lid-1}[i])) - {self.binary_word_size};
							}}
						}}
					"""
				elif isinstance(l, Step):
					if l.threshold_is_high:
						comp = ">="
					else:
						comp = ">"

					if isinstance(l.threshold, (list, np.ndarray)):
						# TODO round to next integer and use the int type 
						tmp_threshold = str(l.threshold.tolist()).replace("[", "{").replace("]","}")
						threshold_array = f"constexpr float layer_{lid}_threshold[{len(l.threshold)}] = {tmp_threshold};"
						alloc += threshold_array + "\n"

						threshold = f"layer_{lid}_threshold[i]"
					else:
						threshold = l.threshold
					
					if self.binary_word_size >= 64:
						bit = "1ULL"
					else:
						bit = "1U"

					code += f"""
						for (unsigned int i = 0; i < {l.output_shape}; i++) {{
							if (layer_{lid-1}[i] {comp} {threshold}) {{
								layer_{lid}[i / {min(l.output_shape, self.binary_word_size)}] |= {bit} << ({self.binary_word_size - 1} - i % {min(l.output_shape, self.binary_word_size)});
							}} else {{
								layer_{lid}[i / {min(l.output_shape, self.binary_word_size)}] &= ~({bit} << ({self.binary_word_size - 1} - i % {min(l.output_shape, self.binary_word_size)}));
							}}
						}}
					"""
					
					padding_size = int(np.ceil(l.output_shape / self.binary_word_size)) * self.binary_word_size
					if l.output_shape % self.binary_word_size != 0:
						code += f"""
							for (unsigned int i = {l.output_shape}; i < {padding_size}; i++) {{
								layer_{lid}[i / {min(l.output_shape, self.binary_word_size)}] &= ~({bit} << ({self.binary_word_size - 1} - i % {min(l.output_shape, self.binary_word_size)}));
							}}
						"""
				else: 
					raise ValueError("Reached a code path in cpp.bnn which should be unreachable. Lol")
		
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