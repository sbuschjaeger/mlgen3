import numpy as np

from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class NHWC(Implementation):

	def __init__(self, model, feature_type="int", label_type="int", internal_type = "float", align = None):
		super().__init__(model,feature_type,label_type)
		self.internal_type = internal_type
		self.align = align

	def implement(self):
		alloc = ""
		code = ""
		header = ""

		for lid, l in enumerate(self.model.layers):
			# TODO This only works for 1d inputs at the moment. 
			alloc += f"static {self.internal_type} layer_{lid}[{l.output_shape}]"
			if self.align is not None and self.align > 0:
				alloc += f"__attribute__((aligned({self.align})));\n"
			else:
				alloc += ";\n"

			if lid == 0:
				input = "x"
			else:
				input = f"layer_{lid-1}"
			
			if isinstance(l, Sign):
				code += f"""
					for (unsigned int i = 0; i < {l.output_shape}; ++i) {{
						if ({input}[i] > 0) layer_{lid}[i] = 1;
						else if ({input}[i] < 0) layer_{lid}[i] = -1;
						else layer_{lid}[i] = 0;
					}}
				"""
			elif isinstance(l, Sigmoid):
				header += "#include <cmath>\n"

				code += f"""
					for (unsigned int i = 0; i < {l.output_shape}; ++i) {{
						layer_{lid}[i] = 1 / (1 + std::exp(-{input}[i]));
					}}
				"""
			elif isinstance(l, Relu):
				code += f"""
					for (unsigned int i = 0; i < {l.output_shape}; ++i) {{
						layer_{lid}[i] = {input}[i] >= 0 ? {input}[i] : 0;
					}}
				"""
			elif isinstance(l, Linear):
				# Convert NumPy arrays to Python lists and format as C++ arrays
				weight_list = l.weight.tolist()
				weight_str = []
				for row in weight_list:
					row_str = "{" + ", ".join(str(val) for val in row) + "}"
					weight_str.append(row_str)
				tmp_weight = ", ".join(weight_str)
				weight_array = f"constexpr {self.internal_type} layer_{lid}_weight[{len(l.weight)}][{len(l.weight[0])}] = {{{tmp_weight}}};"

				bias_str = "{" + ", ".join(str(val) for val in l.bias.tolist()) + "}"
				bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(l.bias)}] = {bias_str};"
				
				# TODO add aligned
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
			elif isinstance(l, BatchNorm):
				# Convert NumPy arrays to Python lists for C++ compatibility
				scale_str = "{" + ", ".join(str(val) for val in l.scale.tolist()) + "}"
				scale_array = f"constexpr {self.internal_type} layer_{lid}_scale[{len(l.scale)}] = {scale_str};"
				
				bias_str = "{" + ", ".join(str(val) for val in l.bias.tolist()) + "}"
				bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(l.bias)}] = {bias_str};"

				alloc += scale_array + "\n"
				alloc += bias_array + "\n"

				code += f"""
					for (unsigned int d = 0; d < {l.output_shape}; d++) {{
						layer_{lid}[d] = {input}[d] * layer_{lid}_scale[d] + layer_{lid}_bias[d];
					}}
				"""
			elif isinstance(l, Step):
				if l.threshold_is_high:
					comp = ">="
				else:
					comp = ">"

				if isinstance(l.threshold, (list, np.ndarray)):
					threshold_str = "{" + ", ".join(str(val) for val in l.threshold.tolist()) + "}"
					threshold_array = f"constexpr {self.internal_type} layer_{lid}_threshold[{len(l.threshold)}] = {threshold_str};"
					alloc += threshold_array + "\n"

					threshold = f"layer_{lid}_threshold[i]"
				else:
					threshold = l.threshold

				code += f"""
					for (unsigned int i = 0; i < {l.output_shape}; i++) {{
						layer_{lid}[i] = {input}[i] {comp} {threshold} ? {l.high} : {l.low};
					}}
				"""
			else:
				raise ValueError(f"Layer of {l} is currently not supported by cpp.nhwc. Cannot generate code!")
		
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
			{header}

			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
		""".strip()