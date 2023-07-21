import numpy as np

from mlgen3.implemantations.implementation import Implementation
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
				alloc += "__attribute__((aligned({self.align})));\n"
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
				tmp_weight = ",".join([str(list(c1)).replace("[", "{").replace("]","}") for c1 in l.weight])
				weight_array = f"constexpr {self.internal_type} layer_{lid}_weight[{len(l.weight)}][{len(l.weight[0])}] = {{{tmp_weight}}};"

				tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
				bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"
				# TODO add alinged
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
				tmp_scale = str(l.scale.tolist()).replace("[", "{").replace("]","}")
				scale_array = f"constexpr {self.internal_type} layer_{lid}_scale[{len(l.scale)}] = {tmp_scale};"
				
				tmp_bias = str(l.bias.tolist()).replace("[", "{").replace("]","}")
				bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{len(l.bias)}] = {tmp_bias};"

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
					tmp_threshold = str(l.threshold.tolist()).replace("[", "{").replace("]","}")
					threshold_array = f"constexpr {self.internal_type} layer_{lid}_threshold[{len(l.threshold)}] = {tmp_threshold};"
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