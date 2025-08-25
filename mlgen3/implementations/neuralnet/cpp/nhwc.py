import numpy as np

from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.conv2d import Conv2D
from mlgen3.models.nn.maxpool2d import MaxPool2D

class NHWC(Implementation):

	def __init__(self, model, feature_type="int", label_type="int", internal_type = "float", align = None):
		super().__init__(model,feature_type,label_type)
		self.internal_type = internal_type
		self.align = align

	def implement(self):
		alloc = ""
		code = ""
		header = "#include <algorithm> // For std::max\n#include <cmath>\n"
		
		# First, check if the first layer is Conv2D to determine if we need input reshaping
		needs_input_reshaping = False
		input_shape = None
		
		# Track dimensions of each layer output - important for flattening conv to linear
		layer_dimensions = {}
		
		if len(self.model.layers) > 0 and isinstance(self.model.layers[0], Conv2D):
			needs_input_reshaping = True
			l = self.model.layers[0]
			in_channels = l.weight.shape[1]
			
			if hasattr(l, 'input_height') and l.input_height is not None:
				in_h, in_w = l.input_height, l.input_width
			else:
				# Default for MNIST/Fashion-MNIST
				in_h, in_w = 28, 28
				
			input_shape = (in_channels, in_h, in_w)
			
			# Add code to reshape the flattened input back to 3D (CHW format)
			code += f"""
				// Reshape flattened input back to [channels, height, width] format
				// Expected input dimensions: {in_channels}x{in_h}x{in_w}
				// Input 'x' is flattened with {in_channels * in_h * in_w} elements
				{self.internal_type} x_reshaped[{in_channels * in_h * in_w}];
				for (unsigned int i = 0; i < {in_channels * in_h * in_w}; ++i) {{
					x_reshaped[i] = x[i];
				}}
			"""

		# First pass: determine dimensions for all layers
		for lid, l in enumerate(self.model.layers):
			if isinstance(l, Conv2D):
				out_channels = l.weight.shape[0]
				in_channels = l.weight.shape[1]
				kernel_h, kernel_w = l.kernel_size
				stride_h, stride_w = l.stride
				pad_h, pad_w = l.padding
				
				# Calculate input dimensions
				if lid == 0:
					if input_shape:
						in_h, in_w = input_shape[1], input_shape[2]
					else:
						in_h, in_w = 28, 28  # Default for MNIST/Fashion-MNIST
				else:
					prev_shape = layer_dimensions[lid-1]
					if len(prev_shape) == 3:  # If previous layer is conv/pool
						in_channels, in_h, in_w = prev_shape
				
				# Calculate output dimensions
				out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
				out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
				
				layer_dimensions[lid] = (out_channels, out_h, out_w)
				
				# Store dimensions in the layer for future reference
				l.input_height = in_h
				l.input_width = in_w
				l.output_height = out_h
				l.output_width = out_w
				
			elif isinstance(l, MaxPool2D):
				if lid > 0:
					prev_shape = layer_dimensions[lid-1]
					if len(prev_shape) == 3:  # If previous layer is conv/pool
						channels, in_h, in_w = prev_shape
					else:
						# If previous layer was flattened, this is an error
						raise ValueError(f"MaxPool2D layer {lid} cannot follow a flattened layer")
				else:
					if input_shape:
						channels, in_h, in_w = input_shape
					else:
						channels, in_h, in_w = 1, 28, 28  # Default
				
				kernel_h, kernel_w = l.kernel_size
				stride_h, stride_w = l.stride
				pad_h, pad_w = l.padding
				
				out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
				out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
				
				layer_dimensions[lid] = (channels, out_h, out_w)
				
				# Store dimensions in the layer
				l.channels = channels
				l.input_height = in_h
				l.input_width = in_w
				l.output_height = out_h
				l.output_width = out_w
				
			elif isinstance(l, Linear):
				# For Linear layer following Conv/Pool, we need to ensure flattening
				if lid > 0:
					prev_shape = layer_dimensions.get(lid-1)
					if prev_shape and len(prev_shape) == 3:  # Previous layer was Conv/Pool
						channels, height, width = prev_shape
						flattened_size = channels * height * width
						# Verify that the Linear layer's input_shape matches the flattened size
						if l.input_shape != flattened_size:
							print(f"Warning: Linear layer {lid} expects {l.input_shape} inputs but previous layer produces {flattened_size} outputs when flattened")
						# Update the layer's input shape for correct code generation
						l.input_shape = flattened_size
				
				# Linear layer output is always 1D
				layer_dimensions[lid] = (l.output_shape,)
				
			else:
				# For activation layers, preserve dimensions
				if lid > 0:
					layer_dimensions[lid] = layer_dimensions[lid-1]
				else:
					# If it's the first layer and not conv/linear, assume it's 1D
					layer_dimensions[lid] = (l.output_shape,)

		# Now generate code with correct dimensions
		for lid, l in enumerate(self.model.layers):
			if lid == 0:
				input = needs_input_reshaping and "x_reshaped" or "x"
			else:
				input = f"layer_{lid-1}"
			
			# Determine if we need to add flattening code
			needs_flattening = False
			if isinstance(l, Linear) and lid > 0:
				prev_shape = layer_dimensions.get(lid-1)
				if prev_shape and len(prev_shape) == 3:  # Previous layer was Conv/Pool
					needs_flattening = True
					channels, height, width = prev_shape
					flattened_size = channels * height * width
					
					# Add flattening code
					code += f"""
						// Flatten 3D tensor to 1D for Linear layer
						// Input shape: [{channels}, {height}, {width}]
						// Output shape: [{flattened_size}]
						{self.internal_type} flattened_{lid}[{flattened_size}];
						for (unsigned int c = 0; c < {channels}; ++c) {{
							for (unsigned int h = 0; h < {height}; ++h) {{
								for (unsigned int w = 0; w < {width}; ++w) {{
									flattened_{lid}[c*{height}*{width} + h*{width} + w] = {input}[c*{height}*{width} + h*{width} + w];
								}}
							}}
						}}
					"""
					# Update input reference to use the flattened array
					input = f"flattened_{lid}"
			
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
				
				alloc += weight_array + "\n"
				alloc += bias_array + "\n"

				# Use the updated input_shape from our dimension tracking
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
			elif isinstance(l, Conv2D):
				# Add implementation for Conv2D layer
				out_channels, out_h, out_w = layer_dimensions[lid]
				in_channels = l.weight.shape[1]
				kernel_h, kernel_w = l.kernel_size
				stride_h, stride_w = l.stride
				pad_h, pad_w = l.padding
				in_h, in_w = l.input_height, l.input_width
				
				# Allocate output tensor
				alloc += f"static {self.internal_type} layer_{lid}[{out_channels}*{out_h}*{out_w}]"
				if self.align is not None and self.align > 0:
					alloc += f"__attribute__((aligned({self.align})));\n"
				else:
					alloc += ";\n"
				
				# Convert weights to C++ format
				weight_list = l.weight.tolist()
				weight_str = []
				for c_out in range(out_channels):
					c_out_str = []
					for c_in in range(in_channels):
						c_in_str = []
						for kh in range(kernel_h):
							kh_str = "{" + ", ".join(str(val) for val in weight_list[c_out][c_in][kh]) + "}"
							c_in_str.append(kh_str)
						c_in_str = "{" + ", ".join(c_in_str) + "}"
						c_out_str.append(c_in_str)
					c_out_str = "{" + ", ".join(c_out_str) + "}"
					weight_str.append(c_out_str)
				weight_str = "{" + ", ".join(weight_str) + "}"
				weight_array = f"constexpr {self.internal_type} layer_{lid}_weight[{out_channels}][{in_channels}][{kernel_h}][{kernel_w}] = {weight_str};"
				
				# Convert bias to C++ format
				bias_str = "{" + ", ".join(str(val) for val in l.bias.tolist()) + "}"
				bias_array = f"constexpr {self.internal_type} layer_{lid}_bias[{out_channels}] = {bias_str};"
				
				alloc += weight_array + "\n"
				alloc += bias_array + "\n"
				
				# Generate convolution code
				code += f"""
					// Conv2D layer
					// Input shape: [1, {in_channels}, {in_h}, {in_w}]
					// Output shape: [1, {out_channels}, {out_h}, {out_w}]
					// Initialize with bias
					for (unsigned int c_out = 0; c_out < {out_channels}; ++c_out) {{
						for (unsigned int h_out = 0; h_out < {out_h}; ++h_out) {{
							for (unsigned int w_out = 0; w_out < {out_w}; ++w_out) {{
								layer_{lid}[c_out*{out_h}*{out_w} + h_out*{out_w} + w_out] = layer_{lid}_bias[c_out];
							}}
						}}
					}}
					
					// Perform convolution
					for (unsigned int c_out = 0; c_out < {out_channels}; ++c_out) {{
						for (unsigned int h_out = 0; h_out < {out_h}; ++h_out) {{
							for (unsigned int w_out = 0; w_out < {out_w}; ++w_out) {{
								// Calculate input region
								int h_start = h_out * {stride_h} - {pad_h};
								int w_start = w_out * {stride_w} - {pad_w};
								
								// Compute convolution for this output position
								for (unsigned int c_in = 0; c_in < {in_channels}; ++c_in) {{
									for (unsigned int kh = 0; kh < {kernel_h}; ++kh) {{
										int h_in = h_start + kh;
										if (h_in >= 0 && h_in < {in_h}) {{
											for (unsigned int kw = 0; kw < {kernel_w}; ++kw) {{
												int w_in = w_start + kw;
												if (w_in >= 0 && w_in < {in_w}) {{
													layer_{lid}[c_out*{out_h}*{out_w} + h_out*{out_w} + w_out] += 
														layer_{lid}_weight[c_out][c_in][kh][kw] * 
														{input}[c_in*{in_h}*{in_w} + h_in*{in_w} + w_in];
												}}
											}}
										}}
									}}
								}}
							}}
						}}
					}}
				"""
			elif isinstance(l, MaxPool2D):
				# Improved implementation for MaxPool2D layer
				channels, in_h, in_w = layer_dimensions[lid-1]
				out_channels, out_h, out_w = layer_dimensions[lid]
				
				kernel_h, kernel_w = l.kernel_size
				stride_h, stride_w = l.stride
				pad_h, pad_w = l.padding
				
				# Allocate output tensor
				alloc += f"static {self.internal_type} layer_{lid}[{out_channels}*{out_h}*{out_w}]"
				if self.align is not None and self.align > 0:
					alloc += f"__attribute__((aligned({self.align})));\n"
				else:
					alloc += ";\n"
				
				# Generate MaxPool code
				code += f"""
					// MaxPool2D layer
					// Input shape: [1, {channels}, {in_h}, {in_w}]
					// Output shape: [1, {out_channels}, {out_h}, {out_w}]
					for (unsigned int c = 0; c < {channels}; ++c) {{
						for (unsigned int h_out = 0; h_out < {out_h}; ++h_out) {{
							for (unsigned int w_out = 0; w_out < {out_w}; ++w_out) {{
								// Calculate input region
								int h_start = h_out * {stride_h} - {pad_h};
								int w_start = w_out * {stride_w} - {pad_w};
								int h_end = std::min(h_start + {kernel_h}, {in_h} + {pad_h});
								int w_end = std::min(w_start + {kernel_w}, {in_w} + {pad_w});
								h_start = std::max(h_start, 0);
								w_start = std::max(w_start, 0);
								h_end = std::min(h_end, {in_h});
								w_end = std::min(w_end, {in_w});
								
								// Apply max pooling
								{self.internal_type} max_val = -std::numeric_limits<{self.internal_type}>::infinity();
								for (int h = h_start; h < h_end; ++h) {{
									for (int w = w_start; w < w_end; ++w) {{
										{self.internal_type} val = {input}[c*{in_h}*{in_w} + h*{in_w} + w];
										max_val = std::max(max_val, val);
									}}
								}}
								layer_{lid}[c*{out_h}*{out_w} + h_out*{out_w} + w_out] = max_val;
							}}
						}}
					}}
				"""
			else:
				raise ValueError(f"Layer of {l} is currently not supported by cpp.nhwc. Cannot generate code!")
		
		# Allocate memory for each layer output - now after calculating dimensions
		for lid, l in enumerate(self.model.layers):
			if isinstance(l, (Conv2D, MaxPool2D)):
				# Already handled above
				pass
			else:
				# For standard layers, use output_shape directly
				if isinstance(l, Linear):
					alloc_size = l.output_shape
				else:
					# For activation layers, use the layer dimensions we calculated
					if lid in layer_dimensions:
						shape = layer_dimensions[lid]
						if len(shape) == 3:
							alloc_size = shape[0] * shape[1] * shape[2]  # channels * height * width
						else:
							alloc_size = shape[0]
					else:
						alloc_size = l.output_shape
						
				alloc_line = f"static {self.internal_type} layer_{lid}[{alloc_size}]"
				if self.align is not None and self.align > 0:
					alloc_line += f"__attribute__((aligned({self.align})));\n"
				else:
					alloc_line += ";\n"
				alloc = alloc_line + alloc
		
		# Set the code and header attributes properly
		self.code = f"""
			#include "model.h"
			#include <limits>
			{alloc}
			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
				{code}
				return std::vector<{self.label_type}>(layer_{len(self.model.layers)-1}, layer_{len(self.model.layers)-1}+{self.model.layers[-1].output_shape});
			}}
		"""

		self.header = f"""
			#pragma once
			#include <vector>
			#include <limits>
			{header}

			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
		""".strip()
