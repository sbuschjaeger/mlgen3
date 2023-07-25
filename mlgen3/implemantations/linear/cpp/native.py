import numpy as np

from ...implementation import Implementation

def simplify_array(array):
	"""Try to simplify the data type of an array

	Args:
		array: The array to simplify

	Returns:
		The simplified array
	"""
	array = np.array(array)

	array_int = array.astype(int)
	if (array_int == array).all():
		minimum = np.min(array_int)
		maximum = np.max(array_int)
		if minimum >= -(2 ** 7) and maximum < 2 ** 7:
			array = array_int.astype(np.int8)
		elif minimum >= -(2 ** 15) and maximum < 2 ** 15:
			array = array_int.astype(np.int16)
		elif minimum >= -(2 ** 31) and maximum < 2 ** 31:
			array = array_int.astype(np.int32)
		else:
			array = array_int
	return array

def ctype(dtype):
	if dtype == "float32":
		return "float"
	elif dtype == "float64":
		return "double"
	elif dtype == "int8":
		return "signed char"
	elif dtype == "int16":
		return "signed short"
	elif dtype == "int32":
		return "signed int"
	elif dtype == "int64":
		return "signed long"
	else:
		# Only go for fixed-sizes data types as a last resort
		return str(dtype) + "_t"

def larger_datatype(dtype1, dtype2):
	types = [
		["unsigned char", "uint8_t"],
		["unsigned short", "uint16_t"],
		["unsigned int", "uint32_t"],
		["unsigned long", "uint64_t"],
		["float"],
		["double"]
	]

	if dtype1 in types[0]:
		return dtype2
	
	if dtype1 in types[1] and dtype2 not in types[0]:
		return dtype2

	if dtype1 in types[2] and (dtype2 not in types[0] + types[1]):
		return dtype2

	if dtype1 in types[3] and (dtype2 not in types[0] + types[1] + types[2]):
		return dtype2

	if dtype1 in types[4] and (dtype2 not in types[0] + types[1] + types[2] + types[3]):
		return dtype2

	return dtype1

class Native(Implementation):

	def __init__(self, model, feature_type="int", label_type="int", internal_type = "float"):
		super().__init__(model,feature_type,label_type)
		self.internal_type = internal_type
	
	def implement(self):
		if self.internal_type is None:
			self.model.coef = simplify_array(self.model.coef)
			self.model.intercept = simplify_array(self.model.intercept)

			internal_type = larger_datatype(ctype(self.model.coef.dtype), ctype(self.model.intercept.dtype))
			internal_type = larger_datatype(internal_type, self.feature_type)
		else:
			internal_type = self.internal_type

		coef = self.model.coef.T.tolist()
		intercept = self.model.intercept.tolist()

		tmp_coef = ",".join([str(list(c1)).replace("[", "{").replace("]","}") for c1 in coef])
		coef_array = f"constexpr {internal_type} coef[{len(coef)}][{len(coef[0])}] = {{{tmp_coef}}};"

		tmp_intercept = str(intercept).replace("[", "{").replace("]","}")
		intercept_array = f"constexpr {internal_type} intercept[{len(intercept)}] = {tmp_intercept};"

		self.code = f"""
			#include "model.h"
			{coef_array}
			{intercept_array}

			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
				std::vector<{self.label_type}> pred({len(coef)}, 0);
				for (unsigned int j = 0; j < {len(coef)}; ++j) {{
					{internal_type} sum = intercept[j]; 
					for (unsigned int i = 0; i < {len(coef[0])}; ++i) {{
						sum += coef[j][i] * x[i];
					}}
					pred[j] += sum; 
				}}
				return pred;
			}}
		""".strip()

		self.header = f"""
			#pragma once
			#include <vector>
			std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
		""".strip()