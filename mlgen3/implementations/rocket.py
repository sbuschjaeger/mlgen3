import numpy as np

from mlgen3.implementations.implementation import Implementation


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

#used to convert numpy arrays to c arrays. if the array consists of weights, and we got only one channel, we wrap that in another list. important for type coherence in C++.
def pyArray_cArray(array, weights=False):
	if type(array) == np.ndarray:
		array = array.tolist()

	tmp = str(array).replace("[", "{").replace("]","}")

	if weights and (tmp.count("{") == 1):
		tmp = f"{{{tmp}}}"
	#print(tmp)
	return tmp

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

class Rocket(Implementation):

	def __init__(self, model, feature_type="int", label_type="int", internal_type = "float"):
		super().__init__(model,feature_type,label_type)
		self.internal_type = internal_type
	
	def implement(self):
		if self.internal_type is None:
			self.model.coef = simplify_array(self.model.linear.coef)
			self.model.intercept = simplify_array(self.model.linear.intercept)

			internal_type = larger_datatype(ctype(self.model.coef.dtype), ctype(self.model.intercept.dtype))
			internal_type = larger_datatype(internal_type, self.feature_type)
		else:
			internal_type = self.internal_type

		# kernel definition

		coef = self.model.coef.T.tolist() #(weights:matrix, channels:list, bias:float, dilation:int, padding:int)
		intercept = self.model.intercept.tolist()
		tmp_kernellist = ",".join([f"Kernel({pyArray_cArray(w, weights=True)}, {b}, {d}, {p}, {pyArray_cArray(c)}) " for (w, c, b, d, p) in self.model.kernellist]).replace("[","{").replace("]","}")
		#tmp_kernellist = tmp_kernellist[:-1]
		#print(tmp_kernellist)
		kernellist = f"Kernel kernels[{len(self.model.kernellist)}] = {{{tmp_kernellist}}};"
        
		kerneldefinition = f"""
		#ifndef KERNEL_H
		#define KERNEL_H
		#include <vector>
		#include <numeric> //for accumulate

		class Kernel {{
		private:
    	std::vector<std::vector<float>> weights;
    	float bias;
    	int dilation;
    	int padding;
    	std::vector<float> channels;

		public:
    	Kernel(std::vector<std::vector<float>> w, float b, int d, int p, std::vector<float> c)
        	: weights(w), bias(b), dilation(d), padding(p), channels(c) {{}}

    	std::vector<std::vector<float>> getWeights() const {{ return weights; }}
    	float getBias() const {{ return bias; }}
    	int getDilation() const {{ return dilation; }}
    	int getPadding() const {{ return padding; }}
    	std::vector<float> getChannels() const {{ return channels; }}
		}};

		#endif
		""".strip()

        ####

		tmp_coef = ",".join([str(list(c1)).replace("[", "{").replace("]","}") for c1 in coef])
		coef_array = f"constexpr {internal_type} coef[{len(coef)}][{len(coef[0])}] = {{{tmp_coef}}};"

		tmp_intercept = str(intercept).replace("[", "{").replace("]","}")
		intercept_array = f"constexpr {internal_type} intercept[{len(intercept)}] = {tmp_intercept};"

		normalisation = ""
		if self.model.normalise:
			normalisation = f"""
					for(unsigned int i=0; i<X.size(); i++){{ //normalise each channel of timeseries X
						float mean = (std::accumulate(X[i].begin(), X[i].end(), 0.0))/X[i].size();

						//calculate variance of channel
						float variance_sum = 0.0;
						for(unsigned int j=0; j<X[i].size(); j++){{
							float value = X[i][j];
							variance_sum += (value - mean) * (value - mean);
						}}
						float variance = variance_sum / X[i].size();
						//normalise input channel
						for(unsigned int j=0; j<X[i].size(); j++){{
							X[i][j] = (X[i][j] - mean) / (variance);
						}}
					}}""".strip()

		####

		self.code = f"""
			{kerneldefinition}
			#include "model.h"
            #include <vector>
            #include <iostream>

            {kernellist}

			{coef_array}
			{intercept_array}

			std::tuple<float, float> applyKernel(std::vector<std::vector<{self.feature_type}>>const &timeseries, std::vector<std::vector<{self.feature_type}>>const &kernel, float const &bias, float const &dilation, int const &padding) {{
				
				float max = -3.4028235e38f; //smallest float number
				int positive_values = 0;
				int kernel_length = kernel[0].size();
				int timeseries_length = timeseries[0].size(); //implies that every channel in one time series sample has the same size
				int dimensions = timeseries.size();

				//size of the convolution; important for calculating the ppv
				int outputsize = timeseries_length + (2 * padding) - (kernel_length - 1) * dilation;
				//endpoint of the convolution for the iteration
				int end = (timeseries_length + padding) - ((kernel_length - 1) * dilation);

				//we skipped to pad the timeseries for efficiency.
				//instead, we just skip the places where the timeseries should be padded (inspired by the sktime apply_kernel methods)

				int result = 0;
				int index = 0;
				for (int i =-padding; i<end; i++){{
					result = bias;
					index = i;

					for (int j=0; j<kernel_length; j++){{
						if (index > -1 && index < timeseries_length){{
							for (int d=0; d<dimensions; d++){{
								result = result + timeseries[d][index] * kernel[d][j];
							}}
						}}
						index = index + dilation;
					}}

					if (result > max){{
						max = result;
					}}
					if (result > 0){{
						positive_values++;
					}}
				}}

				return std::make_tuple(positive_values/outputsize, max);

			}}

			//describes the rocket transformation that generates the features for the linear model. Gets a batch of (multivariate) timeseries
			std::vector<float> transform(std::vector<std::vector<{self.feature_type}>>X){{
				std::vector<float> features;

					{normalisation}

					for (Kernel kernel : kernels){{
						//timeseries got every channel we want to convolute on
						std::vector<std::vector<float>> timeseries;
						for(int channel: kernel.getChannels()){{
							timeseries.push_back(X[channel]);
						}}
						std::tuple<float, float> results = applyKernel(timeseries, kernel.getWeights(), kernel.getBias(), kernel.getDilation(), kernel.getPadding());
						//append ppv. if model not minirocket, also append max
						features.push_back(std::get<0>(results));
						{"" if self.model.minirocket else "features.push_back(std::get<1>(results));"}


					}}

			
				return features;
			}}

			std::vector<{self.label_type}> predict(std::vector<std::vector<{self.feature_type}>> &timeseries) {{
				//transform layer
				std::vector<float> x = transform(timeseries);
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
			std::vector<{self.label_type}> predict(std::vector<std::vector<{self.feature_type}>> &timeseries);
		""".strip()