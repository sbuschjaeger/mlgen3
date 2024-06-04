# implementation of ROCKET https://arxiv.org/abs/1910.13051

import numpy as np
import sys
import pandas as pd

from .linear import Linear
from .model import Model, PredictionType

# from mlgen3.models.nn.linear import Linear
from sktime.transformations.panel.rocket import Rocket as SK_Rocket
from sktime.transformations.panel.rocket import MiniRocket as SK_MiniRocket
from sktime.classification.kernel_based import RocketClassifier
from sktime.pipeline import Pipeline


class Rocket(Model):

    # constructor taking rocket panel
    def __init__(self):
        super().__init__(PredictionType.CLASSIFICATION)

        self.multivariate = False
        self.normalise = False
        self.dict = 0
        self.kernel = 0
        self.weights = 0
        self.lengths = 0
        self.biases = 0
        self.dilations = 0
        self.paddings = 0
        self.num_channels = 0
        self.channel_indices = 0
        self.linear = 0
        self.clases = 0
        self.kernellist = 0

    def __call__(self):
        pass

    @classmethod
    def from_sklearn(cls, sk_model):
        if isinstance(sk_model, RocketClassifier):
            print(
                "RocketClassifier is currently not supported, due to missing kernel weights in the classifier.",
                "Please use an sktime.transformation.panel.rocket object for initialization, and add a linear model with 'model.addLinear(linear)'.",
            )
            # ...

        elif isinstance(sk_model, SK_Rocket) or isinstance(sk_model, SK_MiniRocket):
            rocket_model = cls()

            rocket_model.normalise = sk_model.normalise
            rocket_model.multivariate = False
            rocket_model.dict = sk_model.__dict__
            rocket_model.kernel = sk_model.__dict__[
                "kernels"
            ]  # kernel = tupel ([weights], [lengths], [biases], [dilations], [paddings])
            rocket_model.weights = rocket_model.kernel[0]
            rocket_model.lengths = rocket_model.kernel[1]
            rocket_model.biases = rocket_model.kernel[2]
            rocket_model.dilations = rocket_model.kernel[3]
            rocket_model.paddings = rocket_model.kernel[4]
            rocket_model.num_channels = rocket_model.kernel[5]
            rocket_model.channel_indices = rocket_model.kernel[6]
            rocket_model.kernellist = rocket_model.getKernellist()

            return rocket_model
        else:
            raise ValueError("Model isn't part of Rocket")

    @classmethod
    def from_dict(cls, data):
        """Generates a new rocket model from the given dictionary. It is assumed that a rocket model has previously been stored with the :meth:`Rocket.to_dict` method.

        Args:
                data (dict): The dictionary from which this linear model should be generated.

        Returns:
                Tree: The newly generated rocket model.
        """

        model = Rocket()
        model.normalise = data["normalise"]
        model.multivariate = data["multivariate"]
        model.dict = data["dict"]
        model.kernel = data["kernel"]
        model.weights = data["weights"]
        model.lengths = data["lengths"]
        model.biases = data["biases"]
        model.dilations = data["dilations"]
        model.paddings = data["paddings"]
        model.num_channels = data["num_channels"]
        model.channel_indices = data["channel_indices"]
        model.linear = data["linear"]
        return model

    def to_dict(self):
        """Stores this rocket model as a dictionary which can be loaded with :meth:`Rocket.from_dict`.

        Returns:
                dict: The dictionary representation of this rocket model.
        """
        return {
            "multivariate": self.multivariate,
            "normalise": self.normalise,
            "dict": self.dict,
            "kernel": self.kernel,
            "weights": self.weights,
            "lengths": self.lengths,
            "biases": self.biases,
            "dilations": self.dilations,
            "paddings": self.paddings,
            "num_channels": self.num_channels,
            "channel_indices": self.channel_indices,
            "linear": self.linear,
            "classes": self.classes,
            "kernellist": self.kernellist,
        }

    def addLinear(self, sk_linear):
        """Adds linear model to rocket. Object creation only takes the rocket transformation, but not the linear model. So we also need this method."""
        linear = Linear.from_sklearn(sk_linear)
        self.linear = linear
        self.classes = sk_linear.classes_

    # returns a list with every entry being a kernel. every kernel is described as a quintuple of (weights:matrix, channels:list, bias:float, dilation:int, padding:int)
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
            channels = [
                channel_indices.pop() for i in range(0, c)
            ]  # due to multivariate rocket, each kernel can target multiple input channels. That's why the kernel-width could variate and is not always equal.
            channels.reverse()  # We also have to remember, which input channels a kernel convolutes over. So we save a kernel with the associated channel_indices
            while c > 0:
                vector = [
                    weights.pop() for i in range(0, length)
                ]  # extract the weights for each vector of a kernel
                vector.reverse()
                c = c - 1
                matrix.append(
                    vector
                )  # matrix consists of multiple vectors for multiple channels

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

    # describes the rocket transformation that generates the features for the linear model
    def transform(self, Xs):
        kernellist = self.kernellist  # (matrix, channels, bias, dilation, padding)
        features = []
        if not isinstance(Xs, np.ndarray):
            Xs = Xs.to_numpy()
        num_samples = len(Xs)

        for sample_index in range(num_samples):
            X = Xs[sample_index]
            X = np.asarray(X)
            if self.normalise:
                for i in range(len(X)):  # we normalise each channel of the timeseries X
                    X[i] = (X[i] - X[i].mean()) / (
                        X[i].std(ddof=0) + 1e-8
                    )  # ddof=0 is very important. Otherwise, normalising isn't similar to sktime

            for kernel, channels, bias, dilation, padding in kernellist:

                X = np.transpose(X)
                timeseries = X[channels]
                max, ppv = self.applyKernel(timeseries, kernel, bias, dilation, padding)
                features.append(max)
                features.append(ppv)

        features = np.asarray(features)
        features = features.reshape(num_samples, -1)
        return features

    # Applies this linear model to the given data and provides the predicted probabilities for each example in X.
    def predict_proba(self, X):
        assert self.linear != 0, "Linear model is missing"
        features = self.transform(X)
        results = self.linear.predict_proba(features)
        argmax_results = np.argmax(results, axis=1, keepdims=False)
        return np.asarray([self.classes[index] for index in argmax_results])

    def implement(self):
        assert self.linear is not None, "Linear model has to be set before implementing"
        assert self.kernel is not None, "Kernels are missing"

        alloc = ""
        code = ""
        header = ""

    def printAll(self):
        print("-----------------")
        print("weights: ", self.weights)
        print("lengths: ", self.lengths)
        print("biases: ", self.biases)
        print("dilations: ", self.dilations)
        print("paddings: ", self.paddings)
        print("num_channels: ", self.num_channels)
        print("channel_indices: ", self.channel_indices)
        print("-----------------")

    ######################################################################################################################################################################

    # we don't need to return a feature map. it is sufficent to return max and ppv
    def applyKernel(self, timeseries, kernel, bias, dilation, padding):
        # important for displaying the resulting features of the rocket panel.
        # they were used for debugging, just ignore them
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        np.set_printoptions(threshold=np.inf)
        # timeseries = vector; each element is a channel of the timeseries

        max = -(sys.float_info.max)
        positive_values = 0
        kernel_length = 0
        timeseries_length = len(timeseries[0])
        dimensions = 1  # filter for multivariate or univariate kernel

        if len(np.shape(kernel)) == 1:
            kernel_length = np.shape(kernel)[0]
        elif len(np.shape(kernel)) == 2:
            kernel_length = np.shape(kernel)[1]
            dimensions = np.shape(kernel)[0]
        else:
            raise AttributeError("Kernel got a weird shape")

        # size of the convolution; important for calculating the ppv
        output_size = (timeseries_length + (2 * padding)) - (
            (kernel_length - 1) * dilation
        )

        # endpopint of the convolution for the iteration
        end = (timeseries_length + padding) - ((kernel_length - 1) * dilation)

        # we skipped to pad the timeseries for efficency.
        # instead, we just skip the places where the timeseries should be padded (inspired by the sktime apply_kernel methods)
        for i in range(-padding, end):
            result = bias
            index = i
            for j in range(kernel_length):
                if index > -1 and index < timeseries_length:
                    if dimensions == 1:
                        result += timeseries[0][index] * kernel[j]
                    else:  # if timeseries got more than one channel
                        for d in range(dimensions):
                            result += timeseries[d][index] * kernel[d][j]
                index = index + dilation

            if result > max:
                max = result

            if result > 0:
                positive_values += 1

        return (np.float32(positive_values / output_size), np.float32(max))
