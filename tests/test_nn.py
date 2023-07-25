#!/usr/bin/env python3

import os
import tempfile
import numpy as np
import unittest

from sklearn import datasets
from sklearn.metrics import accuracy_score
from mlgen3.implemantations.neuralnet.cpp.bnn import BNN
from mlgen3.implemantations.neuralnet.cpp.nhwc import NHWC
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.nn.batchnorm import BatchNorm 
from mlgen3.models.nn.activations import Relu, Sigmoid, Sign, Step
from mlgen3.models.nn.linear import Linear

import torch
import torch.nn.functional as tnf

from mlgen3.models.nn.neuralnet import NeuralNet

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

    def test_activations(self):
        x = np.random.uniform(low=0, high=1, size=(25,25))
        
        xr = Relu(x.shape)(x) 
        self.assertSequenceEqual(x.shape, xr.shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(tnf.relu(torch.tensor(x)).numpy(), xr))

        xr = Sigmoid(x.shape)(x) 
        self.assertSequenceEqual(x.shape, xr.shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(tnf.sigmoid(torch.tensor(x)).numpy(), xr))

        xr = Sign(x.shape)(x) 
        self.assertSequenceEqual(x.shape, xr.shape)
        self.assertIsNone(np.testing.assert_array_almost_equal(torch.sign(torch.tensor(x)).numpy(), xr))

    def test_linear(self):
        # batch size, in_features, out_features
        batch_size = 32
        in_features = 64
        out_features = 16

        bias = np.random.uniform(low=0, high=1, size=(out_features))
        weight = np.random.uniform(low=0, high=1, size=(out_features, in_features))
        torch_linear = torch.nn.Linear(in_features, out_features)
        torch_linear.weight = torch.nn.Parameter(torch.tensor(weight))
        torch_linear.bias = torch.nn.Parameter(torch.tensor(bias))

        linear = Linear(weight, bias)
        input = np.random.uniform(low=0, high=1, size=(batch_size, in_features))

        self.assertIsNone(np.testing.assert_array_almost_equal(torch_linear(torch.tensor(input)).detach().numpy(), linear(input)))
    
    def test_batchnorm(self):        
        batch_size = 32
        n_features = 64
        x = np.random.uniform(low=0, high=1, size=(batch_size, n_features))

        running_mean = np.random.uniform(low=0, high=1, size=(n_features))
        running_var = np.random.uniform(low=0, high=1, size=(n_features))
        weight = np.random.uniform(low=0, high=1, size=(n_features))
        bias = np.random.uniform(low=0, high=1, size=(n_features))
        eps=1e-05

        batchnorm = BatchNorm(weight, bias, running_mean, running_var, eps)

        tbn = tnf.batch_norm(torch.tensor(x), running_mean = torch.tensor(running_mean), running_var = torch.tensor(running_var), weight = torch.tensor(weight), bias = torch.tensor(bias), eps = eps, training=False)

        self.assertIsNone(np.testing.assert_array_almost_equal(tbn.detach().numpy(), batchnorm(x)))

    def test_mlp(self): 
        class TSign(torch.nn.Module):

            def forward(self, x):
                return torch.sign(x)

        class TStep(torch.nn.Module):

            def __init__(self, threshold=0, low=-1, high=1):
                super().__init__()

                self.threshold = threshold
                self.low = low 
                self.high = high
                self.threshold_is_high = True

            def forward(self, x):
                if self.threshold_is_high:    
                    x[x >= self.threshold] = self.high
                    x[x < self.threshold] = self.low
                else:
                    x[x > self.threshold] = self.high
                    x[x <= self.threshold] = self.low
                
                return x

        # TODO Add the other activations in here as well...
        torch_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
            torch.nn.BatchNorm1d(num_features=16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=16, out_features=8),
            torch.nn.BatchNorm1d(num_features=8),
            TSign(), 
            torch.nn.Linear(in_features=8, out_features=8),
            torch.nn.BatchNorm1d(num_features=8),
            TStep() 
        )

        layers = [
            Linear(torch_layers[0].weight.detach().numpy(), torch_layers[0].bias.detach().numpy()),
            BatchNorm(torch_layers[1].weight.detach().numpy(), torch_layers[1].bias.detach().numpy(), torch_layers[1].running_mean.detach().numpy(), torch_layers[1].running_var.detach().numpy(), torch_layers[1].eps),
            Relu(32),
            Linear(torch_layers[3].weight.detach().numpy(), torch_layers[3].bias.detach().numpy()),
            BatchNorm(torch_layers[4].weight.detach().numpy(), torch_layers[4].bias.detach().numpy(), torch_layers[4].running_mean.detach().numpy(), torch_layers[4].running_var.detach().numpy(), torch_layers[4].eps),
            Sigmoid(16),
            Linear(torch_layers[6].weight.detach().numpy(), torch_layers[6].bias.detach().numpy()),
            BatchNorm(torch_layers[7].weight.detach().numpy(), torch_layers[7].bias.detach().numpy(), torch_layers[7].running_mean.detach().numpy(), torch_layers[7].running_var.detach().numpy(), torch_layers[7].eps),
            Sign(8), 
            Linear(torch_layers[9].weight.detach().numpy(), torch_layers[9].bias.detach().numpy()),
            BatchNorm(torch_layers[10].weight.detach().numpy(), torch_layers[10].bias.detach().numpy(), torch_layers[10].running_mean.detach().numpy(), torch_layers[10].running_var.detach().numpy(), torch_layers[10].eps),
            Step(8)
        ]

        x = np.random.uniform(size=(128, 64)).astype("float32")
        net = NeuralNet.from_layers(layers)
        pred = net.predict_proba(x)
        
        torch_layers.eval()
        predtorch = torch_layers(torch.tensor(x)).detach().numpy()

        self.assertIsNone(np.testing.assert_array_almost_equal(pred, predtorch))

    # TODO THIS DOES NOT FULLY WORK AT THE MOMENT!
    @unittest.skip
    def test_bnn_linuxstandalone(self):
        layers = [
            Linear(np.random.choice([-1,1],size=(32,self.X.shape[1])), np.random.choice([-1,1],size=32)),
            BatchNorm(np.random.uniform(size=32),np.random.uniform(size=32),np.random.uniform(size=32),np.random.uniform(size=32), 1e-5),
            Step(32),
            Linear(np.random.choice([-1,1],size=(16,32)), np.random.choice([-1,1],size=16)),
            BatchNorm(np.random.uniform(size=16),np.random.uniform(size=16),np.random.uniform(size=16),np.random.uniform(size=16), 1e-5),
            Step(16),
            Linear(np.random.choice([-1,1],size=(8,16)), np.random.choice([-1,1],size=8)),
            BatchNorm(np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8), 1e-5),
            Step(8),
            Linear(np.random.choice([-1,1],size=(8,8)), np.random.choice([-1,1],size=8)),
            BatchNorm(np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8), 1e-5),
            Step(8),
            Linear(np.random.choice([-1,1],size=(3,8)), np.random.choice([-1,1],size=3))
        ]

        net = NeuralNet.from_layers(layers)
        scores = net.score(self.X,self.y)
        acc = scores["Accuracy"]

        implementation = BNN(net, feature_type="float", label_type="float")
        implementation.implement()
        
        materializer = LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
        materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestMLPBNN"))
        materializer.deploy() 
        output = materializer.run(True) 
        self.assertAlmostEqual(float(output["Accuracy"]), acc*100.0, places=3)

    # TODO THIS DOES NOT FULLY WORK AT THE MOMENT!
    @unittest.skip
    def test_nhwc_linuxstandalone(self):
        layers = [
            Linear(np.random.uniform(size=(32,self.X.shape[1])), np.random.uniform(size=32)),
            BatchNorm(np.random.uniform(size=32),np.random.uniform(size=32),np.random.uniform(size=32),np.random.uniform(size=32), 1e-5),
            Relu(32),
            Linear(np.random.uniform(size=(16,32)), np.random.uniform(size=16)),
            BatchNorm(np.random.uniform(size=16),np.random.uniform(size=16),np.random.uniform(size=16),np.random.uniform(size=16), 1e-5),
            Sigmoid(16),
            Linear(np.random.uniform(size=(8,16)), np.random.uniform(size=8)),
            BatchNorm(np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8), 1e-5),
            Step(8),
            Linear(np.random.uniform(size=(8,8)), np.random.uniform(size=8)),
            BatchNorm(np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8),np.random.uniform(size=8), 1e-5),
            Sign(8),
            Linear(np.random.uniform(size=(3,8)), np.random.uniform(size=3))
        ]

        net = NeuralNet.from_layers(layers)
        scores = net.score(self.X,self.y)
        acc = scores["Accuracy"]

        implementation = NHWC(net, feature_type="float", label_type="float",internal_type="float")
        implementation.implement()
        
        materializer = LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
        materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestMLPNHWC"))
        materializer.deploy() 
        output = materializer.run(True) 
        self.assertAlmostEqual(float(output["Accuracy"]), acc*100.0, places=3)

        materializer.clean()

if __name__ == '__main__':
    unittest.main()