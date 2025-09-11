#!/usr/bin/env python3

import os
import sys
import tempfile
import numpy as np
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets import get_dataset
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu, Sigmoid
from mlgen3.implementations.neuralnet.cpp.nhwc_onnx import NHWC_ONNX
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

class TestCustomMLPONNX(unittest.TestCase):
    
    def setUp(self):
        # Load MNIST dataset
        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset("mnist")
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
    
    def test_mlp_onnx_export_and_deploy(self):
        # Define a simple MLP model
        class SimpleMLP(nn.Module):
            def __init__(self):
                super(SimpleMLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(784, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        # Create and train the model
        model = SimpleMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        batch_size = 64
        epochs = 3
        
        print("Training MLP model...")
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(0, min(10000, len(self.X_train)), batch_size):  # Limit training for faster testing
                inputs = torch.tensor(self.X_train[i:i+batch_size], dtype=torch.float32)
                targets = torch.tensor(self.y_train[i:i+batch_size], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(min(10000, len(self.X_train))/batch_size):.4f}")
        
        # Test the model accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(self.X_test), batch_size):
                inputs = torch.tensor(self.X_test[i:i+batch_size], dtype=torch.float32)
                targets = torch.tensor(self.y_test[i:i+batch_size], dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f"\nPyTorch Model Accuracy: {accuracy:.2f}%")

        # Create output directory for ONNX model
        output_dir = os.path.join("generated_code", "onnx_models")
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "simple_mlp.onnx")
        
        # Export to ONNX format
        print(f"Exporting model to ONNX format")
        dummy_input = torch.randn(1, 784, requires_grad=False)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"\nONNX Model exported successfully to {onnx_path}")
        
        # Create MLGen3 model for testing
        # We still need a MLGen3 model structure, but weights will be loaded from ONNX
        layers = []
        for i in range(0, len(model.layers), 3):
            
            linear_layer = model.layers[i]
            weight = linear_layer.weight.detach().numpy()
            bias = linear_layer.bias.detach().numpy()
            layers.append(Linear(weight, bias))
            
            if i+1 < len(model.layers) and isinstance(model.layers[i+1], nn.BatchNorm1d):
                bn_layer = model.layers[i+1]
                weight = bn_layer.weight.detach().numpy()
                bias = bn_layer.bias.detach().numpy()
                running_mean = bn_layer.running_mean.detach().numpy()
                running_var = bn_layer.running_var.detach().numpy()
                eps = bn_layer.eps
                layers.append(BatchNorm(weight, bias, running_mean, running_var, eps))
            
            if i+2 < len(model.layers) and isinstance(model.layers[i+2], nn.ReLU):
                output_shape = weight.shape[0]  # Output shape from previous linear layer
                layers.append(Relu(output_shape))
        
        # Create MLGen3 model
        mlgen_model = NeuralNet.from_layers(layers)
        mlgen_model.XTest = self.X_test
        mlgen_model.YTest = self.y_test
        
        # Generate C++ code using NHWC_ONNX implementation
        print("Generating C++ code using NHWC_ONNX implementation...")
        implementation = NHWC_ONNX(
            mlgen_model,
            onnx_path,
            feature_type="float",
            label_type="float",
            internal_type="float"
        )
        implementation.implement()
        
        # Deploy and test the model using LinuxStandalone materializer with ONNX support
        print("Deploying model...")
        materializer = LinuxStandalone(
            implementation,
            measure_accuracy=True,
            measure_time=True,
            use_onnx=True  # Use the ONNX-specific Makefile template
        )
        
        output_path = os.path.join("generated_code", "mlp_onnx")
        materializer.materialize(output_path)
        print(f"Model materialized at: {output_path}")
        
        # Copy ONNX model to deployment directory
        import shutil
        shutil.copy(onnx_path, output_path)
        
        materializer.deploy()
        print("Model deployed successfully.\n")
        
        results = materializer.run(verbose=True)
        print(f"Model test results: {results}")
        

if __name__ == '__main__':
    unittest.main()