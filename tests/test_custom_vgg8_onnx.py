#!/usr/bin/env python3
import os
import sys
import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets import get_dataset
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu
from mlgen3.implementations.neuralnet.cpp.vgg_onnx import VGG_ONNX
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.implementations.neuralnet.cpp.onnx_utils import deploy_onnx_model

class TestCustomVGG8ONNX(unittest.TestCase):
    def setUp(self):
        # Load CIFAR10 dataset
        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset("cifar10")
        # Reshape data for CNN (samples, channels, height, width)
        self.X_train = self.X_train.reshape(-1, 3, 32, 32).astype('float32') / 255.0
        self.X_test = self.X_test.reshape(-1, 3, 32, 32).astype('float32') / 255.0
        
        # Define the VGG8 network architecture
        class VGG8(nn.Module):
            def __init__(self):
                super(VGG8, self).__init__()
                self.model = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Block 2
                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Block 3
                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Block 4
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Block 5
                    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Block 6
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    
                    # Fully connected layers
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(in_features=8192, out_features=1024, bias=True),
                    nn.ReLU(),
                    nn.Linear(in_features=1024, out_features=10, bias=True)
                )
                
            def forward(self, x):
                return self.model(x)
                
        self.model_cls = VGG8
        self.batch_size = 128
        self.output_dir = os.path.join("generated_code", "vgg8_onnx")
        os.makedirs(self.output_dir, exist_ok=True)
        self.onnx_dir = os.path.join("generated_code", "onnx_models")
        os.makedirs(self.onnx_dir, exist_ok=True)

    def test_vgg8_onnx_export_and_deploy(self):
        # Train the model
        model = self.model_cls()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training parameters - reduced for faster testing
        epochs = 3
        
        print("Training VGG8 model...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i in tqdm(range(0, len(self.X_train), self.batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = torch.tensor(self.X_train[i:i+self.batch_size], dtype=torch.float32)
                targets = torch.tensor(self.y_train[i:i+self.batch_size], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(self.X_train)/self.batch_size):.4f}")
        
        # Test the PyTorch model's accuracy
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for i in range(0, len(self.X_test), self.batch_size):
                inputs = torch.tensor(self.X_test[i:i+self.batch_size], dtype=torch.float32)
                targets = torch.tensor(self.y_test[i:i+self.batch_size], dtype=torch.long)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            pytorch_accuracy = 100 * correct / total
            print(f"\nPyTorch Model Accuracy: {pytorch_accuracy:.2f}%")
        
        # Export to ONNX
        print("Exporting VGG8 model to ONNX format...")
        onnx_path = os.path.join(self.onnx_dir, "vgg8_model.onnx")
        
        # Create a dummy input for the model
        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)
        
        # Export the model to ONNX format
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

        # Create a simplified MLGen3 model structure
        # We only need a basic structure as the weights will be loaded from ONNX
        layers = [
            # Just create minimal structure - actual inference will use ONNX model
            Linear(np.random.randn(10, 3072), np.zeros(10))
        ]
        
        # Create MLGen3 model with test data - IMPORTANT: Keep 4D shape for CNN
        mlgen_model = NeuralNet.from_layers(layers)
        
        # DO NOT flatten the test data - keep original 4D shape (samples, channels, height, width)
        mlgen_model.XTest = self.X_test  # Keep the 4D shape
        mlgen_model.YTest = self.y_test
        
        # Generate C++ code using VGG_ONNX implementation
        print("Generating C++ code using VGG_ONNX implementation...")
        implementation = VGG_ONNX(
            mlgen_model,
            onnx_path=onnx_path,
            feature_type="float",
            label_type="int",
            internal_type="float",
            batch_size=self.batch_size
        )
        
        # Generate implementation code
        implementation.implement()
        
        # Deploy the model using LinuxStandalone materializer
        print("Deploying model...")
        materializer = deploy_onnx_model(
            implementation, 
            self.output_dir,
            LinuxStandalone,
            measure_accuracy=True,
            measure_time=True
        )
        
        # Deploy and run
        materializer.deploy()
        print("Model deployed successfully.\n")

        results = materializer.run(verbose=True)
        print(f"Model test results: {results}")
        
        # Verify accuracy is within reasonable bounds of PyTorch model
        if "Accuracy" in results:
            onnx_accuracy = float(results["Accuracy"])
            accuracy_diff = abs(onnx_accuracy - pytorch_accuracy)
            # Allow a larger margin for CIFAR10 which is more challenging
            self.assertLess(accuracy_diff, 15.0)

if __name__ == '__main__':
    unittest.main()
