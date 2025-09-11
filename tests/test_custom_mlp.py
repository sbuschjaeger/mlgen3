import os
import sys
# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Datasets import get_dataset

# Load MNIST dataset
X_train, y_train, X_test, y_test = get_dataset("mnist")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define a simple neural network model
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

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
epochs = 3

print("Training model...")
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
        targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")

# Test model accuracy
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for i in range(0, len(X_test), batch_size):
        inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
        targets = torch.tensor(y_test[i:i+batch_size], dtype=torch.long)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f"PyTorch Model Accuracy: {100 * correct / total:.2f}%")

print("Converting to MLGen3 model...")
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu

# Extract layers and parameters
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
mlgen_model.XTest = X_test
mlgen_model.YTest = y_test

# Generate C++ code with NHWC implementation
from mlgen3.implementations.neuralnet.cpp.nhwc import NHWC
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

implementation = NHWC(
    mlgen_model, 
    feature_type="float", 
    label_type="float",
    internal_type="float"
)
implementation.implement()

# Deploy and test the model
print("Deploying model...")
materializer = LinuxStandalone(
    implementation, 
    measure_accuracy=True, 
    measure_time=True
)
output_path = os.path.join("generated_code", "custom_mnist_mlp")
os.makedirs(output_path, exist_ok=True)

materializer.materialize(output_path)
print("Model materialized at:", output_path)

materializer.deploy()
print("Model deployed successfully.")

results = materializer.run(verbose=True)
print(f"Deployment results: {results}")