import os
import sys
# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Datasets import get_dataset

# Load MNIST dataset
X_train, y_train, X_test, y_test = get_dataset("mnist")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define a simple neural network model with quantization support
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
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

# Custom quantization function for weights
def quantize_weights(weights, num_bits=8):
    w_min = weights.min().item()
    w_max = weights.max().item()
    
    # Avoid division by zero
    if w_min == w_max:
        return weights.clone(), 1.0, 0.0
        
    scale = (w_max - w_min) / (2**num_bits - 1)
    zero_point = -w_min / scale if scale != 0 else 0
    
    # Quantize
    quantized_weights = torch.clamp(torch.round(weights / scale + zero_point), 0, 2**num_bits - 1)
    
    # Dequantize for forward pass
    dequantized_weights = (quantized_weights - zero_point) * scale
    
    return dequantized_weights, scale, zero_point

# Apply Straight-Through Estimator for backpropagation
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantized_input):
        return quantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

ste = StraightThroughEstimator.apply

# Quantization hook for model training
def add_quantization_hooks(model, num_bits=8):
    def quantize_hook(module, input, output):
        # Apply only to Linear layers
        if isinstance(module, nn.Linear):
            # Quantize weights
            quantized_weight, scale, zero_point = quantize_weights(module.weight.data, num_bits)
            module.weight.data = ste(module.weight, quantized_weight)
            
            # Store quantization parameters for later use
            if not hasattr(module, 'scale'):
                module.register_buffer('scale', torch.tensor(scale))
                module.register_buffer('zero_point', torch.tensor(zero_point))
            else:
                module.scale = torch.tensor(scale)
                module.zero_point = torch.tensor(zero_point)
    
    # Register forward hook for each layer
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(quantize_hook)
            hooks.append(hook)
    
    return hooks

# Train the model
model = QuantizedMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Add quantization hooks
num_bits = 8
hooks = add_quantization_hooks(model, num_bits)

batch_size = 64
epochs = 5

print("Training quantized model...")
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
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")

# Remove hooks after training
for hook in hooks:
    hook.remove()

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

print("Converting to MLGen3 quantized model...")
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

# Generate C++ code with quantized implementation
from mlgen3.implementations.neuralnet.cpp.qnn_qat import QNN
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

implementation = QNN(
    mlgen_model, 
    feature_type="float", 
    label_type="float",
    internal_type="float",
    num_bits=num_bits
)
implementation.implement()

# Deploy and test the model
print("Deploying quantized model...")
materializer = LinuxStandalone(
    implementation, 
    measure_accuracy=True, 
    measure_time=True
)

output_path = os.path.join("generated_code", "custom_q_mnist_mlp")
os.makedirs(output_path, exist_ok=True)

materializer.materialize(output_path)
print("Model materialized at:", output_path)

materializer.deploy()
print("Model deployed successfully.")

results = materializer.run(verbose=True)
print(f"Deployment results: {results}")
