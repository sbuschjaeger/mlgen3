import os
import sys
# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Datasets import get_dataset

# Load Fashion-MNIST dataset
X_train, y_train, X_test, y_test = get_dataset("fashion")

# Reshape data for CNN (adding channel dimension)
X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0

# Define the VGG4 network architecture
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=3136, out_features=2048, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=10, bias=True)
        )
        
    def forward(self, x):
        return self.model(x)

# Train the model
model = VGG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# Training parameters
batch_size = 64
epochs = 1  # Reduced for faster execution in test environment

print("Training VGG4 model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
        targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    scheduler.step()
    # Print epoch statistics
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")

# Test the PyTorch model's accuracy
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(X_test), batch_size), desc="Testing"):
        inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
        targets = torch.tensor(y_test[i:i+batch_size], dtype=torch.long)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f"PyTorch Model Accuracy: {100 * correct / total:.2f}%")

print("Converting to MLGen3 model...")

# Flatten test data for MLGen3 compatibility
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Extract the fully connected part of the network only
# (CNN layers would require additional implementation in MLGen3)
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.activations import Relu

# We need to run a forward pass to get the output of the flattened layer
dummy_input = torch.tensor(X_test[0:1], dtype=torch.float32)
with torch.no_grad():
    # Get activations at each layer for the dummy input
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for the layer activations we need
    handles = []
    handles.append(model.model[8].register_forward_hook(get_activation('flatten')))  # Flatten layer
    model(dummy_input)  # Forward pass to populate activations
    
    # Clean up hooks
    for handle in handles:
        handle.remove()

# Get the flattened features for the test set
flattened_features = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
        # Run forward pass up to flatten layer
        output = model.model[:9](inputs)  # Up to and including flatten
        flattened_features.append(output)

flattened_X_test = torch.cat(flattened_features, dim=0).numpy()

# Extract the fully connected layers
fc_layers = []

# First Linear + ReLU
linear1 = model.model[9]  # First linear layer
weight1 = linear1.weight.detach().numpy()
bias1 = linear1.bias.detach().numpy()
fc_layers.append(Linear(weight1, bias1))
fc_layers.append(Relu(2048))  # Output shape from first linear layer

# Second Linear layer
linear2 = model.model[11]  # Second linear layer
weight2 = linear2.weight.detach().numpy()
bias2 = linear2.bias.detach().numpy()
fc_layers.append(Linear(weight2, bias2))

# Create MLGen3 model
mlgen_model = NeuralNet.from_layers(fc_layers)
mlgen_model.XTest = flattened_X_test
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
print("Deploying FC layers of VGG4 model...")
materializer = LinuxStandalone(
    implementation, 
    measure_accuracy=True, 
    measure_time=True
)

output_path = os.path.join("generated_code", "custom_vgg4_fc")
os.makedirs(output_path, exist_ok=True)

materializer.materialize(output_path)
print("Model materialized at:", output_path)
materializer.deploy()
print("Model deployed successfully.")
results = materializer.run(verbose=True)
print(f"Deployment results for FC layers: {results}")

# Note: For full CNN support, additional implementations would be needed in MLGen3
print("Note: This implementation only deploys the fully connected layers of the VGG4 model.")
print("For full CNN support including Conv2d and MaxPool2d, additional implementations would be needed in MLGen3.")
