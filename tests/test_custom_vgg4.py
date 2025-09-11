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
X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0

# Define VGG4 network architecture (same model as in qnn matquant framework)
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

# Train model
model = VGG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# Training parameters
batch_size = 64
epochs = 3

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
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")

# Test model accuracy
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

print("Converting to MLGen3 model with full CNN support...")

# Convert to MLGen3 model
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.activations import Relu
from mlgen3.models.nn.conv2d import Conv2D
from mlgen3.models.nn.maxpool2d import MaxPool2D
from mlgen3.models.nn.batchnorm import BatchNorm

# Extract layers and parameters from the trained model
layers = []

conv1 = model.model[0]
conv1_weight = conv1.weight.detach().numpy()
conv1_bias = conv1.bias.detach().numpy()
layers.append(Conv2D(conv1_weight, conv1_bias, 
                    kernel_size=conv1.kernel_size, 
                    stride=conv1.stride, 
                    padding=conv1.padding))

maxpool1 = model.model[1]
layers.append(MaxPool2D(kernel_size=maxpool1.kernel_size, 
                        stride=maxpool1.stride, 
                        padding=maxpool1.padding))

bn1 = model.model[2]
bn1_weight = bn1.weight.detach().numpy()
bn1_bias = bn1.bias.detach().numpy()
bn1_mean = bn1.running_mean.detach().numpy()
bn1_var = bn1.running_var.detach().numpy()
layers.append(BatchNorm(bn1_weight, bn1_bias, bn1_mean, bn1_var, bn1.eps))

layers.append(Relu(64))

conv2 = model.model[4]
conv2_weight = conv2.weight.detach().numpy()
conv2_bias = conv2.bias.detach().numpy()
layers.append(Conv2D(conv2_weight, conv2_bias, 
                    kernel_size=conv2.kernel_size, 
                    stride=conv2.stride, 
                    padding=conv2.padding))

maxpool2 = model.model[5]
layers.append(MaxPool2D(kernel_size=maxpool2.kernel_size, 
                        stride=maxpool2.stride, 
                        padding=maxpool2.padding))

bn2 = model.model[6]
bn2_weight = bn2.weight.detach().numpy()
bn2_bias = bn2.bias.detach().numpy()
bn2_mean = bn2.running_mean.detach().numpy()
bn2_var = bn2.running_var.detach().numpy()
layers.append(BatchNorm(bn2_weight, bn2_bias, bn2_mean, bn2_var, bn2.eps))

layers.append(Relu(64))

fc1 = model.model[9]
fc1_weight = fc1.weight.detach().numpy()
fc1_bias = fc1.bias.detach().numpy()
layers.append(Linear(fc1_weight, fc1_bias))

layers.append(Relu(2048))

fc2 = model.model[11]
fc2_weight = fc2.weight.detach().numpy()
fc2_bias = fc2.bias.detach().numpy()
layers.append(Linear(fc2_weight, fc2_bias))

# Create MLGen3 model
mlgen_model = NeuralNet.from_layers(layers)

# Ensure test data is properly formatted for the Conv2D implementation
# No need to reshape X_test as we want the Conv2D implementation to process it directly
mlgen_model.XTest = X_test
mlgen_model.YTest = y_test

print(f"Input shape for MLGen3 model: {X_test.shape}")

# Generate C++ code
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
print("Deploying full VGG4 CNN model...")
materializer = LinuxStandalone(
    implementation, 
    measure_accuracy=True, 
    measure_time=True
)

output_path = os.path.join("generated_code", "custom_vgg4_full_cnn")
os.makedirs(output_path, exist_ok=True)

materializer.materialize(output_path)
print("Model materialized at:", output_path)
materializer.deploy()
print("Model deployed successfully.")
results = materializer.run(verbose=True)
print(f"Deployment results for full VGG4 CNN: {results}")

print("Full CNN implementation of VGG4 now supported in MLGen3!")
