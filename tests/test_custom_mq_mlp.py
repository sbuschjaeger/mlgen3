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
import torch.nn.functional as F
from tqdm import tqdm

from Datasets import get_dataset
from matquant import MatQuant, MQ_ActivationQuantizer

# Create output directories
os.makedirs("models/matquant/mnist", exist_ok=True)
os.makedirs("generated_code/matquant_mnist", exist_ok=True)

# Load MNIST dataset
print("Loading MNIST dataset...")
X_train, y_train, X_test, y_test = get_dataset("mnist")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert to PyTorch tensors
train_x = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)
test_x = torch.tensor(X_test, dtype=torch.float32)
test_y = torch.tensor(y_test, dtype=torch.long)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
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

# Create model
print("\nCreating MLP model...")
model = MLP()

# Create MatQuant config
config = {
    'quantization': {
        'use_matquant': True,
        'use_codistillation': False,
        'use_qat': False,
        'fx_mode': False,
        'target_bits': [8, 4, 2],
        'loss_weights': {8: 0.4, 4: 0.4, 2: 0.2},
        'quantize_bias': True,
        'quantize_target': 'weights_and_activations',
        'quantize_layers': [
            'layers.0.weight',
            'layers.3.weight',
            'layers.6.weight'
        ]
    }
}

# Create LayerRegistry class to register layers for MatQuant
class LayerRegistry:
    def __init__(self):
        self.layer_paths = []
    
    def register_model(self, model):
        """Register all quantizable layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_name = f"{name}.weight"
                if weight_name not in self.layer_paths:
                    self.layer_paths.append(weight_name)
        return self.layer_paths

# Create and register layers
layer_registry = LayerRegistry()
all_layers = layer_registry.register_model(model)
print(f"Registered layers: {all_layers}")

# Initialize MatQuant wrapper
print("\nInitializing MatQuant wrapper...")
mq_model = MatQuant(model, config)
mq_model.set_quantized_layers(config['quantization']['quantize_layers'])

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
batch_size = 64
epochs = 1

# Training loop
print(f"\nTraining MLP with MatQuant for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
        inputs = train_x[i:i+batch_size]
        targets = train_y[i:i+batch_size]
        
        optimizer.zero_grad()
        
        # Multi-precision forward pass
        outputs = mq_model.multi_precision_forward(inputs)
        
        # Calculate weighted loss
        loss, individual_losses = mq_model.matquant_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    scheduler.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Test for each bit-width
        accuracies = {}
        for bits in config['quantization']['target_bits']:
            correct = 0
            total = 0
            
            for i in tqdm(range(0, len(X_test), batch_size), desc=f"Testing {bits}-bit"):
                inputs = test_x[i:i+batch_size]
                targets = test_y[i:i+batch_size]
                
                outputs = mq_model.forward_with_quant(inputs, bits)
                _, predicted = torch.max(outputs, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            accuracy = 100 * correct / total
            accuracies[bits] = accuracy
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")
    for bits, acc in accuracies.items():
        print(f"  {bits}-bit Accuracy: {acc:.2f}%")

# Save the trained MatQuant model
model_path = "models/matquant/mnist/mq_model.pt"

## Option 1: save weights along with config information
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'config': config,
#     'test_accuracy': accuracies
# }, model_path)

## Option 2: save only the model weights
torch.save(model.state_dict(), model_path)

print(f"\nSaved MatQuant model to {model_path}")

# Extract models with different bit-widths
print("\nExtracting models with different bit-widths...")
extracted_models = {}
for bits in config['quantization']['target_bits']:
    extracted_models[bits] = mq_model.extract_model(bits)

# Create mix-and-match model
mix_config = {
    'layers.0.weight': 8, 
    'layers.3.weight': 4, 
    'layers.6.weight': 2
}
mix_model = mq_model.mix_and_match(mix_config)

print("\nTesting extracted and mix-and-match models...")
# Test extracted models
for bits, ext_model in extracted_models.items():
    ext_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(0, len(X_test), batch_size):
            inputs = test_x[i:i+batch_size]
            targets = test_y[i:i+batch_size]
            
            outputs = ext_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Extracted {bits}-bit model accuracy: {accuracy:.2f}%")

print("")

# Test mix-and-match model
mix_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        inputs = test_x[i:i+batch_size]
        targets = test_y[i:i+batch_size]
        
        outputs = mix_model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
    accuracy = 100 * correct / total
    print(f"Mix-and-match model accuracy: {accuracy:.2f}%")

print("")

# Now convert to MLGen3 model and generate C++ code
print("Converting to MLGen3 model...")
import argparse
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu

# Function to extract model for MLGen3
def extract_mlgen3_model(pytorch_model):
    layers = []
    
    # Extract parameters from the PyTorch model
    for i in range(0, len(pytorch_model.layers), 3):
        if i+2 < len(pytorch_model.layers):
            # Linear layer
            linear = pytorch_model.layers[i]
            weight = linear.weight.detach().numpy()
            bias = linear.bias.detach().numpy()
            
            # Store layer index in name for easier mapping to mix-and-match config
            layer_name = f"layers.{i // 3}.weight"
            linear_layer = Linear(weight, bias)
            linear_layer.layer_name = layer_name  # Store name for reference
            layers.append(linear_layer)
            
            # BatchNorm layer
            bn = pytorch_model.layers[i+1]
            scale = bn.weight.detach().numpy()
            bias = bn.bias.detach().numpy()
            mean = bn.running_mean.detach().numpy()
            var = bn.running_var.detach().numpy()
            eps = bn.eps
            layers.append(BatchNorm(scale, bias, mean, var, eps))
            
            # Activation layer
            if isinstance(pytorch_model.layers[i+2], nn.ReLU):
                output_shape = weight.shape[0]
                layers.append(Relu(output_shape))
        else:
            # Final Linear layer
            linear = pytorch_model.layers[i]
            weight = linear.weight.detach().numpy()
            bias = linear.bias.detach().numpy()
            layers.append(Linear(weight, bias))
    
    # Create MLGen3 model
    mlgen_model = NeuralNet.from_layers(layers)
    return mlgen_model

def generate_uniform_model(bit_width, extracted_models, X_test, y_test):
    """Generate and deploy a uniform bit-width model"""
    print(f"\nGenerating C++ code for {bit_width}-bit uniform model...")
    from mlgen3.implementations.neuralnet.cpp.matquant import MatQuant as MLGenMatQuant
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    model = extracted_models[bit_width]
    mlgen_model = extract_mlgen3_model(model)
    mlgen_model.XTest = X_test
    mlgen_model.YTest = y_test

    implementation = MLGenMatQuant(
        mlgen_model, 
        feature_type="float", 
        label_type="float",
        internal_type="float",
        target_bits=bit_width
    )
    
    # Create materializer first to set the filename
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        filename=f"matquant_{bit_width}bit"
    )
    
    # Now implement after the filename has been set
    implementation.implement()
    
    output_path = f"generated_code/matquant_mnist/uniform_{bit_width}bit"
    os.makedirs(output_path, exist_ok=True)

    materializer.materialize(output_path)
    print(f"Model materialized at: {output_path}")
    materializer.deploy()
    print(f"{bit_width}-bit model deployed successfully.")
    results = materializer.run(verbose=True)
    print(f"{bit_width}-bit model results: {results}")
    
    return results

def generate_mix_model(mix_config, mq_model, X_test, y_test):
    """Generate and deploy a mix-and-match model"""
    print(f"\nGenerating C++ code for mix-and-match model with config: {mix_config}")
    from mlgen3.implementations.neuralnet.cpp.matquant import MatQuant as MLGenMatQuant
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    mix_model = mq_model.mix_and_match(mix_config)
    mlgen_model = extract_mlgen3_model(mix_model)
    mlgen_model.XTest = X_test
    mlgen_model.YTest = y_test

    implementation = MLGenMatQuant(
        mlgen_model, 
        feature_type="float", 
        label_type="float",
        internal_type="float",
        mix_and_match_config=mix_config
    )

    config_str = '_'.join([f"{layer.split('.')[-2]}{bits}" for layer, bits in mix_config.items()])
    
    # Create materializer first to set the filename
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        filename=f"matquant_mix_{config_str}"
    )
    
    # Now implement after the filename has been set
    implementation.implement()
    
    output_path = f"generated_code/matquant_mnist/mix_{config_str}"
    os.makedirs(output_path, exist_ok=True)

    materializer.materialize(output_path)
    print(f"Model materialized at: {output_path}")
    materializer.deploy()
    print("Mix-and-match model deployed successfully.")
    results = materializer.run(verbose=True)
    print(f"Mix-and-match model results: {results}")
    
    return results

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generate MatQuant models with specified bit-widths')
    parser.add_argument('--uniform', type=int, nargs='+', help='Uniform bit-width models to generate (e.g. 8 4 2)')
    parser.add_argument('--mix', action='store_true', help='Generate mix-and-match model')
    parser.add_argument('--custom-mix', type=str, help='Custom mix-and-match configuration in format "layer1:bits,layer2:bits" (e.g. "layers.0.weight:8,layers.3.weight:4,layers.6.weight:2")')
    return parser.parse_args()

args = parse_args()

# Process the models based on arguments
results = {}

# Generate uniform bit-width models
if args.uniform:
    for bit_width in args.uniform:
        if bit_width in extracted_models:
            results[f"uniform_{bit_width}bit"] = generate_uniform_model(bit_width, extracted_models, X_test, y_test)
        else:
            print(f"Warning: No {bit_width}-bit model available. Available bit-widths: {list(extracted_models.keys())}")

# Generate default mix-and-match model
if args.mix:
    default_mix = {
        'layers.0.weight': 8, 
        'layers.3.weight': 4, 
        'layers.6.weight': 2
    }
    results["default_mix"] = generate_mix_model(default_mix, mq_model, X_test, y_test)

# Generate custom mix-and-match model
if args.custom_mix:
    try:
        custom_mix = {}
        for pair in args.custom_mix.split(','):
            layer, bits = pair.split(':')
            custom_mix[layer] = int(bits)
        results["custom_mix"] = generate_mix_model(custom_mix, mq_model, X_test, y_test)
    except Exception as e:
        print(f"Error parsing custom mix-and-match configuration: {e}")
        print("Format should be: 'layers.0.weight:8,layers.3.weight:4,layers.6.weight:2'")

# If no arguments provided, generate a default 4-bit model
if not args.uniform and not args.mix and not args.custom_mix:
    print("No specific models requested. Generating default 4-bit model.")
    results["uniform_4bit"] = generate_uniform_model(4, extracted_models, X_test, y_test)

print("\nMatQuant implementation complete!")
print(f"Generated models: {list(results.keys())}")
