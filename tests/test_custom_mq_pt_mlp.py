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
import argparse

from Datasets import get_dataset
from matquant import MatQuant, MQ_ActivationQuantizer
from mlgen3.utils.seed import set_seed, get_seed_from_config

# Create output directories
os.makedirs("models/matquant/mnist_pt", exist_ok=True)
os.makedirs("generated_code/matquant_pt_mnist", exist_ok=True)

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
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.model(x)

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
        'quantize_target': 'weights_and_activations', # 'weights_and_activations' or 'weights_only'
        'quantize_layers': [
            'model.0.weight',
            'model.3.weight',
            'model.6.weight'
        ]
    },
    'training': {
        'seed': 707  # Random seed for reproducibility
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

def train_model(args):
    # Set random seed for reproducibility
    seed = get_seed_from_config(config)
    if args.seed is not None:
        seed = args.seed
    set_seed(seed)
    
    # Create model
    print("\nCreating MLP model...")
    model = MLP()

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
    epochs = args.epochs

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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(X_train)/batch_size):.4f}")


        # Evaluate forward_with_quant()
        model.eval()
        mq_model.eval()
        with torch.no_grad():
            # Test for each bit-width
            accuracies = {}
            for bits in config['quantization']['target_bits']:
                correct = 0
                total = 0
                
                for i in tqdm(range(0, len(X_test), batch_size), desc=f"(1)Testing {bits}-bit"):
                    inputs = test_x[i:i+batch_size]
                    targets = test_y[i:i+batch_size]
                    
                    outputs = mq_model.forward_with_quant(inputs, bits)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                accuracy = 100 * correct / total
                accuracies[bits] = accuracy
        
        for bits, acc in accuracies.items():
            print(f"(1)  {bits}-bit Accuracy: {acc:.2f}%")

        print("")


    # Save the trained model parameters
    model_path = "models/matquant/mnist_pt/mq_pt_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved MatQuant model to {model_path}")
    
    return model, mq_model

def extract_and_test_models(mq_model):
    batch_size = 64
    
    # Extract models with different bit-widths
    print("\nExtracting models with different bit-widths...")
    extracted_models = {}
    for bits in config['quantization']['target_bits']:
        extracted_models[bits] = mq_model.extract_model(bits)

    # Create mix-and-match model
    mix_config = {
        'model.0.weight': 8, 
        'model.3.weight': 4, 
        'model.6.weight': 2
    }
    mix_model = mq_model.mix_and_match(mix_config)

    print("\n(2)Testing extracted and mix-and-match models...")
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
            print(f"(2)Extracted {bits}-bit model accuracy: {accuracy:.2f}%")

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
    
    return extracted_models, mix_config

# Convert to MLGen3 model
print("Converting to MLGen3 model...")
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm
from mlgen3.models.nn.activations import Relu

# Function to extract model parameters from PyTorch to MLGen3 format
def extract_mlgen3_model(pytorch_model):
    layers = []
    
    # Extract parameters from the PyTorch model
    for i in range(0, len(pytorch_model.model), 3):
        if i+2 < len(pytorch_model.model):
            # Linear layer
            linear = pytorch_model.model[i]
            weight = linear.weight.detach().numpy()
            bias = linear.bias.detach().numpy()
            
            # Store layer index in name for easier mapping to mix-and-match config
            layer_name = f"model.{i}.weight"
            linear_layer = Linear(weight, bias)
            linear_layer.layer_name = layer_name  # Store name for reference
            layers.append(linear_layer)
            
            # BatchNorm layer
            bn = pytorch_model.model[i+1]
            scale = bn.weight.detach().numpy()
            bias = bn.bias.detach().numpy()
            mean = bn.running_mean.detach().numpy()
            var = bn.running_var.detach().numpy()
            eps = bn.eps
            layers.append(BatchNorm(scale, bias, mean, var, eps))
            
            # Activation layer
            if isinstance(pytorch_model.model[i+2], nn.ReLU):
                output_shape = weight.shape[0]
                layers.append(Relu(output_shape))
        else:
            # Final Linear layer
            linear = pytorch_model.model[i]
            weight = linear.weight.detach().numpy()
            bias = linear.bias.detach().numpy()
            layers.append(Linear(weight, bias))
    
    # Create MLGen3 model
    mlgen_model = NeuralNet.from_layers(layers)
    return mlgen_model

def generate_uniform_model(bit_width, model_path=None):
    """Generate and deploy a uniform bit-width model with 8-bit storage and runtime slicing"""
    print(f"\nGenerating C++ code for {bit_width}-bit uniform model with 8-bit storage and runtime slicing...")
    from mlgen3.implementations.neuralnet.cpp.matquant_pt import MatQuantPT
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

    # Load the saved model
    if not model_path:
        model_path = "models/matquant/mnist_pt/mq_pt_model.pt"
    
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    
    # Create MatQuant wrapper for extraction
    mq_model_temp = MatQuant(model, config)
    mq_model_temp.set_quantized_layers(config['quantization']['quantize_layers'])
    
    mq_model_8bit = mq_model_temp.extract_model(8)
    mlgen_model = extract_mlgen3_model(mq_model_8bit)
    mlgen_model.XTest = X_test
    mlgen_model.YTest = y_test

    # Create binary directory for model parameters
    binary_dir = f"generated_code/matquant_pt_mnist/uniform_{bit_width}bit/mq_pt_model_binary"
    os.makedirs(binary_dir, exist_ok=True)
    
    implementation = MatQuantPT(
        mlgen_model, 
        feature_type="float", 
        label_type="float",
        internal_type="float",
        target_bits=bit_width
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        test_samples=1000,
        filename=f"matquant_pt_{bit_width}bit"
    )
    
    # Now implement after the filename has been set
    implementation.implement()
    
    output_path = f"generated_code/matquant_pt_mnist/uniform_{bit_width}bit"
    os.makedirs(output_path, exist_ok=True)

    materializer.materialize(output_path)
    print(f"Model materialized at: {output_path}")
    materializer.deploy()
    print(f"{bit_width}-bit model deployed successfully (using 8-bit storage with runtime slicing).")
    results = materializer.run(verbose=True)
    print(f"{bit_width}-bit model results: {results}")
    
    return results

def generate_mix_model(mix_config, model_path=None):
    """Generate and deploy a mix-and-match model with 8-bit storage and runtime slicing"""
    print(f"\nGenerating C++ code for mix-and-match model with 8-bit storage and runtime slicing...")
    from mlgen3.implementations.neuralnet.cpp.matquant_pt import MatQuantPT
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Load the saved model
    if not model_path:
        model_path = "models/matquant/mnist_pt/mq_pt_model.pt"
    
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    
    # Create MatQuant wrapper for extraction
    mq_model_temp = MatQuant(model, config)
    mq_model_temp.set_quantized_layers(config['quantization']['quantize_layers'])
    
    mq_model_8bit = mq_model_temp.extract_model(8)
    mlgen_model = extract_mlgen3_model(mq_model_8bit)
    mlgen_model.XTest = X_test
    mlgen_model.YTest = y_test

    # Create configuration string for filename
    config_str = '_'.join([f"{layer.split('.')[-2]}{bits}" for layer, bits in mix_config.items()])
    
    # Create binary directory for model parameters
    binary_dir = f"generated_code/matquant_pt_mnist/mix_{config_str}/mq_pt_model_binary"
    os.makedirs(binary_dir, exist_ok=True)
    
    implementation = MatQuantPT(
        mlgen_model, 
        feature_type="float", 
        label_type="float",
        internal_type="float",
        mix_and_match_config=mix_config
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        filename=f"matquant_pt_mix_{config_str}"
    )
    
    # Now implement after the filename has been set
    implementation.implement()
    
    output_path = f"generated_code/matquant_pt_mnist/mix_{config_str}"
    os.makedirs(output_path, exist_ok=True)

    materializer.materialize(output_path)
    print(f"Model materialized at: {output_path}")
    materializer.deploy()
    print("Mix-and-match model deployed successfully (using 8-bit storage with runtime slicing).")
    results = materializer.run(verbose=True)
    print(f"Mix-and-match model results: {results}")
    
    return results

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generate MatQuant models with specified bit-widths using PyTorch binary loading')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
    parser.add_argument('--generate', action='store_true', help='Generate C++ code')
    parser.add_argument('--uniform', type=int, nargs='+', help='Uniform bit-width models to generate (e.g. 8 4 2)')
    parser.add_argument('--mix', action='store_true', help='Generate mix-and-match model')
    parser.add_argument('--custom-mix', type=str, help='Custom mix-and-match configuration in format "layer1:bits,layer2:bits"')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (overrides config)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.train:
        model, mq_model = train_model(args)
        extracted_models, mix_config = extract_and_test_models(mq_model)
    
    if args.generate:
        # Process the models based on arguments
        results = {}

        # Generate uniform bit-width models
        if args.uniform:
            for bit_width in args.uniform:
                results[f"uniform_{bit_width}bit"] = generate_uniform_model(bit_width)
        else:
            # Default to 8-bit if no uniform bit-widths specified
            results["uniform_8bit"] = generate_uniform_model(8)

        # Generate default mix-and-match model
        if args.mix:
            default_mix = {
                'model.0.weight': 8, 
                'model.3.weight': 4, 
                'model.6.weight': 2
            }
            results["default_mix"] = generate_mix_model(default_mix)

        # Generate custom mix-and-match model
        if args.custom_mix:
            try:
                custom_mix = {}
                for pair in args.custom_mix.split(','):
                    layer, bits = pair.split(':')
                    custom_mix[layer] = int(bits)
                results["custom_mix"] = generate_mix_model(custom_mix)
            except Exception as e:
                print(f"Error parsing custom mix-and-match configuration: {e}")
                print("Format should be: 'model.0.weight:8,model.3.weight:4,model.6.weight:2'")

        print("\nMatQuant PyTorch implementation complete!")
        print(f"Generated models: {list(results.keys())}")
