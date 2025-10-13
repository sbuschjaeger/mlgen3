import os
import sys
# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse

from Datasets import get_dataset
from matquant import MatQuant
from mlgen3.utils.seed import set_seed, get_seed_from_config

# Create output directories
os.makedirs("models/matquant/cifar10_pt", exist_ok=True)
os.makedirs("generated_code/matquant_pt_vgg8", exist_ok=True)

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
X_train, y_train, X_test, y_test = get_dataset("cifar10")
X_train = X_train.reshape(-1, 3, 32, 32).astype('float32') / 255.0
X_test = X_test.reshape(-1, 3, 32, 32).astype('float32') / 255.0

# Convert to PyTorch tensors
train_x = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)
test_x = torch.tensor(X_test, dtype=torch.float32)
test_y = torch.tensor(y_test, dtype=torch.long)

# Define VGG8 network architecture (same model as in qnn matquant framework)
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=8192, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10, bias=True)
        )

        print(self.model)
        
    def forward(self, x):
        return self.model(x)

# Create LayerRegistry class to register layers for MatQuant
class LayerRegistry:
    def __init__(self):
        self.layer_paths = []
    
    def register_model(self, model):
        """Register all quantizable layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_name = f"{name}.weight"
                if weight_name not in self.layer_paths:
                    self.layer_paths.append(weight_name)
        return self.layer_paths

# Create MatQuant config based on config_vgg8_mq842.yaml
config = {
    'quantization': {
        'use_matquant': True,
        'use_codistillation': False,
        'use_qat': False,
        'fx_mode': False,
        'target_bits': [8, 4, 2],
        'loss_weights': {8: 0.4, 4: 0.8, 2: 0.8},
        'quantize_bias': True,
        'quantize_target': 'weights_only', # 'weights_and_activations' or 'weights_only'
        'quantize_layers': [
            # Will be filled by layer_registry
        ]
    },
    'model': {
        'name': 'vgg',
        'dataset': 'cifar10',
        'num_channels': 3,
        'num_blocks': 3,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'input_size': 32,
        'num_classes': 10
    },
    'training': {
        'model_dir': './models/matquant/cifar10_pt',
        'model_savename': 'mq_pt_vgg8_model',
        'optimizer': 'sgd',
        'batch_size': 64,
        'num_epochs': 1,
        'learning_rate': 0.01,
        'lr_scheduler': 'step',
        'gamma': 0.1,
        'step_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'seed': 707  # Random seed for reproducibility
    },
    'evaluation': {
        'model_path': './models/matquant/cifar10_pt/mq_pt_vgg8_model.pt',
        'batch_size': 128,
        'num_iterations': 1,
    }
}

def train_model(args):
    
    print("\nCreating VGG8 model...")
    model = VGG()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Register layers for MatQuant
    layer_registry = LayerRegistry()
    all_layers = layer_registry.register_model(model)
    
    # Set quantize_layers in config
    # For VGG8, quantizing the 6 conv layers and 2 linear layers
    # TODO add batchnorm layers too
    config['quantization']['quantize_layers'] = [
        "model.0.weight",   # Conv2d(3, 128)
        "model.4.weight",   # Conv2d(128, 128)
        "model.7.weight",   # Conv2d(128, 256)
        "model.11.weight",  # Conv2d(256, 256)
        "model.14.weight",  # Conv2d(256, 512)
        "model.18.weight",  # Conv2d(512, 512)
        "model.22.weight",  # Linear(8192, 1024)
        "model.24.weight"   # Linear(1024, 10)
    ]
    
    print(f"Registered layers for quantization: {config['quantization']['quantize_layers']}")
    
    # Initialize MatQuant wrapper
    print("\nInitializing MatQuant wrapper...")
    mq_model = MatQuant(model, config)
    mq_model.set_quantized_layers(config['quantization']['quantize_layers'])
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'], 
                          weight_decay=config['training']['weight_decay'])
    scheduler = StepLR(optimizer, 
                       step_size=config['training']['step_size'], 
                       gamma=config['training']['gamma'])
    batch_size = config['training']['batch_size']
    epochs = args.epochs if args.epochs else config['training']['num_epochs']
    
    # Training loop
    print(f"\nTraining VGG8 with MatQuant for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = train_x[i:i+batch_size].to(device)
            targets = train_y[i:i+batch_size].to(device)
            
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
                
        # Evaluate
        model.eval()
        mq_model.eval()
        with torch.no_grad():
            # Test for each bit-width
            accuracies = {}
            for bits in config['quantization']['target_bits']:
                correct = 0
                total = 0
                
                for i in tqdm(range(0, len(X_test), batch_size), desc=f"Testing {bits}-bit"):
                    inputs = test_x[i:i+batch_size].to(device)
                    targets = test_y[i:i+batch_size].to(device)
                    
                    outputs = mq_model.forward_with_quant(inputs, bits)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                accuracy = 100 * correct / total
                accuracies[bits] = accuracy
        
        for bits, acc in accuracies.items():
            print(f"  {bits}-bit Accuracy: {acc:.2f}%")

        print("")
    

    # Save the trained model parameters
    model_path = config['evaluation']['model_path']
    # Move model to CPU before saving to ensure compatibility
    model = model.cpu()
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved MatQuant model to {model_path}")
    return model, mq_model

def extract_and_test_models(mq_model):
    batch_size = config['evaluation']['batch_size']
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract models with different bit-widths
    print("\nExtracting models with different bit-widths...")
    extracted_models = {}
    for bits in config['quantization']['target_bits']:
        extracted_models[bits] = mq_model.extract_model(bits).to(device)
    
    # Create mix-and-match model
    # TODO add batchnorm layers too
    mix_config = {
        'model.0.weight': 8,
        'model.4.weight': 4,
        'model.7.weight': 8,
        'model.11.weight': 4,
        'model.14.weight': 2,
        'model.18.weight': 2,
        'model.22.weight': 8,
        'model.24.weight': 8
    }
    mix_model = mq_model.mix_and_match(mix_config).to(device)
    
    print("\nTesting extracted and mix-and-match models...")
    # Test extracted models
    for bits, ext_model in extracted_models.items():
        ext_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(0, len(X_test), batch_size):
                inputs = test_x[i:i+batch_size].to(device)
                targets = test_y[i:i+batch_size].to(device)
                
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
            inputs = test_x[i:i+batch_size].to(device)
            targets = test_y[i:i+batch_size].to(device)
            
            outputs = mix_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Mix-and-match model accuracy: {accuracy:.2f}%")
    
    print("")
    # Move models back to CPU before returning
    for bits in extracted_models:
        extracted_models[bits] = extracted_models[bits].cpu()
    mix_model = mix_model.cpu()
    
    return extracted_models, mix_config

def generate_cpp_model(bit_width, mix_config=None, model_path=None, seed=707):
    """Generate C++ code for MatQuant PyTorch VGG8 model"""
    if not model_path:
        model_path = config['evaluation']['model_path']
        
    print(f"\nGenerating C++ code for {'mix-and-match' if mix_config else bit_width}-bit VGG8 model...")
    
    # Load the saved model state dict
    saved_model_state = torch.load(model_path)
    
    # Determine configuration name
    if mix_config:
        config_name = "mix_" + "_".join([f"{k.split('.')[-2]}{v}" for k, v in mix_config.items()])
    else:
        config_name = f"uniform_{bit_width}bit"
    
    # Create binary directory for model parameters
    binary_dir = f"generated_code/matquant_pt_vgg8/{config_name}/mq_pt_model_binary"
    os.makedirs(binary_dir, exist_ok=True)
    
    # Generate MatQuant PyTorch VGG implementation
    from mlgen3.implementations.neuralnet.cpp.matquant_pt_vgg import MatQuantPT_VGG
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Create a simple class to hold necessary properties for the implementation
    class SimpleModelWrapper:
        def __init__(self, state_dict):
            self.state_dict = state_dict
            self.XTest = X_test
            self.YTest = y_test
            
            # Create a model with the same structure as the one that generated the state dict
            # This is needed for the MatQuantPT_VGG.analyze_model() function
            # TODO just copy from existing model defined in VGG class above
            self.model = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=8192, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=10, bias=True)
            )
    
    # Create a simple model wrapper with the saved state dict
    simple_model = SimpleModelWrapper(saved_model_state)
    
    # Set quantization parameters
    simple_model.quantization_config = bit_width
    simple_model.mix_and_match_config = mix_config
    
    implementation = MatQuantPT_VGG(
        simple_model, 
        feature_type="float", 
        label_type="float",
        internal_type="float",
        target_bits=bit_width if not mix_config else 8,
        mix_and_match_config=mix_config,
        input_height=32,
        input_width=32,
        input_channels=3
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        test_samples=1000,
        filename=f"matquant_pt_vgg8_{config_name}",
        seed=seed
    )
    
    output_path = f"generated_code/matquant_pt_vgg8/{config_name}"
    os.makedirs(output_path, exist_ok=True)
    
    materializer.materialize(output_path)
    print(f"\nModel materialized at: {output_path}")

    materializer.deploy()
    print(f"\nModel deployed successfully.")

    results = materializer.run(verbose=True)
    print(f"\nModel results: {results}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Train and deploy MatQuant VGG8 model on CIFAR-10')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--generate', action='store_true', help='Generate C++ code')
    parser.add_argument('--uniform', type=int, nargs='+', default=[8], help='Bit-widths for uniform quantization models')
    parser.add_argument('--mix', action='store_true', help='Generate mix-and-match model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--inference-seed', type=int, default=707, help='Random seed for C++ inference (default: 707)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set random seed for reproducibility
    seed = get_seed_from_config(config)
    if args.seed is not None:
        seed = args.seed
    set_seed(seed)
    
    if args.train:
        model, mq_model = train_model(args)
        # Ensure mq_model's model is on CPU for extract_and_test_models
        if torch.cuda.is_available():
            mq_model.model = mq_model.model.cpu()
        extracted_models, mix_config = extract_and_test_models(mq_model)
    
    if args.generate:
        results = {}
        inference_seed = args.inference_seed
        
        # Generate uniform bit-width models
        for bit_width in args.uniform:
            results[f"uniform_{bit_width}bit"] = generate_cpp_model(bit_width, seed=inference_seed)
        
        # Generate mix-and-match model
        if args.mix:
            # TODO add batchnorm layers too
            default_mix = {
                'model.0.weight': 8,
                'model.4.weight': 4,
                'model.7.weight': 8,
                'model.11.weight': 4,
                'model.14.weight': 2,
                'model.18.weight': 2,
                'model.22.weight': 8,
                'model.24.weight': 8
            }
            results["mix_and_match"] = generate_cpp_model(8, default_mix, seed=inference_seed)
        
        print("\nGeneration complete. Results summary:")
        for model_name, res in results.items():
            print(f"{model_name}: {res}")
