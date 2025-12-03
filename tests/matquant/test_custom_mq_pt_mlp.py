import os
import sys
# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import argparse

from mlgen3.implementations.matquant.matquant import MatQuant
from mlgen3.utils import (
    get_dataset, create_model, LayerRegistry,
    load_config, ModelTrainer, ModelEvaluator,
    set_seed, get_seed_from_config
)

# Create output directories
os.makedirs("models/matquant/mnist_pt", exist_ok=True)
os.makedirs("generated_code/matquant_pt_mnist", exist_ok=True)

def load_test_config(config_path=None):
    """Load configuration from YAML file or create default."""
    if config_path and os.path.exists(config_path):
        return load_config(config_path=config_path)
    else:
        from mlgen3.utils import create_default_config
        return create_default_config(
            model_name='mlp',
            dataset_name='mnist',
            target_bits=[8, 4, 2],
            num_epochs=1,
            batch_size=64
        )

# Load config
config = load_test_config('tests/matquant/configs/config_mlp_mq842.yaml')

# Load dataset based on config
dataset_name = config['model']['dataset']
print(f"Loading {dataset_name} dataset...")
X_train, y_train, X_test, y_test = get_dataset(dataset_name, as_tensors=True, flatten=True)  # Added flatten=True

def train_model(args):
    
    # Create model
    print("\nCreating MLP model...")
    model = create_model(config)
    
    # Register layers
    layer_registry = LayerRegistry()
    quantize_bias = config['quantization'].get('quantize_bias', False)
    print(quantize_bias)
    all_layers = layer_registry.register_model(model, include_bias=quantize_bias)
    
    # Use quantize_layers from config
    quantize_layers = config['quantization'].get('quantize_layers', [])
    
    # Handle "all" keyword
    if quantize_layers == ["all"] or quantize_layers == "all":
        quantize_layers = all_layers
        config['quantization']['quantize_layers'] = quantize_layers
    elif not quantize_layers:
        # Fallback to default
        quantize_layers = [
            'model.0.weight',
            'model.3.weight',
            'model.6.weight'
        ]
        config['quantization']['quantize_layers'] = quantize_layers
    
    print(f"Quantizing layers: {quantize_layers}")
    
    # Initialize MatQuant
    mq_model = MatQuant(model, config)
    mq_model.set_quantized_layers(config['quantization']['quantize_layers'])
    
    # Train
    trainer = ModelTrainer(model, mq_model, config)
    return trainer.train(X_train, y_train, X_test, y_test, num_epochs=args.epochs)

def extract_and_test_models(mq_model):
    evaluator = ModelEvaluator(mq_model, config)
    extracted_models = evaluator.extract_and_test_models(X_test, y_test)
    
    # Test mix-and-match from config or use default
    eval_config = config.get('evaluation', {})
    mix_configs = eval_config.get('mix_and_match_configs', [])
    
    if mix_configs:
        # Use first mix config from YAML
        mix_config = mix_configs[0]['config']
    else:
        # Fallback to default
        mix_config = {
            'model.0.weight': 8,
            'model.3.weight': 4,
            'model.6.weight': 2
        }
    
    evaluator.test_mix_and_match(mix_config, X_test, y_test)
    
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
            
            layer_name = f"model.{i}.weight"
            linear_layer = Linear(weight, bias)
            linear_layer.layer_name = layer_name
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
    
    mlgen_model = NeuralNet.from_layers(layers)
    
    # Copy quantization parameters if available
    if hasattr(pytorch_model, 'quantization_params'):
        mlgen_model.quantization_params = pytorch_model.quantization_params
    
    return mlgen_model

def generate_uniform_model(bit_width, model_path=None, seed=707):
    """Generate and deploy a uniform bit-width model with 8-bit storage and runtime slicing"""
    print(f"\nGenerating C++ code for {bit_width}-bit uniform model with 8-bit storage and runtime slicing...")
    from mlgen3.implementations.matquant.matquant_pt_mlp import MatQuantPT
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Load the saved model
    if not model_path:
        model_path = config['evaluation']['model_path']
    
    # Create model and load state
    model = create_model(config)
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
        target_bits=bit_width,
        quantize_signed=config['quantization'].get('quantize_signed', False)
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 1000)

    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        test_samples=test_samples,
        filename=f"matquant_pt_{bit_width}bit",
        seed=seed
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

def generate_mix_model(mix_config, model_path=None, seed=707):
    """Generate and deploy a mix-and-match model with 8-bit storage and runtime slicing"""
    print(f"\nGenerating C++ code for mix-and-match model with 8-bit storage and runtime slicing...")
    from mlgen3.implementations.matquant.matquant_pt_mlp import MatQuantPT
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Load the saved model
    if not model_path:
        model_path = config['evaluation']['model_path']
    
    # Create model and load state
    model = create_model(config)
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
        mix_and_match_config=mix_config,
        quantize_signed=config['quantization'].get('quantize_signed', False)
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        filename=f"matquant_pt_mix_{config_str}",
        seed=seed
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
    parser = argparse.ArgumentParser(description='Train and deploy MatQuant MLP model on MNIST')
    parser.add_argument('--config', type=str, default='config_mlp_mq_842.yaml', help='Path to config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--generate', action='store_true', help='Generate C++ code')
    parser.add_argument('--uniform', type=int, nargs='+', default=None, help='Uniform bit-width models (overrides config)')
    parser.add_argument('--mix', action='store_true', help='Generate mix-and-match model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--inference-seed', type=int, default=707, help='Inference seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Reload config if different path specified
    if args.config != 'config_mlp_mq_842.yaml':
        config = load_test_config(args.config)
        dataset_name = config['model']['dataset']
        X_train, y_train, X_test, y_test = get_dataset(dataset_name, as_tensors=True, flatten=True)  # Added flatten=True
    
    seed = get_seed_from_config(config)
    if args.seed is not None:
        seed = args.seed
    set_seed(seed)
    
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    if args.train:
        model, mq_model = train_model(args)
        extracted_models, mix_config = extract_and_test_models(mq_model)
    
    if args.generate:
        results = {}
        inference_seed = args.inference_seed
        
        # Use bit-widths from args or config
        target_bits = args.uniform if args.uniform else config['quantization']['target_bits']
        
        # Generate uniform bit-width models
        for bit_width in target_bits:
            results[f"uniform_{bit_width}bit"] = generate_uniform_model(bit_width, seed=inference_seed)
        
        # Generate mix-and-match model
        if args.mix:
            eval_config = config.get('evaluation', {})
            mix_configs = eval_config.get('mix_and_match_configs', [])
            
            if mix_configs:
                default_mix = mix_configs[0]['config']
            else:
                default_mix = {
                    'model.0.weight': 8,
                    'model.3.weight': 4,
                    'model.6.weight': 2
                }
            
            results["mix_and_match"] = generate_mix_model(default_mix, seed=inference_seed)
        
        print("\nGeneration complete. Results summary:")
        for model_name, res in results.items():
            print(f"{model_name}: {res}")

