import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import argparse
import numpy as np

from mlgen3.implementations.matquant.matquant import MatQuant
from mlgen3.utils import LayerRegistry

from mlgen3.utils import (
    get_dataset, create_model, load_config, set_seed, get_seed_from_config
)

# Create output directories
os.makedirs("generated_code/matquant_pt_mnist_c", exist_ok=True)

def load_test_config(config_path='models-tuned/mlp/q864/config.yaml'):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        return load_config(config_path=config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

def run_baseline_inference(mq_model, X_test, y_test, config):
    """Run baseline inference with the PyTorch model for each bit-width."""
    print("\nRunning baseline MatQuant inference for each bit-width...")
    
    # # Create MatQuant wrapper
    # mq_model = MatQuant(model, config)
    
    # # Register layers
    # layer_registry = LayerRegistry()
    # quantize_bias = config['quantization'].get('quantize_bias', True)
    # all_layers = layer_registry.register_model(model, include_bias=quantize_bias)
    
    # # Set quantized layers
    # quantize_layers = config['quantization'].get('quantized_params_list', ['all'])
    # if quantize_layers == ["all"] or quantize_layers == "all":
    #     quantize_layers = all_layers
    
    # mq_model.set_quantization(quantize_layers, set())  # Empty set for activations in C impl
    
    # Extract and test models at each bit-width
    target_bits = config['quantization']['target_bits']
    baseline_results = {}
    
    for bits in target_bits:
        print(f"\nExtracting {bits}-bit model...")
        extracted_model = mq_model.extract_model(bits)
        extracted_model.eval()
        
        # Test on first 100 samples for quick verification
        with torch.no_grad():
            X_batch = X_test
            y_batch = y_test
            
            outputs = extracted_model(X_batch)
            _, predicted = torch.max(outputs, 1)
            
            correct = (predicted == y_batch).sum().item()
            accuracy = 100 * correct / len(y_batch)
            
            baseline_results[bits] = accuracy
            print(f"{bits}-bit PyTorch accuracy: {accuracy:.2f}%")
    
    return baseline_results

def extract_mlgen3_model(pytorch_model):
    """Extract model parameters from PyTorch to MLGen3 format."""
    from mlgen3.models.nn.neuralnet import NeuralNet
    from mlgen3.models.nn.linear import Linear
    from mlgen3.models.nn.batchnorm import BatchNorm
    from mlgen3.models.nn.activations import Relu
    
    layers = []
    
    # Dynamically extract layers based on actual model structure
    for i, module in enumerate(pytorch_model.model):
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            
            layer_name = f"model.{i}.weight"
            linear_layer = Linear(weight, bias)
            linear_layer.layer_name = layer_name
            layers.append(linear_layer)
            
        elif isinstance(module, nn.BatchNorm1d):
            scale = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            mean = module.running_mean.detach().cpu().numpy()
            var = module.running_var.detach().cpu().numpy()
            eps = module.eps
            layers.append(BatchNorm(scale, bias, mean, var, eps))
            
        elif isinstance(module, nn.ReLU):
            # Get output shape from previous layer
            if layers and isinstance(layers[-1], Linear):
                output_shape = layers[-1].output_shape
            else:
                # Fallback: try to infer from model
                output_shape = 512  # Default for hidden layers
            layers.append(Relu(output_shape))
    
    mlgen_model = NeuralNet.from_layers(layers)
    return mlgen_model

def generate_c_model(model_path, config, seed=707):
    """Generate C code for the model."""
    print(f"\nGenerating C code for model: {model_path}")
    
    from mlgen3.implementations.matquant.matquant_pt_mlp_c import MatQuantPT_C
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Create model with current config
    model = create_model(config)
    
    # Load state dict and handle key mismatches
    checkpoint = torch.load(model_path, map_location='cpu')

    # print(checkpoint)
    
    # Debug: print checkpoint keys to understand structure
    print(f"Checkpoint keys : {list(checkpoint.keys())}")
    print(f"Model keys : {list(model.state_dict().keys())}")
    
    # Normalize checkpoint keys - remove extra 'model.' prefix if present
    normalized_checkpoint = {}
    for key, value in checkpoint.items():
        # Remove 'model.model.' prefix and replace with 'model.'
        if key.startswith('model.model.'):
            new_key = key.replace('model.model.', 'model.', 1)
            normalized_checkpoint[new_key] = value
        else:
            normalized_checkpoint[key] = value
    
    print(f"Normalized checkpoint keys : {list(normalized_checkpoint.keys())}")
    
    # Count Linear layers in normalized checkpoint
    linear_layers_in_ckpt = []
    for key in normalized_checkpoint.keys():
        if 'weight' in key and 'running' not in key and 'num_batches' not in key:
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_num = int(parts[1])
                if layer_num not in linear_layers_in_ckpt:
                    linear_layers_in_ckpt.append(layer_num)
    
    linear_layers_in_ckpt = sorted(linear_layers_in_ckpt)
    print(f"Linear layers in checkpoint: {linear_layers_in_ckpt}")
    
    # Create a mapping from checkpoint linear layers to current model linear layers
    current_linear_idx = 0
    new_checkpoint = {}
    
    # Get use_bias flag from config
    use_bias = config['quantization'].get('use_bias', True)
    
    for ckpt_layer_idx in linear_layers_in_ckpt:
        ckpt_weight_key = f"model.{ckpt_layer_idx}.weight"
        ckpt_bias_key = f"model.{ckpt_layer_idx}.bias"
        
        if ckpt_weight_key in normalized_checkpoint:
            # Find the corresponding Linear layer in current model
            current_layer_idx = None
            linear_count = 0
            for i, module in enumerate(model.model):
                if isinstance(module, nn.Linear):
                    if linear_count == current_linear_idx:
                        current_layer_idx = i
                        break
                    linear_count += 1
            
            if current_layer_idx is not None:
                new_checkpoint[f"model.{current_layer_idx}.weight"] = normalized_checkpoint[ckpt_weight_key]
                
                # Only load bias if use_bias is True and bias exists in checkpoint
                if use_bias and ckpt_bias_key in normalized_checkpoint:
                    new_checkpoint[f"model.{current_layer_idx}.bias"] = normalized_checkpoint[ckpt_bias_key]
                
                current_linear_idx += 1
    
    print(f"Mapped checkpoint keys : {list(new_checkpoint.keys())}")
    
    # Load the mapped state dict (only Linear layers, strict=False to ignore missing BatchNorm)
    model.load_state_dict(new_checkpoint, strict=False)
    model.eval()
    
    # Get dataset for testing
    dataset_name = config['model']['dataset']
    X_train, y_train, X_test, y_test = get_dataset(dataset_name, as_tensors=True, flatten=True)
    

    # Extract to MLGen3 format using the highest bit-width model
    from mlgen3.implementations.matquant.matquant import MatQuant
    from mlgen3.utils import LayerRegistry
    
    # Create MatQuant wrapper for extraction
    mq_model_temp = MatQuant(model, config)
    
    # Register layers
    layer_registry = LayerRegistry(model, config)

    
    # Set quantized layers
    quantize_layers = config['quantization'].get('quantized_params_list', ['all'])
    if quantize_layers == ["all"] or quantize_layers == "all":
        quantize_layers = layer_registry.quantizable_params
    
    mq_model_temp.set_quantization(quantize_layers, set())

    # Run baseline MatQuant inference for all bit-widths
    baseline_results = run_baseline_inference(mq_model_temp, X_test, y_test, config)
    print("\nBaseline inference completed.\n")
    
    # Extract model at highest bit-width (8-bit for storage)
    max_bits = max(config['quantization']['target_bits'])
    mq_model_8bit = mq_model_temp.extract_model(max_bits)
    
    mlgen_model = extract_mlgen3_model(mq_model_8bit)
    mlgen_model.XTest = X_test.cpu().numpy()
    mlgen_model.YTest = y_test.cpu().numpy()
    
    # Create binary directory for model parameters
    binary_dir = "generated_code/matquant_pt_mnist_c/mq_pt_model_binary"
    os.makedirs(binary_dir, exist_ok=True)
    
    # Create C implementation with use_bias parameter
    implementation = MatQuantPT_C(
        mlgen_model,
        feature_type="float",
        label_type="float",
        internal_type="float",
        quantize_signed=config['quantization'].get('quantize_signed', True),
        use_bias=use_bias  # Pass use_bias to implementation
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 1000)
    
    materializer = LinuxStandalone(
        implementation,
        measure_accuracy=True,
        measure_time=True,
        test_samples=test_samples,
        filename="matquant_pt_c",
        seed=seed,
        compiler="gcc"  # Use gcc for C code
    )
    
    # Implement and materialize
    implementation.implement()
    
    output_path = "generated_code/matquant_pt_mnist_c"
    materializer.materialize(output_path)
    print(f"Model materialized at: {output_path}")
    
    materializer.deploy()
    print("C model deployed successfully.")
    
    results = materializer.run(verbose=True)
    print(f"\nC model results: {results}")
    
    return results, baseline_results

def parse_args():
    parser = argparse.ArgumentParser(description='Generate C code for MatQuant MLP model')
    parser.add_argument('--config', type=str, default='models-tuned/mlp/q864/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (overrides config)')
    parser.add_argument('--seed', type=int, default=707, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    config = load_test_config(args.config)
    
    # Set seed
    seed = get_seed_from_config(config) if args.seed is None else args.seed
    set_seed(seed)
    
    # Get model path
    model_path = args.model_path if args.model_path else config['evaluation']['model_path']
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using test_custom_mq_pt_mlp.py")
        sys.exit(1)
    
    # Generate C code and run
    results, baseline_results = generate_c_model(model_path, config, seed=seed)
    
    print("\n" + "="*80)
    print("Summary:")
    print("\nBaseline PyTorch MatQuant Accuracies:")
    for bits, acc in sorted(baseline_results.items(), reverse=True):
        print(f"  {bits}-bit: {acc:.2f}%")
    print(f"\nC Implementation Results: {results}")
    print("="*80)
