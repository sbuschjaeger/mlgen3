import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mlgen3.implementations.matquant.matquant import MatQuant
from mlgen3.utils import LayerRegistry

from mlgen3.utils import (
    get_dataset, create_model, load_config, set_seed, get_seed_from_config
)

# Create output directories
os.makedirs("generated_code/matquant_pt_mnist_c-bitplane", exist_ok=True)

def load_test_config(config_path='models-tuned/mlp/q864/config.yaml'):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        return load_config(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

def run_baseline_inference(mq_model, X_test, y_test, config):
    """Run baseline inference with the PyTorch model for each bit-width."""
    print("\nRunning baseline MatQuant inference for each bit-width...")
    
    # Extract and test models at each bit-width
    target_bits = config['quantization']['target_bits']
    baseline_results = {}
    
    for bits in target_bits:
        extracted_model = mq_model.extract_model(bits)
        extracted_model.eval()
        
        with torch.no_grad():
            outputs = extracted_model(X_test)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item() * 100
            baseline_results[bits] = accuracy
            print(f"  {bits}-bit accuracy: {accuracy:.2f}%")
    
    return baseline_results

def extract_mlgen3_model(pytorch_model, use_bias=True):
    """Extract model parameters from PyTorch to MLGen3 format."""
    from mlgen3.models.nn.neuralnet import NeuralNet
    from mlgen3.models.nn.linear import Linear
    from mlgen3.models.nn.batchnorm import BatchNorm
    from mlgen3.models.nn.activations import Relu
    
    layers = []
    
    # Dynamically extract layers based on actual model structure
    for i, module in enumerate(pytorch_model.model):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            # Handle bias - can be None if use_bias=False
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
            layers.append(Linear(weight, bias))
        elif isinstance(module, torch.nn.BatchNorm1d):
            # Extract fused BatchNorm parameters for inference
            gamma = module.weight.detach().cpu().numpy()
            beta = module.bias.detach().cpu().numpy()
            mean = module.running_mean.detach().cpu().numpy()
            var = module.running_var.detach().cpu().numpy()
            eps = module.eps
            layers.append(BatchNorm(gamma, beta, mean, var, eps))
        elif isinstance(module, torch.nn.ReLU):
            # Get input shape from previous layer
            if layers:
                prev_layer = layers[-1]
                input_shape = prev_layer.output_shape
            else:
                input_shape = pytorch_model.model[0].in_features
            layers.append(Relu(input_shape))
    
    mlgen_model = NeuralNet.from_layers(layers)
    return mlgen_model

def generate_c_model(model_path, config, seed=707):
    """Generate C code for the model."""
    print(f"\nGenerating C code for model: {model_path}")
    
    from mlgen3.implementations.matquant.matquant_pt_mlp_c_bitplane import MatQuantPT_C_Bitplane
    from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
    
    # Get use_bias from config (check both model and quantization sections)
    use_bias = config['model'].get('use_bias', config['quantization'].get('use_bias', True))
    use_batchnorm = config['model'].get('use_batchnorm', False)
    
    print(f"Model config: use_bias={use_bias}, use_batchnorm={use_batchnorm}")
    
    # Create model with current config
    model = create_model(config)

    # print(model)
    # exit(0)
    
    # Load state dict and handle key mismatches
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Debug: print checkpoint keys to understand structure
    print(f"Checkpoint keys : {list(checkpoint.keys())}")
    print(f"Model keys : {list(model.state_dict().keys())}")
    
    # Normalize checkpoint keys - remove extra 'model.' prefix if present
    normalized_checkpoint = {}
    for key, value in checkpoint.items():
        # Handle different checkpoint formats
        if key.startswith('model.model.'):
            # Checkpoint has 'model.model.X.weight' format, model expects 'model.X.weight'
            new_key = key[6:]  # Remove first 'model.'
        elif key.startswith('model.'):
            # Already in correct format
            new_key = key
        else:
            # No prefix, add 'model.'
            new_key = 'model.' + key
        normalized_checkpoint[new_key] = value
    
    print(f"Normalized checkpoint keys : {list(normalized_checkpoint.keys())}")
    print(f"Model state dict keys: {list(model.state_dict().keys())}")
    
    # Check if normalized keys match model keys directly
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(normalized_checkpoint.keys())
    
    if model_keys == checkpoint_keys:
        # Keys match exactly, use normalized checkpoint directly
        print("Checkpoint keys match model keys exactly, loading directly.")
        new_checkpoint = normalized_checkpoint
    else:
        # Keys don't match, need to figure out mapping
        print(f"Key mismatch detected. Model keys: {model_keys}, Checkpoint keys: {checkpoint_keys}")
        
        # Check for common keys
        common_keys = model_keys & checkpoint_keys
        print(f"Common keys: {common_keys}")
        
        # For mismatched keys, try to map based on layer structure
        new_checkpoint = {}
        for ckpt_key, ckpt_value in normalized_checkpoint.items():
            if ckpt_key in model_keys:
                # Direct match
                new_checkpoint[ckpt_key] = ckpt_value
            else:
                print(f"Warning: Checkpoint key '{ckpt_key}' not found in model, skipping.")
    
    print(f"Final checkpoint keys to load: {list(new_checkpoint.keys())}")
    
    # Load the state dict
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
        quantize_layers = layer_registry.get_layers_by_indices(None)
    
    mq_model_temp.set_quantization(quantize_layers, set())

    # Run baseline MatQuant inference for all bit-widths
    baseline_results = run_baseline_inference(mq_model_temp, X_test, y_test, config)
    print("\nBaseline inference completed.\n")
    
    # Extract model at highest bit-width (8-bit for storage)
    max_bits = max(config['quantization']['target_bits'])
    mq_model_8bit = mq_model_temp.extract_model(max_bits)
    
    mlgen_model = extract_mlgen3_model(mq_model_8bit, use_bias=use_bias)
    mlgen_model.XTest = X_test.cpu().numpy()
    mlgen_model.YTest = y_test.cpu().numpy()
    
    # Create binary directory for model parameters
    binary_dir = "generated_code/matquant_pt_mnist_c/mq_pt_model_binary"
    os.makedirs(binary_dir, exist_ok=True)
    
    # Create C implementation with use_bias parameter
    implementation = MatQuantPT_C_Bitplane(
        mlgen_model,
        feature_type="float",
        label_type="float",
        internal_type="float",
        quantize_signed=config['quantization'].get('quantize_signed', True),
        use_bias=use_bias,
        use_header_weights=True  # Use header weights for bitplane packing
    )
    
    implementation.set_filename("mq_pt_mlp_bitplane")
    
    # Create materializer
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 1000)
    
    materializer = LinuxStandalone(
        implementation,
        filename="mq_pt_mlp_bitplane",
        measure_accuracy=True,
        measure_time=True,
        compiler="gcc",
        test_samples=test_samples,
        seed=seed
    )
    
    # Materialize and deploy
    output_path = "generated_code/matquant_pt_mnist_c-bitplane"
    materializer.materialize(output_path)
    
    # Save the debug header with unpacked weights
    if hasattr(implementation, 'weights_debug_header') and implementation.weights_debug_header:
        debug_header_path = os.path.join(output_path, "mq_pt_mlp_bitplane_weights_debug.h")
        with open(debug_header_path, 'w') as f:
            f.write(implementation.weights_debug_header)
        print(f"Debug header saved to: {debug_header_path}")
    
    materializer.deploy()
    
    print(f"\nC code generated in: {output_path}")
    print(f"Use 'cd {output_path} && make && ./mq_pt_mlp_bitplane testing.csv 1' to run")
    
    return materializer, baseline_results


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate C code for MatQuant MLP model')
    parser.add_argument('--config', type=str, default='models-tuned/mlp/q8642-nobias/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=707, help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    config = load_test_config(args.config)
    
    # Get model path from config
    model_path = config['evaluation']['model_path']
    
    # Generate C code
    materializer, baseline_results = generate_c_model(model_path, config, args.seed)
    
    # Run the generated code
    print("\nRunning generated C code...")
    results = materializer.run(verbose=True, seed=args.seed)
    
    print("\n=== Results Comparison ===")
    print("PyTorch baseline results:")
    for bits, acc in baseline_results.items():
        print(f"  {bits}-bit: {acc:.2f}%")
    print(f"\nC implementation result: {results}")
