import os
import sys
import torch
import numpy as np
import struct
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mlgen3.implementations.matquant.matquant import MatQuant
from mlgen3.utils import LayerRegistry
from mlgen3.utils.bitplane import pack_weights_bitplane, pack_weights_2d_bitplane

from mlgen3.utils import (
    get_dataset, create_model, load_config, set_seed, get_seed_from_config
)

# Create output directories
os.makedirs("generated_code/matquant_pt_mnist_c", exist_ok=True)

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
    
    from mlgen3.implementations.matquant.matquant_pt_mlp_c import MatQuantPT_C
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
    implementation = MatQuantPT_C(
        mlgen_model,
        feature_type="float",
        label_type="float",
        internal_type="float",
        quantize_signed=config['quantization'].get('quantize_signed', True),
        use_bias=use_bias,
        use_header_weights=False
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    # Create materializer
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 1000)
    
    materializer = LinuxStandalone(
        implementation,
        filename="mq_pt_mlp",
        measure_accuracy=True,
        measure_time=True,
        compiler="gcc",
        test_samples=test_samples,
        seed=seed
    )
    
    # Materialize and deploy
    output_path = "generated_code/matquant_pt_mnist_c"
    materializer.materialize(output_path)
    materializer.deploy()
    
    print(f"\nC code generated in: {output_path}")
    print(f"Use 'cd {output_path} && make && ./mq_pt_mlp testing.csv 1' to run")
    
    return materializer, baseline_results


def generate_bitplane_headers_from_binary(binary_dir, output_dir=None, signed=True, use_2d_arrays=False, layer_shapes=None):
    """
    Generate C header files with bitplane-packed weights from binary files.
    
    Args:
        binary_dir: Directory containing the .bin weight files
        output_dir: Output directory for header files (defaults to binary_dir)
        signed: Whether weights are signed int8
        use_2d_arrays: If True, format weight arrays as 2D arrays with layer dimensions
        layer_shapes: Dict mapping layer index to (input_size, output_size) tuples.
                      If None and use_2d_arrays=True, will try to infer from qparams files.
    """
    if output_dir is None:
        output_dir = binary_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    int_type = "int8_t" if signed else "uint8_t"
    
    # Find all weight/bias binary files
    weight_files = glob.glob(os.path.join(binary_dir, "layer_*_weight.bin"))
    bias_files = glob.glob(os.path.join(binary_dir, "layer_*_bias.bin"))
    scale_files = glob.glob(os.path.join(binary_dir, "layer_*_scale.bin"))
    
    all_bin_files = weight_files + bias_files + scale_files
    
    if not all_bin_files:
        print(f"No binary files found in {binary_dir}")
        return
    
    print(f"\nGenerating bitplane-packed headers from {len(all_bin_files)} binary files...")
    if use_2d_arrays:
        print("Using 2D array format for weight matrices.")
    
    for bin_file in sorted(all_bin_files):
        base_name = os.path.basename(bin_file).replace('.bin', '')
        qparams_file = bin_file.replace('.bin', '_qparams.bin')
        
        # Read binary weights
        dtype = np.int8 if signed else np.uint8
        weights = np.fromfile(bin_file, dtype=dtype)
        
        # Read qparams if available
        scale, zero_point = 1.0, 0.0
        if os.path.exists(qparams_file):
            with open(qparams_file, 'rb') as f:
                scale, zero_point = struct.unpack('ff', f.read(8))
        
        # Determine if this is a 2D weight matrix
        is_weight_matrix = 'weight' in base_name and 'lin' in base_name
        
        # Extract layer index from filename (e.g., "layer_0_lin_weight" -> 0)
        layer_idx = None
        import re
        match = re.search(r'layer_(\d+)', base_name)
        if match:
            layer_idx = int(match.group(1))
        
        # Determine shape for 2D arrays
        rows, cols = None, None
        if use_2d_arrays and is_weight_matrix and layer_shapes and layer_idx is not None:
            if layer_idx in layer_shapes:
                input_size, output_size = layer_shapes[layer_idx]
                rows, cols = output_size, input_size  # Weight matrix is (output, input)
        
        # Generate header for this layer
        header_name = f"{base_name}_packed.h"
        header_path = os.path.join(output_dir, header_name)
        
        header_guard = base_name.upper() + "_PACKED_H"
        
        header_content = f"""/* Auto-generated bitplane-packed weights for {base_name} */
#ifndef {header_guard}
#define {header_guard}

#include <stdint.h>

/* Original size: {len(weights)} bytes */
/* Scale: {scale:.10f}, Zero Point: {zero_point:.10f} */
"""
        
        if use_2d_arrays and is_weight_matrix and rows is not None and cols is not None:
            header_content += f"/* Shape: [{rows}][{cols}] (output_size x input_size) */\n\n"
            
            # Reshape weights to 2D and pack row by row
            weights_2d = weights.reshape(rows, cols)
            packed_weights_2d = pack_weights_2d_bitplane(weights_2d, signed=signed)
            
            # Format as 2D C array
            header_content += f"static const {int_type} {base_name}_packed[{rows}][{cols}] = {{\n"
            
            items_per_line = 16
            row_strs = []
            for row_idx in range(rows):
                row = packed_weights_2d[row_idx]
                row_parts = []
                for i in range(0, len(row), items_per_line):
                    chunk = row[i:min(i + items_per_line, len(row))]
                    chunk_str = ", ".join(str(int(v)) for v in chunk)
                    row_parts.append(chunk_str)
                row_str = ", ".join(row_parts)
                row_strs.append(f"    {{{row_str}}}")
            
            header_content += ",\n".join(row_strs)
            header_content += "\n};\n\n"
        else:
            header_content += "\n"
            # Pack as 1D array
            packed_weights = pack_weights_bitplane(weights, signed=signed)
            
            # Format as 1D C array
            header_content += f"static const {int_type} {base_name}_packed[{len(packed_weights)}] = {{\n"
            
            items_per_line = 16
            lines = []
            for i in range(0, len(packed_weights), items_per_line):
                chunk = packed_weights[i:min(i + items_per_line, len(packed_weights))]
                chunk_str = ", ".join(str(int(v)) for v in chunk)
                lines.append(f"    {chunk_str}")
            header_content += ",\n".join(lines)
            header_content += "\n};\n\n"
        
        # Add qparams
        header_content += f"static const float {base_name}_scale = {scale:.10f}f;\n"
        header_content += f"static const float {base_name}_zero_point = {zero_point:.10f}f;\n\n"
        
        header_content += f"#endif /* {header_guard} */\n"
        
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        if use_2d_arrays and is_weight_matrix and rows is not None:
            print(f"  Generated: {header_path} (shape [{rows}][{cols}])")
        else:
            print(f"  Generated: {header_path} ({len(weights)} bytes)")
    
    # Also generate a combined header that includes all individual headers
    combined_header_path = os.path.join(output_dir, "all_layers_packed.h")
    combined_content = """/* Auto-generated combined header for all bitplane-packed layers */
#ifndef ALL_LAYERS_PACKED_H
#define ALL_LAYERS_PACKED_H

"""
    for bin_file in sorted(all_bin_files):
        base_name = os.path.basename(bin_file).replace('.bin', '')
        combined_content += f'#include "{base_name}_packed.h"\n'
    
    combined_content += "\n#endif /* ALL_LAYERS_PACKED_H */\n"
    
    with open(combined_header_path, 'w') as f:
        f.write(combined_content)
    
    print(f"  Generated combined header: {combined_header_path}")
    print(f"\nBitplane header generation complete.")


def extract_layer_shapes_from_config(config):
    """
    Extract layer shapes from model configuration.
    
    Returns:
        Dict mapping layer index to (input_size, output_size) tuples
    """
    layer_shapes = {}
    
    model_config = config.get('model', {})
    hidden_layers = model_config.get('hidden_layers', [])
    input_size = model_config.get('input_size', 28)
    output_size = model_config.get('output_size', 10)
    
    # For MNIST-like datasets, input is typically flattened
    dataset = model_config.get('dataset', '')
    if dataset.lower() == 'mnist':
        input_dim = input_size * input_size  # 28*28 = 784
    else:
        input_dim = input_size
    
    # Build layer shapes
    # Layer indices depend on model structure (Linear, ReLU, etc.)
    # For MLP without batchnorm: Linear(0), ReLU(1), Linear(2), ReLU(3), Linear(4)
    prev_size = input_dim
    linear_layer_idx = 0
    
    for i, layer_cfg in enumerate(hidden_layers):
        layer_size = layer_cfg.get('size', layer_cfg) if isinstance(layer_cfg, dict) else layer_cfg
        # Linear layers are at indices 0, 2, 4, ... (interleaved with ReLU)
        layer_shapes[linear_layer_idx] = (prev_size, layer_size)
        prev_size = layer_size
        linear_layer_idx += 2  # Skip ReLU layer
    
    # Output layer
    layer_shapes[linear_layer_idx] = (prev_size, output_size)
    
    return layer_shapes


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate C code for MatQuant MLP model')
    parser.add_argument('--config', type=str, default='models-tuned/mlp/q8642-nobias/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=707, help='Random seed')
    parser.add_argument('--generate-headers', action='store_true',
                        help='Generate bitplane-packed header files from binary weights')
    parser.add_argument('--binary-dir', type=str, default=None,
                        help='Directory containing binary weight files (default: generated_code/matquant_pt_mnist_c/mq_pt_model_binary)')
    parser.add_argument('--header-output-dir', type=str, default=None,
                        help='Output directory for generated headers (default: same as binary-dir)')
    parser.add_argument('--unsigned', action='store_true',
                        help='Treat weights as unsigned (default: signed)')
    parser.add_argument('--2d-arrays', dest='use_2d_arrays', action='store_true',
                        help='Generate weight arrays as 2D arrays with layer dimensions')
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
    
    # Generate bitplane-packed headers if requested
    if args.generate_headers:
        binary_dir = args.binary_dir or "generated_code/matquant_pt_mnist_c/mq_pt_model_binary"
        output_dir = args.header_output_dir or binary_dir
        signed = not args.unsigned
        
        # Get signed setting from config if available
        if 'quantization' in config:
            signed = config['quantization'].get('quantize_signed', True)
        
        # Extract layer shapes from config for 2D array formatting
        layer_shapes = None
        if args.use_2d_arrays:
            layer_shapes = extract_layer_shapes_from_config(config)
            print(f"Layer shapes extracted from config: {layer_shapes}")
        
        generate_bitplane_headers_from_binary(
            binary_dir, output_dir, signed=signed,
            use_2d_arrays=args.use_2d_arrays, layer_shapes=layer_shapes
        )
