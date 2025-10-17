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
os.makedirs("models/matquant/cifar10_pt", exist_ok=True)
os.makedirs("generated_code/matquant_pt_vgg8", exist_ok=True)

def load_test_config(config_path=None):
    """Load configuration from YAML file or create default."""
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        return load_config(config_path=config_path)
    else:
        print("Using default configuration")
        from mlgen3.utils import create_default_config
        return create_default_config(
            model_name='vgg8',
            dataset_name='cifar10',
            target_bits=[8, 4, 2],
            num_epochs=1,
            batch_size=64
        )

# Load config
config = load_test_config('tests/matquant/configs/config_vgg8_mq842.yaml')

# Load dataset based on config
dataset_name = config['model']['dataset']
print(f"Loading {dataset_name} dataset...")
X_train, y_train, X_test, y_test = get_dataset(dataset_name, as_tensors=True)

def train_model(args):
    
    print("\nCreating VGG8 model...")
    model = create_model(config)
    
    # Register layers for MatQuant
    layer_registry = LayerRegistry()
    quantize_bias = config['quantization'].get('quantize_bias', False)
    all_layers = layer_registry.register_model(model, include_bias=quantize_bias)
    
    # Use quantize_layers from config
    quantize_layers = config['quantization'].get('quantize_layers', [])

    
    # Handle "all" keyword
    if quantize_layers == ["all"] or quantize_layers == "all":
        print("Quantizing all layers.")
        quantize_layers = all_layers
        config['quantization']['quantize_layers'] = quantize_layers
    elif not quantize_layers:
        print("No quantize_layers specified in config.")
        # Fallback to default
        quantize_layers = [
            "model.0.weight",
            "model.4.weight",
            "model.7.weight",
            "model.11.weight",
            "model.14.weight",
            "model.18.weight",
            "model.22.weight",
            "model.24.weight"
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
            'model.4.weight': 4,
            'model.7.weight': 8,
            'model.11.weight': 4,
            'model.14.weight': 2,
            'model.18.weight': 2,
            'model.22.weight': 8,
            'model.24.weight': 8
        }
    
    evaluator.test_mix_and_match(mix_config, X_test, y_test)
    
    return extracted_models, mix_config

def generate_cpp_model(bit_width, mix_config=None, model_path=None, seed=707, debug=False):
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
    from mlgen3.implementations.matquant.matquant_pt_vgg import MatQuantPT_VGG
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
        input_channels=3,
        quantize_signed=config['quantization'].get('quantize_signed', False)
    )
    
    implementation.set_model_binary_dir(binary_dir)
    
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 100)

    # Create materializer
    materializer = LinuxStandalone(
        implementation, 
        measure_accuracy=True, 
        measure_time=True,
        test_samples=test_samples,
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
    parser = argparse.ArgumentParser(description='Train and deploy MatQuant VGG8 model')
    parser.add_argument('--config', type=str, default='config_vgg8_mq842.yaml', help='Path to config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--generate', action='store_true', help='Generate C++ code')
    parser.add_argument('--uniform', type=int, nargs='+', default=None, help='Bit-widths')
    parser.add_argument('--mix', action='store_true', help='Generate mix-and-match model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--inference-seed', type=int, default=707, help='Inference seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Reload config if different path specified
    if args.config != 'config_vgg8_mq842.yaml':
        config = load_test_config(args.config)
        dataset_name = config['model']['dataset']
        X_train, y_train, X_test, y_test = get_dataset(dataset_name, as_tensors=True)
    
    seed = get_seed_from_config(config)
    if args.seed is not None:
        seed = args.seed
    set_seed(seed)
    
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    if args.train:
        model, mq_model = train_model(args)
        # Ensure mq_model's model is on CPU for extract_and_test_models
        if torch.cuda.is_available():
            mq_model.model = mq_model.model.cpu()
        extracted_models, mix_config = extract_and_test_models(mq_model)
    
    if args.generate:
        results = {}
        inference_seed = args.inference_seed
        
        # Use bit-widths from args or config
        target_bits = args.uniform if args.uniform else config['quantization']['target_bits']
        
        # Generate uniform bit-width models
        for bit_width in target_bits:
            results[f"uniform_{bit_width}bit"] = generate_cpp_model(bit_width, seed=inference_seed)
        
        # Generate mix-and-match model
        if args.mix:
            eval_config = config.get('evaluation', {})
            mix_configs = eval_config.get('mix_and_match_configs', [])
            
            if mix_configs:
                default_mix = mix_configs[0]['config']
            else:
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
