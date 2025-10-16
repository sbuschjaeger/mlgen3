"""Configuration parsing utilities."""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from .check_config import check_config


def load_config(config_path: Optional[str] = None, 
                config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from file or dictionary.
    
    Args:
        config_path: Path to YAML config file
        config_dict: Configuration dictionary (takes precedence over file)
    
    Returns:
        Configuration dictionary (validated)
    """
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Either config_path or config_dict must be provided")
    
    # Validate the configuration
    check_config(config)
    
    return config


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return config.get('training', {})


def get_quantization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quantization configuration."""
    return config.get('quantization', {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration."""
    return config.get('model', {})


def get_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract evaluation configuration."""
    return config.get('evaluation', {})


def create_default_config(
    model_name: str,
    dataset_name: str,
    target_bits: list = [8, 4, 2],
    num_epochs: int = 1,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Args:
        model_name: Name of the model ('mlp', 'vgg4', 'vgg8')
        dataset_name: Name of the dataset ('mnist', 'fashion', 'cifar10')
        target_bits: List of target bit-widths
        num_epochs: Number of training epochs
        batch_size: Training batch size
    
    Returns:
        Configuration dictionary (validated)
    """
    # Dataset-specific parameters
    dataset_params = {
        'mnist': {'num_channels': 1, 'input_size': 28, 'num_classes': 10},
        'fashion': {'num_channels': 1, 'input_size': 28, 'num_classes': 10},
        'cifar10': {'num_channels': 3, 'input_size': 32, 'num_classes': 10},
    }
    
    params = dataset_params.get(dataset_name, dataset_params['mnist'])
    
    config = {
        'quantization': {
            'use_matquant': True,
            'use_codistillation': False,
            'use_qat': False,
            'fx_mode': False,
            'target_bits': target_bits,
            'loss_weights': {8: 0.4, 4: 0.4, 2: 0.2},
            'quantize_bias': True,
            'quantize_target': 'weights_and_activations',
            'quantize_signed': True,
            'quantize_layers': []  # Will be filled by layer registry
        },
        'model': {
            'name': model_name,
            'dataset': dataset_name,
            'num_channels': params['num_channels'],
            'input_size': params['input_size'],
            'num_classes': params['num_classes']
        },
        'training': {
            'model_dir': f'./models/matquant/{dataset_name}_pt',
            'model_savename': f'mq_pt_{model_name}_model',
            'optimizer': 'sgd',
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 0.01,
            'lr_scheduler': 'step',
            'gamma': 0.1,
            'step_size': 5,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'seed': 707
        },
        'evaluation': {
            'model_path': f'./models/matquant/{dataset_name}_pt/mq_pt_{model_name}_model.pt',
            'batch_size': 128,
            'num_iterations': 1,
        }
    }
    
    # Validate before returning
    check_config(config)
    
    return config
