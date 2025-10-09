import random
import numpy as np
import torch
import os


def set_seed(seed=707):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value (default: 707)
    """
    # Python built-in random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed (for Python 3.3+)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


def get_seed_from_config(config, default_seed=707):
    """
    Extract seed from config dictionary.
    
    Args:
        config (dict): Configuration dictionary
        default_seed (int): Default seed if not found in config
        
    Returns:
        int: Seed value
    """
    # Check multiple possible locations in config
    if 'training' in config and 'seed' in config['training']:
        return config['training']['seed']
    elif 'seed' in config:
        return config['seed']
    else:
        return default_seed
