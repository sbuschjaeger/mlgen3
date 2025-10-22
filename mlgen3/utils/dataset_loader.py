"""Dataset loading utilities for MLGen3."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os

# Import from existing Datasets module
try:
    # Add tests directory to path
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tests')
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    
    from Datasets import get_dataset as _get_dataset_raw
except ImportError as e:
    print(f"Warning: Datasets module not found: {e}")
    _get_dataset_raw = None


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    name: str
    input_channels: int
    input_height: int
    input_width: int
    num_classes: int
    normalize: bool = True
    
    @classmethod
    def from_name(cls, name: str) -> 'DatasetConfig':
        """Create dataset config from dataset name."""
        configs = {
            'mnist': cls('mnist', 1, 28, 28, 10),
            'fashion': cls('fashion', 1, 28, 28, 10),
            'cifar10': cls('cifar10', 3, 32, 32, 10),
        }
        
        if name not in configs:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(configs.keys())}")
        
        return configs[name]


def get_dataset(
    dataset_name: str,
    as_tensors: bool = True,
    normalize: bool = True,
    flatten: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess dataset.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion', 'cifar10')
        as_tensors: If True, return PyTorch tensors instead of numpy arrays
        normalize: If True, normalize pixel values to [0, 1]
        flatten: If True, flatten images to 1D (for MLP). If False, keep as 4D (for CNN)
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if _get_dataset_raw is None:
        raise ImportError("Datasets module not available")
    
    # Load raw data
    X_train, y_train, X_test, y_test = _get_dataset_raw(dataset_name)
    
    # Get dataset config for reshaping
    config = DatasetConfig.from_name(dataset_name)
    
    if flatten:
        # Flatten for MLP: (batch, channels * height * width)
        total_features = config.input_channels * config.input_height * config.input_width
        X_train = X_train.reshape(-1, total_features)
        X_test = X_test.reshape(-1, total_features)
    else:
        # Keep 4D for CNN: (batch, channels, height, width)
        X_train = X_train.reshape(-1, config.input_channels, config.input_height, config.input_width)
        X_test = X_test.reshape(-1, config.input_channels, config.input_height, config.input_width)
    
    # Normalize if requested
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    else:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
    
    # Convert to tensors if requested
    if as_tensors:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def get_dataset_quantized(
    dataset_name: str,
    input_bits: int,
    input_signed: bool = False,
    flatten: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and quantize dataset.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion', 'cifar10')
        input_bits: Bit-width for quantization (e.g., 8, 4, 2)
        input_signed: If True, use signed quantization
        flatten: If True, flatten images to 1D (for MLP). If False, keep as 4D (for CNN)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """

    X_train, y_train, X_test, y_test = get_dataset(
        dataset_name,
        as_tensors=True,
        normalize=True,
        flatten=flatten
    )
    print("\nQuantizing input data...")

    # Print sample of full 28x28 image data
    print("===before quantization===")

    # print("\nSample of first training image (28x28):")
    # print("Shape:", X_train[0].shape)
    # for i in range(5, 6):
    #     print(f"{X_train[0][i*28:(i+1)*28]}")

    print("\nSample of first test image (28x28):")
    print("Shape:", X_test[0].shape)
    for i in range(7, 8):
        print(f"{X_test[0][i*28:(i+1)*28]}")

    # Quantize input data
    def quantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        scale = (2 ** input_bits - 1) / 1.0 #255.0 #TODO check range for other datasets
        if input_signed:
            tensor = tensor * scale - (2 ** (input_bits - 1))
        else:
            tensor = tensor * scale
        return torch.clamp(tensor.round(), 0, 2 ** input_bits - 1)

    X_train = quantize_tensor(X_train)
    X_test = quantize_tensor(X_test)

    return X_train, y_train, X_test, y_test