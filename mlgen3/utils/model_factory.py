"""Model factory for creating neural network architectures."""

import torch
import torch.nn as nn
from typing import Dict, Any


class ModelFactory:
    """Factory class for creating neural network models."""
    
    @staticmethod
    def create_mlp(
        input_size: int = 784,
        hidden_sizes: list = [128, 64],
        num_classes: int = 10,
        use_batchnorm: bool = True
    ) -> nn.Module:
        """Create an MLP model."""
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                layers = []
                
                # Input layer
                layers.append(nn.Linear(input_size, hidden_sizes[0]))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_sizes[0]))
                layers.append(nn.ReLU())
                
                # Hidden layers
                for i in range(len(hidden_sizes) - 1):
                    layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                    if use_batchnorm:
                        layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
                    layers.append(nn.ReLU())
                
                # Output layer
                layers.append(nn.Linear(hidden_sizes[-1], num_classes))
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        return MLP()
    
    @staticmethod
    def create_vgg4(
        input_channels: int = 1,
        num_classes: int = 10
    ) -> nn.Module:
        """Create a VGG4 model (2 conv blocks + 2 FC layers)."""
        class VGG4(nn.Module):
            def __init__(self):
                super(VGG4, self).__init__()
                self.model = nn.Sequential(
                    # Conv block 1
                    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 2
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    # Flatten and FC layers
                    nn.Flatten(),
                    nn.Linear(3136, 2048),  # For 28x28 input: 64 * 7 * 7 = 3136
                    nn.ReLU(),
                    nn.Linear(2048, num_classes)
                )
            
            def forward(self, x):
                return self.model(x)
        
        return VGG4()
    
    @staticmethod
    def create_vgg8(
        input_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32  # Add input_size parameter
    ) -> nn.Module:
        """Create a VGG8 model (6 conv layers + 2 FC layers)."""
        class VGG8(nn.Module):
            def __init__(self):
                super(VGG8, self).__init__()
                
                # Calculate the size after convolutions and pooling
                # Input: input_size x input_size
                # After Conv1 + MaxPool (stride=2): input_size/2 x input_size/2
                # After Conv2 (no pool): input_size/2 x input_size/2
                # After Conv3 + MaxPool (stride=2): input_size/4 x input_size/4
                # After Conv4 (no pool): input_size/4 x input_size/4
                # After Conv5 + MaxPool (stride=2): input_size/8 x input_size/8
                # After Conv6 (no pool): input_size/8 x input_size/8
                
                feature_size = input_size // 8  # After 3 max pooling layers with stride 2
                flattened_size = 512 * feature_size * feature_size
                
                self.model = nn.Sequential(
                    # Conv block 1
                    nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 2
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 3
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 4
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 5
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    
                    # Conv block 6
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    
                    # Flatten and FC layers
                    nn.Flatten(),
                    nn.Linear(flattened_size, 1024),  # Use calculated flattened_size
                    nn.ReLU(),
                    nn.Linear(1024, num_classes)
                )
            
            def forward(self, x):
                return self.model(x)
        
        return VGG8()


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Configuration dictionary with 'model' section
    
    Returns:
        PyTorch model
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'mlp').lower()
    
    if model_name == 'mlp':
        # Extract hidden layer sizes from config
        input_size = model_config.get('input_size', 28)
        num_classes = model_config.get('num_classes', model_config.get('output_size', 10))
        
        # Support both formats: hidden_sizes list or hidden_layers list of dicts
        if 'hidden_sizes' in model_config:
            hidden_sizes = model_config['hidden_sizes']
        elif 'hidden_layers' in model_config:
            # Extract sizes from list of dicts
            hidden_sizes = [layer['size'] for layer in model_config['hidden_layers']]
        else:
            hidden_sizes = [128, 64]  # Default
        
        return ModelFactory.create_mlp(
            input_size=input_size * input_size,  # Flatten 2D input
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            use_batchnorm=model_config.get('use_batchnorm', True)
        )
    elif model_name == 'vgg4':
        return ModelFactory.create_vgg4(
            input_channels=model_config.get('num_channels', 1),
            num_classes=model_config.get('num_classes', 10)
        )
    elif model_name == 'vgg8':
        return ModelFactory.create_vgg8(
            input_channels=model_config.get('num_channels', 3),
            num_classes=model_config.get('num_classes', 10),
            input_size=model_config.get('input_size', 32)  # Pass input_size
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
