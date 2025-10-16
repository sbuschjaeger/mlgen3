"""Layer registry for identifying quantizable layers in models."""

import torch.nn as nn
from typing import List


class LayerRegistry:
    """Registry for identifying quantizable layers in a model."""
    
    def __init__(self):
        """Initialize the layer registry."""
        self.layer_paths = []
    
    def register_model(self, model: nn.Module) -> List[str]:
        """
        Register all quantizable layers in the model.
        
        Args:
            model: PyTorch model to register
        
        Returns:
            List of layer paths (e.g., 'model.0.weight')
        """
        self.layer_paths = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                weight_name = f"{name}.weight"
                if weight_name not in self.layer_paths:
                    self.layer_paths.append(weight_name)
        
        return self.layer_paths
    
    def get_layer_paths(self) -> List[str]:
        """Get the list of registered layer paths."""
        return self.layer_paths
    
    def filter_layers(self, include_patterns: List[str] = None, 
                     exclude_patterns: List[str] = None) -> List[str]:
        """
        Filter layer paths based on include/exclude patterns.
        
        Args:
            include_patterns: List of patterns to include (e.g., ['conv', 'linear'])
            exclude_patterns: List of patterns to exclude
        
        Returns:
            Filtered list of layer paths
        """
        filtered = self.layer_paths.copy()
        
        if include_patterns:
            filtered = [p for p in filtered if any(pat in p for pat in include_patterns)]
        
        if exclude_patterns:
            filtered = [p for p in filtered if not any(pat in p for pat in exclude_patterns)]
        
        return filtered
