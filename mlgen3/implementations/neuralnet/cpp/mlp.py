import torch
import torch.nn as nn

from ..quantization.input_quantizer import InputQuantizer
from ..utils.batchnorm_folding import fold_model_batchnorm


class MLP(nn.Module):
    def __init__(self, config):
        """
        Initialize the MLP model based on the provided configuration.

        Args:
            config (dict): Configuration dictionary containing the following parameters:
                model:
                    - name (str): Name of the NN Model (e.g. MLP, VGG, RESNET)
                    - dataset (str): Name of the dataset (e.g. MNIST, FASHION, CIFAR10)
                    - input_size (int): Size of the input image (<=> width <=> height)
                    - output_size (int): Number of output classes.
                    - hidden_layers (list[dict]): List of dictionaries containing sizes for each hidden layer.
        """

        super(MLP, self).__init__()
        
        name = config['model']['name'].upper()
        dataset = config['model']['dataset'].upper()
        input_size = config['model']['input_size']
        output_size = config['model']['output_size']
        hidden_layers = config['model']['hidden_layers']
        
        # Add input quantization if enabled
        self.use_input_quantization = config['quantization'].get('quantize_input', True)
        if self.use_input_quantization:
            # Determine input range based on dataset normalization
            if 'MNIST' in dataset or 'FASHION' in dataset:
                # Normalized to mean=0.1307/0.5, std=0.3081/0.5
                input_range = (-1.0, 1.0)  # Approximate range after normalization
            elif 'CIFAR' in dataset:
                input_range = (-2.5, 2.5)  # Wider range for CIFAR datasets
            else:
                input_range = (-3.0, 3.0)  # Wider range for other datasets
            
            self.input_quantizer = InputQuantizer(input_range=input_range, signed=True)
        
        # Create layers dynamically based on config
        layers = []
        
        # First layer (input to first hidden)
        prev_size = input_size ** 2
        
        # Add all hidden layers
        for layer_config in hidden_layers:
            layer_size = layer_config['size']

            layers.append(nn.Linear(prev_size, layer_size))

            # Depending on the target bits, add different activation functions
            if config['quantization']['use_matquant'] is False:
                if config['quantization']['use_qat'] and max(config['quantization']['target_bits']) < 4:
                    # Only add BatchNorm if not folding during inference
                    if not self.fold_bn_inference:
                        layers.append(nn.Sequential(nn.BatchNorm1d(layer_size), nn.Hardtanh()))
                    else:
                        layers.append(nn.Hardtanh())
                    
                elif config['quantization']['use_qat'] is False or max(config['quantization']['target_bits']) >= 4:
                    layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
            
            prev_size = layer_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Add flag for BatchNorm folding
        self.fold_bn_inference = config['quantization'].get('fold_bn_inference', True)
        self.dry_mode = False

    def set_dry_mode(self, dry_mode):
        """
        Set the dry mode for the model.

        Args:
            dry_mode (bool): If True, the model will operate in dry mode (full precision).
        """
        self.dry_mode = dry_mode

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after passing through the model.
        """

        # Quantize input if enabled
        if self.use_input_quantization and not self.dry_mode:
            x = self.input_quantizer(x)
        
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)
    
    def get_folded_model(self):
        """
        Get a copy of the model with BatchNorm layers folded into Conv/Linear layers.
        This should only be used during inference.
        
        Returns:
            MLP: New model instance with folded BatchNorm
        """
        if self.training:
            raise RuntimeError("Cannot fold BatchNorm during training. Set model to eval mode first.")
        
        return fold_model_batchnorm(self, inplace=False)
