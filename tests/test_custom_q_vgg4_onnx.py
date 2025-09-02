#!/usr/bin/env python3
import os
import sys
import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import get_default_qat_qconfig
from torch.quantization import prepare_qat, convert
import torch.quantization as quant
from tqdm import tqdm

# Add the parent directory to the Python path to find the mlgen3 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets import get_dataset
from mlgen3.models.nn.neuralnet import NeuralNet
from mlgen3.models.nn.linear import Linear
from mlgen3.implementations.neuralnet.cpp.vgg_qat_onnx import VGG_QAT_ONNX
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.implementations.neuralnet.cpp.onnx_utils import deploy_onnx_model

class TestCustomQVGG4ONNX(unittest.TestCase):
    def setUp(self):
        # Load Fashion-MNIST dataset
        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset("fashion")
        # Reshape data for CNN (adding channel dimension)
        self.X_train = self.X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
        self.X_test = self.X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0
        
        # Configuration for quantization
        self.bit_width = 8  # Options: 8, 4, or 2 bit
        
        # Define the Quantization-Aware VGG4 network architecture
        class QVGG(nn.Module):
            def __init__(self):
                super(QVGG, self).__init__()
                # Quantization stubs - needed for QAT
                self.quant = QuantStub()
                self.dequant = DeQuantStub()
                
                # Main model architecture
                self.features = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 10)
                )
            
            def forward(self, x):
                x = self.quant(x)
                x = self.features(x)
                x = self.classifier(x)
                x = self.dequant(x)
                return x
                
            # Helper function to fuse modules - required for proper quantization
            def fuse_model(self):
                # Explicitly specify which modules to fuse - this approach is more reliable
                torch.quantization.fuse_modules(
                    self.features,
                    [['0', '1'], ['4', '5']],  # Conv-BN pairs
                    inplace=True
                )
        
        self.model_cls = QVGG
        self.batch_size = 128
        self.output_dir = os.path.join("generated_code", "q_vgg4_onnx")
        os.makedirs(self.output_dir, exist_ok=True)
        self.onnx_dir = os.path.join("generated_code", "onnx_models")
        os.makedirs(self.onnx_dir, exist_ok=True)
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def configure_qat(self, model, bit_width):
        """Configure model for Quantization Aware Training with specified bit width"""
        if bit_width == 8:
            # Use default 8-bit quantization config
            qconfig = get_default_qat_qconfig('fbgemm')
        elif bit_width == 4:
            # 4-bit quantization config using 8-bit tensors with restricted range
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=0, quant_max=15, dtype=torch.quint8),  # Use quint8 with restricted range (0-15)
                weight=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=-8, quant_max=7, dtype=torch.qint8)     # Use qint8 with restricted range (-8 to 7)
            )
        elif bit_width == 2:
            # 2-bit quantization config using 8-bit tensors with restricted range
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=0, quant_max=3, dtype=torch.quint8),    # Use quint8 with restricted range (0-3)
                weight=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=-2, quant_max=1, dtype=torch.qint8)     # Use qint8 with restricted range (-2 to 1)
            )
        else:
            raise ValueError(f"Unsupported bit width: {bit_width}. Use 2, 4, or 8.")
            
        model.qconfig = qconfig
        
        # # Switch to eval mode for fusion (required by PyTorch)
        # model.eval()
        # model.fuse_model()
        
        # Prepare the model for QAT and switch back to training mode
        model_prepared = prepare_qat(model.train())
        
        print(f"Model prepared for {bit_width}-bit QAT")
        return model_prepared

    def test_qvgg4_onnx_export_and_deploy(self):
        # Create and train the quantized model
        model = self.model_cls()
        model = self.configure_qat(model, self.bit_width)
        
        # Move model to device
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        if self.bit_width == 8:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        elif self.bit_width == 4:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        elif self.bit_width == 2:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # Training parameters - reduced for faster testing
        epochs = 1
        
        print(f"Training VGG4 model with {self.bit_width}-bit QAT...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i in tqdm(range(0, len(self.X_train), self.batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = torch.tensor(self.X_train[i:i+self.batch_size], dtype=torch.float32).to(self.device)
                targets = torch.tensor(self.y_train[i:i+self.batch_size], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(self.X_train)/self.batch_size):.4f}")
        
        # Test the QAT model's accuracy
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for i in range(0, len(self.X_test), self.batch_size):
                inputs = torch.tensor(self.X_test[i:i+self.batch_size], dtype=torch.float32).to(self.device)
                targets = torch.tensor(self.y_test[i:i+self.batch_size], dtype=torch.long).to(self.device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            qat_accuracy = 100 * correct / total
            print(f"\nQAT Model Accuracy (before conversion): {qat_accuracy:.2f}%")
        
        # Convert model to quantized model
        print("Converting QAT model to quantized model...")
        model.to('cpu')  # Conversion must happen on CPU
        
        # Save the prepared model before quantization conversion for ONNX export
        pre_quantized_model = model.eval()
        
        # Now convert the model to fully quantized for testing accuracy
        quantized_model = convert(model.eval())
        
        # Test the quantized model's accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            
            for i in range(0, len(self.X_test), self.batch_size):
                inputs = torch.tensor(self.X_test[i:i+self.batch_size], dtype=torch.float32)
                targets = torch.tensor(self.y_test[i:i+self.batch_size], dtype=torch.long)
                
                outputs = quantized_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            quantized_accuracy = 100 * correct / total
            print(f"\nQuantized Model Accuracy: {quantized_accuracy:.2f}%")
        
        # Extract quantization parameters from the model
        print(f"Extracting quantization parameters from {self.bit_width}-bit model...")
        quant_params = {}
        
        # Helper function to extract parameters from quantized modules
        def collect_qparams(model, prefix=''):
            for name, module in model.named_modules():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Handle various quantized module types
                if hasattr(module, '_packed_params') and module._packed_params is not None:
                    # Extract quantization parameters from packed modules (like quantized Conv2d)
                    packed_params = module._packed_params
                    if hasattr(packed_params, 'scale') and hasattr(packed_params, 'zero_point'):
                        # Handle both tensor and scalar cases
                        scale_val = packed_params.scale().item() if hasattr(packed_params.scale(), 'item') else float(packed_params.scale())
                        zero_point_val = packed_params.zero_point().item() if hasattr(packed_params.zero_point(), 'item') else int(packed_params.zero_point())
                        
                        quant_params[f"{full_name}.weight"] = {
                            'scale': scale_val,
                            'zero_point': zero_point_val
                        }
            
                # Extract activation quantization parameters
                if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
                    # Handle both tensor and scalar cases
                    scale_val = module.scale.item() if hasattr(module.scale, 'item') else float(module.scale)
                    zero_point_val = module.zero_point.item() if hasattr(module.zero_point, 'item') else int(module.zero_point)
                    
                    quant_params[f"{full_name}"] = {
                        'scale': scale_val,
                        'zero_point': zero_point_val
                    }
                    
                # For QAT, extract observer stats
                if hasattr(module, 'activation_post_process'):
                    observer = module.activation_post_process
                    if hasattr(observer, 'calculate_qparams'):
                        scale, zero_point = observer.calculate_qparams()
                        if scale is not None and zero_point is not None:
                            # Handle both tensor and scalar cases
                            scale_val = scale.item() if hasattr(scale, 'item') else float(scale)
                            zero_point_val = zero_point.item() if hasattr(zero_point, 'item') else int(zero_point)
                            
                            quant_params[f"{full_name}.activation"] = {
                                'scale': scale_val,
                                'zero_point': zero_point_val,
                                'bit_width': self.bit_width
                            }
    
        # Collect parameters
        collect_qparams(quantized_model)
        
        # Save quantization parameters to JSON
        import json
        quant_params_path = os.path.join(self.onnx_dir, f"vgg4_quantized_{self.bit_width}bit_params.json")
        with open(quant_params_path, 'w') as f:
            json.dump(quant_params, f, indent=2)
    
        # Export the pre-quantized model to ONNX (much more reliable)
        print(f"Exporting pre-quantized VGG4 model to ONNX format...")
        onnx_path = os.path.join(self.onnx_dir, f"vgg4_quantized_{self.bit_width}bit.onnx")
        
        # Create a dummy input for the model
        dummy_input = torch.randn(1, 1, 28, 28, requires_grad=False)
        
        # Export the pre-quantized model to ONNX format (this has quantization preparation but is still a floating-point model)
        torch.onnx.export(
            pre_quantized_model,  # Using the model before final quantization conversion
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )

        print(f"\nPre-quantized ONNX Model with quantization parameters exported successfully")
        print(f"ONNX model: {onnx_path}")
        print(f"Quantization parameters: {quant_params_path}")

        # Create a simplified MLGen3 model structure
        # We only need a basic structure as the weights will be loaded from ONNX
        layers = [
            # Just create minimal structure - actual inference will use ONNX model
            Linear(np.random.randn(10, 784), np.zeros(10))
        ]
        
        # Create MLGen3 model with test data
        mlgen_model = NeuralNet.from_layers(layers)
        mlgen_model.XTest = self.X_test
        mlgen_model.YTest = self.y_test
        
        # Generate C++ code using VGG_QAT_ONNX implementation with quant_params_path
        print("Generating C++ code using VGG_QAT_ONNX implementation...")
        implementation = VGG_QAT_ONNX(
            mlgen_model,
            onnx_path=onnx_path,
            quant_params_path=quant_params_path,  # Add quantization parameters path
            feature_type="float",
            label_type="int",
            internal_type="float",
            batch_size=self.batch_size,
            bit_width=self.bit_width
        )

        implementation.implement()

        # Deploy the model using LinuxStandalone materializer
        print("Deploying model...")
        materializer = deploy_onnx_model(
            implementation, 
            self.output_dir,
            LinuxStandalone,
            measure_accuracy=True,
            measure_time=True
        )
        
        materializer.deploy()
        print("Model deployed successfully.\n")
        
        # Run the model and verify results
        results = materializer.run(verbose=True)
        print(f"Model test results: {results}")
        
        # Verify accuracy is within reasonable bounds of PyTorch model
        if "Accuracy" in results:
            onnx_accuracy = float(results["Accuracy"])
            accuracy_diff = abs(onnx_accuracy - quantized_accuracy)
            self.assertLess(accuracy_diff, 10.0)  # Allow up to 10% difference due to numerical differences

if __name__ == '__main__':
    unittest.main()
