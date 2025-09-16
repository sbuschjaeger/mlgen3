import ai_edge_torch
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import tensorflow as tf

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ensure_dir(directory):
    """Make sure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"\nCreated directory: {directory}")

def save_model(model, path):
    """Save PyTorch model with directory creation."""
    # Ensure directory exists
    directory = os.path.dirname(path)
    ensure_dir(directory)
    
    # Save model
    print(f"\nSaving PyTorch model to {path}")
    torch.save(model.state_dict(), path)
    
def load_model(path):
    print(f"\nLoading PyTorch model from {path}")
    model = SimpleModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def convert_to_tflite(model, output_path):
    """Convert to TFLite with directory creation."""
    # Ensure directory exists
    directory = os.path.dirname(output_path)
    ensure_dir(directory)
    
    print("\nConverting model to TFLite format...")
    sample_input = (torch.randn(1, 3, 32, 32),)
    edge_model = ai_edge_torch.convert(model, sample_input)
    output = edge_model(*sample_input)
    print(f"\nSample output shape: {output.shape}")
    
    print(f"\nSaving TFLite model to {output_path}")
    edge_model.export(output_path)
    print("\nConversion complete!")
    
    return sample_input

def verify_model_equivalence(pt_model, tflite_path, sample_input):
    """Verify that PyTorch and TFLite models produce equivalent outputs."""
    print("\n========== VERIFYING MODEL EQUIVALENCE ==========")
    
    # Get PyTorch model prediction
    with torch.no_grad():
        pt_output = pt_model(*sample_input).numpy()
    print(f"PyTorch output shape: {pt_output.shape}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input details for debugging
    print(f"TFLite expects input shape: {input_details[0]['shape']}")
    
    # Prepare input data - matching the expected input shape
    input_tensor = sample_input[0]
    if input_details[0]['shape'][3] == 3:  # If TFLite expects NHWC
        input_data = input_tensor.permute(0, 2, 3, 1).numpy()  # Convert from NCHW to NHWC
    else:
        # Keep original format if TFLite expects NCHW
        input_data = input_tensor.numpy()
    
    # Resize input tensor if necessary
    if list(input_data.shape) != input_details[0]['shape']:
        print(f"Reshaping input from {input_data.shape} to {input_details[0]['shape']}")
        input_data = np.resize(input_data, input_details[0]['shape'])
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    print(f"TFLite output shape: {tflite_output.shape}")
    
    # Compare outputs
    max_diff = np.max(np.abs(pt_output - tflite_output))
    mean_diff = np.mean(np.abs(pt_output - tflite_output))
    
    print(f"\nMaximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check if the models are equivalent (allowing for small numerical differences)
    if max_diff < 1e-3:
        print("\n✅ Models are equivalent!")
    else:
        print("\n⚠️ Warning: Models show significant differences!")
        
    return max_diff, mean_diff

if __name__ == "__main__":
    # Set up directory structure
    PT_DIR = "models/pt"
    TF_DIR = "models/tf"
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TFLite')
    parser.add_argument('--mode', type=str, choices=['create', 'load'], default='create',
                        help='Create new model or load existing .pt model')
    parser.add_argument('--pt_path', type=str, default=os.path.join(PT_DIR, 'simple_model.pt'),
                        help='Path for saving/loading PyTorch model')
    parser.add_argument('--tflite_path', type=str, default=os.path.join(TF_DIR, 'simple_model.tflite'),
                        help='Path for saving TFLite model')
    parser.add_argument('--verify', action='store_true',
                        help='Verify that the PyTorch and TFLite models produce equivalent outputs')
    args = parser.parse_args()
    
    if args.mode == 'create':
        # Create and save new model
        model = SimpleModel().eval()
        save_model(model, args.pt_path)
    else:
        # Load existing model
        if not os.path.exists(args.pt_path):
            print(f"Error: Model file {args.pt_path} does not exist!")
            exit(1)
        model = load_model(args.pt_path)
    
    # Convert to TFLite
    sample_input = convert_to_tflite(model, args.tflite_path)
    
    # Optionally verify model equivalence
    if args.verify:
        verify_model_equivalence(model, args.tflite_path, sample_input)