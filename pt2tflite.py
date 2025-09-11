import ai_edge_torch
import torch
import torch.nn as nn
import argparse
import os

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
    convert_to_tflite(model, args.tflite_path)