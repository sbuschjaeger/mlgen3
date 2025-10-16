import os
import numpy as np
import struct
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

def read_binary_file(file_path):
    """Read a binary file and return its contents as bytes."""
    with open(file_path, 'rb') as f:
        return f.read()

def parse_weight_file(data, print_stats=True, plot_hist=True, max_elements=20, is_signed=False):
    """Parse weight binary file and print stats."""
    # Choose dtype based on signed/unsigned
    dtype = np.int8 if is_signed else np.uint8
    weights = np.frombuffer(data, dtype=dtype)
    
    print(f"Shape: {weights.shape}")
    print(f"Data type: {dtype}")
    print(f"Min value: {weights.min()}")
    print(f"Max value: {weights.max()}")
    print(f"Mean value: {weights.mean():.4f}")
    print(f"First {max_elements} values: {weights[:max_elements]}")
    # print(f"All values: {weights}")
    
    if plot_hist:
        plt.figure(figsize=(10, 5))
        plt.hist(weights, bins=50)
        plt.title('Weight Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return weights

def parse_qparams_file(data):
    """Parse quantization parameters file."""
    # Assuming qparams files contain scale and zero_point as float32
    if len(data) >= 8:
        scale = struct.unpack('f', data[0:4])[0]
        zero_point = struct.unpack('f', data[4:8])[0]
        print(f"Scale: {scale}")
        print(f"Zero point: {zero_point}")
        return scale, zero_point
    else:
        print(f"Unexpected data length: {len(data)}")
        return None

def analyze_bin_files(directory, is_signed=False):
    """Analyze all bin files in the directory."""
    bin_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    bin_files.sort()  # Sort for consistent output
    
    # Group files by layer and parameter type
    grouped_files = defaultdict(dict)
    for file_name in bin_files:
        match = re.match(r'layer_(\d+)_(\w+)(?:_qparams)?\.bin', file_name)
        if match:
            layer_num = match.group(1)
            param_type = match.group(2)
            if "_qparams" in file_name:
                grouped_files[f"layer_{layer_num}_{param_type}"]["qparams"] = file_name
            else:
                grouped_files[f"layer_{layer_num}_{param_type}"]["data"] = file_name
    
    # Process each group
    for param_name, files in grouped_files.items():
        print(f"\n{'='*50}\n{param_name.upper()}\n{'='*50}")
        
        # Process data file if exists
        if "data" in files:
            data_path = os.path.join(directory, files["data"])
            print(f"\nAnalyzing {files['data']}:")
            data = read_binary_file(data_path)
            weights = parse_weight_file(data, is_signed=is_signed)
        
        # Process qparams file if exists
        if "qparams" in files:
            qparams_path = os.path.join(directory, files["qparams"])
            print(f"\nAnalyzing {files['qparams']}:")
            qparams_data = read_binary_file(qparams_path)
            scale, zero_point = parse_qparams_file(qparams_data)
            
            # If we have both data and qparams, show dequantized values
            if "data" in files and scale is not None:
                print("\nDequantized Statistics:")
                dequantized = (weights.astype(np.float32) - zero_point) * scale
                print(f"Min value: {dequantized.min():.6f}")
                print(f"Max value: {dequantized.max():.6f}")
                print(f"Mean value: {dequantized.mean():.6f}")
                print(f"First 10 weights: {weights[:10]}")
                print(f"First 10 dequantized values: {[f'{v:.6f}' for v in dequantized[:10]]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze binary weight/bias files')
    parser.add_argument('--dir', type=str, 
                        default="generated_code/matquant_pt_vgg4/uniform_8bit/mq_pt_model_binary",
                        help='Directory containing binary files')
    parser.add_argument('--signed', action='store_true',
                        help='Use signed int8 instead of unsigned uint8 for weights/biases')
    
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        print(f"Analyzing binary files in {args.dir}...")
        print(f"Using {'signed' if args.signed else 'unsigned'} integers\n")
        analyze_bin_files(args.dir, is_signed=args.signed)
    else:
        print(f"Directory {args.dir} does not exist.")
