import os
import numpy as np
import struct
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def read_binary_file(file_path):
    """Read a binary file and return its contents as bytes."""
    with open(file_path, 'rb') as f:
        return f.read()

def parse_weight_file(data, print_stats=True, plot_hist=True, max_elements=20):
    """Parse weight binary file and print stats."""
    # Assuming int8 data for weights
    weights = np.frombuffer(data, dtype=np.int8)
    
    print(f"Shape: {weights.shape}")
    print(f"Data type: int8")
    print(f"Min value: {weights.min()}")
    print(f"Max value: {weights.max()}")
    print(f"Mean value: {weights.mean():.4f}")
    print(f"First {max_elements} values: {weights[:max_elements]}")
    
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

def analyze_bin_files(directory):
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
            weights = parse_weight_file(data)
        
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
                print(f"First 10 dequantized values: {[f'{v:.6f}' for v in dequantized[:10]]}")

if __name__ == "__main__":
    # Path to the directory containing binary files
    # bin_dir = "generated_code/matquant_pt_mnist/uniform_4bit/mq_pt_model_binary"
    bin_dir = "generated_code/matquant_pt_vgg4/uniform_8bit/mq_pt_vgg_binary"
    
    if os.path.exists(bin_dir):
        print(f"Analyzing binary files in {bin_dir}...\n")
        analyze_bin_files(bin_dir)
    else:
        print(f"Directory {bin_dir} does not exist.")
