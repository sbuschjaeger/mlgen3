import re
from pathlib import Path
import numpy as np

# ---------------- PARAMETERS ----------------
SIGNED = True  # Set True for signed interpretation, False for unsigned
M_d = 10  # Number of input rows (batch size)
K_d = 512  # Input/weight columns (must be multiple of 16 and K_d * 2 must be a multiple of 64 for bus bandwith alignment)
N_d = 10  # Number of output rows (output features)

# ---------------- UTILITY FUNCTIONS ----------------
def detect_dims(text):
    """Extract M_d, K_d, N_d from header text (#define lines)."""
    def get_dim(name):
        m = re.search(rf"#define\s+{name}\s+(\d+)", text)
        return int(m.group(1)) if m else None
    return get_dim("M_d"), get_dim("K_d"), get_dim("N_d")

# def pack_block_2bitplanes(block16):
#     """Pack 16 uint8 values into 4 groups of 2-bit planes (MSB-first)."""
#     assert len(block16) == 16
#     packed = []
#     for group in range(3, -1, -1):  # bits [7:6], [5:4], [3:2], [1:0]
#         shift = group * 2
#         word = 0
#         for i, val in enumerate(block16):
#             two_bits = (val >> shift) & 0x3
#             word |= (two_bits << (2 * i))
#         for b in range(4):
#             packed.append((word >> (8 * b)) & 0xFF)
#     return packed

def pack_block_2bitplanes(block16):
    """Pack 16 uint8 values into 4 groups of 2-bit planes (MSB-first).
    
    For each bitplane (bits [7:6], [5:4], [3:2], [1:0]):
      - Take 2 bits from values 0-3, pack into byte 0 (value 0 at MSB)
      - Take 2 bits from values 4-7, pack into byte 1 (value 4 at MSB)
      - Take 2 bits from values 8-11, pack into byte 2 (value 8 at MSB)
      - Take 2 bits from values 12-15, pack into byte 3 (value 12 at MSB)
    """
    assert len(block16) == 16
    packed = []
    for group in range(3, -1, -1):  # bits [7:6], [5:4], [3:2], [1:0]
        shift = group * 2
        # Process 4 bytes, each containing 4 values' 2-bit slices
        for byte_idx in range(4):
            byte_val = 0
            for i in range(4):
                val_idx = byte_idx * 4 + i
                two_bits = (block16[val_idx] >> shift) & 0x3
                # Pack with first value at MSB (bits 7:6), last at LSB (bits 1:0)
                byte_val |= (two_bits << (2 * (3 - i)))
            packed.append(byte_val)
    return packed

def rearrange_bitplanes(values, cols):
    """Rearrange row-major values into packed 2-bit planes per 16-element block."""
    output = []
    num_rows = len(values) // cols
    if len(values) % cols != 0:
        num_rows += 1
    
    for row_idx in range(num_rows):
        row_start = row_idx * cols
        row_end = min(row_start + cols, len(values))
        row = values[row_start:row_end]
        
        # Pad row to be multiple of 16
        if len(row) % 16 != 0:
            padding_needed = 16 - (len(row) % 16)
            row = row + [0] * padding_needed
        
        for blk_start in range(0, len(row), 16):
            block = row[blk_start:blk_start+16]
            assert len(block) == 16, f"Block at {blk_start} has {len(block)} elements, expected 16"
            output.extend(pack_block_2bitplanes(block))
    return output

def extract_2bit_matrix(packed, col_start, signed=False):
    """Extract 16 2-bit elements from first 4 bytes of a row or block."""
    word = sum((b & 0xFF) << (8*i) for i, b in enumerate(packed[col_start:col_start+4]))
    result = []
    for i in range(16):
        v = (word >> (2*i)) & 0x3
        if signed and v >= 2:
            v -= 4
        result.append(v)
    return result

def extract_4bit_matrix(packed, col_start, signed=False):
    """Extract 16 4-bit elements from first 8 bytes of a row or block."""
    msb_bytes  = packed[col_start:col_start+4]
    next_bytes = packed[col_start+4:col_start+8]
    word_msb  = sum((b & 0xFF) << (8*i) for i, b in enumerate(msb_bytes))
    word_next = sum((b & 0xFF) << (8*i) for i, b in enumerate(next_bytes))
    result = []
    for i in range(16):
        v = ((word_msb >> (2*i)) & 0x3) << 2 | ((word_next >> (2*i)) & 0x3)
        if signed and v >= 8:
            v -= 16
        result.append(v)
    return result

def parse_flat_array(name, text):
    """Extract flat array values from a C header file, ignoring commented lines.
    Supports both 1D arrays (input[512]) and 2D arrays (input[10][512]).
    Returns: (values_list, dimensions_tuple)
    """
    # Remove C-style comments first
    text_no_comments = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    
    # Match both 1D: int8_t input[512] and 2D: int8_t input[10][512]
    pattern = re.compile(
        rf"(?:u?int8_t|int8_t)\s+{name}\s*"
        rf"(\[(\d+)\])?(\[(\d+)\])"  # Capture dimensions
        rf"\s*=\s*{{(.*?)}};",
        re.S
    )
    match = pattern.search(text_no_comments)
    if not match:
        raise ValueError(f"Array {name} not found")
    
    # Extract dimensions
    dim1 = int(match.group(2)) if match.group(2) else None
    dim2 = int(match.group(4))
    
    numbers = re.findall(r"-?\d+", match.group(5))
    values = list(map(int, numbers))
    
    if dim1 is None:
        # 1D array
        return values, (len(values),)
    else:
        # 2D array
        return values, (dim1, dim2)

def write_array(name, values, signed=False):
    ctype = "int8_t" if signed else "uint8_t"
    lines = [", ".join(str(int(v)) for v in values[i:i+16]) for i in range(0, len(values), 16)]
    return f"{ctype} {name}[{len(values)}] = {{\n    " + ",\n    ".join(lines) + "\n};"

def to_signed_8bit_binary(val):
    """Convert a signed 8-bit integer to its binary representation string."""
    # Handle signed values using two's complement
    if val < 0:
        val = (1 << 8) + val  # Convert to unsigned representation
    return f"0b{val:08b}"

def write_array_binary(name, values, signed=False):
    """Write array with values in binary representation."""
    ctype = "int8_t" if signed else "uint8_t"
    lines = [", ".join(to_signed_8bit_binary(int(v)) for v in values[i:i+16]) for i in range(0, len(values), 16)]
    return f"// Binary representation of {name}\n{ctype} {name}_binary[{len(values)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n\n"

def write_array_2d(name, values, rows, cols, signed=False):
    """Write array in 2D format with proper row/column structure."""
    ctype = "int8_t" if signed else "uint8_t"
    result = [f"{ctype} {name}[{len(values)}] = {{"]
    
    for row_idx in range(rows):
        row_start = row_idx * cols
        row_end = min(row_start + cols, len(values))
        row_data = values[row_start:row_end]
        
        # Format row data in chunks of 16 for readability
        row_lines = [", ".join(str(int(v)) for v in row_data[i:i+16]) 
                     for i in range(0, len(row_data), 16)]
        row_text = ",\n    ".join(row_lines)
        
        if row_idx < rows - 1:
            result.append(f"    {row_text},")
        else:
            result.append(f"    {row_text}")
    
    result.append("};")
    return "\n".join(result)

def write_array_2d_binary(name, values, rows, cols, signed=False):
    """Write 2D array with values in binary representation."""
    ctype = "int8_t" if signed else "uint8_t"
    result = [f"// Binary representation of {name}", f"{ctype} {name}_binary[{len(values)}] = {{"]
    
    for row_idx in range(rows):
        row_start = row_idx * cols
        row_end = min(row_start + cols, len(values))
        row_data = values[row_start:row_end]
        
        # Format row data in chunks of 16 for readability
        row_lines = [", ".join(to_signed_8bit_binary(int(v)) for v in row_data[i:i+16]) 
                     for i in range(0, len(row_data), 16)]
        row_text = ",\n    ".join(row_lines)
        
        if row_idx < rows - 1:
            result.append(f"    {row_text},")
        else:
            result.append(f"    {row_text}")
    
    result.append("};")
    return "\n".join(result) + "\n\n"

def write_array_compact(name, values, signed=False):
    """Write array in compact single-line format for template."""
    ctype = "int8_t" if signed else "uint8_t"
    values_str = ", ".join(str(int(v)) for v in values)
    return f"{ctype} {name}[{len(values)}] = {{{values_str}}};"

def to_signed_32bit_binary(val):
    """Convert a signed 32-bit integer to its binary representation string."""
    if val < 0:
        val = (1 << 32) + val
    return f"0b{val:032b}"

def write_product_array(name, mat, signed=False):
    ctype = "int32_t" if signed else "uint32_t"
    flat = mat.flatten()
    lines = [", ".join(str(int(v)) for v in flat[i:i+16]) for i in range(0, len(flat), 16)]
    return f"{ctype} {name}[{len(flat)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n\n"

def write_product_array_binary(name, mat, signed=False, cols_per_line=16):
    """Write product array with values in binary representation."""
    ctype = "int32_t" if signed else "uint32_t"
    flat = mat.flatten()
    lines = [", ".join(to_signed_32bit_binary(int(v)) for v in flat[i:i+cols_per_line]) 
             for i in range(0, len(flat), cols_per_line)]
    return f"// Binary representation of {name}\n{ctype} {name}_binary[{len(flat)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n\n"

def generate_random_matrices(M, K, N, signed=True):
    """Generate random input and weight matrices."""
    if signed:
        input_vals = np.random.randint(-128, 128, M * K, dtype=np.int8).tolist()
        weights_vals = np.random.randint(-128, 128, N * K, dtype=np.int8).tolist()
    else:
        input_vals = np.random.randint(0, 256, M * K, dtype=np.uint8).tolist()
        weights_vals = np.random.randint(0, 256, N * K, dtype=np.uint8).tolist()
    return input_vals, weights_vals

# ---------------- MAIN SCRIPT ----------------
def main(infile=None, outfile="output.h", datafile="data.h"):
    """
    Process or generate matrices for bitplane packing.
    
    Args:
        infile: Input header file (optional). If None, generates random matrices.
        outfile: Output file with diagnostic arrays (packed, 2bit, 4bit, products).
        datafile: Final data file generated from template_data.h.
    """
    if infile:
        # Read from existing file
        text = Path(infile).read_text()
        M, K, N = detect_dims(text)
        if not all([M, K, N]):
            raise ValueError("Could not detect M_d, K_d, N_d from header")
        
        # Parse arrays with dimension info
        input_vals, input_dims = parse_flat_array("input", text)
        weights_vals, weights_dims = parse_flat_array("weights", text)
        
        # Determine if arrays are 2D
        is_input_2d = len(input_dims) == 2
        is_weights_2d = len(weights_dims) == 2
        
        if is_input_2d:
            M, K = input_dims
        if is_weights_2d:
            N, K_w = weights_dims
            if K_w != K:
                raise ValueError(f"Dimension mismatch: input K={K}, weights K={K_w}")
    else:
        # Generate random matrices
        M, K, N = M_d, K_d, N_d
        if K % 16 != 0:
            raise ValueError(f"K={K} must be multiple of 16 for bitplane packing")
        print(f"Generating random matrices: M={M}, K={K}, N={N}")
        input_vals, weights_vals = generate_random_matrices(M, K, N, SIGNED)
        is_input_2d = M > 1
        is_weights_2d = N > 1
        input_dims = (M, K) if is_input_2d else (M * K,)
        weights_dims = (N, K) if is_weights_2d else (N * K,)
        text = ""

    # Pack into bitplanes (works the same for both 1D and 2D)
    input_packed   = rearrange_bitplanes(input_vals, K)
    weights_packed = rearrange_bitplanes(weights_vals, K)

    # Convert packed arrays to signed if needed
    if SIGNED:
        input_packed   = [b - 256 if b >= 128 else b for b in input_packed]
        weights_packed = [b - 256 if b >= 128 else b for b in weights_packed]

    # Extract 2-bit and 4-bit matrices row by row
    input_2bit, input_4bit = [], []
    weights_2bit, weights_4bit = [], []
    num_blocks = K // 16
    for row in range(M):
        row_start = row * num_blocks * 16  # total packed bytes per row
        for blk in range(num_blocks):
            base = row_start + blk * 16
            input_2bit.extend(extract_2bit_matrix(input_packed, base, signed=SIGNED))
            input_4bit.extend(extract_4bit_matrix(input_packed, base, signed=SIGNED))
    
    for row in range(N):
        row_start = row * num_blocks * 16
        for blk in range(num_blocks):
            base = row_start + blk * 16
            weights_2bit.extend(extract_2bit_matrix(weights_packed, base, signed=SIGNED))
            weights_4bit.extend(extract_4bit_matrix(weights_packed, base, signed=SIGNED))

    # Convert to matrices
    dtype8  = np.int8 if SIGNED else np.uint8
    dtype32 = np.int32 if SIGNED else np.uint32
    
    in_mat = np.array(input_vals, dtype=dtype8).reshape(M, K)
    wt_mat = np.array(weights_vals, dtype=dtype8).reshape(N, K)
    in2_mat = np.array(input_2bit, dtype=dtype8).reshape(M, K)
    in4_mat = np.array(input_4bit, dtype=dtype8).reshape(M, K)
    wt2_mat = np.array(weights_2bit, dtype=dtype8).reshape(N, K)
    wt4_mat = np.array(weights_4bit, dtype=dtype8).reshape(N, K)

    # Compute products (input @ weights.T)
    prod_8bit = in_mat.astype(dtype32) @ wt_mat.T.astype(dtype32)
    prod_2bit = in2_mat.astype(dtype32) @ wt2_mat.T.astype(dtype32)
    prod_4bit = in4_mat.astype(dtype32) @ wt4_mat.T.astype(dtype32)

    # Calculate packed dimensions
    packed_cols = num_blocks * 16  # 16 bytes per block
    
    # Build output text
    text = ""
    
    # Write unpacked arrays
    text += "// ===== UNPACKED ARRAYS =====\n"
    if is_input_2d:
        text += write_array_2d("input_unpacked", input_vals, M, K, SIGNED) + "\n"
        text += write_array_2d_binary("input_unpacked", input_vals, M, K, SIGNED)
    else:
        text += write_array("input_unpacked", input_vals, SIGNED) + "\n"
        text += write_array_binary("input_unpacked", input_vals, SIGNED)
    
    if is_weights_2d:
        text += write_array_2d("weights_unpacked", weights_vals, N, K, SIGNED) + "\n"
        text += write_array_2d_binary("weights_unpacked", weights_vals, N, K, SIGNED)
    else:
        text += write_array("weights_unpacked", weights_vals, SIGNED) + "\n"
        text += write_array_binary("weights_unpacked", weights_vals, SIGNED)
    
    # Write packed arrays
    text += "// ===== PACKED ARRAYS =====\n"
    if is_input_2d:
        text += write_array_2d("input_packed", input_packed, M, packed_cols, SIGNED) + "\n"
        text += write_array_2d_binary("input_packed", input_packed, M, packed_cols, SIGNED)
    else:
        text += write_array("input_packed", input_packed, SIGNED) + "\n"
        text += write_array_binary("input_packed", input_packed, SIGNED)
    
    if is_weights_2d:
        text += write_array_2d("weights_packed", weights_packed, N, packed_cols, SIGNED) + "\n"
        text += write_array_2d_binary("weights_packed", weights_packed, N, packed_cols, SIGNED)
    else:
        text += write_array("weights_packed", weights_packed, SIGNED) + "\n"
        text += write_array_binary("weights_packed", weights_packed, SIGNED)

    # Write 2-bit/4-bit arrays
    text += "// ===== 2-BIT ARRAYS =====\n"
    if is_input_2d:
        text += write_array_2d("input_2bit", input_2bit, M, K, SIGNED) + "\n"
        text += write_array_2d_binary("input_2bit", input_2bit, M, K, SIGNED)
        text += write_array_2d("input_4bit", input_4bit, M, K, SIGNED) + "\n"
        text += write_array_2d_binary("input_4bit", input_4bit, M, K, SIGNED)
        text += write_array_2d("input_8bit", input_vals, M, K, SIGNED) + "\n"
        text += write_array_2d_binary("input_8bit", input_vals, M, K, SIGNED)
    else:
        text += write_array("input_2bit", input_2bit, SIGNED) + "\n"
        text += write_array_binary("input_2bit", input_2bit, SIGNED)
        text += write_array("input_4bit", input_4bit, SIGNED) + "\n"
        text += write_array_binary("input_4bit", input_4bit, SIGNED)
        text += write_array("input_8bit", input_vals, SIGNED) + "\n"
        text += write_array_binary("input_8bit", input_vals, SIGNED)
    
    if is_weights_2d:
        text += write_array_2d("weights_2bit", weights_2bit, N, K, SIGNED) + "\n"
        text += write_array_2d_binary("weights_2bit", weights_2bit, N, K, SIGNED)
        text += write_array_2d("weights_4bit", weights_4bit, N, K, SIGNED) + "\n"
        text += write_array_2d_binary("weights_4bit", weights_4bit, N, K, SIGNED)
        text += write_array_2d("weights_8bit", weights_vals, N, K, SIGNED) + "\n"
        text += write_array_2d_binary("weights_8bit", weights_vals, N, K, SIGNED)
    else:
        text += write_array("weights_2bit", weights_2bit, SIGNED) + "\n"
        text += write_array_binary("weights_2bit", weights_2bit, SIGNED)
        text += write_array("weights_4bit", weights_4bit, SIGNED) + "\n"
        text += write_array_binary("weights_4bit", weights_4bit, SIGNED)
        text += write_array("weights_8bit", weights_vals, SIGNED) + "\n"
        text += write_array_binary("weights_8bit", weights_vals, SIGNED)
    
    # Write products
    text += write_product_array("product_2bit", prod_2bit, SIGNED)
    text += write_product_array_binary("product_2bit", prod_2bit, SIGNED, cols_per_line=N)
    text += write_product_array("product_4bit", prod_4bit, SIGNED)
    text += write_product_array_binary("product_4bit", prod_4bit, SIGNED, cols_per_line=N)
    text += write_product_array("product_8bit", prod_8bit, SIGNED)
    text += write_product_array_binary("product_8bit", prod_8bit, SIGNED, cols_per_line=N)

    Path(outfile).write_text(text)
    print(f"Diagnostic header written to {outfile} with SIGNED={SIGNED}, dims=({M},{K},{N})")
    if is_input_2d or is_weights_2d:
        print(f"  Input: {'2D' if is_input_2d else '1D'} {input_dims}, Weights: {'2D' if is_weights_2d else '1D'} {weights_dims}")
        print(f"  Packed dimensions: input[{M}][{packed_cols}], weights[{N}][{packed_cols}]")

    # Generate data.h from template
    template = Path("template_data.h").read_text()
    data_content = template.replace("{{M}}", str(M))
    data_content = data_content.replace("{{K}}", str(K))
    data_content = data_content.replace("{{N}}", str(N))
    data_content = data_content.replace("{{INPUT_ARRAY}}", write_array_compact("input", input_vals, SIGNED))
    data_content = data_content.replace("{{WEIGHTS_ARRAY}}", write_array_compact("weights", weights_vals, SIGNED))
    
    # Bias values (all zeros)
    bias_vals = ", ".join(["0"] * N)
    data_content = data_content.replace("{{BIAS_VALUES}}", bias_vals)
    
    # Golden values (8-bit product)
    golden_vals = ", ".join(str(int(v)) for v in prod_8bit.flatten())
    data_content = data_content.replace("{{GOLDEN_VALUES}}", golden_vals)
    
    # Path(datafile).write_text(data_content)
    # print(f"Data header written to {datafile}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments: generate random matrices
        main(infile=None, outfile="output.h", datafile="data.h")
    elif len(sys.argv) == 2:
        # One argument: input file
        main(infile=sys.argv[1], outfile="output.h", datafile="data.h")
    elif len(sys.argv) == 3:
        # Two arguments: input and output
        main(infile=sys.argv[1], outfile=sys.argv[2], datafile="data.h")
    elif len(sys.argv) == 4:
        # Three arguments: input, output, and data file
        main(infile=sys.argv[1], outfile=sys.argv[2], datafile=sys.argv[3])
    else:
        print(f"Usage: {sys.argv[0]} [input.h [output.h [data.h]]]")
        print("  No args: Generate random matrices with M_d, K_d, N_d from script")
        print("  input.h: Process existing header file")
        print("  output.h: Diagnostic output with packed/2bit/4bit arrays (default: output.h)")
        print("  data.h: Final data file from template (default: data.h)")
        sys.exit(1)

import numpy as np

def int8_to_binary_str(value):
    """Convert int8 value to 8-bit binary string representation."""
    # Handle two's complement for negative numbers
    if value < 0:
        value = (1 << 8) + value
    return format(value, '08b')

def pack_weights_bitplane(weights, output_file="out_real.h"):
    """
    Pack weights using bitplane representation while maintaining original array dimensions.
    Input: weights array of shape (M, K)
    Output: Generates header file with unpacked and binary representations
    """
    M, K = weights.shape
    total_elements = M * K
    
    # Flatten for processing
    weights_flat = weights.flatten()
    
    # Generate header file
    with open(output_file, 'w') as f:
        f.write("// ===== UNPACKED ARRAYS =====\n")
        
        # Write unpacked input array (flattened)
        f.write(f"int8_t input_unpacked[{total_elements}] = {{\n")
        for i in range(0, total_elements, 16):
            line_vals = weights_flat[i:i+16]
            f.write("    " + ", ".join(str(v) for v in line_vals) + ",\n")
        f.write("};\n")
        
        # Write binary representation
        f.write("// Binary representation of input_unpacked\n")
        f.write(f"int8_t input_unpacked_binary[{total_elements}] = {{\n")
        for i in range(0, total_elements, 16):
            line_vals = weights_flat[i:i+16]
            binary_vals = [f"0b{v & 0xFF:08b}" for v in line_vals]
            f.write("    " + ", ".join(binary_vals) + ",\n")
        f.write("};\n\n")
        
        # Write unpacked weights array (same as input in this context)
        f.write(f"int8_t weights_unpacked[{total_elements}] = {{\n")
        for i in range(0, total_elements, 16):
            line_vals = weights_flat[i:i+16]
            f.write("    " + ", ".join(str(v) for v in line_vals) + ",\n")
        f.write("};\n")
        
        # Write binary representation of weights
        f.write("// Binary representation of weights_unpacked\n")

def generate_input_header(input_data, weights_data, output_file="in_real.h"):
    """
    Generate input header file maintaining original 2D array dimensions.
    Input: input_data (M, K), weights_data (N, K)
    """
    M, K = input_data.shape
    N, K_w = weights_data.shape
    
    with open(output_file, 'w') as f:
        f.write("#ifndef __ARRAY_INT_8_1_32_1_H__\n")
        f.write("#define __ARRAY_INT_8_1_32_1_H__\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"#define M_d {M}\n")
        f.write(f"#define K_d {K}\n")
        f.write(f"#define N_d {N}\n\n")
        
        f.write("const int8_t zp_input = 0;\n")
        f.write("const int8_t zp_weights = 0;\n\n")
        
        # Write input array as 2D array
        f.write(f"int8_t input[{M}][{K}] = {{\n")
        for i in range(M):
            f.write("    {")
            f.write(", ".join(str(v) for v in input_data[i]))
            f.write("},\n")
        f.write("};\n\n")
        
        # Write weights array as 2D array
        f.write(f"int8_t weights[{N}][{K_w}] = {{\n")
        for i in range(N):
            f.write("    {")
            f.write(", ".join(str(v) for v in weights_data[i]))
            f.write("},\n")
        f.write("};\n\n")
        
        # Write bias array
        f.write(f"\nint32_t bias[N_d] = {{\n")
        f.write("    " + ", ".join(["0"] * N))
        f.write("\n};\n\n")
        
        # Placeholder for golden output
        f.write(f"int32_t golden[M_d * N_d] = {{\n")
        f.write("    -19908\n")
        f.write("};\n\n")
        
        f.write("#endif  // __ARRAY_INT_8_1_32_1_H__\n")

# Main execution
if __name__ == "__main__":
    # Load data from existing header file or generate sample data
    # For this example, create sample data matching the dimensions
    M, K, N = 10, 512, 10
    
    # Generate random sample data for demonstration
    np.random.seed(42)
    input_data = np.random.randint(-100, 100, size=(M, K), dtype=np.int8)
    weights_data = np.random.randint(-100, 100, size=(N, K), dtype=np.int8)
    
    # Generate header files
    pack_weights_bitplane(input_data, "out_real.h")
    generate_input_header(input_data, weights_data, "in_real.h")
    
    print(f"Generated header files with dimensions:")
    print(f"  input[{M}][{K}]")
    print(f"  weights[{N}][{K}]")