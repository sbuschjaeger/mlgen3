import re
from pathlib import Path
import numpy as np

# ---------------- PARAMETERS ----------------
SIGNED = True  # Set True for signed interpretation, False for unsigned

# ---------------- UTILITY FUNCTIONS ----------------
def detect_dims(text):
    """Extract M_d, K_d, N_d from header text (#define lines)."""
    def get_dim(name):
        m = re.search(rf"#define\s+{name}\s+(\d+)", text)
        return int(m.group(1)) if m else None
    return get_dim("M_d"), get_dim("K_d"), get_dim("N_d")

def pack_block_2bitplanes(block16):
    """Pack 16 uint8 values into 4 groups of 2-bit planes (MSB-first)."""
    assert len(block16) == 16
    packed = []
    for group in range(3, -1, -1):  # bits [7:6], [5:4], [3:2], [1:0]
        shift = group * 2
        word = 0
        for i, val in enumerate(block16):
            two_bits = (val >> shift) & 0x3
            word |= (two_bits << (2 * i))
        for b in range(4):
            packed.append((word >> (8 * b)) & 0xFF)
    return packed

def rearrange_bitplanes(values, cols):
    """Rearrange row-major values into packed 2-bit planes per 16-element block."""
    output = []
    for row_start in range(0, len(values), cols):
        row = values[row_start:row_start+cols]
        for blk_start in range(0, cols, 16):
            output.extend(pack_block_2bitplanes(row[blk_start:blk_start+16]))
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
    """Extract flat array values from a C header file."""
    pattern = re.compile(rf"(?:u?int8_t|int8_t)\s+{name}\s*\[\s*\d+\s*\]\s*=\s*{{(.*?)}};", re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Array {name} not found")
    numbers = re.findall(r"-?\d+", match.group(1))
    return list(map(int, numbers))

def write_array(name, values, signed=False):
    ctype = "int8_t" if signed else "uint8_t"
    lines = [", ".join(str(int(v)) for v in values[i:i+16]) for i in range(0, len(values), 16)]
    return f"{ctype} {name}[{len(values)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n\n"

def write_product_array(name, mat, signed=False):
    ctype = "int32_t" if signed else "uint32_t"
    flat = mat.flatten()
    lines = [", ".join(str(int(v)) for v in flat[i:i+16]) for i in range(0, len(flat), 16)]
    return f"{ctype} {name}[{len(flat)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n\n"

# ---------------- MAIN SCRIPT ----------------
def main(infile, outfile):
    text = Path(infile).read_text()
    M, K, N = detect_dims(text)
    if not all([M, K, N]):
        raise ValueError("Could not detect M_d, K_d, N_d from header")

    input_vals   = parse_flat_array("input", text)
    weights_vals = parse_flat_array("weights", text)

    # Pack into bitplanes
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
            weights_2bit.extend(extract_2bit_matrix(weights_packed, base, signed=SIGNED))
            weights_4bit.extend(extract_4bit_matrix(weights_packed, base, signed=SIGNED))

    # Convert to matrices
    dtype8  = np.int8 if SIGNED else np.uint8
    dtype32 = np.int32 if SIGNED else np.uint32
    in2_mat = np.array(input_2bit, dtype=dtype8).reshape(M, K)
    in4_mat = np.array(input_4bit, dtype=dtype8).reshape(M, K)
    wt2_mat = np.array(weights_2bit, dtype=dtype8).reshape(N, K)
    wt4_mat = np.array(weights_4bit, dtype=dtype8).reshape(N, K)

    # Compute products
    prod_2bit = in2_mat.astype(dtype32) @ wt2_mat.T.astype(dtype32)
    prod_4bit = in4_mat.astype(dtype32) @ wt4_mat.T.astype(dtype32)

    # Replace original input/weights with packed
    text = re.sub(r"(?:u?int8_t|int8_t)\s+input\s*\[\s*\d+\s*\]\s*=\s*{.*?};",
                  write_array("input", input_packed, SIGNED), text, flags=re.S)
    text = re.sub(r"(?:u?int8_t|int8_t)\s+weights\s*\[\s*\d+\s*\]\s*=\s*{.*?};",
                  write_array("weights", weights_packed, SIGNED), text, flags=re.S)

    # Append 2-bit/4-bit matrices and products
    text += "\n" + write_array("input_2bit", input_2bit, SIGNED)
    text += write_array("input_4bit", input_4bit, SIGNED)
    text += write_array("weights_2bit", weights_2bit, SIGNED)
    text += write_array("weights_4bit", weights_4bit, SIGNED)
    text += write_product_array("product_2bit", prod_2bit, SIGNED)
    text += write_product_array("product_4bit", prod_4bit, SIGNED)

    Path(outfile).write_text(text)
    print(f"Header written to {outfile} with SIGNED={SIGNED}, dims=({M},{K},{N})")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.h output.h")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
