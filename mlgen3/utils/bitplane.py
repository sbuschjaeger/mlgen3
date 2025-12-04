"""
Bitplane packing utilities for quantized weights.

This module provides functions to pack 8-bit quantized weights into a bitplane-interleaved
format that allows extracting lower bit-width versions (2-bit, 4-bit) by reading fewer bytes.

The packing format interleaves bits from blocks of 16 values such that:
- 2-bit version: read first 2 bytes per block
- 4-bit version: read first 4 bytes per block  
- 8-bit version: read all 8 bytes per block
"""

import numpy as np
import re
from pathlib import Path


# ---------------- PARAMETERS ----------------
BLOCK_SIZE = 16  # Process 16 values at a time
SIGNED = True  # Set True for signed interpretation, False for unsigned


# ---------------- UTILITY FUNCTIONS ----------------
def detect_dims(text):
    """Extract M_d, K_d, N_d from header text (#define lines)."""
    def get_dim(name):
        match = re.search(rf'#define\s+{name}\s+(\d+)', text)
        return int(match.group(1)) if match else None
    
    return get_dim('M_d'), get_dim('K_d'), get_dim('N_d')


def pack_block_bitplanes(block16, signed=True):
    """
    Pack a block of 16 int8 values into bitplane-interleaved format.
    
    The output is arranged so that extracting the first N bytes gives
    an N-bit approximation of the original values.
    
    For 8-bit signed values packed into 16 bytes:
    - Bytes 0-1: bits 7-6 (MSBs) of all 16 values -> 2-bit precision
    - Bytes 2-3: bits 5-4 of all 16 values -> adds to 4-bit precision
    - Bytes 4-5: bits 3-2 of all 16 values -> adds to 6-bit precision
    - Bytes 6-7: bits 1-0 (LSBs) of all 16 values -> full 8-bit precision
    
    Args:
        block16: Array of 16 int8 values
        signed: Whether values are signed (default: True)
    
    Returns:
        Array of 16 packed uint8 values
    """
    block16 = np.array(block16, dtype=np.int8 if signed else np.uint8)
    if len(block16) != BLOCK_SIZE:
        raise ValueError(f"Block must have exactly {BLOCK_SIZE} values, got {len(block16)}")
    
    # Convert to unsigned for bit manipulation
    if signed:
        unsigned = block16.view(np.uint8)
    else:
        unsigned = block16.astype(np.uint8)
    
    packed = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    
    # Pack bits in pairs (2 bits at a time) for efficient extraction
    # Output byte layout:
    # packed[0]: bit7 of values 0-7 (one bit each)
    # packed[1]: bit7 of values 8-15 (one bit each)
    # packed[2]: bit6 of values 0-7
    # packed[3]: bit6 of values 8-15
    # ... and so on for bits 5,4,3,2,1,0
    
    for bit_pair in range(4):  # 4 pairs of bits (7-6, 5-4, 3-2, 1-0)
        high_bit = 7 - (bit_pair * 2)
        low_bit = high_bit - 1
        
        # Pack high bit of the pair
        for val_idx in range(8):
            bit = (unsigned[val_idx] >> high_bit) & 1
            packed[bit_pair * 4] |= (bit << (7 - val_idx))
        for val_idx in range(8, 16):
            bit = (unsigned[val_idx] >> high_bit) & 1
            packed[bit_pair * 4 + 1] |= (bit << (15 - val_idx))
        
        # Pack low bit of the pair
        for val_idx in range(8):
            bit = (unsigned[val_idx] >> low_bit) & 1
            packed[bit_pair * 4 + 2] |= (bit << (7 - val_idx))
        for val_idx in range(8, 16):
            bit = (unsigned[val_idx] >> low_bit) & 1
            packed[bit_pair * 4 + 3] |= (bit << (15 - val_idx))
    
    return packed.view(np.int8) if signed else packed


def unpack_block_bitplanes(packed16, target_bits=8, signed=True):
    """
    Unpack a block of 16 packed bytes back to original values at specified precision.
    
    Args:
        packed16: Array of 16 packed bytes
        target_bits: Target bit precision (2, 4, 6, or 8)
        signed: Whether to return signed values
    
    Returns:
        Array of 16 int8 (or uint8) values at the specified precision
    """
    packed16 = np.array(packed16, dtype=np.uint8)
    if len(packed16) != BLOCK_SIZE:
        raise ValueError(f"Packed block must have exactly {BLOCK_SIZE} values")
    
    unpacked = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    
    # Determine how many bit pairs to extract based on target_bits
    num_bit_pairs = (target_bits + 1) // 2  # 2->1, 4->2, 6->3, 8->4
    
    for bit_pair in range(num_bit_pairs):
        high_bit = 7 - (bit_pair * 2)
        low_bit = high_bit - 1
        
        # Unpack high bit
        for val_idx in range(8):
            bit = (packed16[bit_pair * 4] >> (7 - val_idx)) & 1
            unpacked[val_idx] |= (bit << high_bit)
        for val_idx in range(8, 16):
            bit = (packed16[bit_pair * 4 + 1] >> (15 - val_idx)) & 1
            unpacked[val_idx] |= (bit << high_bit)
        
        # Unpack low bit
        for val_idx in range(8):
            bit = (packed16[bit_pair * 4 + 2] >> (7 - val_idx)) & 1
            unpacked[val_idx] |= (bit << low_bit)
        for val_idx in range(8, 16):
            bit = (packed16[bit_pair * 4 + 3] >> (15 - val_idx)) & 1
            unpacked[val_idx] |= (bit << low_bit)
    
    if signed:
        return unpacked.view(np.int8)
    return unpacked


def pack_weights_bitplane(weights_flat, signed=True):
    """
    Pack a flat array of int8 weights into bitplane-interleaved format.
    
    The array is processed in blocks of 16 values. If the array length
    is not a multiple of 16, it is padded with zeros.
    
    Args:
        weights_flat: 1D numpy array of int8 weights
        signed: Whether weights are signed (default: True)
    
    Returns:
        Packed numpy array of same length (padded to multiple of 16)
    """
    weights_flat = np.array(weights_flat, dtype=np.int8 if signed else np.uint8).flatten()
    
    # Pad to multiple of BLOCK_SIZE
    original_len = len(weights_flat)
    pad_len = (BLOCK_SIZE - (original_len % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len > 0:
        weights_flat = np.concatenate([weights_flat, np.zeros(pad_len, dtype=weights_flat.dtype)])
    
    num_blocks = len(weights_flat) // BLOCK_SIZE
    packed = np.zeros_like(weights_flat)
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = weights_flat[start:end]
        packed[start:end] = pack_block_bitplanes(block, signed=signed)
    
    return packed[:original_len] if pad_len > 0 else packed


def pack_weights_2d_bitplane(weights_2d, signed=True):
    """
    Pack a 2D weight matrix into bitplane-interleaved format.
    
    Each row is packed separately, maintaining the row structure.
    
    Args:
        weights_2d: 2D numpy array of shape [rows, cols]
        signed: Whether weights are signed (default: True)
    
    Returns:
        Packed 2D numpy array of same shape
    """
    weights_2d = np.array(weights_2d, dtype=np.int8 if signed else np.uint8)
    rows, cols = weights_2d.shape
    
    packed = np.zeros_like(weights_2d)
    
    for row_idx in range(rows):
        row = weights_2d[row_idx]
        # Pad row to multiple of BLOCK_SIZE
        pad_len = (BLOCK_SIZE - (cols % BLOCK_SIZE)) % BLOCK_SIZE
        if pad_len > 0:
            row_padded = np.concatenate([row, np.zeros(pad_len, dtype=row.dtype)])
        else:
            row_padded = row
        
        # Pack each block in the row
        num_blocks = len(row_padded) // BLOCK_SIZE
        packed_row = np.zeros_like(row_padded)
        
        for block_idx in range(num_blocks):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = row_padded[start:end]
            packed_row[start:end] = pack_block_bitplanes(block, signed=signed)
        
        packed[row_idx] = packed_row[:cols]
    
    return packed


def rearrange_bitplanes(values, cols):
    """
    Rearrange a flat array of int8 values into bitplane-packed format.
    Processes in blocks of 16 values.
    
    Args:
        values: flat numpy array of int8 values
        cols: number of columns (typically 16 for our use case)
    
    Returns:
        Packed numpy array
    """
    values = np.array(values, dtype=np.int8).flatten()
    return pack_weights_bitplane(values, signed=True)


def extract_2bit_matrix(packed, col_start, signed=True):
    """Extract 2-bit values from packed representation."""
    # Unpack with 2-bit precision
    return unpack_block_bitplanes(packed, target_bits=2, signed=signed)


def extract_4bit_matrix(packed, col_start, signed=True):
    """Extract 4-bit values from packed representation."""
    return unpack_block_bitplanes(packed, target_bits=4, signed=signed)


def parse_flat_array(name, text):
    """Parse a C array from header text."""
    pattern = rf'{name}\s*\[[^\]]*\]\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    values_str = match.group(1)
    values = [int(x.strip()) for x in values_str.split(',') if x.strip() and x.strip() != '']
    return np.array(values, dtype=np.int8)


def write_array(name, values, signed=True, indent="    "):
    """
    Format a numpy array as a C array declaration.
    
    Args:
        name: C variable name
        values: numpy array of values
        signed: whether values are signed int8
        indent: indentation string
    
    Returns:
        C code string for array declaration
    """
    dtype_str = "int8_t" if signed else "uint8_t"
    values = np.array(values).flatten()
    
    # Format values in rows of 16
    lines = []
    for i in range(0, len(values), 16):
        row = values[i:i+16]
        row_str = ", ".join(str(int(v)) for v in row)
        lines.append(f"{indent}{row_str}")
    
    values_str = ",\n".join(lines)
    return f"static const {dtype_str} {name}[{len(values)}] = {{\n{values_str}\n}};\n"


def write_2d_array(name, values, rows, cols, signed=True, indent="    "):
    """
    Format a 2D numpy array as a C array declaration.
    """
    dtype_str = "int8_t" if signed else "uint8_t"
    values = np.array(values).reshape(rows, cols)
    
    lines = []
    for i in range(rows):
        row = values[i]
        # Format in chunks of 16 for readability
        row_parts = []
        for j in range(0, cols, 16):
            chunk = row[j:min(j + 16, cols)]
            chunk_str = ", ".join(str(int(v)) for v in chunk)
            row_parts.append(chunk_str)
        row_str = ", ".join(row_parts)
        lines.append(f"{indent}{{{row_str}}}")
    
    values_str = ",\n".join(lines)
    return f"static const {dtype_str} {name}[{rows}][{cols}] = {{\n{values_str}\n}};\n"


def write_product_array(name, mat, signed=False):
    """Write a product array (int32)."""
    dtype_str = "int32_t"
    values = np.array(mat).flatten()
    
    lines = []
    for i in range(0, len(values), 16):
        row = values[i:i+16]
        row_str = ", ".join(str(int(v)) for v in row)
        lines.append(f"    {row_str}")
    
    values_str = ",\n".join(lines)
    return f"static const {dtype_str} {name}[{len(values)}] = {{\n{values_str}\n}};\n"


# ---------------- MAIN SCRIPT ----------------
def main(infile, outfile):
    """Main function to process input file and generate packed output."""
    with open(infile, 'r') as f:
        text = f.read()
    
    M, K, N = detect_dims(text)
    print(f"Detected dimensions: M={M}, K={K}, N={N}")
    
    # Parse arrays
    input_arr = parse_flat_array('input', text)
    weights_arr = parse_flat_array('weights', text)
    
    output = f"""#ifndef __PACKED_ARRAYS_H__
#define __PACKED_ARRAYS_H__

#include <stdint.h>

"""
    if M:
        output += f"#define M_d {M}\n"
    if K:
        output += f"#define K_d {K}\n"
    if N:
        output += f"#define N_d {N}\n"
    output += "\n"
    
    if input_arr is not None:
        print(f"Input array: {len(input_arr)} elements")
        packed_input = pack_weights_bitplane(input_arr, signed=True)
        print(f"Packed input: {len(packed_input)} elements")
        
        output += "/* Packed input array (bitplane-interleaved) */\n"
        output += write_array('input_packed', packed_input, signed=True)
        output += "\n"
    
    if weights_arr is not None:
        print(f"Weights array: {len(weights_arr)} elements")
        packed_weights = pack_weights_bitplane(weights_arr, signed=True)
        print(f"Packed weights: {len(packed_weights)} elements")
        
        output += "/* Packed weights array (bitplane-interleaved) */\n"
        output += write_array('weights_packed', packed_weights, signed=True)
        output += "\n"
    
    output += "#endif  /* __PACKED_ARRAYS_H__ */\n"
    
    with open(outfile, 'w') as f:
        f.write(output)
    
    print(f"Output written to {outfile}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python bitplane.py <input.h> <output.h>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
