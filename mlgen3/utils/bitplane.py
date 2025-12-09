"""
Bitplane packing utilities for weight compression.

Packing strategy for 16 values:
- Extract 2-bit slices from each value (bits [7:6], [5:4], [3:2], [1:0])
- For each 2-bit slice group:
  - Pack bits from values 0-3 into byte 0 (value 0 at MSB, value 3 at LSB)
  - Pack bits from values 4-7 into byte 1
  - Pack bits from values 8-11 into byte 2
  - Pack bits from values 12-15 into byte 3
- This creates 16 packed bytes from 16 original bytes (4 bytes per 2-bit slice group)
"""

import numpy as np


def pack_weights_bitplane(weights_1d, signed=True):
    """
    Pack a 1D array of weights using bitplane interleaving strategy.
    Works on 16-value blocks.
    
    Args:
        weights_1d: 1D numpy array of int8 weights
        signed: Whether weights are signed
        
    Returns:
        Packed 1D numpy array
    """
    weights_1d = np.array(weights_1d).flatten()
    
    # Pad to multiple of 16
    pad_size = (16 - len(weights_1d) % 16) % 16
    if pad_size > 0:
        weights_1d = np.pad(weights_1d, (0, pad_size), mode='constant')
    
    packed = []
    for i in range(0, len(weights_1d), 16):
        block = weights_1d[i:i+16]
        packed.extend(pack_block_bitplane(block, signed))
    
    return np.array(packed[:len(weights_1d) - pad_size] if pad_size > 0 else packed)


def pack_weights_2d_bitplane(weights_2d, signed=True):
    """
    Pack a 2D array of weights using bitplane interleaving strategy.
    Works row-by-row on 16-value blocks.
    
    Args:
        weights_2d: 2D numpy array of int8 weights
        signed: Whether weights are signed
        
    Returns:
        Packed 2D numpy array with same shape
    """
    rows, cols = weights_2d.shape
    
    # Pad columns to multiple of 16
    pad_cols = (16 - cols % 16) % 16
    if pad_cols > 0:
        weights_2d = np.pad(weights_2d, ((0, 0), (0, pad_cols)), mode='constant')
    
    packed_rows = []
    for row in weights_2d:
        packed_row = []
        for i in range(0, len(row), 16):
            block = row[i:i+16]
            packed_row.extend(pack_block_bitplane(block, signed))
        packed_rows.append(packed_row[:cols] if pad_cols > 0 else packed_row)
    
    return np.array(packed_rows)


def pack_block_bitplane(block, signed=True):
    """
    Pack a block of 16 int8 values into bitplane-interleaved format.
    
    Args:
        block: List or array of 16 int8 values
        signed: If True, handle signed int8 values; if False, handle unsigned uint8
    
    Returns:
        List of 16 packed bytes (bitplane-interleaved format)
    """
    assert len(block) == 16, f"Block must have exactly 16 elements, got {len(block)}"
    
    # Convert to appropriate dtype
    if signed:
        block_array = np.array(block, dtype=np.int8)
    else:
        block_array = np.array(block, dtype=np.uint8)
    
    packed = []
    
    # Process each bitplane (MSB to LSB: bits [7:6], [5:4], [3:2], [1:0])
    for group in range(3, -1, -1):
        shift = group * 2
        
        # Process 4 bytes per bitplane (4 values per byte)
        for byte_idx in range(4):
            byte_val = 0
            for i in range(4):
                val_idx = byte_idx * 4 + i
                
                # Extract 2-bit value - work with Python int to avoid overflow
                val = int(block_array[val_idx])
                
                # For signed values, convert to unsigned representation for bit operations
                if signed and val < 0:
                    val = val + 256  # Convert signed to unsigned representation
                
                two_bits = (val >> shift) & 0x3
                
                # Pack with first value at MSB (bits 7:6), last at LSB (bits 1:0)
                byte_val |= (two_bits << (2 * (3 - i)))
            
            # Convert back to signed if needed
            if signed and byte_val >= 128:
                byte_val = byte_val - 256
            
            packed.append(byte_val)
    
    return packed


def unpack_weights_bitplane(packed_1d, signed=True):
    """
    Unpack a 1D array of bitplane-packed weights.
    
    Args:
        packed_1d: 1D numpy array of packed int8 values
        signed: Whether weights are signed
        
    Returns:
        Unpacked 1D numpy array
    """
    packed_1d = np.array(packed_1d).flatten()
    
    unpacked = []
    for i in range(0, len(packed_1d), 16):
        block = packed_1d[i:i+16]
        unpacked.extend(unpack_block_bitplane(block, signed))
    
    return np.array(unpacked)


def unpack_block_bitplane(block16, signed=True):
    """
    Unpack 16 bytes back to 16 int8 values.
    
    Args:
        block16: Array of 16 packed bytes
        signed: Whether values are signed
        
    Returns:
        List of 16 unpacked int8 values
    """
    assert len(block16) == 16, "Block must contain exactly 16 values"
    
    # Convert to unsigned for bit operations
    if signed:
        block16 = np.array(block16, dtype=np.int8).astype(np.uint8)
    else:
        block16 = np.array(block16, dtype=np.uint8)
    
    # Initialize 16 output values
    values = np.zeros(16, dtype=np.uint8)
    
    # Process each 2-bit slice group
    for group in range(3, -1, -1):  # bits [7:6], [5:4], [3:2], [1:0]
        shift = group * 2
        byte_offset = (3 - group) * 4  # Which 4 bytes in the packed block
        
        # Extract bits from the 4 bytes for this group
        for byte_idx in range(4):
            packed_byte = block16[byte_offset + byte_idx]
            for i in range(4):
                val_idx = byte_idx * 4 + i
                two_bits = (packed_byte >> (2 * (3 - i))) & 0x3
                values[val_idx] |= (two_bits << shift)
    
    # Convert back to signed if needed
    if signed:
        values = values.astype(np.int8)
    
    return values.tolist()


def format_binary_int8(value, signed=True):
    """Format an int8 value as binary string."""
    if signed:
        # Handle negative values using two's complement
        if value < 0:
            value = (1 << 8) + value
    return f"0b{value:08b}"


def format_binary_array(values, signed=True, items_per_line=16):
    """Format array of int8 values as binary strings."""
    lines = []
    for i in range(0, len(values), items_per_line):
        chunk = values[i:min(i + items_per_line, len(values))]
        chunk_str = ", ".join(format_binary_int8(int(v), signed) for v in chunk)
        lines.append(f"    {chunk_str}")
    return ",\n".join(lines)
