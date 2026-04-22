# mlgen3/mlgen3/implementations/neuralnet/matquant_c/dyadic.py
"""
Dyadic arithmetic conversion for middle-plate BN fusion.

STATUS: Scaffolded — currently a float passthrough.
When MatQuant training adds fused-BN, implement to_dyadic() to convert
float scale into (multiplier, shift) pairs:
  output = clamp((accumulator * multiplier) >> shift, qmin, qmax)
This eliminates float ops from the FPGA critical path entirely.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DyadicParams:
    """
    Holds either float or dyadic (multiplier >> shift) BN+requant params.
    is_dyadic=False -> float ops used (current default).
    """
    scale: np.ndarray         # float32 if not dyadic; int32 multiplier if dyadic
    bias: np.ndarray          # float32 offset (used when is_dyadic=False)
    shift: np.ndarray | None  # int8 shift amounts (only when is_dyadic=True)
    is_dyadic: bool = False


def float_bn_params(scale_out: np.ndarray, bias_out: np.ndarray) -> DyadicParams:
    """Return float-mode DyadicParams. Used for all BN layers currently."""
    return DyadicParams(
        scale=scale_out.astype(np.float32),
        bias=bias_out.astype(np.float32),
        shift=None,
        is_dyadic=False,
    )


def to_dyadic(scale: np.ndarray, n_bits: int = 32) -> DyadicParams:
    """
    Convert float scale to (multiplier, shift). NOT YET IMPLEMENTED.

    Future formula for each s:
      shift = floor(log2(1/s)) + (n_bits - 1)
      multiplier = round(s * 2**shift)
    """
    # Passthrough until dyadic training support is added in MatQuant
    return DyadicParams(
        scale=scale.astype(np.float32),
        bias=np.zeros_like(scale, dtype=np.float32),
        shift=None,
        is_dyadic=False,
    )
