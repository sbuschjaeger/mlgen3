# mlgen3/tests/matquant_c/conftest.py
"""
Shared fixtures for MatQuant C export tests.

Set env vars before running:
  MATQUANT_TEST_CKPT=<path>/model.pt
  MATQUANT_TEST_CFG=<path>/resnet18_cifar100_new212-ray2.yaml
  pytest mlgen3/tests/matquant_c/ -v

Both paths are relative to the repo root (matquant/).
"""
import os
import sys
import pytest
import torch
import yaml
from pathlib import Path

# Allow importing from the matquant package (repo root sibling of mlgen3/)
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../matquant/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow importing the mlgen3 package itself.
# The repo layout has an orphan mlgen3/__init__.py that shadows the real
# package (mlgen3/mlgen3/) when only _REPO_ROOT is on sys.path.
# Inserting the mlgen3 subdirectory at position 0 makes the real package win.
_MLGEN3_ROOT = _REPO_ROOT / "mlgen3"
if str(_MLGEN3_ROOT) not in sys.path:
    sys.path.insert(0, str(_MLGEN3_ROOT))

from matquant.config import MatQuantConfig
from matquant.models.model_factory import build_model
from matquant.quantization.quant_wrapper import quantize_model, set_active_bits


def _load_config(cfg_path: str) -> MatQuantConfig:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return MatQuantConfig(**raw)


@pytest.fixture(scope="session")
def cfg() -> MatQuantConfig:
    path = os.environ.get(
        "MATQUANT_TEST_CFG",
        str(_REPO_ROOT / "configs/resnet18_cifar100/resnet18_cifar100_new212-ray2.yaml"),
    )
    return _load_config(path)


@pytest.fixture(scope="session")
def model_int8(cfg) -> torch.nn.Module:
    """ResNet18 loaded from checkpoint, quantized, eval mode, 8-bit active."""
    ckpt_path = os.environ.get(
        "MATQUANT_TEST_CKPT",
        str(_REPO_ROOT / "models-best/resnet18_cifar100_new212-ray2/kd/model.pt"),
    )
    model = build_model(cfg.model, cfg.dataset)
    model = quantize_model(model, cfg.quantization)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    model.eval()
    set_active_bits(model, 8)
    return model


@pytest.fixture(scope="session")
def dummy_input() -> torch.Tensor:
    """Single CIFAR-100 image (1, 3, 32, 32) with values in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 32, 32)
