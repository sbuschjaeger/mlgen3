# mlgen3/mlgen3/implementations/neuralnet/matquant_c/graph_extractor.py
"""
Extract an ordered computation graph from a MatQuant model using torch.fx.

Returns a list of NodeInfo dicts in topological order. Each NodeInfo carries
the layer metadata the weight exporter and template engine need.

The model must be in eval mode with set_active_bits() called before tracing,
so all dynamic branches collapse to static code paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn

# Allow importing matquant from the parent repo root.
# File location: .../matquant/mlgen3/mlgen3/implementations/neuralnet/matquant_c/graph_extractor.py
# parents[5] resolves to .../matquant/
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from matquant.quantization.quant_layers import QuantConv2d, QuantLinear

NodeInfo = dict[str, Any]


def extract_graph(model: nn.Module) -> list[NodeInfo]:
    """
    Trace *model* with torch.fx and return a topologically-ordered
    list of NodeInfo dicts, one per graph node.
    """
    module_map: dict[str, nn.Module] = dict(model.named_modules())
    traced: fx.GraphModule = fx.symbolic_trace(model)

    nodes: list[NodeInfo] = []
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            nodes.append({"op_type": "input", "name": node.name, "inputs": []})

        elif node.op == "output":
            arg = node.args[0]
            inputs = (
                [_arg_name(a) for a in arg if isinstance(a, fx.Node)]
                if isinstance(arg, (list, tuple))
                else [_arg_name(arg)]
            )
            nodes.append({"op_type": "output", "name": node.name, "inputs": inputs})

        elif node.op == "call_module":
            mod = module_map.get(node.target)
            nodes.append(_module_node(node, mod))

        elif node.op == "call_function":
            nodes.append(_function_node(node))

        elif node.op == "call_method":
            nodes.append({
                "op_type": f"method_{node.target}",
                "name": node.name,
                "inputs": [_arg_name(a) for a in node.args if isinstance(a, fx.Node)],
            })

        elif node.op == "get_attr":
            nodes.append({"op_type": "get_attr", "name": node.name,
                          "target": node.target, "inputs": []})

    return nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arg_name(a: Any) -> str:
    return a.name if isinstance(a, fx.Node) else str(a)


def _input_names(node: fx.Node) -> list[str]:
    return [_arg_name(a) for a in node.args if isinstance(a, fx.Node)]


def _quant_fields(mod: Any) -> dict:
    """Extract quantization metadata from a _QuantMixin layer."""
    return {
        "is_quantized": True,
        "weight_bits": getattr(mod, "weight_bits", 8),
        "use_slicing": getattr(mod, "use_slicing", False),
        "weight_signed": getattr(mod, "weight_signed", True),
        "per_channel_weight": getattr(mod, "per_channel_weight", True),
        "weight_symmetric": getattr(mod, "weight_symmetric", True),
        "weight_clip_method": getattr(mod, "weight_clip_method", "minmax"),
        "weight_percentile": getattr(mod, "weight_percentile", 99.9),
        "use_ema": getattr(mod, "use_ema", False),
        "_module_ref": mod,
    }


def _plain_fields(mod: Any) -> dict:
    return {
        "is_quantized": False,
        "weight_bits": 8,
        "use_slicing": False,
        "weight_signed": True,
        "per_channel_weight": True,
        "weight_symmetric": True,
        "weight_clip_method": "minmax",
        "weight_percentile": 99.9,
        "use_ema": False,
        "_module_ref": mod,
    }


def _module_node(node: fx.Node, mod: nn.Module | None) -> NodeInfo:
    base = {
        "name": node.name,
        "module_path": node.target,
        "inputs": _input_names(node),
    }

    if isinstance(mod, (QuantConv2d, nn.Conv2d)):
        fields = _quant_fields(mod) if isinstance(mod, QuantConv2d) else _plain_fields(mod)
        ks = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,) * 2
        st = mod.stride if isinstance(mod.stride, tuple) else (mod.stride,) * 2
        pd = mod.padding if isinstance(mod.padding, tuple) else (mod.padding,) * 2
        dl = mod.dilation if isinstance(mod.dilation, tuple) else (mod.dilation,) * 2
        return {**base, **fields, "op_type": "conv2d",
                "in_channels": mod.in_channels, "out_channels": mod.out_channels,
                "kernel_size": ks, "stride": st, "padding": pd, "dilation": dl,
                "groups": mod.groups, "has_bias": mod.bias is not None}

    if isinstance(mod, (QuantLinear, nn.Linear)):
        fields = _quant_fields(mod) if isinstance(mod, QuantLinear) else _plain_fields(mod)
        return {**base, **fields, "op_type": "linear",
                "in_features": mod.in_features, "out_features": mod.out_features,
                "has_bias": mod.bias is not None}

    if isinstance(mod, nn.BatchNorm2d):
        return {**base, "op_type": "batch_norm",
                "num_features": mod.num_features, "eps": mod.eps,
                "affine": mod.affine, "_module_ref": mod}

    if isinstance(mod, (nn.ReLU, nn.ReLU6)):
        return {**base, "op_type": "relu"}

    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return {**base, "op_type": "global_avg_pool", "output_size": mod.output_size}

    if isinstance(mod, nn.MaxPool2d):
        return {**base, "op_type": "maxpool2d",
                "kernel_size": mod.kernel_size, "stride": mod.stride,
                "padding": mod.padding}

    if isinstance(mod, nn.Identity):
        return {**base, "op_type": "identity"}

    return {**base, "op_type": f"unknown_{type(mod).__name__}", "_module_ref": mod}


def _function_node(node: fx.Node) -> NodeInfo:
    fn = node.target
    fn_name = getattr(fn, "__name__", str(fn))
    inputs = _input_names(node)

    if fn is torch.add or fn_name in ("add", "iadd"):
        return {"op_type": "add", "name": node.name, "inputs": inputs}
    if fn is torch.flatten or fn_name == "flatten":
        return {"op_type": "flatten", "name": node.name, "inputs": inputs}

    return {"op_type": f"fn_{fn_name}", "name": node.name, "inputs": inputs}
