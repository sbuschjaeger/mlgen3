# mlgen3/tests/matquant_c/test_graph_extractor.py
import pytest
import torch
from mlgen3.implementations.neuralnet.matquant_c.graph_extractor import (
    extract_graph,
    NodeInfo,
)


def test_extract_returns_list(model_int8):
    nodes = extract_graph(model_int8)
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_all_entries_are_dicts_with_op_type(model_int8):
    nodes = extract_graph(model_int8)
    for n in nodes:
        assert isinstance(n, dict)
        assert "op_type" in n
        assert "name" in n


def test_has_conv2d_nodes(model_int8):
    nodes = extract_graph(model_int8)
    assert any(n["op_type"] == "conv2d" for n in nodes)


def test_has_batch_norm_nodes(model_int8):
    nodes = extract_graph(model_int8)
    assert any(n["op_type"] == "batch_norm" for n in nodes)


def test_has_add_node_for_residual(model_int8):
    nodes = extract_graph(model_int8)
    assert any(n["op_type"] == "add" for n in nodes)


def test_has_global_avg_pool(model_int8):
    nodes = extract_graph(model_int8)
    assert any(n["op_type"] == "global_avg_pool" for n in nodes)


def test_has_linear_node(model_int8):
    nodes = extract_graph(model_int8)
    assert any(n["op_type"] == "linear" for n in nodes)


def test_topological_order(model_int8):
    """Every node's inputs must appear before it in the list."""
    nodes = extract_graph(model_int8)
    seen = set()
    for n in nodes:
        for inp in n.get("inputs", []):
            assert inp in seen, f"Input '{inp}' of '{n['name']}' appears before its producer"
        seen.add(n["name"])


def test_conv_node_has_required_fields(model_int8):
    nodes = extract_graph(model_int8)
    conv = next(n for n in nodes if n["op_type"] == "conv2d")
    required = (
        "name", "op_type", "module_path", "inputs",
        "in_channels", "out_channels", "kernel_size",
        "stride", "padding", "groups",
        "weight_bits", "use_slicing", "is_quantized",
        "weight_signed", "per_channel_weight", "weight_symmetric",
        "weight_clip_method",
    )
    for field in required:
        assert field in conv, f"Conv node missing field: '{field}'"


def test_resnet18_node_count_in_expected_range(model_int8):
    """ResNet18 traced graph should have 50-120 nodes."""
    nodes = extract_graph(model_int8)
    assert 50 < len(nodes) < 120, f"Unexpected node count: {len(nodes)}"


def test_module_ref_present_for_conv(model_int8):
    nodes = extract_graph(model_int8)
    conv = next(n for n in nodes if n["op_type"] == "conv2d")
    assert "_module_ref" in conv
    assert conv["_module_ref"] is not None
