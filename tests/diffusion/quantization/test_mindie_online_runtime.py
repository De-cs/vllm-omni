# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online MXFP8 lifecycle regression tests with a tensor-only fake Runtime."""

import sys
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, NPUMxfp8OnlineLinearMethod

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def runtime(monkeypatch):
    @dataclass
    class State:
        weight: torch.Tensor
        weight_scale: torch.Tensor

    def prepare(weight, *, dtype):
        assert weight.device.type != "meta"
        # Distinct storage and shape ensure registration/offload checks are meaningful.
        return State(weight.detach().to(dtype).T.contiguous(), torch.ones(1, weight.shape[0], 2, dtype=dtype))

    def forward(x, state, bias=None, *, output_dtype=None):
        out = x @ state.weight
        return out if bias is None else out + bias

    instance = Mock()
    instance.prepare.side_effect = prepare
    instance.forward.side_effect = forward
    mod = ModuleType("mindiesd.layers.quant_linear")
    mod.MXFP8OnlineLinearRuntime = Mock(return_value=instance)
    mod.MXFP8LinearState = State
    for name in ("mindiesd", "mindiesd.layers"):
        parent = ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return instance


def make_linear(runtime):
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.bfloat16).reshape(3, 8), requires_grad=False)
    method = NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config())
    return method, layer


def test_prepare_once_and_current_storage_after_offload(runtime):
    method, layer = make_linear(runtime)
    original = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    runtime.prepare.assert_called_once()
    assert set(dict(layer.named_parameters())) == {"weight", "weight_scale"}
    assert layer.weight.shape == (8, 3)
    # Model an offloader replacing storage; stale Runtime state would retain old values.
    layer.weight = torch.nn.Parameter(layer.weight.detach().clone() + 1, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(layer.weight_scale.detach().clone() + 2, requires_grad=False)
    x = torch.ones(2, 5, 8, dtype=torch.bfloat16)
    bias = torch.ones(3, dtype=torch.bfloat16)
    out = method.apply(layer, x, bias)
    torch.testing.assert_close(out, x @ (original.T + 1) + bias)
    state = runtime.forward.call_args.args[1]
    assert state.weight is layer.weight and state.weight_scale is layer.weight_scale
    assert runtime.forward.call_args.args[0].shape == (10, 8)
    assert out.shape == (2, 5, 3)


def test_partial_qkv_loading_prepares_only_complete_local_weight(runtime):
    method = NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config())
    layer = torch.nn.Module()

    def load(param, weight, shard):
        with torch.no_grad():
            param[shard * 2 : (shard + 1) * 2].copy_(weight)

    method.create_weights(
        layer,
        input_size_per_partition=8,
        output_partition_sizes=[2, 2, 2],
        input_size=16,
        output_size=12,
        params_dtype=torch.bfloat16,
        weight_loader=load,
    )
    assert layer.weight.is_meta
    for shard in range(3):
        layer.weight.weight_loader(layer.weight, torch.full((2, 8), shard + 1, dtype=torch.bfloat16), shard)
        assert runtime.prepare.call_count == (1 if shard == 2 else 0)
    local = runtime.prepare.call_args.args[0]
    assert local.shape == (6, 8)  # Already partitioned, never the global (12, 16).
    assert layer.weight.shape == (8, 6)
    method.process_weights_after_loading(layer)
    runtime.prepare.assert_called_once()


def test_dummy_meta_weight_materializes_before_prepare(runtime):
    method = NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=8,
        output_partition_sizes=[4],
        input_size=8,
        output_size=4,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a: None,
    )
    method.process_weights_after_loading(layer)
    assert not layer.weight.is_meta
    assert runtime.prepare.call_args.args[0].shape == (4, 8)
    runtime.prepare.assert_called_once()


def test_prepare_error_is_not_marked_complete(runtime):
    method, layer = make_linear(runtime)
    runtime.prepare.side_effect = RuntimeError("quantization failed")
    with pytest.raises(RuntimeError, match="quantization failed"):
        method.process_weights_after_loading(layer)
    assert not getattr(layer, "_already_called_process_weights_after_loading", False)
    assert layer.weight.shape == (3, 8)
