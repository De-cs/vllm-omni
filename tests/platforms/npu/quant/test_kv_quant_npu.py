# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for NPU FP8 KV quantization helpers.

These tests load ``kv_quant_npu`` from its source file via ``importlib`` so
the test module itself does not ``import vllm_omni`` (which would pull
``patch`` → ``aenum``, vLLM, etc.).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _repo_root() -> Path:
    """Resolve checkout root (parent of ``vllm_omni/``), not ``tests/``."""
    here = Path(__file__).resolve()
    marker = Path("vllm_omni") / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    for parent in here.parents:
        if (parent / marker).is_file():
            return parent
    msg = f"could not locate repo root (no {marker}) starting from {here}"
    raise FileNotFoundError(msg)


def _load_kv_quant_npu() -> ModuleType:
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    if not path.is_file():
        msg = f"kv_quant_npu source not found: {path}"
        raise FileNotFoundError(msg)
    name = "vllm_omni_test_kv_quant_npu_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kv_quant_npu = _load_kv_quant_npu()


def _npu_smoke_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    except ImportError:
        return False
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False
    a5 = getattr(AscendDeviceType, "A5", None)
    return a5 is not None and get_ascend_device_type() == a5


npu_smoke = pytest.mark.skipif(
    not _npu_smoke_available(), reason="Ascend A5 quantization device/dependencies not available."
)


def test_is_quantized_kv_cache() -> None:
    assert kv_quant_npu.is_quantized_kv_cache("fp8")
    assert kv_quant_npu.is_quantized_kv_cache("mxfp8")
    assert kv_quant_npu.is_quantized_kv_cache("mxfp4")
    assert not kv_quant_npu.is_quantized_kv_cache("float")
    assert not kv_quant_npu.is_quantized_kv_cache(None)
    assert not kv_quant_npu.is_quantized_kv_cache("int8")


@pytest.mark.parametrize("layout,shape", [("BSND", (1, 128, 2, 64)), ("BNSD", (1, 2, 128, 64))])
def test_compatibility_wrapper_delegates(monkeypatch, layout, shape):
    import sys
    from unittest.mock import Mock

    runtime = Mock(side_effect=lambda q, k, v, **kwargs: q)
    module = ModuleType("mindiesd.layers.flash_attn.quant_flash_attn")
    setattr(module, "fp8_rotate_quant_fa", runtime)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    q = torch.randn(shape, dtype=torch.bfloat16)
    assert kv_quant_npu.fp8_rotate_quant_fa(q, q, q, layout=layout, softmax_scale=0.25) is q
    assert runtime.call_args.args[0] is q
    assert runtime.call_args.kwargs == {"layout": layout, "softmax_scale": 0.25, "rotation_seed": 425500}


@npu_smoke
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_fp8_rotate_quant_fa_real_npu_shape_contract(layout):
    pytest.importorskip("mindiesd.layers.flash_attn.quant_flash_attn")
    query = torch.randn(1, 256, 2, 64, dtype=torch.float16, device="npu")
    if layout == "BNSD":
        query = query.transpose(1, 2)
    out = kv_quant_npu.fp8_rotate_quant_fa(query, query, query, layout=layout)
    assert out.shape == query.shape and out.dtype == query.dtype
    assert torch.isfinite(out).all()


@npu_smoke
@pytest.mark.npu
@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_dense_runtime_backend_real_npu(method, layout):
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    # On NPU, missing Runtime symbols must fail this qualification test.
    query = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16, device="npu")
    if layout == "BNSD":
        query = query.transpose(1, 2)
    impl = FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.125,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": method}},
    )
    out = impl.forward_npu(query, query, query)
    assert out.shape == query.shape and out.dtype == query.dtype
    assert torch.isfinite(out).all()
