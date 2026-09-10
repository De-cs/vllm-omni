# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-repository CPU dispatch contracts, requiring the matching MindIE-SD source.

Only native operators are replaced. Public adapters, rotations, layouts, padding,
scale preparation, sequence metadata and output cropping execute their real code.
Operator substitutes carry floating tensors; these are not numerical quantization tests.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flash_attn, rainfusion_attn
from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

mindiesd = pytest.importorskip("mindiesd")
if not hasattr(mindiesd, "quant_attention_forward"):
    pytest.skip("MindIE-SD public quantized attention API is required", allow_module_level=True)
qfa = importlib.import_module("mindiesd.layers.flash_attn.quant_attention_forward")

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_omni_calls_real_dense_public_api(monkeypatch, precision, layout):
    monkeypatch.setattr(flash_attn, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    quantized_inputs = []

    def dynamic_quant(tensor, **kwargs):
        quantized_inputs.append((tensor.clone(), kwargs))
        shape = list(tensor.shape)
        axis = kwargs.get("axis", -1)
        shape[axis] = max(1, shape[axis] // 32)
        return tensor, torch.ones(shape, dtype=torch.uint8)

    monkeypatch.setattr(qfa.torch_npu, "npu_dynamic_block_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(qfa.torch_npu, "npu_dynamic_mx_quant", dynamic_quant, raising=False)
    execute = Mock(side_effect=lambda query, *args, **kwargs: (query, None))
    metadata = Mock(return_value=torch.zeros(1, dtype=torch.int32))
    if precision == "fp8":
        monkeypatch.setattr(torch.ops.mindiesd, "fused_infer_attention_score_v2", execute, raising=False)
    elif precision == "mxfp8":
        monkeypatch.setattr(qfa.torch_npu, "npu_fused_infer_attention_score_v2", execute, raising=False)
    else:
        monkeypatch.setattr(torch.ops.mindiesd, "quant_flash_attn", execute, raising=False)
        monkeypatch.setattr(torch.ops.mindiesd, "quant_flash_attn_metadata", metadata, raising=False)

    query = torch.randn(1, 130, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        query = query.transpose(1, 2)
    impl = flash_attn.FlashAttentionImpl(
        num_heads=2, head_size=64, softmax_scale=0.37, qkv_layout=layout,
        backend_kwargs={"quant": {"method": precision, "fallback": []}},
    )
    output = impl.forward_fa_quant_npu(query, query, query)
    expected = query
    if precision != "mxfp4":
        rotation = get_quant_attention_rotation(query.device, query.dtype, 64, 425500)
        expected = query @ rotation
    torch.testing.assert_close(output, expected)
    assert output.shape == query.shape and len(quantized_inputs) == 3
    execute.assert_called_once()
    kwargs = execute.call_args.kwargs
    assert kwargs["softmax_scale"] == 0.37
    if precision == "fp8":
        assert kwargs["input_layout"] == "BNSD"
        assert quantized_inputs[0][0].shape == (2, 130, 64)
        assert quantized_inputs[0][1]["row_block_size"] == 128
    elif precision == "mxfp8":
        assert kwargs["input_layout"] == "TND"
        assert kwargs["actual_seq_qlen"] == kwargs["actual_seq_kvlen"] == [130]
        assert [entry[1]["axis"] for entry in quantized_inputs] == [-1, -1, 0]
    else:
        seq_axis = 1 if layout == "BSND" else 2
        assert quantized_inputs[0][0].shape[seq_axis] == 512
        assert kwargs["seqused_q"].tolist() == kwargs["seqused_kv"].tolist() == [130]
        assert kwargs["layout_q"] == kwargs["layout_out"] == layout
        assert kwargs["q_dtype"] == qfa.torch_npu.float4_e2m1fn_x2
        assert kwargs["q_descale_dtype"] == qfa.torch_npu.float8_e8m0fnu
        assert [entry[1]["axis"] for entry in quantized_inputs] == [-1, -1, seq_axis]
        assert kwargs["metadata"] is metadata.return_value
        metadata.assert_called_once()


@pytest.mark.parametrize("precision", ["fp8", "mxfp4"])
@pytest.mark.parametrize("length", [65, 600])
def test_omni_calls_real_sparse_public_api(monkeypatch, precision, length):
    rf = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3")
    monkeypatch.setattr(flash_attn, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    for module in (flash_attn, rainfusion_attn):
        monkeypatch.setattr(module, "get_current_diffusion_config_or_none", lambda: None)
    rainfusion_attn._mindiesd_supports_precision.cache_clear()
    monkeypatch.setattr(rf, "do_tensor_rearrange_only", lambda q, k, v, *a, **kw: (q, k, v))
    monkeypatch.setattr(rf, "avgpool", lambda *a, **kw: torch.ones(1))
    monkeypatch.setattr(rf, "_generate_mask_direct", lambda *a, **kw: torch.ones(1, 2, 8, 8))
    monkeypatch.setattr(rf, "_bsa_inv_rearrange", lambda out, *a: out)

    def quantize(tensor, **kwargs):
        shape = list(tensor.shape)
        axis = kwargs.get("axis", -1)
        shape[axis] = max(1, shape[axis] // 32)
        return tensor, torch.ones(shape, dtype=torch.uint8)

    monkeypatch.setattr(qfa.torch_npu, "npu_dynamic_mx_quant", quantize, raising=False)
    monkeypatch.setattr(qfa.torch_npu, "npu_dynamic_block_quant", quantize, raising=False)
    execute = Mock(side_effect=lambda **kw: (torch.ones_like(kw["query"]), None))
    names = ("quant_mode", "dst_type_max", "q_dtype", "k_dtype", "v_dtype",
             "q_scale_dtype", "k_scale_dtype", "v_scale_dtype")
    execute.default = SimpleNamespace(_schema=SimpleNamespace(arguments=[SimpleNamespace(name=n) for n in names]))
    monkeypatch.setattr(torch.ops.mindiesd, "block_sparse_attention", execute, raising=False)
    monkeypatch.setattr(torch.ops.mindiesd, "block_sparse_attention_version", lambda: 3, raising=False)
    query = torch.randn(1, length, 2, 64, dtype=torch.bfloat16)
    impl = rainfusion_attn.RainFusionAttentionImpl(
        num_heads=2, head_size=64, softmax_scale=0.37, qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8, "quant": {"method": precision, "fallback": []}},
    )
    plan = rainfusion_attn.RainFusionPlan(used_len=length, prefix_len=0, latent_shape=(1, 1, length))
    try:
        output = impl._forward_sparse_npu(query, query, query, plan)
    finally:
        rainfusion_attn._mindiesd_supports_precision.cache_clear()
    torch.testing.assert_close(output, torch.ones_like(query))
    execute.assert_called_once()
    kwargs = execute.call_args.kwargs
    assert kwargs["actual_seq_lengths"] == kwargs["actual_seq_lengths_kv"] == [length]
    assert kwargs["q_input_layout"] == "BNSD"
    assert kwargs["block_sparse_mask"] is not None
    if precision == "mxfp4":
        assert kwargs["query"].shape[2] == (length + 63) // 64 * 64
        assert kwargs["quant_mode"] == 2
        assert kwargs["q_dtype"] == qfa.torch_npu.float4_e2m1fn_x2
        assert kwargs["v_scale_dtype"] == qfa.torch_npu.float8_e8m0fnu
