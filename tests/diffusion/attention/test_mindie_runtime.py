# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for MindIE adapters; no NPU kernels are mocked in production."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends import flash_attn, rainfusion_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def runtime(monkeypatch):
    mod = ModuleType("mindiesd")

    def execute(q, k, v, *, precision, layout, scale, q_rot=None, k_rot=None):
        return q

    setattr(mod, "quant_attention", Mock(side_effect=execute))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    for module in (flash_attn, layer_mod):
        monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: False)
    supports_precision = rainfusion_attn._mindiesd_supports_precision
    supports_precision.cache_clear()
    yield mod
    supports_precision.cache_clear()


def flash(method="mxfp8", layout="BSND"):
    return flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": method}},
    )


def sparse(**kwargs):
    return rainfusion_attn.RainFusionAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8, **kwargs},
    )


def video_metadata(**extra):
    return AttentionMetadata(
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(4, 32, 32), used_len=4096),
        extra={"max_seqlen_q": 4096, **extra},
    )


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4", "float"])
def test_method_configuration_reaches_backend(method):
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": method})
    assert spec.quant.enabled
    assert spec.backend_kwargs()["quant"] == {"method": method}


@pytest.mark.parametrize(
    "quant",
    [
        {"method": "invalid"},
        {"method": "mxfp8", "dtype_qk": "int8"},
        {"method": "fp8", "skip_steps": "4-2"},
    ],
)
def test_invalid_quant_configuration(quant):
    with pytest.raises(ValueError):
        AttnQuantSpec(**quant)


def test_gpu_config_and_sparse_mix_preserved():
    assert (
        AttentionSpec(backend="TRTLLM_ATTN", quant={"dtype_qk": "int8"}).backend_kwargs()["quant"]["dtype_qk"] == "int8"
    )
    assert (
        AttentionSpec(backend="RAINFUSION_ATTN", block_sparse={"precision": "mix"}).backend_kwargs()["precision"]
        == "mix"
    )
    with pytest.raises(ValueError, match="Conflicting"):
        AttentionSpec(backend="RAINFUSION_ATTN", block_sparse={"precision": "mix"}, quant={"method": "fp8"})


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", [None, "BSND", "BNSD"])
def test_exact_runtime_layout_scale_and_rotation(runtime, method, layout):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    impl = flash(method, layout=layout)
    assert impl.forward_npu(q, q, q) is q
    args, kwargs = runtime.quant_attention.call_args
    assert args[0] is q  # no unconditional transpose or extra quantization
    assert kwargs["layout"] == (layout or "BSND") and kwargs["scale"] == 0.37 and kwargs["precision"] == method
    assert "attn_mask" not in kwargs
    if method == "mxfp4":
        assert "q_rot" not in kwargs
    else:
        from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

        assert kwargs["q_rot"] is kwargs["k_rot"]
        torch.testing.assert_close(kwargs["q_rot"], get_quant_attention_rotation(q.device, q.dtype, 64))


def test_missing_runtime_requires_config_change(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    del runtime.quant_attention
    with pytest.raises(ImportError, match="select another quant.method"):
        flash("mxfp4").forward_npu(q, q, q)


def test_operator_failure_never_falls_back(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    runtime.quant_attention.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator failure"):
        flash("mxfp4").forward_npu(q, q, q)
    runtime.quant_attention.assert_called_once()


def test_unsupported_shape_requires_config_change(runtime):
    q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="select a supported quant.method"):
        flash("mxfp8").forward_npu(q, q, q)
    runtime.quant_attention.assert_not_called()


def test_packed_metadata_never_reaches_quant_runtime(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="packed/varlen"):
        flash().forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_not_called()


def make_layer(runtime, monkeypatch):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.attention = flash()
    layer.attn_backend = flash_attn.FlashAttentionBackend
    layer._disable_kv_quant = False
    layer.layer_idx = 3
    cfg = SimpleNamespace(diffusion_kv_cache_dtype=None, parallel_config=SimpleNamespace(ring_degree=1))
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": "mxfp8", "skip_steps": "0,1,38,39"})
    layer._init_kv_cache_quantization(cfg, spec)
    return layer


@pytest.mark.parametrize(
    "quant,expected_steps,expected_layers",
    [
        ({"method": "mxfp8"}, {0, 1}, {2, 3}),
        ({"method": "mxfp8", "skip_steps": "38,39", "skip_layers": []}, {38, 39}, set()),
    ],
)
def test_role_skip_selectors_override_or_inherit_global(runtime, quant, expected_steps, expected_layers):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.attention = flash()
    layer.attn_backend = flash_attn.FlashAttentionBackend
    layer._disable_kv_quant = False
    layer.layer_idx = 3
    cfg = SimpleNamespace(
        diffusion_kv_cache_dtype=None,
        diffusion_kv_cache_skip_step_indices={0, 1},
        diffusion_kv_cache_skip_layer_indices={2, 3},
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    layer._init_kv_cache_quantization(cfg, AttentionSpec(backend="FLASH_ATTN", quant=quant))
    assert layer._kv_cache_skip_steps == expected_steps
    assert layer._kv_cache_skip_layers == expected_layers


def test_step_skip_is_request_local_and_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata()
    for step in list(range(40)) * 2:
        ctx.denoise_step_idx = step
        resolved = layer._with_kv_cache_dtype(source)
        assert resolved.extra["kv_cache_dtype"] == ("float" if step in (0, 1, 38, 39) else "mxfp8")
        assert resolved.video_layout is source.video_layout
    assert source.extra == {"max_seqlen_q": 4096}
    layer._disable_kv_quant = True
    assert layer._with_kv_cache_dtype(source).extra["kv_cache_dtype"] == "float"


def test_legacy_conflict_and_ring_rejected(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": "mxfp8"})
    cfg = SimpleNamespace(diffusion_kv_cache_dtype="fp8", parallel_config=SimpleNamespace(ring_degree=1))
    with pytest.raises(ValueError, match="Conflicting"):
        layer._init_kv_cache_quantization(cfg, spec)
    cfg.diffusion_kv_cache_dtype = None
    cfg.parallel_config.ring_degree = 2
    with pytest.raises(ValueError, match="ring"):
        layer._init_kv_cache_quantization(cfg, spec)


def test_sparse_tail_steps_and_dense_quant_dispatch(runtime, monkeypatch):
    ctx = SimpleNamespace(denoise_step_idx=0, total_denoise_steps=50)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(rainfusion_attn, "get_forward_context", lambda: ctx)
    impl = sparse(start_step=2, end_step=2, quant={"method": "fp8"})
    metadata = video_metadata()
    for step in (0, 1, 2, 47, 48, 49):
        ctx.denoise_step_idx = step
        assert (impl._resolve_plan(metadata) is not None) == (2 <= step < 48)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    impl.forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_called_once()
    assert runtime.quant_attention.call_args.kwargs["precision"] == "fp8"


def test_sparse_fp8_uses_high_level_runtime_and_skip_uses_bf16(runtime, monkeypatch):
    calls = []

    def sparse_attention(q, k, v, *, precision="bf16", **kwargs):
        calls.append((precision, kwargs))
        return q

    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", sparse_attention, raising=False)
    impl = sparse(precision="fp8")
    q = torch.randn(1, 4224, 2, 64, dtype=torch.bfloat16)
    out = impl.forward_npu(q, q, q, video_metadata())
    assert out.shape == q.shape and not out[:, 4096:].any()
    assert calls[-1][0] == "fp8"
    assert calls[-1][1]["sparse_type"] == "rf_v3"
    impl.forward_npu(q, q, q, video_metadata(kv_cache_dtype="float"))
    assert calls[-1][0] == "bf16" and calls[-1][1]["sparse_type"] == "rf_v2"
    assert len(calls) == 2
    runtime.quant_attention.assert_not_called()


def test_sparse_unsupported_method_requires_config_change(runtime, monkeypatch):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", call, raising=False)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Automatic precision fallback is not performed"):
        sparse(quant={"method": "mxfp8"}).forward_npu(q, q, q, video_metadata())
    call.assert_not_called()


def test_sparse_mxfp4_uses_public_entrypoint(runtime, monkeypatch):
    calls = []

    def sparse_attention(q, k, v, *, precision="bf16", **kwargs):
        calls.append((precision, kwargs))
        return q

    monkeypatch.setattr(runtime, "sparse_attention", sparse_attention, raising=False)
    spec = AttentionSpec(backend="RAINFUSION_ATTN", block_sparse={"precision": "mxfp4"})
    assert spec.backend_kwargs()["precision"] == "mxfp4"
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(sparse(quant={"method": "mxfp4"}).forward_npu(q, q, q, video_metadata()), q)
    assert calls[0][0] == "mxfp4"
    assert calls[0][1]["sparse_type"] == "rf_v3" and calls[0][1]["inner_precise"] == 4
    runtime.quant_attention.assert_not_called()


def test_sparse_custom_mask_stays_dense(runtime):
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 4096, dtype=torch.bool),
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(4, 32, 32)),
        extra={"max_seqlen_q": 4096},
    )
    assert sparse()._resolve_plan(metadata) is None


def test_sparse_model_padding_mask_is_equivalent_to_trimming(runtime):
    metadata = video_metadata()
    metadata.attn_mask = torch.arange(4224)[None] < 4096
    assert sparse()._resolve_plan(metadata).used_len == 4096


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_bsa_uses_requested_precision_without_dense_precheck_or_native_query(runtime, monkeypatch, method):
    def execute(q, k, v, *, precision="bf16", **kwargs):
        return q

    runtime.sparse_attention = Mock(wraps=execute)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    impl = sparse(quant={"method": method})
    impl.dense_fallback._validate_quant_request = Mock(side_effect=AssertionError("dense precheck called"))
    assert impl.forward_npu(q, q, q, video_metadata()) is not None
    assert runtime.sparse_attention.call_args.kwargs["precision"] == method
    runtime.sparse_attention.assert_called_once()
    runtime.quant_attention.assert_not_called()


def test_bsa_operator_failure_never_retries(runtime, monkeypatch):
    runtime.sparse_attention = Mock(side_effect=RuntimeError("BSA operator failure"))
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="BSA operator failure"):
        sparse(quant={"method": "mxfp4"}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_called_once()


def test_bsa_missing_precision_api_requires_config_change(runtime, monkeypatch):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: False)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Update diffusion_attention_config"):
        sparse(quant={"method": "mxfp4"}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()


def test_layer_selector_requires_index_except_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    layer.layer_idx = None
    config = SimpleNamespace(diffusion_kv_cache_dtype=None, parallel_config=SimpleNamespace(ring_degree=1))
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": "mxfp8", "skip_layers": "0,3"})
    with pytest.raises(ValueError, match="parseable transformer block index"):
        layer._init_kv_cache_quantization(config, spec)
    layer._disable_kv_quant = True
    layer._init_kv_cache_quantization(config, spec)
    assert layer._with_kv_cache_dtype(None).extra["kv_cache_dtype"] == "float"


@pytest.mark.parametrize("prefix", ["transformer.blocks.3.attn1", "transformer_2.blocks.3.attn1"])
@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_expert_layer_and_step_skip_remain_sparse(runtime, monkeypatch, prefix, method):
    def execute(q, k, v, *, precision="bf16", **kwargs):
        return q

    runtime.sparse_attention = Mock(wraps=execute)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    layer = make_layer(runtime, monkeypatch)
    layer.layer_idx = rainfusion_attn._try_extract_layer_index(prefix)
    assert layer.layer_idx == 3
    config = SimpleNamespace(diffusion_kv_cache_dtype=None, parallel_config=SimpleNamespace(ring_degree=1))
    spec = AttentionSpec(
        backend="RAINFUSION_ATTN", quant={"method": method, "skip_layers": "3", "skip_steps": "0,1,38,39"}
    )
    impl = sparse(quant={"method": method})
    layer.attention = impl
    layer.attn_backend = rainfusion_attn.RainFusionAttentionBackend
    layer._init_kv_cache_quantization(config, spec)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata(kv_cache_dtype="mxfp8")
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    for layer_idx, step in ((3, 0), (3, 2), (4, 0), (4, 1), (4, 2), (4, 37), (4, 38), (4, 39), (4, 0)):
        expected = "bf16" if layer_idx == 3 or step in (0, 1, 38, 39) else method
        layer.layer_idx = layer_idx
        ctx.denoise_step_idx = step
        metadata = layer._with_kv_cache_dtype(source)
        impl.forward_npu(q, q, q, metadata)
        assert runtime.sparse_attention.call_args.kwargs["precision"] == expected
    runtime.quant_attention.assert_not_called()
    assert source.extra["kv_cache_dtype"] == "mxfp8"


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
@pytest.mark.parametrize("head_dim", [32, 96])
def test_bsa_unsupported_rotation_shape_requires_config_change(runtime, monkeypatch, method, head_dim):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, head_dim, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Automatic precision fallback is not performed"):
        sparse(quant={"method": method}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_bsa_batch_above_one_requires_config_change(runtime, monkeypatch, method):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(2, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Automatic precision fallback is not performed"):
        sparse(quant={"method": method}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()


def test_gpu_variant_only_configuration_is_preserved():
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant={"flashinfer_backend": "trtllm-gen"})
    assert spec.quant.enabled
    assert spec.backend_kwargs()["quant"] == {"flashinfer_backend": "trtllm-gen"}


def test_custom_attention_initialization_keeps_upstream_optout(monkeypatch):
    monkeypatch.setattr(layer_mod, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: object())
    monkeypatch.setattr(layer_mod, "NoParallelAttention", lambda: object())
    custom = torch.nn.Identity()
    layer = layer_mod.Attention(
        num_heads=2,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
        custom_attention=custom,
        skip_sequence_parallel=True,
    )
    assert layer.attention is custom and layer._kv_cache_dtype is None


def test_legacy_global_fp8_uses_implicit_bsnd(runtime):
    q = torch.randn(1, 17, 2, 64, dtype=torch.bfloat16)
    impl = flash(layout=None)
    impl.quant = {}
    assert impl.forward_npu(q, q, q, AttentionMetadata(extra={"kv_cache_dtype": "fp8"})) is q
    assert runtime.quant_attention.call_args.kwargs["layout"] == "BSND"
    assert runtime.quant_attention.call_args.kwargs["precision"] == "fp8"


@pytest.mark.parametrize("dtype,disabled", [(None, False), ("mxfp8", False), ("mxfp8", True), ("float", False)])
@pytest.mark.parametrize("stale", [False, True])
def test_policy_metadata_is_copied_only_when_needed(dtype, disabled, stale):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer._kv_cache_dtype, layer._disable_kv_quant = dtype, disabled
    layer._kv_cache_skip_steps = layer._kv_cache_skip_layers = None
    keys = {"kv_cache_dtype": "fp8"}
    marker = torch.ones(1)
    metadata = AttentionMetadata(extra={"unrelated": marker, **(keys if stale else {})})
    output = layer_mod.Attention._with_kv_cache_dtype(layer, metadata)
    assert (output is metadata) == (dtype is None and not disabled and not stale)
    assert output.extra["unrelated"] is marker
    assert set(metadata.extra) == {"unrelated", *(keys if stale else {})}
    expected = {"unrelated": marker}
    if disabled or dtype == "float":
        expected["kv_cache_dtype"] = "float"
    elif dtype is not None:
        expected["kv_cache_dtype"] = dtype
    assert output.extra == expected
    fresh = layer_mod.Attention._with_kv_cache_dtype(layer, None)
    assert (fresh is None) == (dtype is None and not disabled)


@pytest.mark.parametrize("layer_idx", [0, 1, 39])
def test_40_step_policy_is_request_local(runtime, monkeypatch, layer_idx):
    layer = object.__new__(layer_mod.Attention)
    torch.nn.Module.__init__(layer)
    layer.layer_idx, layer._kv_cache_dtype, layer._disable_kv_quant = layer_idx, "mxfp8", False
    layer._kv_cache_skip_steps, layer._kv_cache_skip_layers = {0, 1, 38, 39}, {0, 39}
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = AttentionMetadata()
    for step in list(range(40)) * 2:
        ctx.denoise_step_idx = step
        output = layer._with_kv_cache_dtype(source)
        disabled = step in {0, 1, 38, 39} or layer_idx in {0, 39}
        assert output.extra["kv_cache_dtype"] == ("float" if disabled else "mxfp8")
    assert source.extra == {}
