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

    mod.quant_attention = Mock(side_effect=execute)
    mod.get_bsa_supported_precisions = lambda: ("bf16", "fp8", "mxfp4")
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


def flash(method="mxfp8", fallback=(), layout="BSND"):
    return flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": method, "fallback": list(fallback)}},
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
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(4, 32, 32)), extra={"max_seqlen_q": 4096, **extra}
    )


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4", "float"])
def test_method_configuration_reaches_backend(method):
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": method})
    assert spec.quant.enabled
    assert spec.backend_kwargs()["quant"] == {"method": method, "fallback": []}


@pytest.mark.parametrize(
    "quant",
    [
        {"method": "invalid"},
        {"method": "fp8", "fallback": ["fp8"]},
        {"method": "fp8", "fallback": ["float", "mxfp8"]},
        {"fallback": ["float"]},
        {"method": "mxfp8", "dtype_qk": "int8"},
        {"method": "fp8", "rotation_seed": True},
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
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_exact_runtime_layout_scale_seed(runtime, method, layout):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    impl = flash(method, layout=layout)
    assert impl.forward_npu(q, q, q) is q
    args, kwargs = runtime.quant_attention.call_args
    assert args[0] is q  # no unconditional transpose or extra quantization
    assert kwargs["layout"] == layout and kwargs["scale"] == 0.37 and kwargs["precision"] == method
    assert "rotation_seed" not in kwargs and "attn_mask" not in kwargs
    if method == "mxfp4":
        assert "q_rot" not in kwargs
    else:
        from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

        assert kwargs["q_rot"] is kwargs["k_rot"]
        torch.testing.assert_close(kwargs["q_rot"], get_quant_attention_rotation(q.device, q.dtype, 64, 425500))


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("q_len,kv_len", [(17, 8), (8, 17)])
def test_mxfp8_forwards_unequal_lengths_and_gqa(runtime, layout, q_len, kv_len):
    q = torch.randn(2, q_len, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, kv_len, 2, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if layout == "BNSD":
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    impl = flash_attn.FlashAttentionImpl(
        num_heads=4,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": "mxfp8", "fallback": []}},
    )
    assert impl.forward_npu(q, k, v) is q
    runtime.quant_attention.assert_called_once()
    assert all(actual is expected for actual, expected in zip(runtime.quant_attention.call_args.args, (q, k, v)))


def test_mask_is_preserved_for_explicit_float_fallback(runtime, monkeypatch):
    q = torch.randn(1, 130, 2, 64, dtype=torch.bfloat16)
    mask = torch.arange(130)[None, :] < 129
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(runtime, "attention_forward", call, raising=False)
    flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q, AttentionMetadata(attn_mask=mask))
    runtime.quant_attention.assert_not_called()
    forwarded = call.call_args.kwargs["attn_mask"]
    assert forwarded.shape == (1, 1, 130, 130)
    assert forwarded[..., :129].all() and not forwarded[..., 129].any()


def test_missing_runtime_falls_back_only_when_configured(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    del runtime.quant_attention
    with pytest.raises(ImportError, match="compatible MindIE"):
        flash("mxfp4").forward_npu(q, q, q)
    impl = flash("mxfp4", ["mxfp8", "float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()


def test_operator_failure_never_falls_back(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    runtime.quant_attention.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator failure"):
        flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q)
    runtime.quant_attention.assert_called_once()


def test_unsupported_shape_requires_explicit_fallback(runtime):
    q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        flash("mxfp8").forward_npu(q, q, q)
    impl = flash("mxfp8", ["float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()
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
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": "mxfp8", "fallback": ["float"], "skip_steps": "0,49"})
    layer._init_kv_cache_quantization(cfg, spec)
    return layer


def test_step_skip_is_request_local_and_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata()
    for step in (0, 12, 49, 12):
        ctx.denoise_step_idx = step
        resolved = layer._with_kv_cache_dtype(source)
        assert bool(resolved.extra.get("disable_attention_quant")) == (step in (0, 49))
        assert resolved.extra.get("kv_cache_dtype") == (None if step in (0, 49) else "mxfp8")
        assert resolved.video_layout is source.video_layout
    assert source.extra == {"max_seqlen_q": 4096}
    layer._disable_kv_quant = True
    assert layer._with_kv_cache_dtype(source).extra["disable_attention_quant"]


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


def test_sparse_tail_steps_and_dense_quant_fallback(runtime, monkeypatch):
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
    impl.forward_npu(q, q, q, video_metadata(disable_attention_quant=True))
    assert calls[-1][0] == "bf16" and calls[-1][1]["sparse_type"] == "rf_v2"
    assert len(calls) == 2
    runtime.quant_attention.assert_not_called()


def test_sparse_unsupported_method_fallback(runtime, monkeypatch):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", call, raising=False)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sparse mxfp8"):
        sparse(quant={"method": "mxfp8"}).forward_npu(q, q, q, video_metadata())
    sparse(quant={"method": "mxfp8", "fallback": ["float"]}).forward_npu(q, q, q, video_metadata())
    assert call.call_args.kwargs["precision"] == "bf16"


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


def test_sparse_custom_seed_is_not_silently_ignored(runtime, monkeypatch):
    def sparse_attention(q, k, v, *, precision="bf16", **kwargs):
        return q

    monkeypatch.setattr(runtime, "sparse_attention", sparse_attention, raising=False)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rotation_seed"):
        sparse(quant={"method": "fp8", "rotation_seed": 42}).forward_npu(q, q, q, video_metadata())


@pytest.mark.parametrize("kv_shape", [(1, 64, 2, 64), (1, 128, 1, 64)])
def test_mxfp8_supported_shape_does_not_use_configured_fallback(runtime, kv_shape):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    kv = torch.randn(kv_shape, dtype=q.dtype)
    assert flash("mxfp8", ["fp8"]).forward_npu(q, kv, kv) is q
    runtime.quant_attention.assert_called_once()
    assert runtime.quant_attention.call_args.kwargs["precision"] == "mxfp8"


def test_float_fallback_does_not_drop_packed_boundaries(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="explicit mask"):
        flash(fallback=["float"]).forward_npu(q, q, q, metadata)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_float_fallback_preserves_mask_layout_and_scale(runtime, monkeypatch, layout):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "attention_forward", call, raising=False)
    q = torch.randn(1, 130, 2, 96, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    mask = torch.arange(130)[None, :] < 129
    flash(fallback=["float"], layout=layout).forward_npu(q, q, q, AttentionMetadata(attn_mask=mask))
    kwargs = call.call_args.kwargs
    assert kwargs["layout"] == layout and kwargs["scale"] == 0.37
    assert kwargs["head_first"] == (layout == "BNSD")
    assert kwargs["attn_mask"].shape == (1, 1, 130, 130)
    assert kwargs["attn_mask"][..., :129].all() and not kwargs["attn_mask"][..., 129].any()


def test_sparse_custom_mask_stays_dense(runtime):
    metadata = video_metadata()
    metadata.attn_mask = torch.ones(1, 4096, dtype=torch.bool)
    assert sparse()._resolve_plan(metadata) is None


def test_sparse_model_padding_mask_is_equivalent_to_trimming(runtime):
    metadata = video_metadata(attn_mask_is_padding=True)
    metadata.attn_mask = torch.arange(4224)[None] < 4096
    assert sparse()._resolve_plan(metadata).used_len == 4096


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_bsa_precheck_is_independent_of_dense_fia(runtime, monkeypatch, method):
    def execute(q, k, v, *, precision="bf16", **kwargs):
        return q

    runtime.sparse_attention = Mock(wraps=execute)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    impl = sparse(quant={"method": method})
    impl.dense_fallback._quant_unsupported_reason = Mock(side_effect=AssertionError("dense precheck called"))
    assert impl.forward_npu(q, q, q, video_metadata()) is not None
    assert runtime.sparse_attention.call_args.kwargs["precision"] == method


@pytest.mark.parametrize("available,expected", [(("bf16", "fp8"), "fp8"), (("bf16",), "bf16")])
def test_bsa_native_capability_selects_configured_chain(runtime, monkeypatch, available, expected):
    def execute(q, k, v, *, precision="bf16", **kwargs):
        return q

    runtime.sparse_attention = Mock(wraps=execute)
    runtime.get_bsa_supported_precisions = lambda: available
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    sparse(quant={"method": "mxfp4", "fallback": ["fp8", "float"]}).forward_npu(q, q, q, video_metadata())
    assert runtime.sparse_attention.call_args.kwargs["precision"] == expected
    runtime.sparse_attention.assert_called_once()
    runtime.quant_attention.assert_not_called()


def test_bsa_missing_capability_query_is_not_assumed_supported(runtime, monkeypatch):
    runtime.sparse_attention = Mock()
    del runtime.get_bsa_supported_precisions
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="get_bsa_supported_precisions"):
        sparse(quant={"method": "fp8"}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()


def test_bsa_operator_failure_never_retries(runtime, monkeypatch):
    runtime.sparse_attention = Mock(side_effect=RuntimeError("BSA operator failure"))
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="BSA operator failure"):
        sparse(quant={"method": "mxfp4", "fallback": ["fp8", "float"]}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_called_once()


def test_layer_selector_requires_index_except_cross_optout(runtime, monkeypatch):
    layer = make_layer(runtime, monkeypatch)
    layer.layer_idx = None
    config = SimpleNamespace(diffusion_kv_cache_dtype=None, parallel_config=SimpleNamespace(ring_degree=1))
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": "mxfp8", "skip_layers": "0,3"})
    with pytest.raises(ValueError, match="parseable transformer block index"):
        layer._init_kv_cache_quantization(config, spec)
    layer._disable_kv_quant = True
    layer._init_kv_cache_quantization(config, spec)
    assert layer._with_kv_cache_dtype(None).extra["disable_attention_quant"]


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
    spec = AttentionSpec(backend="RAINFUSION_ATTN", quant={"method": method, "skip_layers": "3", "skip_steps": "0"})
    impl = sparse(quant={"method": method})
    layer.attention = impl
    layer.attn_backend = rainfusion_attn.RainFusionAttentionBackend
    layer._init_kv_cache_quantization(config, spec)
    ctx = SimpleNamespace(denoise_step_idx=0)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(layer_mod, "get_forward_context", lambda: ctx)
    source = video_metadata(kv_cache_dtype="mxfp8", disable_attention_quant=True)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    for layer_idx, step, expected in ((3, 0, "bf16"), (3, 2, "bf16"), (4, 0, "bf16"), (4, 2, method), (4, 0, "bf16")):
        layer.layer_idx = layer_idx
        ctx.denoise_step_idx = step
        metadata = layer._with_kv_cache_dtype(source)
        impl.forward_npu(q, q, q, metadata)
        assert runtime.sparse_attention.call_args.kwargs["precision"] == expected
    runtime.quant_attention.assert_not_called()
    assert source.extra["kv_cache_dtype"] == "mxfp8"


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
@pytest.mark.parametrize("head_dim", [32, 96])
def test_bsa_unsupported_rotation_shape_uses_float_sparse(runtime, monkeypatch, method, head_dim):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(1, 4096, 2, head_dim, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        sparse(quant={"method": method}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()
    sparse(quant={"method": method, "fallback": ["float"]}).forward_npu(q, q, q, video_metadata())
    assert runtime.sparse_attention.call_args.kwargs["precision"] == "bf16"
    runtime.quant_attention.assert_not_called()


@pytest.mark.parametrize("method", ["fp8", "mxfp4"])
def test_bsa_batch_above_one_requires_explicit_float_fallback(runtime, monkeypatch, method):
    runtime.sparse_attention = Mock(side_effect=lambda q, k, v, **kwargs: q)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    q = torch.randn(2, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch size 1"):
        sparse(quant={"method": method}).forward_npu(q, q, q, video_metadata())
    runtime.sparse_attention.assert_not_called()
    sparse(quant={"method": method, "fallback": ["float"]}).forward_npu(q, q, q, video_metadata())
    assert runtime.sparse_attention.call_args.kwargs["precision"] == "bf16"
    runtime.quant_attention.assert_not_called()
