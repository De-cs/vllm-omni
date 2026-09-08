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
    mod = ModuleType("mindiesd.layers.flash_attn.quant_flash_attn")
    for name in ("fp8_rotate_quant_fa", "mxfp8_rotate_quant_fa", "mxfp4_quant_fa", "fp8_rotate_quant_bsa"):
        setattr(mod, name, Mock(side_effect=lambda q, k, v, **kw: q))
    for name in ("mindiesd", "mindiesd.layers", "mindiesd.layers.flash_attn"):
        parent = ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    npu = ModuleType("torch_npu")
    for name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
        setattr(npu, name, torch.float16)
    for name in ("npu_dynamic_block_quant", "npu_dynamic_mx_quant", "npu_fused_infer_attention_score_v2"):
        setattr(npu, name, Mock(side_effect=AssertionError("preflight must not execute an operator")))
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    ascend = ModuleType("vllm_ascend")
    ascend.__path__ = []
    utils = ModuleType("vllm_ascend.utils")
    setattr(utils, "AscendDeviceType", SimpleNamespace(A5="A5"))
    setattr(utils, "get_ascend_device_type", lambda: "A5")
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    for name in (
        "fused_infer_attention_score_v2",
        "quant_flash_attn_metadata",
        "quant_flash_attn",
        "block_sparse_attention",
    ):
        monkeypatch.setattr(
            torch.ops.mindiesd,
            name,
            Mock(side_effect=AssertionError("preflight must not execute an operator")),
            raising=False,
        )
    for module in (flash_attn, layer_mod):
        monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: False)
    rainfusion_attn._mindiesd_supports_precision.cache_clear()
    precision_probe = rainfusion_attn._mindiesd_supports_precision
    yield mod
    precision_probe.cache_clear()


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


@pytest.mark.parametrize(
    "method,name",
    [
        ("fp8", "fp8_rotate_quant_fa"),
        ("mxfp8", "mxfp8_rotate_quant_fa"),
        ("mxfp4", "mxfp4_quant_fa"),
    ],
)
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_exact_runtime_layout_scale_seed(runtime, method, name, layout):
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    impl = flash(method, layout=layout)
    assert impl.forward_npu(q, q, q) is q
    args, kwargs = getattr(runtime, name).call_args
    assert args[0] is q  # no unconditional transpose or extra quantization
    assert kwargs["layout"] == layout and kwargs["softmax_scale"] == 0.37
    if method == "mxfp4":
        assert "rotation_seed" not in kwargs
    else:
        assert kwargs["rotation_seed"] == 425500


def test_mask_polarity_and_masked_mxfp4_fallback(runtime):
    q = torch.randn(1, 130, 2, 64, dtype=torch.bfloat16)
    mask = torch.arange(130)[None, :] < 129
    flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q, AttentionMetadata(attn_mask=mask))
    runtime.mxfp4_quant_fa.assert_not_called()
    forwarded = runtime.mxfp8_rotate_quant_fa.call_args.kwargs["attn_mask"]
    assert forwarded.shape == (1, 1, 130, 130)
    assert not forwarded[..., :129].any() and forwarded[..., 129].all()


def test_missing_runtime_falls_back_only_when_configured(runtime):
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    del runtime.mxfp4_quant_fa
    with pytest.raises(ImportError, match="compatible MindIE"):
        flash("mxfp4").forward_npu(q, q, q)
    flash("mxfp4", ["mxfp8"]).forward_npu(q, q, q)
    runtime.mxfp8_rotate_quant_fa.assert_called_once()


def test_operator_failure_never_falls_back(runtime):
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    runtime.mxfp4_quant_fa.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator failure"):
        flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q)
    runtime.mxfp8_rotate_quant_fa.assert_not_called()


def test_unsupported_shape_requires_explicit_fallback(runtime):
    q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        flash("mxfp8").forward_npu(q, q, q)
    impl = flash("mxfp8", ["float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()
    runtime.mxfp8_rotate_quant_fa.assert_not_called()


def test_packed_metadata_never_reaches_quant_runtime(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="packed/varlen"):
        flash().forward_npu(q, q, q, metadata)
    runtime.mxfp8_rotate_quant_fa.assert_not_called()


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
    runtime.fp8_rotate_quant_fa.assert_called_once()


def test_sparse_fp8_uses_high_level_runtime_and_skip_uses_bf16(runtime, monkeypatch):
    calls = []

    def sparse_attention(q, k, v, *, precision="bf16", **kwargs):
        calls.append((precision, kwargs))
        if precision == "fp8":
            return runtime.fp8_rotate_quant_bsa(q, k, v, block_sparse_mask="owned by MindIE")
        return q

    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", sparse_attention, raising=False)
    impl = sparse(precision="fp8")
    q = torch.randn(1, 4224, 2, 64, dtype=torch.bfloat16)
    out = impl.forward_npu(q, q, q, video_metadata())
    assert out.shape == q.shape and not out[:, 4096:].any()
    runtime.fp8_rotate_quant_bsa.assert_called_once()
    assert calls[-1][1]["sparse_type"] == "rf_v3"
    impl.forward_npu(q, q, q, video_metadata(disable_attention_quant=True))
    assert calls[-1][0] == "bf16" and calls[-1][1]["sparse_type"] == "rf_v2"
    assert runtime.fp8_rotate_quant_bsa.call_count == 1


def test_sparse_unsupported_method_fallback(runtime, monkeypatch):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", call, raising=False)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sparse mxfp4"):
        sparse(quant={"method": "mxfp4"}).forward_npu(q, q, q, video_metadata())
    sparse(quant={"method": "mxfp4", "fallback": ["float"]}).forward_npu(q, q, q, video_metadata())
    assert call.call_args.kwargs["precision"] == "bf16"


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


@pytest.mark.parametrize("length", [128, 256, 384, 600])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("short_kv", [False, True])
def test_mxfp4_rejects_internal_padding_before_runtime(runtime, length, layout, short_kv):
    q = torch.randn(1, 512 if short_kv else length, 2, 64, dtype=torch.bfloat16)
    k = torch.randn(1, length, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q, k = q.transpose(1, 2), k.transpose(1, 2)
    with pytest.raises(ValueError, match="aligned to 512"):
        flash("mxfp4", layout=layout).forward_npu(q, k, k)
    flash("mxfp4", ["mxfp8"], layout).forward_npu(q, k, k)
    runtime.mxfp4_quant_fa.assert_not_called()
    runtime.mxfp8_rotate_quant_fa.assert_called_once()
    assert runtime.mxfp8_rotate_quant_fa.call_args.args[0] is q
    assert runtime.mxfp8_rotate_quant_fa.call_args.args[1] is k


@pytest.mark.parametrize("mask_rank", [2, 4])
def test_aligned_mxfp4_mask_requires_explicit_fallback(runtime, mask_rank):
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    keep = torch.arange(512)[None, :] < 500
    mask = keep if mask_rank == 2 else keep[:, None, None, :]
    metadata = AttentionMetadata(attn_mask=mask)
    with pytest.raises(ValueError, match="mask/valid-length"):
        flash("mxfp4").forward_npu(q, q, q, metadata)
    flash("mxfp4", ["mxfp8"]).forward_npu(q, q, q, metadata)
    runtime.mxfp4_quant_fa.assert_not_called()
    forwarded = runtime.mxfp8_rotate_quant_fa.call_args.kwargs["attn_mask"]
    assert not forwarded[..., :500].any() and forwarded[..., 500:].all()
    assert torch.equal(metadata.attn_mask, mask)


@pytest.mark.parametrize(
    "method,owner,name",
    [
        ("fp8", "npu", "float8_e4m3fn"),
        ("mxfp8", "npu", "float8_e8m0fnu"),
        ("mxfp4", "npu", "float4_e2m1fn_x2"),
        ("fp8", "npu", "npu_dynamic_block_quant"),
        ("mxfp8", "npu", "npu_dynamic_mx_quant"),
        ("mxfp8", "npu", "npu_fused_infer_attention_score_v2"),
        ("fp8", "mindie", "fused_infer_attention_score_v2"),
        ("mxfp4", "mindie", "quant_flash_attn_metadata"),
        ("mxfp4", "mindie", "quant_flash_attn"),
    ],
)
def test_missing_capability_is_checked_before_execution(runtime, monkeypatch, method, owner, name):
    target = sys.modules["torch_npu"] if owner == "npu" else torch.ops.mindiesd
    monkeypatch.setattr(target, name, None)
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=name):
        flash(method).forward_npu(q, q, q)
    impl = flash(method, ["float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()
    for name in ("fp8_rotate_quant_fa", "mxfp8_rotate_quant_fa", "mxfp4_quant_fa"):
        getattr(runtime, name).assert_not_called()


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
def test_unsupported_chip_uses_configured_fallback(runtime, monkeypatch, method):
    monkeypatch.setattr(sys.modules["vllm_ascend.utils"], "get_ascend_device_type", lambda: "A3")
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Ascend A5"):
        flash(method).forward_npu(q, q, q)
    impl = flash(method, ["float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()


def test_mxfp4_capability_fallback_can_select_fp8(runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "float4_e2m1fn_x2", None)
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    flash("mxfp4", ["fp8", "float"]).forward_npu(q, q, q)
    runtime.mxfp4_quant_fa.assert_not_called()
    runtime.fp8_rotate_quant_fa.assert_called_once()


def test_operator_attribute_error_is_not_treated_as_missing_capability(runtime):
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    runtime.mxfp4_quant_fa.side_effect = AttributeError("execution bug")
    with pytest.raises(AttributeError, match="execution bug"):
        flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q)
    runtime.mxfp8_rotate_quant_fa.assert_not_called()


def test_sparse_fp8_missing_operator_falls_back_before_quantization(runtime, monkeypatch):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", call, raising=False)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    monkeypatch.setattr(torch.ops.mindiesd, "block_sparse_attention", None)
    q = torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16)
    sparse(quant={"method": "fp8", "fallback": ["float"]}).forward_npu(q, q, q, video_metadata())
    assert call.call_args.kwargs["precision"] == "bf16"
    runtime.fp8_rotate_quant_bsa.assert_not_called()


def test_mxfp8_accepts_torch_fp8_dtype_without_npu_alias(runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "float8_e4m3fn", None)
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    flash("mxfp8").forward_npu(q, q, q)
    runtime.mxfp8_rotate_quant_fa.assert_called_once()


def test_mxfp8_rejects_missing_fp8_dtype_in_both_modules(runtime, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "float8_e4m3fn", None)
    monkeypatch.setattr(torch, "float8_e4m3fn", None)
    q = torch.randn(1, 512, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        flash("mxfp8").forward_npu(q, q, q)
    runtime.mxfp8_rotate_quant_fa.assert_not_called()
