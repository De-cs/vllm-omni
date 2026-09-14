# Quantized KV Cache

## Overview

In DiT-based image and video generation, Flash Attention can take a large share
of denoising time, especially for high-resolution or long-frame workloads.
vLLM-Omni supports online FP8 quantization for eligible diffusion Flash
Attention (FA) to reduce FA latency while keeping model weights in their
original dtype.

This feature is configured through `diffusion_kv_cache_dtype` on
`OmniDiffusionConfig` (CLI: `--diffusion-kv-cache-dtype`). It is intentionally
**not** the same as vLLM's `--kv-cache-dtype`, which controls autoregressive
language-model KV cache storage and defaults to `"auto"`. Diffusion FA
quantization uses the dedicated diffusion flags so omni serve does not inherit
that default.

In vLLM-Omni diffusion pipelines, this is a runtime FA path: Q/K/V tensors are
dynamically quantized before the attention operator. It does not quantize model
weights and is separate from [FP8 W8A8](fp8.md), [Int8 W8A8](int8.md), or
pre-quantized checkpoint formats.

When neither the legacy dtype flag nor a per-role `quant.method` is set,
attention uses its existing precision.

## Hardware Support

| Device | FP8 FA |
|--------|--------|
| Ascend NPU | ✅ |
| NVIDIA GPU | ❌ |
| AMD ROCm | ❌ |
| Intel XPU | ❌ |

Legend: `✅` supported, `❌` unsupported.

Dense FP8, MXFP8 and MXFP4 adapters use MindIE-SD on NPU. Explicit unsupported
NPU methods raise; alternative methods run only when listed in `quant.fallback`.
[RainFusion](../diffusion/attention_backends/rainfusion.md) also supports FP8
sparse attention and independently controls when to return to dense attention.
Hardware numerical and performance validation is required for each Runtime build.

## Model Type Support

### Diffusion Model

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Wan2.2 | Eligible DiT full-attention FA on Ascend NPU | Pre-migration FP8 tested; Runtime migration pending NPU qualification | Compare each Runtime method against a BF16 baseline before production use |
| Other diffusion models | Eligible DiT full-attention FA on Ascend NPU | Not tested | You can try `diffusion_kv_cache_dtype="fp8"`; tune `diffusion_kv_cache_skip_steps` and `diffusion_kv_cache_skip_layers` when higher precision is needed |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

Not tested for FP8 FA. Treat any use as experimental unless a model-specific
guide documents support.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

Not tested. If the diffusion stage uses the same NPU Flash Attention backend,
`diffusion_kv_cache_dtype` may apply in theory; validate quality and latency for
each stage and model.

## Configuration

Offline diffusion example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
    --model <your-wan2.2-model> \
    --prompt "A cat sitting on a surfboard at the beach" \
    --height 1280 \
    --width 720 \
    --num-frames 61 \
    --num-inference-steps 4 \
    --ulysses-degree 4 \
    --vae-patch-parallel-size 4 \
    --diffusion-kv-cache-dtype fp8 \
    --diffusion-kv-cache-skip-steps "0,1" \
    --diffusion-kv-cache-skip-layers "0-2"
```

Online serving:

```bash
vllm serve <your-model> --omni --diffusion-kv-cache-dtype fp8
```

Deploy config:

```yaml
stages:
  - stage_id: 0
    diffusion_kv_cache_dtype: "fp8"
    diffusion_kv_cache_skip_steps: "0,1"
    diffusion_kv_cache_skip_layers: "0-2"
```

The `model_stage` and diffusion execution type belong to the registered
`PipelineConfig`; the deploy YAML only carries runtime overrides.

The legacy keyword aliases `kv_cache_dtype`, `kv_cache_skip_steps`, and
`kv_cache_skip_layers` remain accepted when constructing
`OmniDiffusionConfig` directly. They are not deploy YAML fields; prefer the
`diffusion_*` names for new code.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diffusion_kv_cache_dtype` | str \| None | `None` | NPU method: `fp8`, `mxfp8`, `mxfp4`; `float` disables quantization and `auto` leaves the default unchanged |
| `diffusion_kv_cache_skip_steps` | str \| None | `None` | Denoising step selector to keep in native dtype, for example `"0,1,4-6"` |
| `diffusion_kv_cache_skip_layers` | str \| None | `None` | Transformer layer selector to keep in native dtype, for example `"0-2,10"` |

Selectors use comma-separated integers and inclusive ranges. Listed steps or
layers skip quantized FA; other eligible full-attention forwards use the selected method.

## Validation and Notes

1. Compare generated images or videos against a BF16 baseline with the same
   seed, prompt, resolution, frame count, and denoising steps.
2. Use `diffusion_kv_cache_skip_steps` for denoising steps where quality is more
   sensitive.
3. Use `diffusion_kv_cache_skip_layers` for transformer layers that show visible quality
   regressions.
4. Report both latency and quality results when enabling this option for a new
   model. For image or video models, include visual comparison and quantitative
   metrics when available, such as PSNR or SSIM.

## Per-role MindIE-SD Runtime configuration

```yaml
diffusion_attention_config:
  per_role:
    self:
      backend: FLASH_ATTN
      quant:
        method: mxfp4
        fallback: [mxfp8, float]
        skip_steps: "0-1,48-49"
        skip_layers: "0,39"
    cross:
      backend: FLASH_ATTN
      quant:
        method: float
```

`float` means unquantized attention in the input dtype, not a cast to FP32.
Fallback is ordered, contains no duplicates, and ends at `float` if present.
Known unsupported geometry or a missing Runtime function can select a configured
fallback; operator execution errors propagate without retrying another method.
Contradictory legacy dtype and per-role methods are rejected. Legacy and per-role
skip selectors are combined; a skip disables quantization regardless of fallback.
Wan cross-attention retains its model-level quantization opt-out.

Models must declare `BSND` or `BNSD`; the adapter passes tensors without an
unconditional transpose. Quantized calls require BF16/FP16 four-dimensional
Q/K/V. The current block-FP8 Runtime requires batch size 1;
MXFP8 permits different Q/KV sequence lengths and head counts, provided Q heads
are divisible by KV heads. K/V shapes, batch sizes and head dimensions must match.
FP8/MXFP8 generated rotations require a power-of-two head dimension. Packed,
varlen and piecewise calls are outside the quantized contract. Float fallback also
requires a supported packed path or an explicit mask preserving visibility.
Caller masks require an explicitly configured `float` fallback; their True=keep
semantics are preserved. Ring SP is unsupported; validate Ulysses including padding.

The required MindIE build exports `mindiesd.quant_attention`, called with
`precision=fp8/mxfp8/mxfp4`, the input `layout`, and `scale`.
Omni supplies cached Q/K rotation matrices for dense FP8/MXFP8 using rotation
seed `425500`, overridable by `quant.rotation_seed`. This is independent of the
video generation seed. MXFP4 does not generate rotations and receives no seed.
Quantization and FA execution remain in MindIE-SD. The corrected MindIE-SD MXFP4
implementation must retain original effective sequence lengths when padding Q/K/V.

## Runtime migration validation

Run the CPU contract tests in an Omni development environment:

```bash
pytest tests/diffusion/attention/test_mindie_runtime.py \
  tests/diffusion/attention/test_rainfusion_plan.py \
  tests/diffusion/quantization/test_mindie_online_runtime.py \
  tests/diffusion/models/wan2_2/test_wan22_mindie_metadata.py \
  tests/platforms/npu/quant/test_kv_quant_npu.py
```

Before qualifying a Runtime release, compare against the pre-migration FP8
commit and BF16 using fixed inputs. Cover TP=1/2 (Row/Column/fused QKV), Ulysses
with padding, real CPU offload/HSDP, two-expert transitions, interleaved requests,
and the full latent trajectory. Record actual selected precision, warmup and
steady-state latency, peak device memory and loading peak memory. VBench quality
limits must specify a fixed prompt set, generation seeds and whether a 1% budget
means relative change or percentage points; mock tests do not establish quality
or speedup.
