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

If neither `diffusion_kv_cache_dtype` nor per-role `quant.method` is set,
attention runs in the native dtype.

## Hardware Support

| Device | FP8 FA |
|--------|--------|
| Ascend NPU | ✅ |
| NVIDIA GPU | ❌ |
| AMD ROCm | ❌ |
| Intel XPU | ❌ |

Legend: `✅` supported, `❌` unsupported.

FP8 FA is currently implemented only for the NPU Flash Attention backend. Other
backends do not support `diffusion_kv_cache_dtype="fp8"` for diffusion attention
and reject an incompatible explicit configuration.

## Model Type Support

### Diffusion Model

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Wan2.2 | Eligible DiT full-attention FA on Ascend NPU | Tested | Compare quality and latency against a BF16 baseline before production use |
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
| `diffusion_kv_cache_dtype` | str \| None | `None` | Set to `"fp8"` to enable dynamic FP8 FA on supported attention backends |
| `diffusion_kv_cache_skip_steps` | str \| None | `None` | Denoising step selector to keep in native dtype, for example `"0,1,4-6"` |
| `diffusion_kv_cache_skip_layers` | str \| None | `None` | Transformer layer selector to keep in native dtype, for example `"0-2,10"` |

Selectors use comma-separated integers and inclusive ranges. Listed steps or
layers skip FP8 FA; all other eligible full-attention forwards use the FP8 path.

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

## Wan2.2 T2V quantized attention on Ascend

Wan2.2 T2V A14B supports quantized self-attention through MindIE-SD. Model
weights are unchanged, and cross-attention should remain at `float`.

| Attention path | `backend` | Supported `quant.method` |
| --- | --- | --- |
| Dense FA | `FLASH_ATTN` | `fp8`, `mxfp8`, `mxfp4` |
| BSA | `RAINFUSION_ATTN` | `fp8`, `mxfp4` |

Configure the self-attention role in the model's deployment YAML:

```yaml
diffusion_attention_config:
  per_role:
    self:
      backend: FLASH_ATTN  # Use RAINFUSION_ATTN for BSA.
      quant:
        method: mxfp8
        skip_layers: "0,39"
        skip_steps: "0,1,38,39"
    cross:
      backend: FLASH_ATTN
      quant:
        method: float
```

For 40 denoising steps, `skip_steps: "0,1,38,39"` and
`skip_layers: "0,39"` are recommended starting points. Steps are zero-based
across the complete request and do not reset when Wan switches transformers;
layer indices are zero-based and local to each transformer. Selectors accept
comma-separated indices and inclusive ranges such as `"0,3-5"`. A selected
forward uses floating-point attention while preserving the Dense or BSA path.

Per-role selectors override their corresponding global selectors. An omitted
selector inherits the global value, and `[]` clears it for that role. Unsupported
precision/input combinations raise an error and must be corrected in the
configuration; operator errors are not retried with another precision.

### Requirements and validation

Until the required APIs and native fixes are available in a MindIE-SD release,
use MindIE-SD revision
[`8637b5333b0225381b215390fd09a8732e671cc4`](https://gitcode.com/zqxu/MindIE-SD/commit/8637b5333b0225381b215390fd09a8732e671cc4).
Build its Python package, PyTorch plugin, and custom operators from the same
checkout. BSA also requires RFv3 `sparse_attention` precision support; see
[RainFusion attention](../diffusion/attention_backends/rainfusion.md).

With CANN sourced and that MindIE-SD build installed, run the NPU interface
tests from the Omni checkout:

```bash
python -m pytest tests/platforms/npu/quant/test_kv_quant_npu.py \
    -k real_npu -vv -s -o addopts=''
```

A skipped NPU test is not a pass. Also run the existing Wan2.2 T2V example with
the deployment policy above, confirm the intended Dense or BSA runtime in the
log, require process exit 0 and a decodable video, and compare quality with a
floating-point run using the same prompt, seed, resolution, and step count.
