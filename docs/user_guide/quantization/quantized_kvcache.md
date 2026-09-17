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
| -------- | -------- |
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
| --- | --- | --- | --- |
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
| ----------- | ------ | --------- | ------------- |
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

For Wan2.2 T2V A14B, `FLASH_ATTN` accepts per-role `quant.method` values
`fp8`, `mxfp8`, `mxfp4`, or `float`. This requires a compatible MindIE-SD
`quant_attention` API and native operators for the selected precision.
Keep cross-attention at `float`; model weights are unaffected.

Dense `FLASH_ATTN` and `RAINFUSION_ATTN` do not change precision automatically.
An unsupported input or unavailable public API is reported with instructions
to select another `quant.method` or use `float`; native execution errors
propagate unchanged. Layer/step selectors are the explicit accuracy fallback
and do not change the configured precision for other forwards.
Conflicting per-role and global quantization settings are rejected.

Configure quantized self-attention in the model's existing deployment YAML. For
example, start with the following policy and select the backend/method pair from
the table below. Keep cross-attention at `float`.

```yaml
diffusion_attention_config:
  per_role:
    self:
      backend: FLASH_ATTN
      quant:
        method: mxfp8
        skip_layers: "0,39"
        skip_steps: "0,1,38,39"
    cross:
      backend: FLASH_ATTN
      quant:
        method: float
```

| Attention path | `backend` | Supported `quant.method` |
| --- | --- | --- |
| Dense FA | `FLASH_ATTN` | `fp8`, `mxfp8`, `mxfp4` |
| BSA | `RAINFUSION_ATTN` | `fp8`, `mxfp4` |

For the standard **40 denoising steps**, `quant.skip_steps: "0,1,38,39"`
and `quant.skip_layers: "0,39"` are recommended starting points. Selected
forwards use floating-point Dense FA or floating-point BSA. Steps are zero-based
across the complete request and do not reset when Wan switches transformers;
layer indices are local to each transformer. These are fixed indices, not a
relative "last two steps" selector. Adjust them when changing the step count,
transformer depth, or quality target.

`quant.skip_layers` and `quant.skip_steps` accept index lists or inclusive ranges
such as `"0,3-5"`. When set, each per-role selector overrides its corresponding
global selector; when omitted, it inherits the global value. Use `[]` to clear
an inherited selector for one role. An unparsable layer index is an error when
layer skips are configured.
A skipped forward uses floating-point attention: floating Dense for FA and
floating sparse for BSA.

### Pinned dependencies

| Purpose | Exact MindIE-SD revision | Status |
| --- | --- | --- |
| Official FP8/MXFP8 public API baseline | [`cf1a89be5803ad246f26c88c54db89a6a77748d4`](https://gitcode.com/Ascend/MindIE-SD/commit/cf1a89be5803ad246f26c88c54db89a6a77748d4) | Upstream commit |
| All four validation configurations below | [`8637b5333b0225381b215390fd09a8732e671cc4`](https://gitcode.com/zqxu/MindIE-SD/commit/8637b5333b0225381b215390fd09a8732e671cc4) | Development integration snapshot with the public APIs and QFA native fixes |
| MXFP4 API proposed to upstream | [`9cbf9948325f15a6d9c5951389df564432b6d739`](https://gitcode.com/zqxu/MindIE-SD/commit/9cbf9948325f15a6d9c5951389df564432b6d739) ([PR 630](https://gitcode.com/Ascend/MindIE-SD/pull/630)) | API/tests only; it does not contain the QFA native fixes |

Use `8637b5333b0225381b215390fd09a8732e671cc4` for the four-path
qualification commands in this section until the required native fixes have an
upstream revision. It is not an upstream release. Build its Python package,
PyTorch plugin and custom operators from that same checkout following its
[installation guide](https://gitcode.com/zqxu/MindIE-SD/blob/8637b5333b0225381b215390fd09a8732e671cc4/docs/en/installation.md);
copying Python files alone is insufficient.

BSA FP8/MXFP4 additionally require `sparse_attention` with an explicit
`precision` argument and RFv3 support. Omni passes the precision directly;
native execution errors propagate without retry. For sparse policy details,
see [RainFusion attention](../diffusion/attention_backends/rainfusion.md).

Use a supported Ascend device and matching CANN/PyTorch/torch_npu stack. For each
qualification run, record the Omni and MindIE source commits, wheel SHA256, CANN
version, device model, `torch`/`torch_npu` versions and loaded native library paths.
An import check alone does not qualify a dependency build. Until an upstream
MXFP4 build passes the checks below, keep that mode experimental.

### NPU validation

From the Omni checkout, with the pinned dependency installed and CANN sourced:

```bash
export MINDIESD_SRC=/path/to/MindIE-SD
export MINDIESD_REV=8637b5333b0225381b215390fd09a8732e671cc4
if [ "$(git -C "$MINDIESD_SRC" rev-parse HEAD)" != "$MINDIESD_REV" ]; then
    echo "MindIE-SD revision does not match $MINDIESD_REV" >&2
    exit 1
fi

python - <<'PY'
import inspect
import mindiesd

assert callable(mindiesd.quant_attention)
assert "precision" in inspect.signature(mindiesd.sparse_attention).parameters
print("mindiesd:", mindiesd.__file__)
print("sparse_attention:", inspect.signature(mindiesd.sparse_attention))
PY

python -m pytest tests/platforms/npu/quant/test_kv_quant_npu.py \
    -k real_npu -vv -s -o addopts=''
```

These tests cover both layouts and D=128, including the Wan sequence length
75600 and a constant-V reference. A skipped test is not a pass. For a short T2V
run using an existing deployment YAML configured with the policy above (no
weight quantization):

```bash
export WAN_MODEL=/path/to/Wan2.2-T2V-A14B-Diffusers
export WAN_DEPLOY_CONFIG=/path/to/wan22-deploy.yaml
export OUT_DIR="$PWD/wan22-attention-validation"
mkdir -p "$OUT_DIR"
set -o pipefail
output="$OUT_DIR/wan22.mp4"
log="$OUT_DIR/wan22.log"
python -u examples/offline_inference/text_to_video/text_to_video.py \
    --model "$WAN_MODEL" \
    --deploy-config "$WAN_DEPLOY_CONFIG" \
    --num-inference-steps 40 --num-frames 17 --height 384 --width 640 \
    --prompt "A cat walking through a sunlit garden" --seed 42 \
    --enable-cpu-offload --vae-use-tiling --enforce-eager \
    --output "$output" 2>&1 | tee "$log"
result=$?
printf 'PROCESS_EXIT_CODE=%s\n' "$result" | tee -a "$log"
if [ "$result" -ne 0 ]; then exit "$result"; fi
python -c 'import imageio.v3 as iio, sys; assert iio.imread(sys.argv[1], index=0).size' "$output"
```

Confirm the selected path in the log: Dense FA reports the configured MindIE-SD
runtime (`fp8`, `mxfp8`, or `mxfp4`); BSA reports
`RainFusion uses MindIE-SD sparse attention` with `precision=fp8` or
`precision=mxfp4`.

For the full Wan geometry, repeat with `--num-frames 81 --height 720 --width 1280`.
Require the intended runtime in the log, a decodable video and process exit 0.
Video creation followed by an exit crash is a failure. These are functional
checks; assess VBench quality separately against a matched floating-point run.
