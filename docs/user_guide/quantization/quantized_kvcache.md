# Quantized KV Cache

## Overview

In DiT-based image and video generation, Flash Attention can take a large share
of denoising time, especially for high-resolution or long-frame workloads.
vLLM-Omni integrates online FP8, MXFP8 and MXFP4 quantization for eligible diffusion Flash
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

The quantized MindIE-SD paths require Ascend A5 (950), a compatible
`torch_npu`, and a MindIE-SD build exporting the selected runtime function.
The adapter checks the device family, dtype symbols and required operators
before executing a quantized candidate. NVIDIA GPU, AMD ROCm and Intel XPU do
not use this adapter.

Explicit unsupported NPU methods raise; alternatives run only when listed in
`quant.fallback`. [RainFusion](../diffusion/attention_backends/rainfusion.md)
also supports FP8 sparse attention and independently controls when to return to
dense attention. Hardware numerical and performance qualification remains
required for each runtime build.

## Model scope

The integration and acceptance scope is **Wan2.2 T2V A14B**. Self-attention
uses the selected FA method; cross-attention retains its model-level opt-out.
Linear layers and checkpoint loading keep their existing behavior. Do not set
`--quantization mxfp8` or `--quantization mxfp4` to enable this FA feature: those
options select separate weight/Linear quantization methods.

I2V, S2V, VACE and other model-specific pipelines are outside this qualification.

## Configuration

Online serving with strict MXFP8 FA (no precision fallback):

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni \
    --diffusion-kv-cache-dtype mxfp8
```

Use `--diffusion-kv-cache-dtype mxfp4` for strict MXFP4, subject to the current
runtime restrictions below. The existing T2V Python API accepts these settings
through `Omni` engine keyword arguments (`diffusion_kv_cache_dtype`,
`diffusion_kv_cache_skip_steps`, and `diffusion_kv_cache_skip_layers`).

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
Known unsupported geometry, hardware or missing runtime/dtype/operator symbols
can select a configured fallback; operator execution errors propagate without retrying another method.
Contradictory legacy dtype and per-role methods are rejected. Legacy and per-role
skip selectors are combined; a skip disables quantization regardless of fallback.
Wan cross-attention retains its model-level quantization opt-out.

Models must declare `BSND` or `BNSD`; the adapter passes tensors without an
unconditional transpose. Quantized calls require BF16/FP16 four-dimensional
Q/K/V. The current block-FP8 Runtime requires batch size 1 and equal head counts;
FP8/MXFP8 generated rotations require a power-of-two head dimension. Packed,
varlen and piecewise calls are outside the quantized contract. Float fallback also
requires a supported packed path or an explicit mask preserving visibility. Boolean Omni masks
use True for allowed attention and are converted to CANN's blocked-mask convention.
With the current MindIE MXFP4 runtime, **both Q and K/V sequence lengths must
be multiples of 512, and caller-supplied masks are rejected**. Its internal
padding currently changes the effective softmax length; returning cropped output
does not correct that error. Configure MXFP8/FP8/float fallback for unsupported
calls, or use strict mode to fail visibly. These guards can be relaxed only
after the MindIE valid-length and mask contract is fixed and verified. An entire
MXFP4 run may otherwise fall back, so a successful generation alone does not
establish MXFP4 coverage. Ring SP is unsupported; validate Ulysses including padding.

The required MindIE build exports the selected function from
`mindiesd.layers.flash_attn.quant_flash_attn`: `fp8_rotate_quant_fa`,
`mxfp8_rotate_quant_fa`, or `mxfp4_quant_fa`. Dense FP8/MXFP8 explicitly use rotation
seed `425500`, overridable by `quant.rotation_seed`. This is independent of the
video generation seed. MXFP4 does not generate rotations and receives no seed.
There is no production copy of rotation, quantization or FA kernels in Omni.

## Runtime migration validation

Run the CPU contract tests in an Omni development environment:

```bash
pytest tests/diffusion/attention/test_mindie_runtime.py \
  tests/diffusion/attention/test_rainfusion_plan.py \
  tests/diffusion/models/wan2_2/test_wan22_mindie_metadata.py \
  tests/platforms/npu/quant/test_kv_quant_npu.py
```

## Wan2.2 T2V A14B acceptance

Compare three runs using the same unquantized checkpoint and Linear layers:
native BF16/FP16 FA, MXFP8 FA, and MXFP4 FA. Use the same prompts, negative
prompts, seeds, resolution, frame count, denoising steps, guidance, scheduler,
parallelism and high/low-noise expert boundary. Run both quantized modes with
`fallback: []` first; report any intentionally skipped steps/layers as part of
the configuration. Test the fallback chain separately so it cannot hide missing
MXFP4 execution during qualification.

The target is a VBench score difference of at most 1% for each quantized FA
mode against the same native-precision baseline. Before measurement, fix the
VBench version, prompt set, score aggregation and whether 1% means relative
change or percentage points. No VBench or NPU performance result is claimed by
the CPU contract tests.

On NPU, cover TP=1/2 local head counts, Ulysses with padding, the two-expert
transition, interleaved requests, and offload when it is part of the deployment.
Record selected precision/fallback warnings, software versions, generation
settings, warmup and steady-state latency, and peak device memory. Unaligned
or masked MXFP4 cases require the MindIE runtime fix before strict qualification.

### RainFusion BSA MXFP4

`RAINFUSION_ATTN` supports `quant.method: mxfp4` with a MindIE installation that
reports CANN BlockSparseAttention V3 support. Use `quant.mxfp4_dst_type_max: 0.0`
and `quant.mxfp4_scale_alg: null` for the default OCP setting, or an explicit
range `7.25` for CX. These options belong to the sparse backend and do not enable
Linear quantization. BSA uses internal 64 alignment rather than the dense FA
512 guard. See [RainFusion BSA MXFP4](../diffusion/attention_backends/rainfusion.md#wan22-t2v-bsa-mxfp4)
for configuration, precision fallback, rotation defaults and validation scope.
