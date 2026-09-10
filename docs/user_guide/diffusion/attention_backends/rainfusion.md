# RainFusion Attention

`RAINFUSION_ATTN` runs MindIE-SD RainFusion (`rf_v2`) block-sparse video
attention on Ascend NPU. It pools 128-token key blocks, ranks them per query
block, and attends to the highest-scoring blocks after arranging video tokens
in `(t, h, w)` order.

Only the video segment is sparse. Prefix rows such as text, visual conditions,
and audio, plus first-frame blocks, remain dense. Unsupported calls—including
warmup steps, skipped layers, missing video geometry, and video segments below
32 blocks—delegate to `FLASH_ATTN`, so compatible models can select RainFusion
globally.

## Configuration

| Key | Valid values | Meaning |
| --- | --- | --- |
| `sparsity` | finite, `[0, 1]` | Nominal dropped-key-block fraction; default `0.8`; `0` disables sparsity |
| `start_step` | integer, `>= 0` | Number of early denoise steps kept dense |
| `end_step` | integer, `>= 0` | Number of final denoise steps kept dense (tail fallback) |
| `precision` | `"bf16"`, `"fp8"`, `"mix"` | Kernel precision mode; default `"bf16"`. Requires MindIE-SD with `sparse_attention(precision=...)` |
| `skip_layers` | selector such as `"0-3,38"` | DiT blocks kept dense |

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN",\
    "block_sparse":{"sparsity":0.8,"start_step":0}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, BlockSparseSpec

config = AttentionConfig(
    default=AttentionSpec(
        backend="RAINFUSION_ATTN",
        block_sparse=BlockSparseSpec(
            sparsity=0.8,
            start_step=0,
            skip_layers="0-1",
        ),
    ),
)
```

Tune `start_step`, then `sparsity`, then `skip_layers`. Increasing the dense
early-step window is usually the cheapest way to recover global structure;
use layer exclusions only after a same-seed dense comparison identifies
sensitive blocks.

## Requirements and compatibility

RainFusion requires Ascend NPU and `mindiesd`. Selecting it on another
platform raises. Ring and AllGather-KV sequence parallelism are unsupported:
sparse planning needs full Q/K/V sequences. Use Ulysses with `ring_degree=1`
and `allgather_degree=1`.

The `precision` knob requires a MindIE-SD release whose `sparse_attention`
accepts `precision=`; older releases accept it through `**kwargs` but
silently ignore it and run the BF16 path. `RAINFUSION_ATTN` raises at runtime
when a non-`bf16` precision is requested without that support, so pin a
compatible MindIE-SD release when using `precision="mix"` or `"fp8"`.

## Geometry handling

RainFusion handles arbitrary video grids by rearranging video spatially and,
when necessary, promoting a real-video suffix to the always-kept prefix. It
does not pad because `rf_v2` does not consume a padding attention mask.

For MiniMax-H3 at 1344x768, grid `(62, 24, 42)` produces 62,496 video rows.
The implementation promotes 2,976 rows and leaves 59,520 sparse rows
(`465 x 128`) so the mask and kernel tiling remain aligned.

Spatial grids aligned to 8x8 are still preferable. A latent height or width
not divisible by 8 creates an always-kept suffix and reduces realized
sparsity. At the image level, that means width and height divisible by 256.
Protected video tails require a compatible MindIE-SD release.

For common configuration and selector behavior, see the
[attention backend overview](../attention_backends.md) and the
[backend selection design](../../../design/feature/attention_backend_selection.md).

## Wan2.2 precision and dense fallback

Wan publishes the post-patch video grid and valid sequence length. MindIE owns
mask generation, video rearrangement and inverse rearrangement; Omni only slices
structural tail padding and restores the output shape.

```yaml
diffusion_attention_config:
  per_role:
    self:
      backend: RAINFUSION_ATTN
      quant:
        method: fp8
        fallback: [float]
        skip_steps: "0,49"
      block_sparse:
        sparsity: 0.8
        start_step: 2
        end_step: 2
    cross:
      backend: FLASH_ATTN
```

For 50 denoise steps, `start_step=2, end_step=2` makes steps 0, 1, 48 and 49
dense. `end_step` is a **count**, not an absolute step index. Missing progress
information keeps the call dense when a step window cannot be evaluated.
Denoise indices continue across the high/low-noise transformer boundary.

Precision and sparsity are independent: quantization skips use sparse BF16
when sparse geometry remains eligible; sparse skips use dense FA with the same
configured quantization chain. `quant.method=float` disables quantization.
Sparse MXFP4 uses MindIE-SD's existing `rf_v3` path. Sparse MXFP8 is unsupported:
list `fp8` or `float` fallback explicitly if selected for a RainFusion role. Legacy
`block_sparse.precision=bf16/fp8/mxfp4/mix` remains supported; without a `quant` spec its
dense fallback remains unquantized unless a legacy diffusion dtype is set.

Single-video quantized calls explicitly use `sparse_type=rf_v3` and require
`sparse_attention(precision=...)`. FP8/MXFP4 also require the public read-only
`mindiesd.get_bsa_supported_precisions()` query: native operator version and
schema must advertise the selected precision before it is dispatched. Missing
capability selects only an explicitly configured fallback; execution errors are
never retried. This check is independent of Dense FIA constraints. Quantized
Wan BSA currently requires batch size one: RFv3 FP8 squeezes the batch before
block quantization, and larger MXFP4 batches are outside this integration's
qualified scope. Explicit `float` fallback still selects unquantized BSA.
No private `quant_flash_attn` module or separate BSA Runtime execution symbol is
required. BF16 keeps the existing `rf_v2` route.
A single-video model does not require `video_spans` support. Multi-video geometry
checks that capability at execution; quantized multi-video attention remains
unsupported unless a configured `float` fallback selects sparse BF16.

Sparse rotation defaults remain owned by MindIE (standard Hadamard in the current
`dev` implementation). An explicit `quant.rotation_seed` is accepted only when the high-level
sparse API exposes that parameter; otherwise it requires a configured precision
fallback or raises. Dense FP8/MXFP8 uses Omni's legacy seed 425500. Do not assume
these two paths have identical numerical baselines.

Caller-supplied masks and piecewise visibility use dense attention. Wan marks its
SP padding-only mask with `extra.attn_mask_is_padding`; RainFusion may remove
that padding by trimming the gathered tensors to `video_layout.used_len`.
