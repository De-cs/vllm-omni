# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility entrypoints for MindIE-SD Q/K/V quantization.

The Runtime owns rotations, quantization and attention execution.
"""

from __future__ import annotations

import torch


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    return kv_cache_dtype in {"fp8", "mxfp8", "mxfp4"}


def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Preserve the legacy layout/scale API and Omni rotation seed."""
    try:
        from mindiesd.layers.flash_attn.quant_flash_attn import fp8_rotate_quant_fa as runtime
    except ImportError as exc:
        raise ImportError("NPU FP8 attention requires MindIE-SD quant_flash_attn Runtime.") from exc
    return runtime(query, key, value, layout=layout, softmax_scale=softmax_scale, rotation_seed=425500)
