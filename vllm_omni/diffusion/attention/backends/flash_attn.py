# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Callable
from functools import partial

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask
from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    piecewise_attn,
    run_paged_piecewise_plan,
)
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_piecewise_spans: bool = True
    supports_paged_kv: bool = True

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # CUDA: forward_cuda dispatches the packed cu_seqlens varlen path
        # before ever reading attn_mask. NPU: forward_fa_npu honors the
        # npu_attn_varlen opt-in and rebuilds the padding mask itself if the
        # packed contract fails. XPU reads attn_mask, so models must keep
        # constructing it there.
        return current_omni_platform.is_cuda() or current_omni_platform.is_npu()

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        # CUDA / ROCm / MUSA all route through ``forward_cuda``, which
        # dispatches ``_forward_varlen_packed`` -> ``flash_attn_varlen_func``
        # over the caller's cu_seqlens without a mask, so an arbitrary
        # N-document packing keeps its boundaries. NPU's forward path
        # (``_resolve_packed_seq_npu``) only accepts a ``[real, pad]``
        # two-document layout (cu_seqlens shape <= 3) and otherwise falls
        # back to a padding-mask rebuild that spans the whole packed row,
        # so N>=2 real documents would silently attend across request
        # boundaries. XPU never consumes cu_seqlens from ``extra``. Both
        # therefore stay False here.
        return current_omni_platform.is_cuda() or current_omni_platform.is_rocm() or current_omni_platform.is_musa()

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl


class FlashAttentionImpl(AttentionImpl):
    # Per-platform FP8 KV quantization support.
    # To enable FP8 on a new platform, add its OmniPlatformEnum value here
    # and handle kv_cache_dtype in the corresponding forward_{platform}().
    #
    # TODO(quant-backend): The FP8 quant path currently lives inside
    # FlashAttentionImpl gated by ``attn_metadata.extra["kv_cache_dtype"]``.
    # Eventually extract it into a dedicated FlashAttentionQuantBackend so
    # backend selection (not metadata) decides quant. Until then, model
    # authors can opt a specific Attention layer out via
    # ``Attention(disable_kv_quant=True)``.
    _supported_kv_cache_dtypes = {
        "npu": {"fp8", "mxfp8", "mxfp4"},
    }

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        role: str = "self",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.qkv_layout = qkv_layout
        self.is_cross_attn = role == "cross"
        cfg = get_current_diffusion_config_or_none()
        self.fa_deterministic = bool(getattr(cfg, "fa_deterministic", False)) if cfg is not None else False
        self.quant = dict((backend_kwargs or {}).get("quant") or {})
        if self.quant and not current_omni_platform.is_npu():
            raise ValueError("FlashAttention quant.method is supported only on NPU.")
        unknown = set(backend_kwargs or {}) - {"quant"}
        if unknown:
            logger.warning("FlashAttentionImpl ignoring backend_kwargs: %s", sorted(unknown))

    def _warn_fa_deterministic_non_dense(self, path: str) -> None:
        if not self.fa_deterministic:
            return
        logger.warning_once(
            "fa_deterministic=True is ignored on the %s FlashAttention path; "
            "only the dense flash_attn_func path passes deterministic=True.",
            path,
        )

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _flash_wrapper(q, k, v, *, attn_func, **kwargs):
        return FlashAttentionImpl._unwrap_flash_output(attn_func(q, k, v, **kwargs))

    @staticmethod
    def _flash_varlen_wrapper(q, k, v, *, attn_func, causal, softmax_scale, **kwargs):
        """Call a varlen-only FlashAttention backend for a dense segment."""
        del kwargs
        batch_size, q_len = q.shape[:2]
        k_len = k.shape[1]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * k_len, k_len, dtype=torch.int32, device=q.device)
        out = attn_func(
            q=q.flatten(0, 1),
            k=k.flatten(0, 1),
            v=v.flatten(0, 1),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        out = FlashAttentionImpl._unwrap_flash_output(out)
        return out.reshape(batch_size, q_len, *out.shape[1:])

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _index_first_axis,
            _pad_input,
            _unpad_input,
            _upad_input,
            flash_attn_varlen_func,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        batch_size, query_length = query.shape[:2]
        if not self.is_cross_attn and query_length == key.size(1):
            q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
                query, key, value, attention_mask, query_length, _unpad_input
            )
        else:
            # Cross-attention: the mask covers keys only, so keep every query row.
            k, indices_k, cu_seq_lens_k, max_length_k, _ = _unpad_input(key, attention_mask)
            v = _index_first_axis(value, indices_k)
            q = query.flatten(0, 1)
            cu_seq_lens_q = torch.arange(
                0, (batch_size + 1) * query_length, query_length, dtype=torch.int32, device=query.device
            )
            max_length_q = query_length
            indices_q = None

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **{
                "causal": self.causal,
                "softmax_scale": self.softmax_scale,
            },
        )
        out_unpad = self._unwrap_flash_output(out_unpad)
        if indices_q is None:
            return out_unpad.reshape(batch_size, query_length, *out_unpad.shape[1:])
        return _pad_input(out_unpad, indices_q, batch_size, query_length)

    def _forward_varlen_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Run FlashAttention directly on an already packed sequence.

        Some diffusion transformers already maintain exact packed-document
        boundaries. Reusing those boundaries avoids rebuilding boolean masks
        and gathering/scattering Q/K/V in every attention layer.
        """
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            flash_attn_varlen_func,
        )

        if flash_attn_varlen_func is None:
            raise ImportError("Packed variable-length attention requires flash_attn_varlen_func")
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError("Packed variable-length attention currently requires batch size 1")

        out = flash_attn_varlen_func(
            q=query.flatten(0, 1),
            k=key.flatten(0, 1),
            v=value.flatten(0, 1),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = self._unwrap_flash_output(out)
        return out.reshape_as(query)

    def _forward_varlen_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Common wrapper for calling flash_attn_varlen_func for XPU and CUDA in vLLM.

        NOTE: careful to keep the kwargs on these aligned and to pass everything as a keyword
        argument, because some of the args differ positionally at the moment.

        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/vllm_flash_attn/flash_attn_interface.py#L176
        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/_xpu_ops.py#L310
        """
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            flash_attn_varlen_func,
        )

        batch_size, q_len = query.size()[:2]
        k_len = key.size(1)
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * k_len, step=k_len, dtype=torch.int32, device=query.device)
        # b s ... -> (b s) ...
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = self._unwrap_flash_output(out)
        # (b s) h d -> b s h d
        return out.reshape(batch_size, q_len, *out.shape[1:])

    def forward_paged(self, paged_kv_context) -> torch.Tensor:
        """Run platform-native paged attention through the Omni backend.

        ``DiffusionPagedAttentionAdapter`` owns BlockTables and prepares the
        rank-local native cache context.  The adapter no longer performs the
        attention call itself; this method is the backend-owned execution
        boundary.  Its native layer wrapper keeps vLLM version-specific cache
        and kernel details out of Omni's common ``Attention`` layer.
        """

        layer = getattr(paged_kv_context, "layer", None)
        if layer is None:
            raise TypeError("paged_kv_context must expose a native diffusion attention layer")
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None:
            raise RuntimeError(f"Native KV cache is not bound for diffusion layer {layer.layer_name!r}")
        native_impl = getattr(layer, "impl", None)
        if native_impl is None:
            raise RuntimeError(f"Native attention implementation is not bound for diffusion layer {layer.layer_name!r}")
        if not layer.attn_backend.forward_includes_kv_cache_update:
            native_impl.do_kv_cache_update(
                layer,
                paged_kv_context.key_write,
                paged_kv_context.value_write,
                kv_cache,
                paged_kv_context.slot_mapping,
            )

        def run_native_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            native_metadata,
        ) -> torch.Tensor:
            output = torch.empty(
                (query.shape[0], layer.num_heads, layer.head_size_v),
                dtype=query.dtype,
                device=query.device,
            )
            return native_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                native_metadata,
                output,
            )

        if paged_kv_context.piecewise_plan is not None:
            output = run_paged_piecewise_plan(
                paged_kv_context.query,
                paged_kv_context.key_write,
                paged_kv_context.value_write,
                paged_kv_context.piecewise_plan,
                paged_kv_context.piecewise_native_metadata,
                run_native_attention,
            )
        else:
            output = run_native_attention(
                paged_kv_context.query,
                paged_kv_context.key_write,
                paged_kv_context.value_write,
                paged_kv_context.native_metadata,
            )
        return paged_kv_context.restore_output(output)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm/MUSA flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            flash_attn_func,
            flash_attn_varlen_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please install one of: fa3-fwd, flash-attention, or flash-attn. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        full_attn_spans = attn_metadata.full_attn_spans if attn_metadata is not None else None
        extra = attn_metadata.extra if attn_metadata is not None else {}

        # Try piecewise attention
        if full_attn_spans is not None:
            self._warn_fa_deterministic_non_dense("piecewise")
            logger.debug("Using piecewise Flash Attention for mixed causal/full mask")
            if flash_attn_func is not None:
                attn_func = partial(
                    FlashAttentionImpl._flash_wrapper,
                    attn_func=flash_attn_func,
                )
            elif flash_attn_varlen_func is not None:
                attn_func = partial(
                    FlashAttentionImpl._flash_varlen_wrapper,
                    attn_func=flash_attn_varlen_func,
                )
            else:
                raise ImportError("Piecewise FlashAttention requires a dense or varlen FlashAttention function")

            return piecewise_attn(
                query,
                key,
                value,
                full_attn_spans,
                self.softmax_scale,
                attn_func,
                query_ranges=attn_metadata.query_ranges,
            )

        packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
        present_packed_keys = [key for key in packed_keys if key in extra]
        if present_packed_keys:
            if len(present_packed_keys) != len(packed_keys):
                missing = sorted(set(packed_keys) - set(present_packed_keys))
                raise ValueError(f"Incomplete packed FlashAttention metadata; missing {missing}")
            self._warn_fa_deterministic_non_dense("packed-varlen")
            return self._forward_varlen_packed(
                query,
                key,
                value,
                cu_seqlens_q=extra["cu_seqlens_q"],
                cu_seqlens_k=extra["cu_seqlens_k"],
                max_seqlen_q=extra["max_seqlen_q"],
                max_seqlen_k=extra["max_seqlen_k"],
            )

        if attention_mask is not None and torch.any(~attention_mask):
            self._warn_fa_deterministic_non_dense("masked-varlen")
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        if flash_attn_func is not None:
            fa_kwargs = {
                "causal": self.causal,
                "softmax_scale": self.softmax_scale,
            }
            if self.fa_deterministic:
                fa_kwargs["deterministic"] = True
            out = flash_attn_func(query, key, value, **fa_kwargs)
            return self._unwrap_flash_output(out)

        self._warn_fa_deterministic_non_dense("dense-varlen-fallback")
        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """XPU flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please assure vllm-xpu-kernels properly installed. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""

        extra = attn_metadata.extra if attn_metadata else {}
        method = extra.get("kv_cache_dtype", self.quant.get("method"))
        if not extra.get("disable_attention_quant") and method not in (None, "float", "auto"):
            return self.forward_fa_quant_npu(query, key, value, attn_metadata)
        return self.forward_fa_npu(query, key, value, attn_metadata)

    @staticmethod
    def _load_quant_runtime(method: str) -> Callable[..., torch.Tensor]:
        # Import only the selected concrete function, and only inside NPU execution.
        if method == "fp8":
            from mindiesd.layers.flash_attn.quant_flash_attn import fp8_rotate_quant_fa

            return fp8_rotate_quant_fa
        if method == "mxfp8":
            from mindiesd.layers.flash_attn.quant_flash_attn import mxfp8_rotate_quant_fa

            return mxfp8_rotate_quant_fa
        if method == "mxfp4":
            from mindiesd.layers.flash_attn.quant_flash_attn import mxfp4_quant_fa

            return mxfp4_quant_fa
        raise ValueError(f"Unsupported NPU attention quantization method {method!r}.")

    @staticmethod
    def _quant_capability_reason(method: str, *, sparse: bool = False) -> str | None:
        """Check dependencies before launching any quantization/attention operator."""
        try:
            import torch_npu
            from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        except ImportError as exc:
            return f"NPU quantization dependencies unavailable ({exc})"
        a5 = getattr(AscendDeviceType, "A5", None)
        if a5 is None or get_ascend_device_type() != a5:
            return "quantized MindIE attention requires Ascend A5 (950)"

        if method == "fp8":
            dtypes: tuple[str, ...] = ("float8_e4m3fn",)
            npu_ops: tuple[str, ...] = ("npu_dynamic_block_quant",)
            mindie_ops: tuple[str, ...] = ("block_sparse_attention" if sparse else "fused_infer_attention_score_v2",)
        elif method == "mxfp8":
            if getattr(torch, "float8_e4m3fn", None) is None and getattr(torch_npu, "float8_e4m3fn", None) is None:
                return "float8_e4m3fn is unavailable in torch and torch_npu"
            dtypes = ("float8_e8m0fnu",)
            npu_ops = ("npu_dynamic_mx_quant", "npu_fused_infer_attention_score_v2")
            mindie_ops = ()
        elif method == "mxfp4":
            dtypes = ("float4_e2m1fn_x2", "float8_e8m0fnu")
            npu_ops = ("npu_dynamic_mx_quant",)
            mindie_ops = ("quant_flash_attn_metadata", "quant_flash_attn")
        else:
            return f"unsupported quantization method {method!r}"
        for name in dtypes:
            if getattr(torch_npu, name, None) is None:
                return f"torch_npu.{name} is unavailable"
        for name in npu_ops:
            if not callable(getattr(torch_npu, name, None)):
                return f"torch_npu.{name} is unavailable"
        for name in mindie_ops:
            if not callable(getattr(torch.ops.mindiesd, name, None)):
                return f"MindIE operator {name} is unavailable"
        return None

    def _quant_unsupported_reason(
        self,
        method: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        layout = self.qkv_layout
        if layout not in ("BSND", "BNSD"):
            return "quantized FA requires an explicit BSND or BNSD layout"
        if self.causal:
            return "causal quantized FA is not supported"
        extra = attn_metadata.extra if attn_metadata else {}
        if any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k")) or extra.get("npu_attn_varlen"):
            return "packed/varlen metadata requires the float attention path"
        if attn_metadata is not None and attn_metadata.full_attn_spans is not None:
            return "piecewise attention requires the float attention path"
        if any(t.ndim != 4 or t.dtype not in (torch.float16, torch.bfloat16) for t in (query, key, value)):
            return "quantized FA requires four-dimensional BF16/FP16 tensors"
        if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
            return "incompatible Q/K/V geometry"
        if query.device != key.device or query.device != value.device:
            return "Q/K/V devices must match"
        if query.dtype != key.dtype or query.dtype != value.dtype:
            return "Q/K/V dtypes must match"
        seq_axis, head_axis = (1, 2) if layout == "BSND" else (2, 1)
        if min(query.shape[head_axis], key.shape[head_axis]) == 0 or query.shape[head_axis] % key.shape[head_axis]:
            return "Q head count must be a positive multiple of K/V head count"
        if query.shape[seq_axis] == 0 or key.shape[seq_axis] == 0:
            return "empty sequences are not supported"
        dim = query.shape[-1]
        if method in ("fp8", "mxfp8") and (dim == 0 or dim & (dim - 1)):
            return "generated Hadamard rotations require a power-of-two head dimension"
        if method == "fp8" and (query.shape[0] != 1 or query.shape[head_axis] != key.shape[head_axis]):
            return "block-FP8 Runtime requires batch size 1 and equal Q/K/V head counts"
        if method == "mxfp4" and (query.shape[seq_axis] % 512 or key.shape[seq_axis] % 512):
            # Current MindIE pads to 512 and uses the padded length as seqused.
            # Do not let padding enter softmax; relax only with a verified runtime fix.
            return "MXFP4 requires sequence lengths aligned to 512 with the current MindIE runtime"
        mask = attn_metadata.attn_mask if attn_metadata else None
        if mask is not None:
            if method == "mxfp4":
                return "MXFP4 caller masks require a verified MindIE mask/valid-length contract"
            if mask.dtype != torch.bool:
                return "quantized FA requires a boolean keep mask"
            q_len, k_len = query.shape[seq_axis], key.shape[seq_axis]
            if mask.ndim == 2 and tuple(mask.shape) not in ((query.shape[0], k_len), (q_len, k_len)):
                return "invalid two-dimensional attention mask shape"
            if mask.ndim not in (2, 4):
                return "quantized FA supports only two- or four-dimensional masks"
            if mask.ndim == 4 and (
                mask.shape[0] not in (1, query.shape[0])
                or mask.shape[1] not in (1, query.shape[head_axis])
                or mask.shape[2] not in (1, q_len)
                or mask.shape[3] != k_len
            ):
                return "invalid four-dimensional attention mask shape"
            if method == "mxfp8" and query.shape[0] != 1:
                return "masked batched MXFP8 TND attention is not supported"
        return None

    def forward_fa_quant_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        extra = attn_metadata.extra if attn_metadata else {}
        method = extra.get("kv_cache_dtype", self.quant.get("method", "fp8"))
        fallback = extra.get("quant_fallback", self.quant.get("fallback", ()))
        reasons: list[str] = []
        for candidate in (method, *fallback):
            if candidate == "float":
                logger.warning_once("NPU attention falling back to float: %s", "; ".join(reasons))
                return self.forward_fa_npu(query, key, value, attn_metadata)
            reason = self._quant_unsupported_reason(candidate, query, key, value, attn_metadata)
            if reason is not None:
                reasons.append(f"{candidate}: {reason}")
                continue
            try:
                runtime = self._load_quant_runtime(candidate)
            except ImportError as exc:
                if not fallback:
                    raise ImportError(
                        f"NPU {candidate} attention requires a compatible MindIE-SD quant Runtime."
                    ) from exc
                reasons.append(f"{candidate}: MindIE-SD Runtime unavailable ({exc})")
                continue
            reason = self._quant_capability_reason(candidate)
            if reason is not None:
                reasons.append(f"{candidate}: {reason}")
                continue
            mask = attn_metadata.attn_mask if attn_metadata else None
            if mask is not None:
                # Omni uses True=keep. The quant Runtime passes its mask directly
                # to CANN, which uses True=blocked (unlike attention_forward).
                q_bsnd = query if self.qkv_layout == "BSND" else query.transpose(1, 2)
                k_bsnd = key if self.qkv_layout == "BSND" else key.transpose(1, 2)
                mask = ~_maybe_reshape_attn_mask(q_bsnd, k_bsnd, mask, mask_mode="full_qk")
            kwargs = dict(layout=self.qkv_layout, attn_mask=mask, softmax_scale=self.softmax_scale)
            # MXFP4 does not generate rotations or accept rotation_seed.
            if candidate in ("fp8", "mxfp8"):
                kwargs["rotation_seed"] = extra.get("rotation_seed", self.quant.get("rotation_seed", 425500))
            if reasons:
                logger.warning_once("NPU attention falling back to %s: %s", candidate, "; ".join(reasons))
            logger.info_once("NPU attention uses MindIE-SD %s Runtime, layout=%s.", candidate, self.qkv_layout)
            # Execution errors propagate. Never retry after an operator failure.
            return runtime(query, key, value, **kwargs)
        raise ValueError("No supported NPU attention quantization method: " + "; ".join(reasons))

    def forward_fa_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        extra = attn_metadata.extra if attn_metadata else {}
        mask = attn_metadata.attn_mask if attn_metadata else None
        packed = any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k"))
        piecewise = attn_metadata is not None and attn_metadata.full_attn_spans is not None
        use_sdpa = self.causal or query.dtype == torch.float32
        if mask is None and (
            piecewise or ((packed or extra.get("npu_attn_varlen")) and (use_sdpa or not extra.get("npu_attn_varlen")))
        ):
            raise ValueError("NPU float attention requires an explicit mask for unsupported packed/piecewise metadata.")
        if use_sdpa:
            layout = self.qkv_layout or "BNSD"
            q, k, v = (tensor.transpose(1, 2) if layout == "BSND" else tensor for tensor in (query, key, value))
            mask = attn_metadata.attn_mask if attn_metadata else None
            if mask is not None:
                mask = _maybe_reshape_attn_mask(q.transpose(1, 2), k.transpose(1, 2), mask)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=self.causal,
                scale=self.softmax_scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            return out.transpose(1, 2) if layout == "BSND" else out
        try:
            from mindiesd import attention_forward
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )
        # Opt-in mask-free paths (mirror the CUDA cu_seqlens behavior): the
        # model marks extra["npu_attn_varlen"] and carries packed metadata, so
        # the padding document is excluded without reading or materializing
        # the quadratic full_qk mask.
        #   - default: packed varlen via mindiesd attention_forward_varlen;
        #   - when MINDIE_SD_FA_TYPE=ascend_laser_attention: prefix K/V
        #     slicing via mindiesd attention_forward (so the env-selected op
        #     type is honored; the TND varlen op cannot be).
        extra = attn_metadata.extra if attn_metadata else {}
        if extra.get("npu_attn_varlen", False):
            if os.environ.get("MINDIE_SD_FA_TYPE") == "ascend_laser_attention":
                out = self._forward_prefix_kv_slice_npu(query, key, value, extra)
            else:
                out = self._forward_varlen_packed_npu(query, key, value, extra)
            if out is not None:
                return out
        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        if attention_mask is None and extra.get("npu_attn_varlen", False):
            # Models skip mask construction when this opt-in is set
            # (FlashAttentionBackend.supports_packed_mask_free). The packed
            # paths above declined (contract mismatch), so rebuild the padding
            # mask here to keep the masked fallback correct.
            used = extra.get("valid_kv_length")
            if not isinstance(used, int) or not 0 < used <= query.shape[1]:
                raise ValueError(
                    "npu_attn_varlen packed metadata is unusable and no attn_mask "
                    f"was constructed (valid_kv_length={used!r}, seq_len={query.shape[1]}); "
                    "refusing to run unmasked attention over padding rows."
                )
            attention_mask = torch.arange(query.shape[1], device=query.device)[None] < used

        # NPU aclnnFlashAttentionScore requires mask shape to be one of:
        # [B, N, Sq, Skv], [B, 1, Sq, Skv], [1, 1, Sq, Skv], or [Sq, Skv]
        # But the incoming mask is 2D [B, S] — reshape to [B, 1, 1, S]
        # So reuse SDPA's mask reshape logic: [B, S] -> [B, 1, Sq, Skv]
        layout = self.qkv_layout or "BNSD"
        q_bsnd = query if layout == "BSND" else query.transpose(1, 2)
        k_bsnd = key if layout == "BSND" else key.transpose(1, 2)
        attention_mask = _maybe_reshape_attn_mask(q_bsnd, k_bsnd, attention_mask, mask_mode="full_qk")

        return attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=self.softmax_scale,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout=layout,
        )

    def _resolve_packed_seq_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        extra: dict,
    ) -> tuple[list[int], list[int]] | None:
        """Resolve packed document boundaries (cumulative end offsets per doc).

        Returns (seq_q, seq_k), e.g. ([used_q, total_q], [used_k, total_k]),
        or None unless the metadata matches the exact contract MiniMax-H3
        produces:
          - q/k/v are [1, T, N, D] (BSND, single packed batch);
          - cu_seqlens describe a [real, pad] two-document packing, with the
            padding document as a strict suffix;
          - max_seqlen_q/k are Python ints equal to the real document length
            (so boundaries are derived without any device sync).
        """
        if self.causal:
            return None
        packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
        if any(k not in extra for k in packed_keys):
            return None
        cu_q, cu_k = extra["cu_seqlens_q"], extra["cu_seqlens_k"]
        used_q, used_k = extra["max_seqlen_q"], extra["max_seqlen_k"]
        if query.shape[0] != 1 or key.shape[0] != 1:
            return None
        if not isinstance(used_q, int) or not isinstance(used_k, int):
            return None
        total_q, total_k = query.shape[1], key.shape[1]
        # .shape is host-side metadata: counting documents never syncs.
        if cu_q.shape[0] != cu_k.shape[0] or cu_q.shape[0] > 3:
            return None
        if not (0 < used_q <= total_q) or not (0 < used_k <= total_k):
            return None
        if used_q == total_q and used_k == total_k:
            # No padding document: one full-length document.
            return [total_q], [total_k]
        if cu_q.shape[0] == 3 and used_q >= total_q - used_q and used_k >= total_k - used_k:
            # [real, pad] packing: the real document must be the longer one
            # (consistent with the max_seqlen naming).
            return [used_q, total_q], [used_k, total_k]
        return None

    def _forward_varlen_packed_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        extra: dict,
    ) -> torch.Tensor | None:
        """Packed varlen attention on NPU via mindiesd attention_forward_varlen.

        Returns None (caller falls back to the mask path) when the packed
        contract does not hold; see _resolve_packed_seq_npu.
        """
        resolved = self._resolve_packed_seq_npu(query, key, extra)
        if resolved is None:
            return None
        seq_q, seq_k = resolved

        try:
            from mindiesd import attention_forward_varlen
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )
        q = query.squeeze(0)  # [T, N, D] == TND
        k = key.squeeze(0)
        v = value.squeeze(0)
        # attention_forward_varlen wants cu_seqlens as host lists of cumulative
        # offsets and forwards cu[1:] as actual_seq_qlen/actual_seq_kvlen.
        out = attention_forward_varlen(
            q,
            k,
            v,
            [0, *seq_q],
            [0, *seq_k],
            softmax_scale=self.softmax_scale,
        )
        return out.unsqueeze(0)

    def _forward_prefix_kv_slice_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        extra: dict,
    ) -> torch.Tensor | None:
        """Mask-free attention by slicing K/V to the valid prefix (zero-copy
        views), then mindiesd attention_forward without a mask.

        Same numerical contract as the varlen path: the padding document is a
        strict suffix, so dropping it from K/V is identical to masking it out.
        Query keeps full length; outputs on padding rows are never consumed
        downstream. Returns None when the packed contract does not hold.

        Used when MINDIE_SD_FA_TYPE=ascend_laser_attention: the TND varlen op
        cannot honor the env-selected op type, while attention_forward can.

        The laser kernel stores unscaled S=QK^T in an fp16 GM workspace, so
        bf16 activations with large outliers overflow 65504 into ±inf and the
        affected rows turn NaN. Models may opt into exact power-of-two input
        pre-scaling via extra["laser_input_scale"] (see abstract.py): q/k/v
        are divided by the factor, the kernel scale_value is multiplied by its
        square, and the output is scaled back. Absent or invalid factor means
        no pre-scaling.
        """
        resolved = self._resolve_packed_seq_npu(query, key, extra)
        if resolved is None:
            return None
        _, seq_k = resolved
        used_k = seq_k[0]  # real document length (first cumulative end)

        try:
            from mindiesd import attention_forward
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )
        # mindiesd always takes BSND input regardless of `layout`; the arg
        # selects the op-internal layout and laser only supports BNSD, so do
        # NOT forward the model's qkv_layout ("BSND" for MiniMax-H3) here.
        layout = "BNSD"
        key = key[:, :used_k]
        value = value[:, :used_k]

        input_scale = extra.get("laser_input_scale")
        preserve_input_range = isinstance(input_scale, (int, float)) and input_scale > 1
        if preserve_input_range:
            query = query / input_scale
            key = key / input_scale
            value = value / input_scale
        scale = self.softmax_scale
        if preserve_input_range:
            scale *= input_scale**2

        def _postprocess(out: torch.Tensor) -> torch.Tensor:
            return out * input_scale if preserve_input_range else out

        return _postprocess(
            attention_forward(
                query,
                key,
                value,
                attn_mask=None,
                opt_mode="manual",
                op_type="fused_attn_score",  # MINDIE_SD_FA_TYPE env overrides this
                layout=layout,
                scale=scale,
            )
        )
