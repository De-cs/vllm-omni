# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan integration contracts required by quantized dense/sparse attention."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_block_publishes_unpadded_grid_without_partial_packed_metadata():
    block = object.__new__(WanTransformerBlock)
    torch.nn.Module.__init__(block)
    block.scale_shift_table = torch.zeros(1, 6, 4)
    block.norm1 = lambda x, *a: x
    block.norm2 = lambda x: x
    block.norm3 = lambda x, *a: x
    block.attn1 = Mock(side_effect=lambda x, *a: torch.zeros_like(x))
    block.attn2 = Mock(side_effect=lambda x, *a: torch.zeros_like(x))
    block.ffn = lambda x: torch.zeros_like(x)
    # Local SP tokens include alignment padding; grid describes the full valid video.
    x = torch.zeros(1, 1152, 4)
    block(
        x,
        x,
        torch.zeros(1, 6, 4),
        None,
        hidden_states_mask=torch.ones(1, 1152, dtype=torch.bool),
        vsa_dit_seq_shape=(5, 16, 27),
    )
    metadata = block.attn1.call_args.args[2]
    assert metadata.video_layout.latent_grid == (5, 16, 27)
    assert metadata.video_layout.used_len == 2160
    assert metadata.extra["attn_mask_is_padding"] is True
    assert metadata.video_layout.prefix_len == 0
    assert "max_seqlen_q" not in metadata.extra  # CUDA must not infer incomplete packed varlen.
    assert block.attn2.call_args.args[2] is None


def test_denoise_loop_preserves_global_step_and_total_across_experts():
    high, low = object(), object()
    calls = []
    record = Mock()
    pipeline = SimpleNamespace(
        transformer=high,
        transformer_2=low,
        expand_timesteps=False,
        is_dmd=False,
        progress_bar=lambda **kw: nullcontext(SimpleNamespace(update=lambda: None)),
        record_denoise_step=record,
        predict_noise_maybe_with_cfg=lambda **kw: calls.append(kw["positive_kwargs"]["current_model"]),
        scheduler_step_maybe_with_cfg=lambda noise, t, latents, cfg: latents,
    )
    latents = torch.zeros(1, 4, 1, 2, 2)
    kwargs = dict(
        latents=latents,
        timesteps=torch.tensor([900, 500, 100]),
        prompt_embeds=torch.zeros(1, 2, 4),
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=1.0,
        boundary_timestep=600,
        dtype=torch.float32,
        attention_kwargs={},
    )
    assert Wan22Pipeline.diffuse(pipeline, **kwargs) is latents
    assert calls == [high, low, low]
    assert [call.args[0] for call in record.call_args_list] == [0, 1, 2]
    assert [call.kwargs["total_steps"] for call in record.call_args_list] == [3, 3, 3]
