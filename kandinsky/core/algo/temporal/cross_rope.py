"""Rotary embeddings on a shared time span for audio-video cross-attention."""

import torch
from torch import Tensor, nn

from ..prepare_ropes import compute_rope1d


def global_cross_ropes(
    dit: nn.Module,
    grid: tuple[int, int, int],
    full_frames: int,
    audio_length: int,
    scale: tuple[float, float, float],
) -> tuple[Tensor, Tensor]:
    """Give the reduced video grid the same endpoints as the full video grid."""
    t, h, w = grid
    rope = dit.visual_rope
    if any(n > limit for n, limit in zip((full_frames, h, w), rope.max_pos, strict=True)):
        raise ValueError("Video grid exceeds the RoPE table capacity")
    if audio_length > dit.audio_rope.max_pos:
        raise ValueError("Audio sequence exceeds the RoPE table capacity")

    positions = torch.linspace(0, full_frames - 1, t, device=rope.args_0.device)
    time = positions[:, None] * rope.args_0[1] / scale[0]
    rows = rope.args_1[:h] / scale[1]
    cols = rope.args_2[:w] / scale[2]
    angles = torch.cat(
        [
            time[:, None, None].expand(t, h, w, -1),
            rows[None, :, None].expand(t, h, w, -1),
            cols[None, None, :].expand(t, h, w, -1),
        ],
        dim=-1,
    )
    cos, sin = angles.cos(), angles.sin()
    video_rope = torch.stack([cos, -sin, sin, cos], dim=-1).view(t * h * w, 1, -1, 2, 2)
    return video_rope, compute_rope1d(dit.audio_rope, audio_length)
