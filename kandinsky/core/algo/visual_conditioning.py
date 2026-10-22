"""Tensor-only first/last reference transforms for native inference and ports."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .prepare_latents import append_i2va_tail_condition
from .prepare_ropes import compute_visual_rope

BATCHED_VIDEO_NDIM = 5


def append_visual_references(
    video: Tensor,
    references: Tensor,
    batch_size: int,
    video_duration: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append (B,R,H,W,C) references using the existing single-reference primitive.

    Video accepts native (B,T,H,W,C) and packed (B*T,H,W,C) layouts.
    Returns video, role ids (0=noise, 1=reference), and the generated-frame mask.
    """
    if references.ndim != BATCHED_VIDEO_NDIM or references.shape[1] not in (1, 2):
        raise ValueError("expected one or two references with shape (B,R,H,W,C)")
    role_ids = torch.empty(0, device=video.device, dtype=torch.long)
    for index in range(references.shape[1]):
        video, role_ids, _ = append_i2va_tail_condition(
            video,
            references[:, index],
            batch_size,
            video_duration + index,
        )
    role_ids.reshape(batch_size, -1)[:, video_duration:] = 1
    return video, role_ids, role_ids == 0


def restore_tail_references(video: Tensor, references: Tensor | None, cu_seqlens: Tensor) -> None:
    """Restore clean tail latents in place; accept legacy (B,H,W,C) or (B,R,H,W,C)."""
    if references is None:
        return
    refs = references if references.ndim == BATCHED_VIDEO_NDIM else references[:, None]
    count = refs.shape[1]
    if video.ndim == BATCHED_VIDEO_NDIM:
        video[:, -count:] = refs.to(video)
    else:
        positions = cu_seqlens[1:, None] - count + torch.arange(count, device=video.device)
        video[positions.flatten()] = refs.to(video).flatten(0, 1)


def build_tail_condition(video: Tensor, references: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Restore references and concatenate zero auxiliary channels plus channel mask."""
    restore_tail_references(video, references, cu_seqlens)
    count = references.shape[1] if references.ndim == BATCHED_VIDEO_NDIM else 1
    mask = torch.zeros_like(video[..., :1])
    if video.ndim == BATCHED_VIDEO_NDIM:
        mask[:, -count:] = 1
    else:
        positions = cu_seqlens[1:, None] - count + torch.arange(count, device=video.device)
        mask[positions.flatten()] = 1
    return torch.cat((video, torch.zeros_like(video), mask), dim=-1)


def append_reference_rope(visual_rope: Tensor, reference_count: int) -> Tensor:
    """Append t0 or t0/t_last rotary matrices without changing spatial coordinates."""
    if reference_count not in (1, 2) or visual_rope.shape[0] < 1:
        raise ValueError("reference RoPE requires generated frames and one or two references")
    positions = [0] if reference_count == 1 else [0, visual_rope.shape[0] - 1]
    return torch.cat((visual_rope, visual_rope[positions]), dim=0)


def compute_reference_rope(
    rope: nn.Module,
    shape: tuple[int, int, int],
    scale: tuple[float, float, float],
    count: int,
) -> Tensor:
    """Recompute generated/reference RoPE; count=0 preserves the existing debug path."""
    if count == 0:
        return compute_visual_rope(rope, shape, scale)
    base = compute_visual_rope(rope, (shape[0] - count, *shape[1:]), scale)
    return append_reference_rope(base, count)
