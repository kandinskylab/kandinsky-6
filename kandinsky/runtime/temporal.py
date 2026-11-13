"""Configuration for temporal multiscale denoising."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from .base import ConfigModel
from .offload import OffloadStrategy

if TYPE_CHECKING:
    from kandinsky.pipeline.config import PipelineConfig


class TemporalConfig(ConfigModel):
    mode: Literal["off", "fixed", "adaptive"] = "off"
    switch_after_calls: int = Field(default=2, ge=1, le=8)
    threshold: float = Field(default=0.00036778922367375344, gt=0, allow_inf_nan=False)


def validate_temporal_and_block_cache(
    cfg: PipelineConfig,
    cache_mode: str | None,
    offload_strategy: OffloadStrategy,
) -> None:
    temporal = cfg.temporal.mode != "off"
    block_cache = cache_mode == "adaptive_block"
    if not temporal and not block_cache:
        return
    if not cfg.piflow.enabled or cfg.checkpoint != "pro-distill":
        raise ValueError("Temporal and Adaptive Block Cache require the pro-distill PiFlow checkpoint")
    gen = cfg.generation
    if (gen.height, gen.width, gen.latent_frames, gen.num_steps) != (480, 864, 31, 10):
        raise ValueError("Temporal and Adaptive Block Cache require 864x480, 31 latent frames and 10 steps")
    if cfg.dit.patch_size[0] != 1 or cfg.dit.num_visual_blocks != 60:
        raise ValueError("Temporal and Adaptive Block Cache require the native 60-block Pro Distill DiT")
    if temporal and not cfg.dit.ca_rope:
        raise ValueError("Temporal requires A/V cross-attention RoPE")
    if temporal and cache_mode not in (None, "none", "adaptive_block"):
        raise ValueError("Temporal supports cache.mode=none or adaptive_block")
    if cfg.paths.dit_export is not None or offload_strategy == "block":
        raise ValueError("Temporal and Adaptive Block Cache support eager or regional torch.compile with none/module offload")
    if cfg.speed.enabled:
        raise ValueError("Temporal and Adaptive Block Cache cannot be combined with spatial SPEED")
