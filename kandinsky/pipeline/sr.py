"""K6 adapter for the packaged super-resolution pipeline.

The SR implementation lives in the external ``kandinsky_sr`` package and knows
nothing about K6 configs or offload strategies. This module bridges the two:
it reads the ``sr:`` and ``offload:`` sections of a K6 pipeline config, builds
the K6 offload handle, and hands both to :func:`kandinsky_sr.pipeline.factory.load_sr_pipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from kandinsky_sr.pipeline.factory import load_sr_pipeline as load_packaged_sr_pipeline

from kandinsky.runtime.offload import OffloadStrategy, build_offload
from kandinsky.pipeline.config import PipelineConfig, load_config


def load_sr_pipeline(  # noqa: PLR0913
    config: PipelineConfig | str | Path | Sequence[str],
    device: str | torch.device,
    *,
    force: bool = False,
    checkpoint_path: str | Path | None = None,
    vae_path: str | Path | None = None,
    latent_upscaler_config: str | Path | None = None,
    resolution_scale: float | None = None,
    source_vae: torch.nn.Module | None = None,
    offload_strategy: OffloadStrategy | None = None,
) -> Any:
    """Load the optional SR model described by a K6 pipeline config.

    ``force=True`` is used by explicit benchmark or notebook flags; the
    selected config still supplies model paths unless an override is given.
    The top-level ``offload`` section also controls SR module residency unless
    ``offload_strategy`` is passed explicitly. Returns ``None`` when SR is
    disabled and not forced.
    """
    cfg = config if isinstance(config, PipelineConfig) else load_config(config)
    if not force and not cfg.sr.enabled:
        return None

    strategy = offload_strategy if offload_strategy is not None else cfg.offload.strategy
    offload = build_offload(strategy, torch.device(device), pin_memory=cfg.offload.pin_memory)
    return load_packaged_sr_pipeline(
        cfg.sr,
        device,
        force=force,
        checkpoint_path=checkpoint_path,
        vae_path=vae_path,
        latent_upscaler_config=latent_upscaler_config,
        resolution_scale=resolution_scale,
        source_vae=source_vae,
        offload=offload,
    )


__all__ = ["load_sr_pipeline"]
