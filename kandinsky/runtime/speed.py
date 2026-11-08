"""SpeedConfig and the factory-time validation for SPEED.

SPEED is a staged multi-resolution denoising strategy: it wraps the base
denoising loop in :func:`Kandinsky6Pipeline._denoise` with a sequence of
stages that denoise at progressively higher spatial resolutions. The
algorithm lives in :mod:`kandinsky.core.algo.denoise_speed`; this module
holds the one typed section the YAML binds to and the checks that keep SPEED
from colliding with features that assume one contiguous denoising loop.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from kandinsky.core.algo.speed_schedule import (
    DEFAULT_DELTA,
    DEFAULT_SCALES,
    DEFAULT_TRANSITION_TIMES,
    SCALE_TOLERANCE,
    SPECTRUM_A,
    SPECTRUM_BETA,
)

from .base import ConfigModel


class SpeedConfig(ConfigModel):
    """Spectral progressive diffusion settings embedded in the pipeline YAML.

    ``scales`` are spatial latent scales per stage and must end at ``1.0``.
    ``transition_times`` are the flow times at which stages hand over; pass
    ``None`` to derive them from the latent power-law spectrum. A strictly
    increasing scale list uses the spectrum for every leaving scale; a list
    that starts at ``1.0`` uses ``0.99`` for the first hop and the spectrum for
    the rest.

    ``audio_alignment="euler"`` re-integrates audio to the video's aligned time
    at each hop, which is what T2VA needs. ``"none"`` makes audio read the
    aligned time without having been advanced to it.
    """

    enabled: bool = False
    scales: tuple[float, ...] = DEFAULT_SCALES
    transition_times: tuple[float, ...] | None = DEFAULT_TRANSITION_TIMES
    delta: float = Field(default=DEFAULT_DELTA, gt=0.0, lt=1.0)
    spectrum_a: float = Field(default=SPECTRUM_A, gt=0.0)
    spectrum_beta: float = Field(default=SPECTRUM_BETA, gt=0.0)
    audio_alignment: Literal["none", "euler"] = "euler"
    audio_alignment_substeps: int = Field(default=1, ge=1)

    @field_validator("scales", "transition_times", mode="before")
    @classmethod
    def _as_tuple(cls, v):
        if v is None:
            return v
        return tuple(v)

    @field_validator("scales")
    @classmethod
    def _ends_at_full_resolution(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("speed.scales must not be empty")
        if any(s <= 0.0 or s > 1.0 for s in v):
            raise ValueError(f"every speed.scales entry must be in (0, 1]; got {list(v)}")
        if abs(v[-1] - 1.0) > SCALE_TOLERANCE:
            raise ValueError(f"speed.scales must end at 1.0 (full resolution); got {v[-1]}")
        return v


def reject_incompatible_speed(cfg: Any, cache_mode: str | None) -> None:
    """Guard the SPEED stage loop against features it does not yet support.

    PiFlow is rejected because the staged loop currently invokes the Euler
    ``denoise_loop`` per stage, not ``piflow_denoise_loop``. MagCache /
    NaviCache / AdaTaylor are supported: the staged loop calls
    ``CacheDiT.invalidate_shape_state()`` at every DCT transition so residuals
    sized for the previous grid are dropped, while per-step counters and
    scalar statistics (Kalman ratio, accumulated error) are preserved.
    """
    del cache_mode  # intentionally unused — all cache modes are supported
    if not cfg.speed.enabled:
        return
    if cfg.piflow.enabled:
        raise ValueError("speed.enabled is incompatible with piflow.enabled")


__all__ = ["SpeedConfig", "reject_incompatible_speed"]
