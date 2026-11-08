"""SPEED resolution schedule — spectrum model and per-stage planning.

The transition schedule is derived from a radial power-law spectrum
``P(omega) = a * |omega|**(-beta)`` fitted to clean VAE latents. A band at
``omega`` becomes distinguishable from noise only after flow time drops below
its activation time, so a stage running at scale ``s`` may be truncated at
``activation_time(P(s * omega_max))`` without exceeding the per-band error
budget ``delta``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Calibrated latent power-spectrum parameters (K6 VAE, 512x768 latents).
SPECTRUM_A: float = 834.507821
SPECTRUM_BETA: float = 2.54376768

DEFAULT_DELTA: float = 0.01

# Full → half → full. The first drop (from 1.0) has no spectrum time, so the
# first entry is a fixed default and the rest come from the spectrum.
DEFAULT_SCALES: tuple[float, ...] = (1.0, 0.5, 1.0)
DEFAULT_TRANSITION_TIMES: tuple[float, ...] = (0.99, 0.92)

SCALE_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class SpeedStage:
    """One constant-resolution segment of the denoising trajectory."""

    step_start: int  # inclusive
    step_end: int  # exclusive
    scale: float
    latent_hw: tuple[int, int]


def power_spectrum(omega: float, a: float, beta: float) -> float:
    return a * abs(omega) ** (-beta)


def activation_time(p_omega: float, delta: float) -> float:
    if delta >= 1.0:
        raise ValueError(f"delta={delta} >= 1, but the error threshold must be < 1.")
    return 1.0 / (1.0 + math.sqrt(delta / (p_omega * (1.0 + p_omega - delta))))


def _validate_strictly_increasing(scales: Sequence[float]) -> None:
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > SCALE_TOLERANCE:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:], strict=True):
        if not a < b:
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def _validate_bounds_only(scales: Sequence[float]) -> None:
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")


def _spectrum_times(
    scales: Sequence[float],
    delta: float,
    a: float,
    beta: float,
    omega_max: float,
) -> list[float]:
    return [activation_time(power_spectrum(scale * omega_max, a, beta), delta) for scale in scales]


def delta_optimal_transitions(  # noqa: PLR0913
    scales: Sequence[float],
    delta: float,
    a: float,
    beta: float,
    height: int,
    width: int,
) -> list[float]:
    """Return the flow time at which each adjacent scale pair should transition.

    A schedule that starts at full resolution has no closed-form time for that
    first drop; it uses ``DEFAULT_TRANSITION_TIMES[0]`` (0.99). Every later
    leaving scale uses the spectrum formula. A strictly increasing list uses
    the formula for every leaving scale.
    """
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    omega_max = min(height, width) / 2.0
    if abs(scales[0] - 1.0) <= SCALE_TOLERANCE:
        if any(s <= 0.0 or s > 1.0 for s in scales):
            raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
        if abs(scales[-1] - 1.0) > SCALE_TOLERANCE:
            raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
        if len(scales) == 1:
            return []
        return [
            DEFAULT_TRANSITION_TIMES[0],
            *_spectrum_times(scales[1:-1], delta, a, beta, omega_max),
        ]
    _validate_strictly_increasing(scales)
    return _spectrum_times(scales[:-1], delta, a, beta, omega_max)


def round_to_multiple(value: float, multiple: int, *, mode: str = "ceil") -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    if mode == "nearest":
        q = math.floor(value / multiple + 0.5)
    elif mode == "floor":
        q = math.floor(value / multiple)
    elif mode == "ceil":
        q = math.ceil(value / multiple)
    else:
        raise ValueError(f"mode must be 'nearest', 'floor', or 'ceil', got {mode!r}")
    return max(multiple, q * multiple)


def scaled_latent_hw(
    scale: float,
    full_h: int,
    full_w: int,
    patch_size: Sequence[int],
    *,
    mode: str = "ceil",
) -> tuple[int, int]:
    """Scale a latent grid, keeping both sides divisible by the DiT patch size."""
    patch_h, patch_w = int(patch_size[1]), int(patch_size[2])
    h = round_to_multiple(scale * full_h, patch_h, mode=mode)
    w = round_to_multiple(scale * full_w, patch_w, mode=mode)
    if abs(scale - 1.0) < SCALE_TOLERANCE:
        h, w = full_h, full_w
    if h % patch_h != 0 or w % patch_w != 0:
        raise ValueError(f"Resolved latent size {(h, w)} is not divisible by patch size {(patch_h, patch_w)}")
    return h, w


def find_first_step_below(sigmas: Iterable[float], threshold: float) -> int:
    values = [float(sigma) for sigma in sigmas]
    n_steps = len(values) - 1
    for i in range(n_steps):
        if values[i] <= threshold:
            return i
    return n_steps


def plan_stages(  # noqa: PLR0913
    scales: Sequence[float],
    timesteps: Iterable[float],
    full_h: int,
    full_w: int,
    patch_size: Sequence[int],
    *,
    transition_times: Sequence[float] | None = None,
    delta: float = DEFAULT_DELTA,
    spectrum_a: float = SPECTRUM_A,
    spectrum_beta: float = SPECTRUM_BETA,
) -> list[SpeedStage]:
    """Resolve a scale schedule into contiguous stages over ``timesteps``.

    ``transition_times`` are flow times; when omitted they are derived from the
    latent power-law spectrum. Each is mapped to the first step whose sigma has
    reached it or dropped below it, and those indices become the stage
    boundaries.
    """
    _validate_bounds_only(scales)
    timesteps = list(timesteps)
    num_steps = len(timesteps) - 1
    if num_steps < 1:
        raise ValueError(f"timesteps must hold at least two points, got {len(timesteps)}")

    if len(scales) == 1:
        return [SpeedStage(0, num_steps, scales[0], scaled_latent_hw(scales[0], full_h, full_w, patch_size))]

    if transition_times is None:
        transition_times = delta_optimal_transitions(
            scales,
            delta=delta,
            a=spectrum_a,
            beta=spectrum_beta,
            height=full_h,
            width=full_w,
        )
    if len(transition_times) != len(scales) - 1:
        raise ValueError(
            f"expected {len(scales) - 1} transition times for {len(scales)} scales, got {len(transition_times)}"
        )

    boundaries = [find_first_step_below(timesteps, t) for t in transition_times]
    if any(a >= b for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)):
        raise ValueError(
            f"transition times {list(transition_times)} map to non-increasing step indices "
            f"{boundaries}; increase num_steps or spread the transitions further apart"
        )
    if boundaries[0] < 1 or boundaries[-1] >= num_steps:
        raise ValueError(
            f"transition times {list(transition_times)} map to step indices {boundaries}, "
            f"which leave an empty stage in a {num_steps}-step schedule"
        )

    starts = [0, *boundaries]
    ends = [*boundaries, num_steps]
    return [
        SpeedStage(start, end, scale, scaled_latent_hw(scale, full_h, full_w, patch_size))
        for start, end, scale in zip(starts, ends, scales, strict=True)
    ]


__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_SCALES",
    "DEFAULT_TRANSITION_TIMES",
    "SCALE_TOLERANCE",
    "SPECTRUM_A",
    "SPECTRUM_BETA",
    "SpeedStage",
    "activation_time",
    "delta_optimal_transitions",
    "find_first_step_below",
    "plan_stages",
    "power_spectrum",
    "round_to_multiple",
    "scaled_latent_hw",
]
