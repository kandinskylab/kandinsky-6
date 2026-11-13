"""Temporal DCT, resolution changes, and high-frequency energy."""

from __future__ import annotations

import math

import torch

from ..dct import align_timestep, kappa

_DCT_MATRIX_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _dct_matrix(length: int, device: torch.device) -> torch.Tensor:
    key = (length, device.type, device.index)
    matrix = _DCT_MATRIX_CACHE.get(key)
    if matrix is None:
        frequency = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        position = torch.arange(length, device=device, dtype=torch.float32)[None, :]
        matrix = torch.cos(math.pi / length * (position + 0.5) * frequency)
        matrix *= math.sqrt(2.0 / length)
        matrix[0] /= math.sqrt(2.0)
        _DCT_MATRIX_CACHE[key] = matrix
    return matrix


def _temporal_dct(x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    frames = x.shape[0]
    with torch.autocast(device_type=x.device.type, enabled=False):
        matrix = _dct_matrix(frames, x.device)
        flat = x.float().reshape(frames, -1)
        result = (matrix.T if inverse else matrix) @ flat
        return result.reshape(x.shape)


def _contract_sigma(sigma: float | torch.Tensor, ratio: float) -> float | torch.Tensor:
    return sigma / (ratio - (ratio - 1.0) * sigma)


def _temporal_contract(
    x: torch.Tensor, target_frames: int, sigma: float,
) -> tuple[torch.Tensor, float]:
    ratio = math.sqrt(x.shape[0] / target_frames)
    sigma_after = _contract_sigma(sigma, ratio)
    with torch.autocast(device_type=x.device.type, enabled=False):
        retained = _temporal_dct(x)[:target_frames]
        output = _temporal_dct(retained, inverse=True) / kappa(sigma_after, ratio)
    return output.to(dtype=x.dtype), sigma_after


def _temporal_expand(
    x: torch.Tensor, target_frames: int, sigma: float, master_noise: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    ratio = math.sqrt(target_frames / x.shape[0])
    with torch.autocast(device_type=x.device.type, enabled=False):
        coefficients = _temporal_dct(master_noise) * float(sigma)
        coefficients[:x.shape[0]] = _temporal_dct(x)
        output = _temporal_dct(coefficients, inverse=True) * kappa(sigma, ratio)
    return output.to(dtype=x.dtype), align_timestep(sigma, ratio)


def temporal_high_ratio(clean: torch.Tensor) -> torch.Tensor:
    """Return the fraction of predicted-clean energy in the temporal high band."""
    time_power = _temporal_dct(clean).square()
    frequency = torch.arange(clean.shape[0], device=clean.device) / max(clean.shape[0] - 1, 1)
    high = time_power[frequency >= 0.25].sum()
    total = time_power.sum()
    return high / total.clamp_min(torch.finfo(torch.float32).eps)
