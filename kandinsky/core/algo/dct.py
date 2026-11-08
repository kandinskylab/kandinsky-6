"""Orthonormal DCT-II spatial resize with flow-time alignment.

Torch-native so the latent never leaves the device. Upscaling keeps the source
coefficients and fills the new high-frequency band with ``t``-scaled noise, then
rescales the whole latent by ``kappa`` so the result is a valid flow-matching
state at the aligned time ``t_tilde = t * kappa``.
"""

from __future__ import annotations

import math

import torch


def _dct_1d(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().reshape(-1, n)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip(dims=[1])], dim=1)
    vc = torch.fft.fft(v, dim=1)

    k = -torch.arange(n, device=x.device, dtype=x.dtype) * math.pi / (2 * n)
    out = vc.real * torch.cos(k) - vc.imag * torch.sin(k)

    out[:, 0] /= math.sqrt(n) * 2
    out[:, 1:] /= math.sqrt(n / 2) * 2

    return (2 * out).reshape(x_shape)


def _idct_1d(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().reshape(-1, n) / 2

    x[:, 0] *= math.sqrt(n) * 2
    x[:, 1:] *= math.sqrt(n / 2) * 2

    k = torch.arange(n, device=x.device, dtype=x.dtype) * math.pi / (2 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_i = torch.cat([torch.zeros_like(x[:, :1]), -x.flip(dims=[1])[:, :-1]], dim=1)
    v_r = x * w_r - v_t_i * w_i
    v_i = x * w_i + v_t_i * w_r

    v = torch.fft.ifft(torch.complex(v_r, v_i), dim=1).real

    out = torch.empty_like(v)
    out[:, ::2] = v[:, : n - n // 2]
    out[:, 1::2] = v.flip(dims=[1])[:, : n // 2]

    return out.reshape(x_shape)


def dct_2d_ortho(x: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal DCT-II over the last two dimensions."""
    return _dct_1d(_dct_1d(x).transpose(-1, -2)).transpose(-1, -2)


def idct_2d_ortho(x: torch.Tensor) -> torch.Tensor:
    """Invert :func:`dct_2d_ortho` over the last two dimensions."""
    return _idct_1d(_idct_1d(x).transpose(-1, -2)).transpose(-1, -2)


def dct_spatial_resize(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize last two dims in DCT space with zero-filled high bands.

    Unlike :func:`dct_crop_resize_and_align` this does not apply flow ``kappa``
    or inject ``t``-scaled noise — use it for clean reference frames.
    """
    h_tgt, w_tgt = target_hw
    h_src, w_src = x.shape[-2], x.shape[-1]
    if h_tgt == h_src and w_tgt == w_src:
        return x

    x_float = x.detach().to(dtype=torch.float32)
    coeffs = dct_2d_ortho(x_float)

    if h_tgt <= h_src and w_tgt <= w_src:
        resized_coeffs = coeffs[..., :h_tgt, :w_tgt]
    else:
        resized_coeffs = torch.zeros(
            *coeffs.shape[:-2],
            h_tgt,
            w_tgt,
            device=x.device,
            dtype=torch.float32,
        )
        resized_coeffs[..., : min(h_src, h_tgt), : min(w_src, w_tgt)] = coeffs[
            ..., : min(h_src, h_tgt), : min(w_src, w_tgt)
        ]

    return idct_2d_ortho(resized_coeffs).to(dtype=x.dtype)


def dct_spatial_resize_bhwc(frames: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize ``(B, H, W, C)`` first-frame latents in DCT space."""
    if frames.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"expected (B, H, W, C), got shape {tuple(frames.shape)}")
    x = frames.permute(0, 3, 1, 2).contiguous()
    return dct_spatial_resize(x, target_hw).permute(0, 2, 3, 1).contiguous()


def kappa(t: float, r: float) -> float:
    """Return the state-rescaling factor after spectral noise expansion."""
    return r / (1.0 + (r - 1.0) * t)


def align_timestep(t: float, r: float) -> float:
    """Return the aligned flow-matching time after spectral noise expansion."""
    return t * kappa(t, r)


def dct_crop_resize_and_align(
    x: torch.Tensor,
    target_hw: tuple[int, int],
    r: float,
    t: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Resize ``x`` in DCT space and return the aligned flow time.

    ``r`` is ``next_scale / current_scale``; ``t`` is the current flow time.
    When upscaling, the new high-frequency band is filled with ``t``-scaled
    noise seeded by ``seed``.
    """
    h_tgt, w_tgt = target_hw
    h_src, w_src = x.shape[-2], x.shape[-1]

    x_float = x.detach().to(dtype=torch.float32)
    coeffs = dct_2d_ortho(x_float)

    if h_tgt <= h_src and w_tgt <= w_src:
        resized_coeffs = coeffs[..., :h_tgt, :w_tgt]
    else:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(int(seed))

        resized_coeffs = t * torch.randn(
            *coeffs.shape[:-2],
            h_tgt,
            w_tgt,
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        resized_coeffs[..., :h_src, :w_src] = coeffs

    scale = kappa(t, r)
    t_tilde = align_timestep(t, r)
    out = scale * idct_2d_ortho(resized_coeffs)

    return out.to(dtype=x.dtype), t_tilde


__all__ = [
    "align_timestep",
    "dct_2d_ortho",
    "dct_crop_resize_and_align",
    "dct_spatial_resize",
    "dct_spatial_resize_bhwc",
    "idct_2d_ortho",
    "kappa",
]
