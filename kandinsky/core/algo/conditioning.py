"""Device-local tensor transforms for clean modality conditioning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..components.hunyuan_vae import AutoencoderKLHunyuanVideo
    from ..components.mmaudio.features_utils import FeaturesUtils


@torch.no_grad()
def encode_audio_condition(waveform: Tensor, audio_vae: FeaturesUtils) -> Tensor:
    """Encode mono samples (N,) on the VAE device to clean latents (1, A, D)."""
    if waveform.ndim != 1:
        raise ValueError("audio conditioning expects mono samples with shape (N,)")
    return (audio_vae.wrapped_encode(waveform.unsqueeze(0)).transpose(1, 2) * audio_vae.scaling_factor).to(
        dtype=torch.bfloat16
    )


@torch.no_grad()
def encode_video_condition(frames: Tensor, vae: AutoencoderKLHunyuanVideo) -> Tensor:
    """Encode RGB uint8 (B, 3, T, H, W) on the VAE device to (B, T_lat, H_lat, W_lat, C)."""
    if frames.ndim != 5 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
        raise ValueError("video conditioning expects RGB uint8 frames with shape (B, 3, T, H, W)")
    pixels = frames.to(dtype=next(vae.parameters()).dtype) / 127.5 - 1.0
    posterior = vae.encode(pixels).latent_dist
    return (posterior.mode() * vae.config["scaling_factor"]).permute(0, 2, 3, 4, 1).to(torch.bfloat16)
