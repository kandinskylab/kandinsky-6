"""Staged multi-resolution denoising (spatial SPEED).

Each stage is a plain single-resolution Euler run delegated to
:func:`kandinsky.core.algo.denoise_loop.denoise_loop` over a slice of a shared
timestep schedule. Between stages the latent is expanded in the DCT domain and
the boundary timestep is replaced by the aligned time, so the next stage
resumes from a consistent flow state.

Video and audio share that one schedule. With ``audio_alignment="euler"`` the
audio latent is integrated up to the aligned time at each hop, which keeps the
shared clock correct; with ``"none"`` audio reads the aligned time without
having been advanced to it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ..types import LatentBundle, TextEmbeds
from .dct import dct_crop_resize_and_align, dct_spatial_resize_bhwc
from .denoise_loop import denoise_loop
from .generation import ImageCondition
from .speed_schedule import SpeedStage
from .visual_conditioning import append_reference_rope

AudioAlignment = Literal["none", "euler"]


def _spatial_hw(video: Tensor) -> tuple[int, int]:
    if video.ndim != 5:  # noqa: PLR2004
        raise ValueError(f"SPEED video latent must be 5D BTHWC, got shape {tuple(video.shape)}")
    return int(video.shape[2]), int(video.shape[3])


def _video_to_spatial(video: Tensor) -> Tensor:
    """Permute ``BTHWC`` so the last two dims are ``H, W`` for DCT."""
    if video.ndim != 5:  # noqa: PLR2004
        raise ValueError(f"SPEED video latent must be 5D BTHWC, got shape {tuple(video.shape)}")
    return video.permute(0, 4, 1, 2, 3).contiguous()


def _spatial_to_video(x: Tensor) -> Tensor:
    """Permute DCT layout ``BCTHW`` back to ``BTHWC``."""
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _stage_visual_rope(
    build_visual_rope: Callable[[int, int], Tensor],
    latent_hw: tuple[int, int],
    patch_size: Sequence[int],
    *,
    reference_count: int,
) -> Tensor:
    visual_rope = build_visual_rope(
        latent_hw[0] // int(patch_size[1]),
        latent_hw[1] // int(patch_size[2]),
    )
    if reference_count:
        visual_rope = append_reference_rope(visual_rope, reference_count)
    return visual_rope


def _align_audio(  # noqa: PLR0913
    bundle: LatentBundle,
    *,
    t_start: float,
    t_end: float,
    substeps: int,
    dtype: torch.dtype,
    loop_kwargs: dict[str, Any],
) -> Tensor:
    """Integrate audio from ``t_start`` to ``t_end`` with video frozen at ``t_end``.

    ``denoise_loop`` pins the non-sampled modality to ``timesteps[-1]``, so
    passing the sub-step grid with ``sample_video=False`` evaluates the video
    branch at the aligned time while only audio advances.
    """
    audio = bundle.audio
    if audio is None:
        raise ValueError("audio alignment needs an audio latent")

    align_timesteps = torch.linspace(float(t_start), float(t_end), substeps + 1, device=audio.device, dtype=dtype)
    aligned = denoise_loop(
        bundle=bundle,
        timesteps=align_timesteps,
        sample_video=False,
        sample_audio=True,
        **loop_kwargs,
    )
    assert aligned.audio is not None
    return aligned.audio


@torch.no_grad()
def denoise_speed(  # noqa: PLR0913
    bundle: LatentBundle,
    dit: nn.Module,
    *,
    stages: Sequence[SpeedStage],
    timesteps: Tensor,
    text_embeds: TextEmbeds,
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    build_visual_rope: Callable[[int, int], Tensor],
    audio_rope: Tensor | None,
    text_rope: Tensor | list[Tensor],
    null_text_rope: Tensor | list[Tensor | None],
    guidance_weight: float,
    patch_size: Sequence[int],
    seed: int,
    audio_alignment: AudioAlignment = "euler",
    audio_alignment_substeps: int = 1,
    attention_mask: Tensor | None = None,
    null_attention_mask: Tensor | None = None,
    visual_token_type_ids: Tensor | None = None,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    image: ImageCondition | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> LatentBundle:
    """Run the SPEED trajectory and return the full-resolution latent bundle.

    ``build_visual_rope(h_patches, w_patches)`` supplies the rotary grid for a
    stage; the caller owns the RoPE module and any caching. ``bundle.video``
    must already be sampled at ``stages[0].latent_hw``. Layout is 5D ``BTHWC``.
    """
    if not stages:
        raise ValueError("stages must not be empty")
    if audio_alignment not in ("none", "euler"):
        raise ValueError(f"audio_alignment must be 'none' or 'euler', got {audio_alignment!r}")
    if audio_alignment_substeps < 1:
        raise ValueError(f"audio_alignment_substeps must be a positive integer, got {audio_alignment_substeps}")
    if bundle.video is None:
        raise ValueError("SPEED needs a video latent to rescale")
    initial_hw = _spatial_hw(bundle.video)
    if initial_hw != tuple(stages[0].latent_hw):
        raise ValueError(
            f"initial latent {initial_hw} does not match the first stage resolution {tuple(stages[0].latent_hw)}"
        )

    schedule = timesteps.clone()
    video: Tensor = bundle.video
    audio: Tensor | None = bundle.audio
    sample_audio = audio is not None

    image = image if image is not None else ImageCondition.parse("pretrain")
    # The DiT input length depends on the visual rope length; for tail_cond the
    # pipeline appended a time slot for each reference frame, so each per-stage
    # visual rope must do the same.
    reference_count = image.reference_count

    forward_kwargs: dict[str, Any] = dict(
        dit=dit,
        text_embeds=text_embeds,
        null_text_embeds=null_text_embeds,
        audio_rope=audio_rope,
        text_rope=text_rope,
        null_text_rope=null_text_rope,
        num_steps=0,  # unused: timesteps= is passed through
        guidance_weight=guidance_weight,
        scheduler_scale=1.0,
        image=image,
        attention_mask=attention_mask,
        null_attention_mask=null_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
        scale_factor=scale_factor,
    )

    for index, stage in enumerate(stages):
        visual_rope = _stage_visual_rope(
            build_visual_rope,
            stage.latent_hw,
            patch_size,
            reference_count=reference_count,
        )
        result = denoise_loop(
            bundle=LatentBundle(
                video=video,
                audio=audio,
                video_cu_seqlens=bundle.video_cu_seqlens,
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            ),
            visual_rope=visual_rope,
            timesteps=schedule[stage.step_start : stage.step_end + 1],
            sample_audio=sample_audio,
            progress_callback=progress_callback,
            **forward_kwargs,
        )
        assert result.video is not None  # denoise_loop preserves the video latent
        video, audio = result.video, result.audio

        if index + 1 == len(stages):
            break

        next_stage = stages[index + 1]
        if next_stage.latent_hw == stage.latent_hw:
            continue

        t_now = float(schedule[stage.step_end].item())
        x_spatial, t_tilde = dct_crop_resize_and_align(
            _video_to_spatial(video),
            target_hw=next_stage.latent_hw,
            r=next_stage.scale / stage.scale,
            t=t_now,
            seed=seed + (index + 1) * 10000,
        )
        video = _spatial_to_video(x_spatial)

        # Residual caches (MagCache / NaviCache / AdaTaylor) are sized for the
        # previous spatial grid — tell the wrapper to drop them so the next
        # forward recomputes at the new shape.
        invalidate = getattr(dit, "invalidate_shape_state", None)
        if callable(invalidate):
            invalidate()

        if image.frames is not None:
            image = image.with_frames(dct_spatial_resize_bhwc(image.frames, next_stage.latent_hw))
            forward_kwargs["image"] = image

        if sample_audio and audio_alignment != "none":
            audio = _align_audio(
                LatentBundle(
                    video=video,
                    audio=audio,
                    video_cu_seqlens=bundle.video_cu_seqlens,
                    audio_cu_seqlens=bundle.audio_cu_seqlens,
                ),
                t_start=t_now,
                t_end=t_tilde,
                substeps=audio_alignment_substeps,
                dtype=schedule.dtype,
                loop_kwargs={
                    **forward_kwargs,
                    "visual_rope": _stage_visual_rope(
                        build_visual_rope,
                        next_stage.latent_hw,
                        patch_size,
                        reference_count=reference_count,
                    ),
                },
            )

        schedule[stage.step_end] = t_tilde

    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=bundle.audio_cu_seqlens,
    )


__all__ = ["AudioAlignment", "denoise_speed"]
