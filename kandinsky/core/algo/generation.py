"""Typed generation request. The pipeline parses a call once and the samplers trust it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..types import LatentBundle
from .visual_conditioning import (
    BATCHED_VIDEO_NDIM,
    build_tail_condition,
    restore_tail_references,
)

ImageKind = Literal["pretrain", "i2v", "tail"]
HeldKind = Literal["none", "audio", "video"]

_GUIDANCE_ONE = 1.0
_GUIDANCE_ATOL = 1e-6
_MIN_TAIL_EMBEDDINGS = 2
_UNIT_TEMPORAL_PATCH = 1
_KIND: dict[str, ImageKind] = {
    "pretrain": "pretrain",
    "i2v": "i2v",
    "tail_cond_first_frame": "tail",
}
_SCHEME: dict[ImageKind, str] = {kind: name for name, kind in _KIND.items()}


@dataclass(frozen=True)
class ImageCondition:
    """Who owns a reference frame during denoising.

    ``pretrain`` writes the frame into the conditioning channels.
    ``i2v`` overwrites the latent at the start of each sequence.
    ``tail`` appends one or two clean frames and restores them after a step.
    ``frames`` is ``None`` when this call has no reference image.
    """

    kind: ImageKind
    frames: Tensor | None = None

    @classmethod
    def parse(cls, scheme: str) -> ImageCondition:
        try:
            kind = _KIND[scheme]
        except KeyError:
            raise ValueError(f"unknown visual_cond_scheme={scheme!r}") from None
        return cls(kind=kind)

    def with_frames(self, frames: Tensor | None) -> ImageCondition:
        return ImageCondition(self.kind, frames)

    @property
    def reference_count(self) -> int:
        if self.frames is None or self.frames.ndim != BATCHED_VIDEO_NDIM:
            return 0
        return int(self.frames.shape[1])

    def apply(self, video: Tensor, *, has_visual_cond: bool, cu_seqlens: Tensor | None) -> Tensor:
        """Return the video latent plus conditioning channels, when the DiT has them."""
        if not has_visual_cond:
            return video
        if self.kind == "tail" and self.frames is not None:
            if cu_seqlens is None:
                raise ValueError("tail_cond_first_frame requires video_cu_seqlens")
            return build_tail_condition(video, self.frames, cu_seqlens)

        cond = torch.zeros_like(video)
        mask = torch.zeros([*video.shape[:-1], 1], dtype=video.dtype, device=video.device)
        if self.frames is not None and self.kind in ("pretrain", "i2v"):
            frames = self.frames.to(device=video.device, dtype=video.dtype)
            if cu_seqlens is None:
                raise ValueError(f"{_SCHEME[self.kind]} requires video_cu_seqlens")
            batched = video.ndim == BATCHED_VIDEO_NDIM
            starts = torch.arange(video.shape[0], device=video.device) if batched else cu_seqlens[:-1]
            if batched and self.kind == "i2v":
                video[:, 0] = frames
            elif batched:
                cond[:, 0] = frames
            elif self.kind == "i2v":
                video[starts] = frames
            else:
                cond[starts] = frames
            if batched:
                mask[:, 0] = 1
            else:
                mask[starts] = 1
        return torch.cat([video, cond, mask], dim=-1)

    def restore_after_step(self, video: Tensor, cu_seqlens: Tensor | None) -> None:
        if self.kind == "tail" and self.frames is not None:
            restore_tail_references(video, self.frames, cu_seqlens)

    def pin(self, video: Tensor | None, cu_seqlens: Tensor | None) -> None:
        """Put the clean reference back after the last step."""
        if video is None or self.frames is None or cu_seqlens is None:
            return
        if self.kind == "tail":
            restore_tail_references(video, self.frames, cu_seqlens)
            return
        if self.kind != "i2v":
            return
        frames = self.frames.to(device=video.device, dtype=video.dtype)
        if video.ndim == BATCHED_VIDEO_NDIM:
            video[:, 0] = frames
        else:
            video[cu_seqlens[:-1]] = frames

    def crop_generated(
        self,
        bundle: LatentBundle,
        mask: Tensor | None,
        *,
        batch_size: int,
        generated_duration: int,
        device: torch.device,
    ) -> LatentBundle:
        if self.kind != "tail" or mask is None or bundle.video is None:
            return bundle
        video = bundle.video
        generated = video[:, mask[0]] if video.ndim == BATCHED_VIDEO_NDIM else video[mask]
        offsets = generated_duration * torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        return LatentBundle(
            video=generated,
            audio=bundle.audio,
            video_cu_seqlens=offsets,
            audio_cu_seqlens=bundle.audio_cu_seqlens,
        )


@dataclass(frozen=True)
class HeldModality:
    """A file supplied for one modality. That modality is not sampled."""

    kind: HeldKind

    @property
    def active(self) -> bool:
        return self.kind != "none"


@dataclass(frozen=True, kw_only=True)
class Generation:
    """Decisions for one ``__call__``. Samplers do not read the scheme string."""

    batch_size: int
    texts: list[str]
    negative_texts: list[str]
    images: list[object] | None
    last_images: list[object] | None
    save_paths: list[object] | None
    seed: int
    height: int
    width: int
    explicit_size: bool
    num_steps: int
    guidance_weight: float
    scheduler_scale: float
    latent_frames: int | None
    time_length: int | None
    image: ImageCondition
    held: HeldModality
    do_audio: bool
    sample_audio: bool
    sample_video: bool
    piflow: bool
    generation_mode: str
    audio_path: object | None
    video_path: object | None
    show_progress: bool
    video_duration: int | None = None
    generated_mask: Tensor | None = None
    reference_rope_count: int = 0

    @classmethod
    def from_call(  # noqa: PLR0912, PLR0913
        cls,
        *,
        batch_size: int,
        texts: list[str],
        negative_texts: list[str],
        images: list[object] | None,
        last_images: list[object] | None,
        save_paths: list[object] | None,
        seed: int,
        height: int | None,
        width: int | None,
        default_height: int,
        default_width: int,
        num_steps: int | None,
        default_num_steps: int,
        guidance_weight: float | None,
        default_guidance_weight: float,
        scheduler_scale: float | None,
        default_scheduler_scale: float,
        latent_frames: int | None,
        time_length: int | None,
        visual_cond_scheme: str | None,
        pipeline_scheme: str,
        sample_audio_arg: bool | None,
        multimodal: bool,
        audio_ready: bool,
        visual_cond: bool,
        token_embeddings: int,
        temporal_patch: int,
        piflow_enabled: bool,
        has_audio_file: bool,
        has_video_file: bool,
        audio_path: object | None,
        video_path: object | None,
        show_progress: bool,
    ) -> Generation:
        explicit_size = height is not None or width is not None
        height = default_height if height is None else height
        width = default_width if width is None else width
        num_steps = default_num_steps if num_steps is None else num_steps
        guidance_weight = default_guidance_weight if guidance_weight is None else guidance_weight
        scheduler_scale = default_scheduler_scale if scheduler_scale is None else scheduler_scale

        scheme = visual_cond_scheme or pipeline_scheme
        if last_images is not None:
            scheme = "tail_cond_first_frame"
        image = ImageCondition.parse(scheme)
        if has_audio_file:
            held = HeldModality("audio")
        elif has_video_file:
            held = HeldModality("video")
        else:
            held = HeldModality("none")

        do_audio = sample_audio_arg if sample_audio_arg is not None else bool(multimodal and audio_ready)
        if held.active:
            if not multimodal:
                raise ValueError("audio/video conditioning requires a multimodal transformer")
            do_audio = True
        if do_audio and not audio_ready:
            raise ValueError("sample_audio=True requires audio_vae and vocoder")
        if piflow_enabled:
            if not do_audio:
                raise ValueError("piflow.enabled requires sample_audio=True")
            if abs(guidance_weight - _GUIDANCE_ONE) > _GUIDANCE_ATOL:
                raise ValueError("piflow.enabled requires guidance_weight=1.0")

        has_image = images is not None
        if has_image and image.kind == "pretrain" and not visual_cond:
            raise ValueError("image conditioning requires dit.visual_cond=True")
        if has_image and image.kind == "tail":
            if token_embeddings < _MIN_TAIL_EMBEDDINGS:
                raise ValueError("tail_cond_first_frame requires dit.visual_token_type_num_embeddings >= 2")
            if last_images is not None and temporal_patch != _UNIT_TEMPORAL_PATCH:
                raise ValueError("two reference images require temporal patch_size=1")

        prefix = "i2v" if has_image else "t2v"
        return cls(
            batch_size=batch_size,
            texts=texts,
            negative_texts=negative_texts,
            images=images,
            last_images=last_images,
            save_paths=save_paths,
            seed=seed,
            height=height,
            width=width,
            explicit_size=explicit_size,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            latent_frames=latent_frames,
            time_length=time_length,
            image=image,
            held=held,
            do_audio=do_audio,
            sample_audio=do_audio and held.kind != "audio",
            sample_video=held.kind != "video",
            piflow=piflow_enabled,
            generation_mode=f"{prefix}a" if do_audio else prefix,
            audio_path=audio_path,
            video_path=video_path,
            show_progress=show_progress,
        )
