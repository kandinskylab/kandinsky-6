from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path

import torch

from ..core.algo.dct import dct_spatial_resize_bhwc
from ..core.algo.denoise_loop import denoise_loop
from ..core.algo.denoise_speed import denoise_speed
from ..core.algo.flow_matching import flow_match_timesteps
from ..core.algo.generation import Generation
from ..core.algo.mux import mux_video_audio
from ..core.algo.piflow_sampler import piflow_denoise_loop
from ..core.algo.speed_schedule import SpeedStage, plan_stages
from ..core.algo.postprocess_audio import postprocess_audio
from ..core.algo.postprocess_video import postprocess_video
from ..core.algo.prepare_latents import (
    audio_latent_duration,
    encode_i2va_first_frame,
    prepare_audio_latents,
    prepare_video_latents,
)
from ..core.algo.prepare_ropes import VisualRopeCache, compute_rope1d, on_device
from ..core.algo.temporal.trajectory import FULL_FRAMES, TemporalTrajectory
from ..core.components.beautifier import NullBeautifier
from ..core.ports import (
    append_reference_rope,
    append_visual_references,
    encode_audio_condition,
    encode_video_condition,
)
from ..core.types import Kandinsky6PipelineOutput, LatentBundle
from ..runtime.cache import CacheDiT, CacheMode
from ..runtime.offload import NoOpOffload, OffloadHandle, OffloadStrategy
from ..runtime.profile import NoOpProfile
from ..runtime.speed import SpeedConfig
from ..runtime.temporal import TemporalConfig
from .config import CacheConfig, MagCacheConfig, NaviCacheConfig, PipelineConfig
from .media import load_audio_condition, load_video_condition
from .progress import denoising_progress
from .reconfigure import reconfigure as _reconfigure

logger = logging.getLogger("kandinsky")


@dataclass(frozen=True)
class _Prepared:
    request: Generation
    bundle: LatentBundle
    text_embeds: dict
    null_embeds: dict
    text_cu_seqlens: torch.Tensor
    null_cu_seqlens: torch.Tensor | None
    attention_mask: torch.Tensor | None
    null_attention_mask: torch.Tensor | None
    visual_rope: torch.Tensor | None
    audio_rope: torch.Tensor | None
    text_rope: torch.Tensor | list
    null_text_rope: torch.Tensor | list | None
    visual_token_type_ids: torch.Tensor | None
    captions: list[str]
    source_frames: torch.Tensor | None
    source_audio: object | None
    # SPEED-only; set when ``self.speed_conf.enabled`` and consumed by ``_denoise``.
    speed_stages: list[SpeedStage] | None = None
    speed_timesteps: torch.Tensor | None = None
    speed_h_patches_full: int | None = None
    speed_w_patches_full: int | None = None
    speed_t_rope: int | None = None


_DEFAULT_NEG = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)


def write_expanded_prompt(video_path: str | Path, prompt: str) -> Path:
    """Write the caption encoded for ``video_path`` next to that file."""
    text_path = Path(video_path).with_suffix(".txt")
    text_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
    logger.info("expanded prompt saved path=%s", text_path)
    return text_path


def _batch_values(value, batch_size: int, name: str) -> list:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) == 1:
        return values * batch_size
    if len(values) != batch_size:
        raise ValueError(f"{name} has batch size {len(values)}, expected {batch_size}")
    return values


def _batch_size(*values) -> int:
    sizes = [len(value) for value in values if isinstance(value, (list, tuple))]
    return max(sizes, default=1)


def _reference_batch(images: list[object] | None) -> tuple[list[object] | None, list[object] | None]:
    if images is None:
        return None, None
    references = [list(item) if isinstance(item, (list, tuple)) else [item] for item in images]
    counts = {len(item) for item in references}
    if not counts.issubset({1, 2}):
        raise ValueError("each image sample must contain one or two reference images")
    if len(counts) != 1:
        raise ValueError("mixed image batch: all samples must have the same number of reference images")
    if any(image is None for item in references for image in item):
        raise ValueError("reference images must not be None")
    return [item[0] for item in references], (
        [item[1] for item in references] if counts == {2} else None
    )


def _release_modules_after_call(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        finally:
            self.offload.release("text_embedder", "dit", "vae", "audio_vae", "vocoder")

    return wrapped


def _profile_call(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        # Block offload refuses a stage while autograd is enabled.
        with self.profile.pipeline(), torch.no_grad():
            return function(self, *args, **kwargs)

    return wrapped


class Kandinsky6Pipeline:
    def __init__(
        self,
        dit,
        text_embedder,
        vae,
        device: torch.device,
        audio_vae=None,
        vocoder=None,
        num_steps: int = 50,
        guidance_weight: float = 5.0,
        scheduler_scale: float = 10.0,
        scale_factor: tuple = (1.0, 2.0, 2.0),
        height: int = 512,
        width: int = 768,
        latent_frames: int | None = None,
        sample_frames: int | None = None,
        cache_conf: CacheConfig | dict | None = None,
        fps: float = 24.0,
        audio_fps: int = 44100,
        offload: OffloadHandle | None = None,
        visual_cond_scheme: str = "pretrain",
        max_area: int | None = None,
        image_divisibility: int = 16,
        piflow_conf=None,
        speed_conf: SpeedConfig | dict | None = None,
        temporal_conf: TemporalConfig | None = None,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.device = device
        self.num_steps = num_steps
        self.guidance_weight = guidance_weight
        self.scheduler_scale = scheduler_scale
        self.scale_factor = scale_factor
        self.height = height
        self.width = width
        self.latent_frames = latent_frames
        self.sample_frames = sample_frames
        self.fps = fps
        self.audio_fps = audio_fps
        self.visual_cond_scheme = visual_cond_scheme
        self.max_area = max_area if max_area is not None else height * width
        self.image_divisibility = image_divisibility
        self.piflow_conf = piflow_conf
        if speed_conf is None:
            self.speed_conf = SpeedConfig()
        elif isinstance(speed_conf, SpeedConfig):
            self.speed_conf = speed_conf
        else:
            self.speed_conf = SpeedConfig.model_validate(speed_conf)
        self.offload: OffloadHandle = offload if offload is not None else NoOpOffload()
        self.profile = NoOpProfile()
        if cache_conf is None:
            self.cache_conf = CacheConfig()
        elif isinstance(cache_conf, CacheConfig):
            self.cache_conf = cache_conf
        else:
            self.cache_conf = CacheConfig.model_validate(cache_conf)
        self.beautifier = NullBeautifier()
        self._visual_rope_cache = VisualRopeCache()
        self.temporal_conf = temporal_conf if temporal_conf is not None else TemporalConfig()
        self.last_acceleration_stats = None

    @property
    def raw_dit(self):
        """Bare DiffusionTransformer3D — target for torch.export / torch.compile."""
        return self.dit.module if isinstance(self.dit, CacheDiT) else self.dit

    def _materialize_ropes(
        self,
        *,
        T_rope: int,
        H_patches: int,
        W_patches: int,
        text_len: int,
        null_text_len: int,
        audio_len: int | None,
    ):
        """Precompute RoPE tensors for DiT.forward (max-canvas visual cache + slice)."""
        dit = self.raw_dit
        scale = (
            float(self.scale_factor[0]),
            float(self.scale_factor[1]),
            float(self.scale_factor[2]),
        )
        # Prefer pipeline default canvas as max table when call is smaller.
        pT, pH, pW = dit.patch_size
        T_def = (self.latent_frames // pT) if self.latent_frames is not None else T_rope
        H_def = (self.height // 8) // pH
        W_def = (self.width // 8) // pW
        cache_shape = (max(T_rope, T_def), max(H_patches, H_def), max(W_patches, W_def))
        self._visual_rope_cache.get(dit.visual_rope, cache_shape, scale)
        visual_rope = self._visual_rope_cache.get(dit.visual_rope, (T_rope, H_patches, W_patches), scale)

        if dit.is_multimodal:
            text_rope = [
                compute_rope1d(dit.video_text_rope, text_len),
                compute_rope1d(dit.audio_text_rope, text_len),
            ]
            null_text_rope = (
                [compute_rope1d(dit.video_text_rope, null_text_len), compute_rope1d(dit.audio_text_rope, null_text_len)]
                if null_text_len > 0
                else None
            )
            audio_rope = compute_rope1d(dit.audio_rope, audio_len) if audio_len is not None else None
        else:
            text_rope = compute_rope1d(dit.text_rope, text_len)
            null_text_rope = compute_rope1d(dit.text_rope, null_text_len) if null_text_len > 0 else None
            audio_rope = None

        return visual_rope, audio_rope, text_rope, null_text_rope

    def set_cache(self, mode: CacheMode | None, mode_name: str | None = None, **params) -> None:
        """Switch MagCache / NaviCache / none without recreating the pipeline.

        Extra Mag/Navi hyperparameters fall back to values from the loaded YAML
        (`cache_conf`) when not passed explicitly.
        """
        if not isinstance(self.dit, CacheDiT):
            self.dit = CacheDiT(self.dit)

        if mode is None or mode.lower() == "none":
            self.dit.set_cache("none")
            return

        if mode == "adaptive_block":
            if mode_name is not None or params:
                raise TypeError("Adaptive Block Cache settings come from cache.adaptive_block")
            self.dit.set_cache(mode, adaptive_block=self.cache_conf.adaptive_block)
            return

        num_steps = params.pop("num_steps", self.num_steps)
        no_cfg = params.pop("no_cfg", abs(self.guidance_weight - 1.0) < 1e-6)

        if mode == "magcache":
            defaults: MagCacheConfig | None = self.cache_conf.magcache
            mag_ratios = params.pop(
                "mag_ratios",
                None if defaults is None else defaults.mag_ratios,
            )
            if mag_ratios is None:
                raise ValueError("set_cache('magcache') requires mag_ratios (or cache.magcache in config)")
            self.dit.set_cache(
                "magcache",
                num_steps=num_steps,
                no_cfg=no_cfg,
                mag_ratios=mag_ratios,
                mode_name=mode_name,
                thresh=params.pop("thresh", 0.12 if defaults is None else defaults.thresh),
                K=params.pop("K", 2 if defaults is None else defaults.K),
                retention_ratio=params.pop(
                    "retention_ratio",
                    0.2 if defaults is None else defaults.retention_ratio,
                ),
            )
            if params:
                raise TypeError(f"Unexpected MagCache params: {sorted(params)}")
            return

        if mode == "navicache":
            defaults_n: NaviCacheConfig | None = self.cache_conf.navicache
            self.dit.set_cache(
                "navicache",
                num_steps=num_steps,
                no_cfg=no_cfg,
                thresh=params.pop("thresh", 0.05 if defaults_n is None else defaults_n.thresh),
                align_steps=params.pop("align_steps", 10 if defaults_n is None else defaults_n.align_steps),
                process_noise=params.pop(
                    "process_noise",
                    0.05 if defaults_n is None else defaults_n.process_noise,
                ),
                measurement_noise=params.pop(
                    "measurement_noise",
                    0.05 if defaults_n is None else defaults_n.measurement_noise,
                ),
            )
            if params:
                raise TypeError(f"Unexpected NaviCache params: {sorted(params)}")
            return

        raise ValueError(f"Unknown cache mode: {mode!r}")

    def reconfigure(
        self,
        cfg: PipelineConfig | str | Path,
        *,
        attention_engine: str | None = None,
        cache_mode: str | None = None,
        offload_strategy: OffloadStrategy | None = None,
    ) -> Kandinsky6Pipeline:
        """Apply config settings to this pipeline without constructing a new one."""
        return _reconfigure(
            self,
            cfg,
            attention_engine=attention_engine,
            cache_mode=cache_mode,
            offload_strategy=offload_strategy,
        )

    @staticmethod
    def _generation_mode(*, do_audio: bool, image: object | None) -> str:
        """Same coeffs for t2v-t2va, i2v-i2va for now."""
        prefix = "i2v" if image is not None else "t2v"
        return f"{prefix}a" if do_audio else prefix

    def _expand_captions(self, captions, *, images, do_audio, seed):
        beautifier = self.beautifier
        if beautifier.name == "none":
            return list(captions)
        image_items = images if images is not None else [None] * len(captions)
        expanded = []
        for index, (caption, image_item) in enumerate(zip(captions, image_items, strict=True)):
            expanded.append(
                beautifier.expand(
                    caption,
                    image=image_item,
                    audio=do_audio,
                    seed=seed if index == 0 else None,
                    text_embedder=self.text_embedder,
                )
            )
        return expanded

    @_profile_call
    @_release_modules_after_call
    def __call__(
        self,
        text: str | list[str],
        height: int | None = None,
        width: int | None = None,
        time_length: int | None = None,
        latent_frames: int | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        guidance_weight: float | None = None,
        scheduler_scale: float | None = None,
        negative_text: str | list[str] = _DEFAULT_NEG,
        sample_audio: bool | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        show_progress: bool = False,
        image: str | object | list[str | object] | None = None,
        visual_cond_scheme: str | None = None,
        audio: str | Path | None = None,
        video: str | Path | None = None,
    ) -> Kandinsky6PipelineOutput:
        self.last_acceleration_stats = None
        batch_size = _batch_size(text, negative_text, image, save_path)
        conditioned = audio is not None or video is not None
        if audio is not None and video is not None:
            raise ValueError("pass either audio or video conditioning, not both")
        if conditioned and image is not None:
            raise ValueError("audio/video conditioning cannot be combined with image conditioning")
        if conditioned and batch_size != 1:
            raise ValueError("audio/video conditioning currently requires batch size 1")
        if conditioned and sample_audio is False:
            raise ValueError("audio/video conditioning requires the audio stream; omit sample_audio=False")
        if conditioned and self.cache_conf.mode != "none":
            raise ValueError("audio/video conditioning requires cache.mode=none")
        texts = _batch_values(text, batch_size, "text")
        negative_texts = _batch_values(negative_text, batch_size, "negative_text")
        images = None if image is None else _batch_values(image, batch_size, "image")
        images, last_images = _reference_batch(images)
        if save_path is not None and batch_size > 1 and not isinstance(save_path, (list, tuple)):
            raise ValueError("save_path must be a list when generating a batch")
        save_paths = None if save_path is None else _batch_values(save_path, batch_size, "save_path")

        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        self.profile.note_call(
            seed=seed,
            height=height,
            width=width,
            time_length=time_length,
            latent_frames=latent_frames,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            sample_audio=sample_audio,
            visual_cond_scheme=visual_cond_scheme,
            image=image,
        )

        dit = self.raw_dit
        request = Generation.from_call(
            batch_size=batch_size,
            texts=texts,
            negative_texts=negative_texts,
            images=images,
            last_images=last_images,
            save_paths=save_paths,
            seed=seed,
            height=height,
            width=width,
            default_height=self.height,
            default_width=self.width,
            num_steps=num_steps,
            default_num_steps=self.num_steps,
            guidance_weight=guidance_weight,
            default_guidance_weight=self.guidance_weight,
            scheduler_scale=scheduler_scale,
            default_scheduler_scale=self.scheduler_scale,
            latent_frames=latent_frames,
            time_length=time_length,
            visual_cond_scheme=visual_cond_scheme,
            pipeline_scheme=self.visual_cond_scheme,
            sample_audio_arg=sample_audio,
            multimodal=bool(getattr(dit, "is_multimodal", False)),
            audio_ready=self.audio_vae is not None and self.vocoder is not None,
            visual_cond=bool(dit.visual_cond),
            token_embeddings=int(getattr(dit, "visual_token_type_num_embeddings", 0)),
            temporal_patch=int(dit.patch_size[0]),
            piflow_enabled=bool(getattr(self.piflow_conf, "enabled", False)),
            has_audio_file=audio is not None,
            has_video_file=video is not None,
            audio_path=audio,
            video_path=video,
            show_progress=show_progress,
        )
        cache_mode = self.dit.mode if isinstance(self.dit, CacheDiT) else "none"
        if self.temporal_conf.mode != "off" or cache_mode == "adaptive_block":
            if cache_mode not in ("none", "adaptive_block"):
                raise ValueError("Temporal supports cache.mode=none or adaptive_block")
            self._validate_acceleration_request(request)
        self._align_cache(request)
        prepared = self._prepare(request)
        bundle = self._denoise(prepared)
        return self._decode(prepared, bundle)

    def _validate_acceleration_request(self, request: Generation) -> None:
        if (not request.piflow or request.generation_mode != "t2va" or request.batch_size != 1
                or request.held.active or request.num_steps != 10):
            raise ValueError("Temporal and Adaptive Block Cache require single-clip 10-step PiFlow T2VA")
        if (request.height, request.width) != (480, 864):
            raise ValueError("Temporal and Adaptive Block Cache require 864x480")
        frames = request.latent_frames
        if frames is None and request.time_length is None:
            frames = self.latent_frames
        if frames not in (None, FULL_FRAMES) or request.time_length not in (None, 5):
            raise ValueError("Temporal and Adaptive Block Cache require a five-second clip")
        if self.speed_conf.enabled:
            raise ValueError("Temporal and Adaptive Block Cache cannot be combined with SPEED")
        execution = getattr(self, "_execution_key", None)
        if self.offload.strategy == "block" or (execution is not None and execution[0] == "aoti"):
            raise ValueError("Temporal and Adaptive Block Cache support eager or regional torch.compile with none/module offload")

    def _align_cache(self, request: Generation) -> None:
        if isinstance(self.dit, CacheDiT) and self.dit.mode != "none":
            self.dit.align_schedule(request.num_steps, request.generation_mode, apply=self.set_cache)
            self.dit.reset_cache_state()

    def _prepare(self, request: Generation) -> _Prepared:  # noqa: PLR0912, PLR0915
        raw = self.raw_dit
        if hasattr(raw, "clear_text_proj_cache"):
            raw.clear_text_proj_cache()
        dit = raw
        height = request.height
        width = request.width
        first_frames = None
        last_frames = None
        if request.images is not None:
            with self.offload.use("vae", prefetch="text_embedder"):
                encoded_frames = []
                for image_item in request.images:
                    if request.explicit_size:
                        encoded, item_height, item_width = encode_i2va_first_frame(
                            image_item, self.vae, self.device, height=height, width=width
                        )
                    else:
                        encoded, item_height, item_width = encode_i2va_first_frame(
                            image_item,
                            self.vae,
                            self.device,
                            max_area=self.max_area,
                            divisibility=self.image_divisibility,
                        )
                    encoded_frames.append(encoded)
                    if len(encoded_frames) == 1:
                        height, width = item_height, item_width
                    elif (item_height, item_width) != (height, width):
                        raise ValueError("all batched images must resolve to the same height and width")
                first_frames = torch.cat(encoded_frames, dim=0)
                if request.last_images is not None:
                    last_frames = torch.cat(
                        [
                            encode_i2va_first_frame(
                                item, self.vae, self.device, height=height, width=width
                            )[0]
                            for item in request.last_images
                        ],
                        dim=0,
                    )
            request = replace(request, height=height, width=width)

        h_lat = height // 8
        w_lat = width // 8
        if request.latent_frames is not None:
            num_frames = request.latent_frames
        elif self.latent_frames is not None and request.time_length is None:
            num_frames = self.latent_frames
        else:
            duration_s = 5 if request.time_length is None else request.time_length
            num_frames = duration_s * 24 // 4 + 1
        h_patches = h_lat // dit.patch_size[1]
        w_patches = w_lat // dit.patch_size[2]
        video_duration = num_frames

        # SPEED: plan stages at the full latent canvas, then shrink the initial
        # bundle (and any reference frames) to stage 0 so the first denoising
        # pass runs at the right grid.
        h_lat_full, w_lat_full = h_lat, w_lat
        h_patches_full, w_patches_full = h_patches, w_patches
        t_rope_full = video_duration // dit.patch_size[0]
        speed_stages: list[SpeedStage] | None = None
        speed_timesteps: torch.Tensor | None = None
        if self.speed_conf.enabled:
            if request.batch_size != 1:
                raise ValueError("speed.enabled requires batch size 1")
            if request.piflow:
                raise ValueError("speed.enabled is incompatible with piflow.enabled")
            if request.video_path is not None or request.audio_path is not None:
                raise ValueError("speed.enabled is incompatible with audio/video conditioning")
            speed_timesteps = flow_match_timesteps(
                request.num_steps, request.scheduler_scale, self.device
            )
            speed_stages = plan_stages(
                self.speed_conf.scales,
                speed_timesteps.tolist(),
                h_lat_full,
                w_lat_full,
                dit.patch_size,
                transition_times=self.speed_conf.transition_times,
                delta=self.speed_conf.delta,
                spectrum_a=self.speed_conf.spectrum_a,
                spectrum_beta=self.speed_conf.spectrum_beta,
            )
            stage0_h, stage0_w = speed_stages[0].latent_hw
            if first_frames is not None and (stage0_h, stage0_w) != (h_lat_full, w_lat_full):
                first_frames = dct_spatial_resize_bhwc(first_frames, (stage0_h, stage0_w))
                if last_frames is not None:
                    last_frames = dct_spatial_resize_bhwc(last_frames, (stage0_h, stage0_w))
            h_lat = stage0_h
            w_lat = stage0_w
            h_patches = h_lat // dit.patch_size[1]
            w_patches = w_lat // dit.patch_size[2]

        source_frames = None
        source_audio = None
        clean_video = None
        clean_audio = None
        if request.video_path is not None:
            source_frames = load_video_condition(
                request.video_path,
                height=height,
                width=width,
                num_frames=(num_frames - 1) * 4 + 1,
                fps=self.fps,
            )
            with self.offload.use("vae", prefetch="text_embedder"):
                clean_video = encode_video_condition(source_frames.to(self.device), self.vae)
        if request.audio_path is not None:
            source_audio = load_audio_condition(
                request.audio_path,
                sample_rate=self.audio_fps,
                fps=self.fps,
                num_samples=round(((num_frames - 1) * 4 + 1) / self.fps * self.audio_fps),
            )
            with self.offload.use("audio_vae", prefetch="text_embedder"):
                clean_audio = encode_audio_condition(
                    torch.as_tensor(source_audio, device=self.device), self.audio_vae
                )

        with self.offload.use("text_embedder", prefetch="dit"):
            # Qwen stays on CPU until this stage. Expanding outside it runs the
            # beautifier on the host.
            captions = self._expand_captions(
                request.texts,
                images=request.images,
                do_audio=request.do_audio,
                seed=request.seed,
            )
            self.profile.note_prompt(captions[0] if len(captions) == 1 else list(captions))
            text_embeds, text_cu_seqlens, attention_mask = self.text_embedder.encode(captions)
            # PiFlow uses implicit CFG, so the negative prompt is not encoded.
            if request.piflow:
                null_embeds, null_cu_seqlens, null_attention_mask = {}, None, None
            else:
                null_embeds, null_cu_seqlens, null_attention_mask = self.text_embedder.encode(
                    request.negative_texts
                )

        text_embeds = {key: value.to(device=self.device, dtype=torch.bfloat16) for key, value in text_embeds.items()}
        null_embeds = {key: value.to(device=self.device, dtype=torch.bfloat16) for key, value in null_embeds.items()}
        text_cu_seqlens = text_cu_seqlens.to(self.device)
        if null_cu_seqlens is not None:
            null_cu_seqlens = null_cu_seqlens.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if null_attention_mask is not None:
            null_attention_mask = null_attention_mask.to(self.device)

        bundle = prepare_video_latents(
            bs=request.batch_size,
            duration=num_frames,
            H_lat=h_lat,
            W_lat=w_lat,
            C=dit.in_visual_dim,
            seed=request.seed,
            device=self.device,
        )
        bundle = LatentBundle(
            video=(
                clean_video
                if clean_video is not None
                else bundle.video.reshape(request.batch_size, num_frames, h_lat, w_lat, dit.in_visual_dim)
            ),
            audio=bundle.audio,
            video_cu_seqlens=bundle.video_cu_seqlens,
            audio_cu_seqlens=bundle.audio_cu_seqlens,
        )

        visual_token_type_ids = None
        if first_frames is not None and request.image.kind == "tail":
            references = first_frames[:, None]
            loop_frames = first_frames
            if last_frames is not None:
                references = torch.stack([first_frames, last_frames], dim=1)
                loop_frames = references
            video, visual_token_type_ids, generated_mask = append_visual_references(
                bundle.video,
                references,
                batch_size=request.batch_size,
                video_duration=video_duration,
            )
            duration_with_ref = video_duration + references.shape[1]
            offsets = duration_with_ref * torch.arange(
                request.batch_size + 1, dtype=torch.int32, device=self.device
            )
            bundle = LatentBundle(
                video=video,
                audio=None,
                video_cu_seqlens=offsets,
                audio_cu_seqlens=None,
            )
            request = replace(
                request,
                image=request.image.with_frames(loop_frames),
                generated_mask=generated_mask,
                video_duration=video_duration,
                reference_rope_count=int(references.shape[1]),
            )
        elif first_frames is not None:
            request = replace(
                request,
                image=request.image.with_frames(first_frames),
                video_duration=video_duration,
            )
        else:
            request = replace(request, video_duration=video_duration)

        audio_len = None
        if request.do_audio:
            downsample = int(getattr(self.audio_vae, "downsample_factor", 1024))
            # Audio length follows generated frames only (exclude tail reference).
            audio_dur = audio_latent_duration(
                video_duration,
                fps=self.fps,
                audio_fps=self.audio_fps,
                downsample_factor=downsample,
            )
            if clean_audio is not None:
                audio_dur = clean_audio.shape[1]
            bundle = prepare_audio_latents(
                bundle,
                audio_duration=audio_dur,
                audio_dim=dit.in_audio_dim,
                seed=request.seed,
                device=self.device,
            )
            bundle = LatentBundle(
                video=bundle.video,
                audio=(
                    clean_audio
                    if clean_audio is not None
                    else bundle.audio.reshape(request.batch_size, audio_dur, dit.in_audio_dim)
                ),
                video_cu_seqlens=bundle.video_cu_seqlens,
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            )
            audio_len = int(audio_dur)

        # Generated frames use positions 0..T-1. A tail reference reuses position 0.
        # SPEED builds the visual rope per stage; prime the cache at the full
        # canvas here so later slices are hits.
        t_rope = video_duration // dit.patch_size[0]
        rope_h_patches = h_patches_full if self.speed_conf.enabled else h_patches
        rope_w_patches = w_patches_full if self.speed_conf.enabled else w_patches
        visual_rope, audio_rope, text_rope, null_text_rope = self._materialize_ropes(
            T_rope=t_rope,
            H_patches=rope_h_patches,
            W_patches=rope_w_patches,
            text_len=(
                text_embeds["text_embeds"].shape[1]
                if text_embeds["text_embeds"].ndim == 3
                else int(text_cu_seqlens[-1].item())
            ),
            null_text_len=(
                0
                if null_cu_seqlens is None
                else (
                    null_embeds["text_embeds"].shape[1]
                    if null_embeds["text_embeds"].ndim == 3
                    else int(null_cu_seqlens[-1].item())
                )
            ),
            audio_len=audio_len,
        )
        if request.reference_rope_count and not self.speed_conf.enabled:
            # denoise_speed appends the reference slot per stage; skip here.
            visual_rope = append_reference_rope(visual_rope, request.reference_rope_count)
        if self.speed_conf.enabled:
            # Full-canvas rope is only used to prime the cache; denoise_speed
            # queries the cache per stage.
            visual_rope = None
        visual_rope = on_device(visual_rope, self.device)
        audio_rope = on_device(audio_rope, self.device)
        text_rope = on_device(text_rope, self.device)
        null_text_rope = on_device(null_text_rope, self.device)

        return _Prepared(
            request=request,
            bundle=bundle,
            text_embeds=text_embeds,
            null_embeds=null_embeds,
            text_cu_seqlens=text_cu_seqlens,
            null_cu_seqlens=null_cu_seqlens,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
            visual_rope=visual_rope,
            audio_rope=audio_rope,
            text_rope=text_rope,
            null_text_rope=null_text_rope,
            visual_token_type_ids=visual_token_type_ids,
            captions=list(captions),
            source_frames=source_frames,
            source_audio=source_audio,
            speed_stages=speed_stages,
            speed_timesteps=speed_timesteps,
            speed_h_patches_full=h_patches_full if self.speed_conf.enabled else None,
            speed_w_patches_full=w_patches_full if self.speed_conf.enabled else None,
            speed_t_rope=t_rope_full if self.speed_conf.enabled else None,
        )

    def _denoise(self, prepared: _Prepared) -> LatentBundle:
        request = prepared.request
        progress_bar = denoising_progress(request.num_steps) if request.show_progress else None
        decode_names = ("vae", "audio_vae", "vocoder") if request.do_audio else ("vae",)
        callback = progress_bar.update if progress_bar is not None else None
        try:
            with self.offload.use("dit", prefetch=decode_names):
                if self.speed_conf.enabled:
                    return self._denoise_speed(prepared, callback)
                if request.piflow:
                    temporal = None
                    if self.temporal_conf.mode != "off":
                        fixed = self.temporal_conf.mode == "fixed"
                        temporal = TemporalTrajectory(
                            prepared.bundle.video, prepared.bundle.audio, self.raw_dit, self.scale_factor,
                            switch_after_calls=self.temporal_conf.switch_after_calls if fixed else None,
                            threshold=self.temporal_conf.threshold,
                        )
                    block_state = (
                        self.dit.new_adaptive_block_state(request.num_steps)
                        if isinstance(self.dit, CacheDiT) else None
                    )
                    result = piflow_denoise_loop(
                        bundle=prepared.bundle,
                        dit=self.dit,
                        text_embeds=prepared.text_embeds,
                        visual_rope=prepared.visual_rope,
                        audio_rope=prepared.audio_rope,
                        text_rope=prepared.text_rope,
                        num_steps=request.num_steps,
                        scheduler_scale=request.scheduler_scale,
                        image=request.image,
                        sample_audio=request.sample_audio,
                        sample_video=request.sample_video,
                        attention_mask=prepared.attention_mask,
                        visual_token_type_ids=prepared.visual_token_type_ids,
                        scale_factor=self.scale_factor,
                        eps=self.piflow_conf.eps,
                        final_step_size_scale=self.piflow_conf.final_step_size_scale,
                        shift=self.piflow_conf.shift,
                        num_policy_substeps=self.piflow_conf.num_policy_substeps,
                        progress_callback=callback,
                        temporal=temporal,
                        dit_kwargs={"adaptive_block_state": block_state} if block_state is not None else None,
                    )
                    if temporal is not None or block_state is not None:
                        self.last_acceleration_stats = {
                            "temporal": None if temporal is None else temporal.summary(),
                            "adaptive_block": None if block_state is None else block_state.summary(),
                        }
                    return result
                return denoise_loop(
                    bundle=prepared.bundle,
                    dit=self.dit,
                    text_embeds=prepared.text_embeds,
                    null_text_embeds=prepared.null_embeds,
                    visual_rope=prepared.visual_rope,
                    audio_rope=prepared.audio_rope,
                    text_rope=prepared.text_rope,
                    null_text_rope=prepared.null_text_rope,
                    num_steps=request.num_steps,
                    guidance_weight=request.guidance_weight,
                    scheduler_scale=request.scheduler_scale,
                    image=request.image,
                    sample_audio=request.sample_audio,
                    sample_video=request.sample_video,
                    attention_mask=prepared.attention_mask,
                    null_attention_mask=prepared.null_attention_mask,
                    visual_token_type_ids=prepared.visual_token_type_ids,
                    scale_factor=self.scale_factor,
                    progress_callback=callback,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _denoise_speed(self, prepared: _Prepared, callback) -> LatentBundle:
        """Staged SPEED denoising. Caller holds ``offload.use('dit', ...)``."""
        if prepared.speed_stages is None or prepared.speed_timesteps is None:
            raise RuntimeError("speed.enabled but _prepare did not produce a SPEED plan")
        request = prepared.request
        dit_raw = self.raw_dit
        scale = (
            float(self.scale_factor[0]),
            float(self.scale_factor[1]),
            float(self.scale_factor[2]),
        )
        t_rope = prepared.speed_t_rope

        def build_visual_rope(h_patches: int, w_patches: int) -> torch.Tensor:
            return self._visual_rope_cache.get(
                dit_raw.visual_rope, (t_rope, h_patches, w_patches), scale
            ).to(self.device)

        return denoise_speed(
            bundle=prepared.bundle,
            dit=self.dit,
            stages=prepared.speed_stages,
            timesteps=prepared.speed_timesteps,
            text_embeds=prepared.text_embeds,
            null_text_embeds=prepared.null_embeds,
            build_visual_rope=build_visual_rope,
            audio_rope=prepared.audio_rope,
            text_rope=prepared.text_rope,
            null_text_rope=prepared.null_text_rope,
            guidance_weight=request.guidance_weight,
            patch_size=dit_raw.patch_size,
            seed=request.seed,
            audio_alignment=self.speed_conf.audio_alignment,
            audio_alignment_substeps=self.speed_conf.audio_alignment_substeps,
            attention_mask=prepared.attention_mask,
            null_attention_mask=prepared.null_attention_mask,
            visual_token_type_ids=prepared.visual_token_type_ids,
            scale_factor=self.scale_factor,
            image=request.image,
            progress_callback=callback,
        )

    def _decode(self, prepared: _Prepared, bundle: LatentBundle) -> Kandinsky6PipelineOutput:
        request = prepared.request
        duration = request.video_duration if request.video_duration is not None else 0
        bundle = request.image.crop_generated(
            bundle,
            request.generated_mask,
            batch_size=request.batch_size,
            generated_duration=duration,
            device=self.device,
        )
        decode_names = ("vae", "audio_vae", "vocoder") if request.do_audio else ("vae",)
        with self.offload.use(*decode_names):
            frames = (
                prepared.source_frames
                if prepared.source_frames is not None
                else postprocess_video(bundle, self.vae, bs=request.batch_size)
            )
            if prepared.source_audio is not None:
                audio = [prepared.source_audio]
            elif request.do_audio:
                audio = postprocess_audio(bundle, self.audio_vae, self.vocoder)
            else:
                audio = None

        path = None
        if request.save_paths is not None:
            paths = []
            for index, output_path in enumerate(request.save_paths):
                saved = mux_video_audio(
                    frames[index],
                    None if audio is None else audio[index],
                    output_path,
                    fps=int(self.fps),
                    audio_sample_rate=self.audio_fps,
                )
                paths.append(str(saved))
                write_expanded_prompt(saved, prepared.captions[index])
            path = paths[0] if request.batch_size == 1 else paths

        return Kandinsky6PipelineOutput(frames=frames, audio=audio, path=path, prompts=list(prepared.captions))
