from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..core.components.dit import DiffusionTransformer3D
from ..core.components.text_embedder import Kandinsky6TextEmbedder
from ..core.components.vae_audio import build_audio_vae, build_vocoder
from ..core.components.vae_video import build_vae
from ..runtime.cache import CacheDiT
from ..runtime.kernels import bind_attention
from ..runtime.offload import OffloadHandle, OffloadStrategy, build_offload
from ..runtime.profile import attach_profile, diff_launch_overrides
from ..runtime.sampler import dit_class, reject_incompatible_cache
from ..runtime.weights import ensure_checkpoint, load_bf16_checkpoint, materialize_meta_buffers
from .config import PipelineConfig, load_config
from .pipeline import Kandinsky6Pipeline
from .reconfigure import _apply_cache, _apply_execution, reconfigure, remember

logger = logging.getLogger("kandinsky")

__all__ = (
    "create_bare_dit",
    "get_pipeline",
    "reconfigure",
)


def _under_snapshot(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return value
    return str(root / path)


def bind_checkpoint_paths(cfg: PipelineConfig) -> PipelineConfig:
    """Join relative component paths to the downloaded checkpoint snapshot."""
    paths = cfg.paths
    relatives = [paths.dit, paths.vae, paths.qwen, paths.clip]
    if paths.dit_export is not None:
        relatives.append(paths.dit_export)
    if cfg.audio_vae is not None:
        relatives.append(cfg.audio_vae.tod_vae_ckpt)
    if cfg.vocoder is not None:
        relatives.append(cfg.vocoder.ckpt)
    if all(Path(item).is_absolute() for item in relatives):
        return cfg
    root = ensure_checkpoint(cfg.checkpoint)
    update: dict = {
        "paths": paths.model_copy(
            update={
                "dit": _under_snapshot(root, paths.dit),
                "vae": _under_snapshot(root, paths.vae),
                "qwen": _under_snapshot(root, paths.qwen),
                "clip": _under_snapshot(root, paths.clip),
                "dit_export": None if paths.dit_export is None else _under_snapshot(root, paths.dit_export),
            }
        )
    }
    if cfg.audio_vae is not None:
        update["audio_vae"] = cfg.audio_vae.model_copy(
            update={"tod_vae_ckpt": _under_snapshot(root, cfg.audio_vae.tod_vae_ckpt)}
        )
    if cfg.vocoder is not None:
        update["vocoder"] = cfg.vocoder.model_copy(update={"ckpt": _under_snapshot(root, cfg.vocoder.ckpt)})
    return cfg.model_copy(update=update)


def get_pipeline(
    conf_path: str,
    device: str | torch.device = "cuda",
    attention_engine: str | None = None,
    cache_mode: str | None = None,
    offload_strategy: OffloadStrategy | None = None,
) -> Kandinsky6Pipeline:
    cfg = load_config(conf_path)
    pipe = _build_pipeline(
        cfg,
        device,
        attention_engine,
        cache_mode=cache_mode,
        offload_strategy=offload_strategy,
    )
    pipe.config_path = conf_path
    pipe.launch_overrides = diff_launch_overrides(
        cfg,
        attention_engine=attention_engine,
        cache_mode=cache_mode,
        offload_strategy=offload_strategy,
    )
    return pipe


def create_bare_dit(
    cfg: PipelineConfig,
    device: str | torch.device,
    attention_engine: str,
) -> DiffusionTransformer3D:
    """Construct DiT and load its checkpoint — no CacheDiT / compile / AOTI.

    Builds on the ``meta`` device so we never allocate the full bf16 skeleton
    (~120 GB / ~2 min for Pro T2VA) before overwriting it from the checkpoint.
    """
    device = torch.device(device)
    dit_cfg = cfg.dit
    with torch.device("meta"):
        dit_type = dit_class(cfg.piflow.enabled)
        dit_kwargs = dict(
            in_visual_dim=dit_cfg.in_visual_dim,
            out_visual_dim=dit_cfg.out_visual_dim,
            in_text_dim=dit_cfg.in_text_dim,
            in_text_dim2=dit_cfg.in_text_dim2,
            time_dim=dit_cfg.time_dim,
            patch_size=dit_cfg.patch_size,
            model_dim=dit_cfg.model_dim,
            ff_dim=dit_cfg.ff_dim,
            num_text_blocks=dit_cfg.num_text_blocks,
            num_visual_blocks=dit_cfg.num_visual_blocks,
            axes_dims=dit_cfg.axes_dims,
            visual_cond=dit_cfg.visual_cond,
            is_multimodal=dit_cfg.is_multimodal,
            in_audio_dim=dit_cfg.in_audio_dim,
            model_dim_a=dit_cfg.model_dim_a,
            time_dim_a=dit_cfg.time_dim_a,
            ff_dim_a=dit_cfg.ff_dim_a,
            axes_dims_a=dit_cfg.axes_dims_a,
            audio_freqs_scaling=dit_cfg.audio_freqs_scaling,
            attention_engine=attention_engine,
            text_token_padding=dit_cfg.text_token_padding,
            ca_rope=dit_cfg.ca_rope,
            cross_gates=dit_cfg.cross_gates,
            fix_modulation=dit_cfg.fix_modulation,
            visual_token_type_num_embeddings=dit_cfg.visual_token_type_num_embeddings,
        )
        if cfg.piflow.enabled:
            dit_kwargs.update(
                n_grid=cfg.piflow.dx_num_grid_points,
                out_visual_dim=dit_cfg.out_visual_dim,
                out_audio_dim=dit_cfg.in_audio_dim,
            )
        dit = dit_type(**dit_kwargs)

    load_bf16_checkpoint(dit, cfg.paths.dit)

    materialize_meta_buffers(dit, torch.device("cpu"))
    dit = dit.to(device).eval()
    return dit


def _build_pipeline(
    cfg: PipelineConfig,
    device: str | torch.device,
    attention_engine: str | None,
    cache_mode: str | None = None,
    offload_strategy: OffloadStrategy | None = None,
) -> Kandinsky6Pipeline:
    cfg = bind_checkpoint_paths(cfg)
    device = torch.device(device)
    # TF32 for any residual float32 matmuls (bf16 hot path unchanged).
    torch.set_float32_matmul_precision("high")
    gen = cfg.generation
    cache_mode = cache_mode if cache_mode is not None else cfg.cache.mode
    reject_incompatible_cache(cfg, cache_mode)
    off_cfg = cfg.offload
    strategy: OffloadStrategy = offload_strategy if offload_strategy is not None else off_cfg.strategy
    if strategy == "block" and cfg.paths.dit_export is not None:
        raise ValueError("Block offload supports eager and regional torch.compile, not paths.dit_export")
    offload: OffloadHandle = build_offload(strategy, device, pin_memory=off_cfg.pin_memory)
    # Module and block offload keep the full model on CPU and stage what a step needs.
    load_device = torch.device("cpu") if strategy != "none" else device

    requested = attention_engine if attention_engine is not None else cfg.attention.engine
    engine = bind_attention(requested)

    # Regional AOTI needs static text length (K5 pad-to-max); force padding when export is bound.
    if cfg.paths.dit_export is not None and not cfg.dit.text_token_padding:
        logger.warning("paths.dit_export is set, so dit.text_token_padding is forced on for a static AOTI text length")
        cfg.dit.text_token_padding = True

    # NF4 Qwen must load on CUDA (bitsandbytes); keep it on compute device even with module offload.
    text_device = device if (cfg.text_embedder.quantized_qwen or cfg.text_embedder.release_after_encode) else load_device
    text_embedder = Kandinsky6TextEmbedder(
        qwen_path=cfg.paths.qwen,
        clip_path=cfg.paths.clip,
        max_length=cfg.text_embedder.max_length,
        device=text_device,
        quantized_qwen=cfg.text_embedder.quantized_qwen,
        text_token_padding=cfg.dit.text_token_padding,
    )
    if cfg.text_embedder.quantized_qwen:
        logger.info("Qwen: NF4 (bitsandbytes)")

    vae = build_vae(cfg.paths.vae, device=load_device)

    audio_vae = None
    vocoder = None
    if cfg.audio_vae is not None:
        av = cfg.audio_vae
        audio_vae = build_audio_vae(
            tod_vae_ckpt=av.tod_vae_ckpt,
            mode=av.mode,
            need_vae_encoder=av.need_vae_encoder,
            need_vae_decoder=av.need_vae_decoder,
            scaling_factor=av.scaling_factor,
            device=load_device,
        )
        if cfg.vocoder is None:
            raise ValueError("audio_vae and vocoder must be configured together")
        vocoder = build_vocoder(ckpt=cfg.vocoder.ckpt, device=load_device)

    dit = create_bare_dit(cfg, load_device, engine)
    _apply_execution(dit, cfg, device)
    dit = CacheDiT(dit)

    if strategy != "none":
        if not cfg.text_embedder.release_after_encode:
            offload.register("text_embedder", text_embedder)
        offload.register("dit", dit)
        offload.register("vae", vae)
        if audio_vae is not None:
            offload.register("audio_vae", audio_vae)
        if vocoder is not None:
            offload.register("vocoder", vocoder)

    pipe = Kandinsky6Pipeline(
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        device=device,
        num_steps=gen.num_steps,
        guidance_weight=gen.guidance_weight,
        scheduler_scale=gen.scheduler_scale,
        scale_factor=gen.scale_factor,
        height=gen.height,
        width=gen.width,
        latent_frames=gen.latent_frames,
        sample_frames=gen.sample_frames,
        cache_conf=cfg.cache,
        offload=offload,
        visual_cond_scheme=gen.visual_cond_scheme,
        max_area=gen.max_area,
        image_divisibility=gen.image_divisibility,
        piflow_conf=cfg.piflow,
        release_text_encoder=cfg.text_embedder.release_after_encode,
    )
    _apply_cache(pipe, cfg.cache, cache_mode, force=True)
    remember(
        pipe,
        cfg,
        attention_engine=engine,
        cache_mode=cache_mode,
        offload_strategy=strategy,
    )
    attach_profile(pipe, cfg.profile.mode)
    return pipe
