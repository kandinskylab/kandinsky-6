"""Reapply pipeline settings on an already constructed pipeline.

Weight files are read again only when the module itself has to be replaced
(another checkpoint, NF4, or a DiT shape / PiFlow head change). Cache,
attention, compile, offload, profile, and generation defaults stay on the same objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from kandinsky.core.algo.prepare_ropes import VisualRopeCache
from kandinsky.core.components.beautifier import build_beautifier
from kandinsky.core.components.text_embedder import Kandinsky6TextEmbedder
from kandinsky.core.components.vae_audio import build_audio_vae, build_vocoder
from kandinsky.core.components.vae_video import build_vae
from kandinsky.runtime.aoti import bind_aoti_to_visual_blocks
from kandinsky.runtime.cache import CacheDiT
from kandinsky.runtime.compile import DiTCompiler, restore_execution
from kandinsky.runtime.kernels import bind_attention, rebind_attention
from kandinsky.runtime.offload import OffloadStrategy, _module_to, build_offload
from kandinsky.runtime.profile import attach_profile, diff_launch_overrides
from kandinsky.runtime.sampler import reject_incompatible_cache

from .config import CacheConfig, PipelineConfig, load_config

logger = logging.getLogger("kandinsky")

_MODULE_NAMES = ("text_embedder", "dit", "vae", "audio_vae", "vocoder")


def _load_bound(path: str | Path) -> PipelineConfig:
    from .factory import bind_checkpoint_paths  # noqa: PLC0415  # factory imports this module

    return bind_checkpoint_paths(load_config(path))


def reconfigure(
    pipe,
    cfg: PipelineConfig | str | Path,
    *,
    attention_engine: str | None = None,
    cache_mode: str | None = None,
    offload_strategy: OffloadStrategy | None = None,
):
    """Apply ``cfg`` to ``pipe`` and return the same object."""
    config_path = getattr(pipe, "config_path", None)
    if not isinstance(cfg, PipelineConfig):
        config_path = str(cfg)
        cfg = _load_bound(cfg)
    launch_overrides = diff_launch_overrides(
        cfg,
        attention_engine=attention_engine,
        cache_mode=cache_mode,
        offload_strategy=offload_strategy,
    )

    engine = bind_attention(attention_engine if attention_engine is not None else cfg.attention.engine)
    mode = cache_mode if cache_mode is not None else cfg.cache.mode
    strategy: OffloadStrategy = offload_strategy if offload_strategy is not None else cfg.offload.strategy
    reject_incompatible_cache(cfg, mode)

    prev: PipelineConfig | None = getattr(pipe, "config", None)
    if _same_runtime(pipe, cfg, engine, mode, strategy):
        pipe.config_path = config_path
        pipe.launch_overrides = launch_overrides
        return pipe

    dit_replaced = _dit_needs_reload(prev, cfg)
    text_replaced = _text_needs_reload(prev, cfg)
    vae_replaced = prev is not None and prev.paths.vae != cfg.paths.vae
    audio_replaced = _audio_needs_reload(prev, cfg)
    vocoder_replaced = _vocoder_needs_reload(prev, cfg)
    modules_replaced = dit_replaced or text_replaced or vae_replaced or audio_replaced or vocoder_replaced

    _restore_when_offload_ends(pipe, strategy)
    load_device = torch.device("cpu") if strategy != "none" else pipe.device
    attention_changed = getattr(pipe, "attention_engine", None) != engine
    if dit_replaced:
        _replace_dit(pipe, cfg, engine, load_device)
    elif attention_changed:
        rebind_attention(pipe.raw_dit, engine)

    # Compiled and AOTI graphs capture the attention kernel. Python forwards
    # (`compile: none`) read the slot on each call, so they stay as they are.
    execution_stale = _execution_key(cfg) != getattr(pipe, "_execution_key", None)
    attention_needs_reexecute = attention_changed and getattr(pipe, "_execution_key", ("compile", "none")) != (
        "compile",
        "none",
    )
    execution_reapplied = dit_replaced or execution_stale or attention_needs_reexecute
    if execution_reapplied:
        _apply_execution(pipe.raw_dit, cfg, pipe.device)
    if dit_replaced and not isinstance(pipe.dit, CacheDiT):
        pipe.dit = CacheDiT(pipe.raw_dit)

    if text_replaced:
        _replace_text(pipe, cfg, load_device)
    if vae_replaced:
        _replace_vae(pipe, cfg, load_device)
    if audio_replaced:
        _replace_audio(pipe, cfg, load_device)
    if vocoder_replaced:
        _replace_vocoder(pipe, cfg, load_device)

    _apply_generation(pipe, cfg, prev)
    _apply_cache(pipe, cfg.cache, mode, force=dit_replaced or prev is None)
    _reject_block_export(strategy, cfg)
    # Block offload wraps each block forward. Recompile replaces those wrappers.
    _apply_offload(
        pipe,
        strategy,
        cfg.offload.pin_memory,
        force=modules_replaced or execution_reapplied,
    )
    attach_profile(pipe, cfg.profile.mode)
    remember(
        pipe,
        cfg,
        attention_engine=engine,
        cache_mode=mode,
        offload_strategy=strategy,
    )
    pipe.config_path = config_path
    pipe.launch_overrides = launch_overrides
    return pipe


def remember(
    pipe,
    cfg: PipelineConfig,
    *,
    attention_engine: str,
    cache_mode: str | None,
    offload_strategy: OffloadStrategy,
) -> None:
    """Record the runtime that is currently attached to ``pipe``."""
    pipe.config = cfg
    pipe.beautifier = build_beautifier(cfg.beautifier.name, model_path=cfg.beautifier.model_path)
    pipe.attention_engine = attention_engine
    pipe.cache_mode = cache_mode
    pipe.offload_strategy = offload_strategy
    pipe._execution_key = _execution_key(cfg)


def _same_runtime(pipe, cfg: PipelineConfig, engine: str, mode: str | None, strategy: str) -> bool:
    prev = getattr(pipe, "config", None)
    if prev is None or prev.model_dump() != cfg.model_dump():
        return False
    return (
        getattr(pipe, "attention_engine", None) == engine
        and getattr(pipe, "cache_mode", None) == mode
        and getattr(pipe, "offload_strategy", None) == strategy
    )


def _dit_needs_reload(prev: PipelineConfig | None, cfg: PipelineConfig) -> bool:
    if prev is None:
        return False
    return (
        prev.dit.model_dump() != cfg.dit.model_dump()
        or prev.paths.dit != cfg.paths.dit
        or prev.piflow.enabled != cfg.piflow.enabled
        or prev.piflow.dx_num_grid_points != cfg.piflow.dx_num_grid_points
    )


def _text_needs_reload(prev: PipelineConfig | None, cfg: PipelineConfig) -> bool:
    if prev is None:
        return False
    return (
        prev.text_embedder.model_dump() != cfg.text_embedder.model_dump()
        or prev.paths.qwen != cfg.paths.qwen
        or prev.paths.clip != cfg.paths.clip
        or prev.dit.text_token_padding != cfg.dit.text_token_padding
    )


def _audio_needs_reload(prev: PipelineConfig | None, cfg: PipelineConfig) -> bool:
    if prev is None:
        return False
    previous = None if prev.audio_vae is None else prev.audio_vae.model_dump()
    current = None if cfg.audio_vae is None else cfg.audio_vae.model_dump()
    return previous != current


def _vocoder_needs_reload(prev: PipelineConfig | None, cfg: PipelineConfig) -> bool:
    if prev is None:
        return False
    previous = None if prev.vocoder is None else prev.vocoder.model_dump()
    current = None if cfg.vocoder is None else cfg.vocoder.model_dump()
    return previous != current


def _execution_key(cfg: PipelineConfig) -> tuple[str, str]:
    if cfg.paths.dit_export is not None:
        return ("aoti", cfg.paths.dit_export)
    return ("compile", cfg.compile.strategy)


def _apply_execution(dit, cfg: PipelineConfig, device: torch.device) -> None:
    restore_execution(dit)
    if cfg.paths.dit_export is not None:
        bind_aoti_to_visual_blocks(dit, cfg.paths.dit_export, device=device)
        return
    DiTCompiler(cfg.compile.strategy).apply(dit)


def _apply_generation(pipe, cfg: PipelineConfig, prev: PipelineConfig | None) -> None:
    pipe.piflow_conf = cfg.piflow
    if prev is not None and prev.generation.model_dump() == cfg.generation.model_dump():
        return
    gen = cfg.generation
    pipe.num_steps = gen.num_steps
    pipe.guidance_weight = gen.guidance_weight
    pipe.scheduler_scale = gen.scheduler_scale
    pipe.scale_factor = gen.scale_factor
    pipe.height = gen.height
    pipe.width = gen.width
    pipe.latent_frames = gen.latent_frames
    pipe.sample_frames = gen.sample_frames
    pipe.visual_cond_scheme = gen.visual_cond_scheme
    pipe.max_area = gen.max_area if gen.max_area is not None else gen.height * gen.width
    pipe.image_divisibility = gen.image_divisibility
    pipe._visual_rope_cache = VisualRopeCache()


def _apply_cache(pipe, cache: CacheConfig, mode: str | None, *, force: bool) -> None:
    if not force and pipe.cache_conf.model_dump() == cache.model_dump() and getattr(pipe, "cache_mode", None) == mode:
        return
    pipe.cache_conf = cache
    if mode is None or mode == "none":
        pipe.set_cache(None)
        return
    if mode == "magcache":
        mag = cache.magcache
        if mag is None:
            raise ValueError("cache.mode='magcache' requires cache.magcache.mag_ratios in the config")
        if isinstance(mag.mag_ratios, dict):
            gen_mode, mag_ratios = next(iter(mag.mag_ratios)), mag.mag_ratios
        else:
            gen_mode, mag_ratios = None, list(mag.mag_ratios)
        pipe.set_cache(
            "magcache",
            mode_name=gen_mode,
            mag_ratios=mag_ratios,
            thresh=mag.thresh,
            K=mag.K,
            retention_ratio=mag.retention_ratio,
        )
        return
    if mode == "navicache":
        navi = cache.navicache
        pipe.set_cache(
            "navicache",
            thresh=0.05 if navi is None else navi.thresh,
            align_steps=10 if navi is None else navi.align_steps,
            process_noise=0.05 if navi is None else navi.process_noise,
            measurement_noise=0.05 if navi is None else navi.measurement_noise,
        )
        return
    raise ValueError(f"Unknown cache mode: {mode!r}")


def _restore_when_offload_ends(pipe, strategy: OffloadStrategy) -> None:
    if pipe.offload.strategy == "none" or strategy != "none":
        return
    if pipe.device.type == "cuda":
        torch.cuda.synchronize(pipe.device)
    _move_modules(pipe, pipe.device)


def _reject_block_export(strategy: OffloadStrategy, cfg: PipelineConfig) -> None:
    if strategy == "block" and cfg.paths.dit_export is not None:
        raise ValueError("Block offload supports eager and regional torch.compile, not paths.dit_export")


def _apply_offload(pipe, strategy: OffloadStrategy, pin_memory: bool, *, force: bool) -> None:
    current_pin = getattr(pipe.offload, "pin_memory", True)
    pin_matters = strategy in ("module", "block")
    unchanged = pipe.offload.strategy == strategy and (not pin_matters or current_pin == pin_memory)
    if unchanged and not force:
        return
    pipe.offload = build_offload(strategy, pipe.device, pin_memory=pin_memory)
    if strategy == "none":
        return
    for name in _MODULE_NAMES:
        if name == "text_embedder" and getattr(pipe, "release_text_encoder", False):
            continue
        module = getattr(pipe, name, None)
        if module is not None:
            pipe.offload.register(name, module)


def _move_modules(pipe, device: torch.device) -> None:
    for name in _MODULE_NAMES:
        if name == "text_embedder" and getattr(pipe, "release_text_encoder", False):
            continue
        module = getattr(pipe, name, None)
        if module is not None:
            _module_to(module, device, non_blocking=False)


def _replace_dit(pipe, cfg: PipelineConfig, engine: str, load_device: torch.device) -> None:
    from .factory import create_bare_dit  # noqa: PLC0415  # factory imports this module

    bare = create_bare_dit(cfg, load_device, engine)
    pipe.dit = CacheDiT(bare)
    pipe.attention_engine = engine


def _replace_text(pipe, cfg: PipelineConfig, load_device: torch.device) -> None:
    pipe.release_text_encoder = cfg.text_embedder.release_after_encode
    text_device = pipe.device if (cfg.text_embedder.quantized_qwen or cfg.text_embedder.release_after_encode) else load_device
    pipe.text_embedder = Kandinsky6TextEmbedder(
        qwen_path=cfg.paths.qwen,
        clip_path=cfg.paths.clip,
        max_length=cfg.text_embedder.max_length,
        device=text_device,
        quantized_qwen=cfg.text_embedder.quantized_qwen,
        text_token_padding=cfg.dit.text_token_padding,
    )
    if cfg.text_embedder.quantized_qwen:
        logger.info("Qwen: NF4 (bitsandbytes)")


def _replace_vae(pipe, cfg: PipelineConfig, load_device: torch.device) -> None:
    pipe.vae = build_vae(cfg.paths.vae, device=load_device)


def _replace_audio(pipe, cfg: PipelineConfig, load_device: torch.device) -> None:
    if cfg.audio_vae is None:
        pipe.audio_vae = None
        return
    audio = cfg.audio_vae
    pipe.audio_vae = build_audio_vae(
        tod_vae_ckpt=audio.tod_vae_ckpt,
        mode=audio.mode,
        need_vae_encoder=audio.need_vae_encoder,
        need_vae_decoder=audio.need_vae_decoder,
        scaling_factor=audio.scaling_factor,
        device=load_device,
    )


def _replace_vocoder(pipe, cfg: PipelineConfig, load_device: torch.device) -> None:
    if cfg.vocoder is None:
        pipe.vocoder = None
        return
    pipe.vocoder = build_vocoder(ckpt=cfg.vocoder.ckpt, device=load_device)
