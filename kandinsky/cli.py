from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cyclopts
import torch

from kandinsky.core.types import Kandinsky6PipelineOutput
from kandinsky.pipeline.sr import load_sr_pipeline
from kandinsky.runtime.home import export_dir, logs_dir, open_generate_run
from kandinsky.runtime.offload import OffloadStrategy
from kandinsky.runtime.profile import NoOpProfile, ProfileHandle

logger = logging.getLogger("kandinsky")

app = cyclopts.App(help="Kandinsky 6 Video inference CLI")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "devices" / "h100.yaml")


@app.command
def generate(  # noqa: PLR0913
    prompt: str,
    *,
    config: str = _DEFAULT_CONFIG,
    out: str | None = None,
    height: int | None = None,
    width: int | None = None,
    time_length: int | None = None,
    latent_frames: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    negative: str = (
        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
        "worst quality, low quality, ugly, deformed, walking backwards"
    ),
    device: str = "cuda",
    attention_engine: str | None = None,
    magcache: bool | None = None,
    navicache: bool = False,
    offload: str | None = None,
    image: str | None = None,
    audio: bool = True,
    progress: bool = True,
    sr: bool = False,
    warmup: bool = False,
):
    """Generate a video from a text prompt.

    With ``sr.enabled: true`` in the config (or ``--sr``) the clip is then
    super-resolved (see ``kandinsky.pipeline.sr``) into ``<out>_sr.<ext>``.
    Geometry / MagCache defaults come from the YAML config unless overridden.
    Pass ``--magcache`` / omit it to force on; use config ``cache.mode`` when
    neither flag is set. ``--navicache`` forces NaviCache.
    ``--offload module`` enables async CUDA module offload (overrides YAML).
    ``--attention-engine`` overrides YAML ``attention.engine``.
    Prompt rewrite follows YAML ``beautifier.name`` (default ``qwen25``).
    ``--image`` enables I2VA / I2V conditioning (path to reference image).
    ``--no-audio`` generates video only (t2v / i2v). Audio is on by default.
    ``--progress`` is the default and shows a tqdm bar over the denoising steps
    (and over the SR tiles). ``--no-progress`` hides it. In a log file each
    denoising step is one short line.
    ``--sr`` runs the SR stage even when the config keeps ``sr.enabled: false``
    (model paths still come from the config's ``sr:`` section).
    Without ``--out``, the clip is written to
    ``$KANDINSKY_HOME/outputs/generate_<YYYY-MM-DDTHH-MM-SS>/generations/output.mp4``.
    That folder also holds ``expanded_prompt.txt``, ``launch.json``, ``logs/``,
    and ``profiles/``. ``--out`` writes the given file instead.
    ``--warmup`` runs the same generation once without saving, then the saved run.
    The profile JSON is the second run. When SR runs, that JSON also records
    ``measurements.sr_time`` for the saved SR pass.
    """
    from kandinsky.pipeline.factory import get_pipeline

    if magcache and navicache:
        raise ValueError("--magcache and --navicache are mutually exclusive")

    if navicache:
        cache_mode = "navicache"
    elif magcache is True:
        cache_mode = "magcache"
    elif magcache is False:
        cache_mode = "none"
    else:
        cache_mode = None  # fall back to YAML cache.mode

    offload_strategy: OffloadStrategy | None = None
    if offload is not None:
        if offload not in ("none", "module", "block"):
            raise ValueError("--offload must be one of: none, module, block")
        offload_strategy = offload  # type: ignore[assignment]

    run = None
    if out is None:
        run = open_generate_run()
        _bind_generate_run(run)
        destination = run / "generations" / "output.mp4"
        _write_launch(run, prompt=prompt, config=config, seed=seed, output="generations/output.mp4")
    else:
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)

    pipe = get_pipeline(
        config,
        device=device,
        attention_engine=attention_engine,
        cache_mode=cache_mode,
        offload_strategy=offload_strategy,
    )

    def _generate(target, save_path: Path | None) -> Kandinsky6PipelineOutput:
        return target(
            text=prompt,
            height=height,
            width=width,
            time_length=time_length,
            latent_frames=latent_frames,
            num_steps=steps,
            guidance_weight=guidance,
            seed=seed,
            negative_text=negative,
            save_path=save_path,
            image=image,
            sample_audio=audio,
            show_progress=progress,
        )

    if warmup:
        timed = pipe.profile
        pipe.profile = NoOpProfile()
        logger.info("warmup generation")
        print("Warmup")
        _generate(pipe, None)
        pipe.profile = timed

    result = _generate(pipe, destination)
    if run is not None:
        _write_run_prompt(run, destination, result.prompts)

    kind = "frames + audio" if result.audio is not None else "frames"
    print(f"Saved {result.frames.shape[2]} {kind} → {result.path}")

    fps, audio_fps = int(pipe.fps), pipe.audio_fps
    profile = pipe.profile
    del pipe  # free the base model before the SR components come in
    gc.collect()
    _super_resolve_if_enabled(
        config,
        device,
        offload_strategy,
        result,
        destination,
        profile,
        fps=fps,
        audio_fps=audio_fps,
        show_progress=progress,
        force=sr,
        warmup=warmup,
    )


def _bind_generate_run(run: Path) -> None:
    """Point this process's logs and profiles at one generate run."""
    os.environ["KANDINSKY_LOG_DIR"] = str(run / "logs")
    os.environ["KANDINSKY_PROFILE_DIR"] = str(run / "profiles")
    configure_logging()


def _write_launch(run: Path, **fields: object) -> None:
    payload = {
        "name": "generate",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **fields,
    }
    (run / "launch.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_run_prompt(run: Path, video_path: Path, prompts: list[str] | None) -> Path:
    """Store the expanded caption as ``expanded_prompt.txt`` in the run folder.

    The pipeline writes that caption beside the video. The run folder keeps
    one copy under the fixed name.
    """
    target = run / "expanded_prompt.txt"
    sidecar = video_path.with_suffix(".txt")
    if sidecar.is_file() and sidecar.resolve() != target.resolve():
        sidecar.replace(target)
        return target
    text = "\n\n".join(item.rstrip() for item in (prompts or []))
    target.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return target


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


@torch.no_grad()
def _super_resolve_if_enabled(  # noqa: PLR0913
    config: str,
    device: str,
    offload_strategy: OffloadStrategy | None,
    result: Kandinsky6PipelineOutput,
    out: str | Path,
    profile: ProfileHandle,
    *,
    fps: int,
    audio_fps: int,
    show_progress: bool = False,
    force: bool = False,
    warmup: bool = False,
) -> None:
    """Run SR on the generated clip when the config's ``sr.enabled`` is true (or ``force``).

    Composition stays at the application layer: the base pipeline knows
    nothing about SR. The SR result is written next to ``out`` with an
    ``_sr`` suffix, carrying the generated audio track.
    """
    torch.cuda.empty_cache()
    sr_pipe = load_sr_pipeline(config, device, force=force, offload_strategy=offload_strategy)
    if sr_pipe is None:
        return
    from kandinsky_sr.pipeline.components import scale_factor_for
    from kandinsky_sr.pipeline.warmup import clip_base_resolution, compile_vae_decode
    from kandinsky_sr.pipeline.warmup import warmup as warmup_sr
    out_path = Path(out)
    sr_path = out_path.with_name(f"{out_path.stem}_sr{out_path.suffix}")
    frames_tchw = result.frames[0].permute(1, 0, 2, 3).contiguous().cpu()  # (3,T,H,W) -> (T,3,H,W)
    # One-time compile work outside the SR pipeline, as kandy-sr does: flex/nabla
    # attention kernels, then the KVAE decode for the base resolution this clip tiles into.
    with sr_pipe.offload.use("dit"):
        warmup_sr(sr_pipe.dit, scale_factor_for(sr_pipe), device)
    with sr_pipe.offload.use("vae"):
        compile_vae_decode(sr_pipe, device, bases=[clip_base_resolution(sr_pipe, tuple(frames_tchw.shape[-2:]))])
    sr_kwargs = {
        "video": frames_tchw,
        "audio": result.audio,
        "fps": fps,
        "audio_sample_rate": audio_fps,
        "show_progress": show_progress,
    }
    if warmup:
        logger.info("warmup SR")
        print("Warmup SR")
        sr_pipe(**sr_kwargs, save_path=None)
    _synchronize(device)
    started = time.perf_counter()
    sr_result = sr_pipe(**sr_kwargs, save_path=str(sr_path))
    _synchronize(device)
    profile.note_sr(time.perf_counter() - started)
    print(f"Saved SR x{sr_pipe.resolution_scale} {tuple(sr_result.frames.shape[-2:])} → {sr_result.path}")


@app.command
def bench(
    *,
    gpu: int | None = None,
    prompt: str | None = None,
    seed: int = 0,
    mem_threshold: int = 1000,
) -> None:
    """Time twelve t2v generations on one GPU.

    Lite and Pro, distilled and not, at SD, HD, and Full HD. Each checkpoint
    is loaded once and reused for its three resolutions. Compile is torch,
    every cell warms up, and non-distilled checkpoints use MagCache from the
    checkpoint file. Profile mode is ``time_mem_module``. JSON files go to
    ``profiles/`` inside ``$KANDINSKY_HOME/outputs/bench_<date-time>/``.
    ``--gpu`` selects one card. When it is omitted, the first free GPU is used.
    """
    from kandinsky.bench import run

    raise SystemExit(run(gpu=gpu, prompt=prompt, seed=seed, mem_threshold=mem_threshold))


@app.command
def download(*names: str, cache_dir: str | None = None) -> None:
    """Download catalog snapshots when they are not already on disk.

    Each name is one of ``pro``, ``pro-distill``, ``pro-pretrain``, ``lite``,
    ``lite-distill``, ``lite-pretrain``. Snapshots go to
    ``$KANDINSKY_HOME/weights``, or to ``--cache-dir``.
    """
    from kandinsky.runtime.weights import CHECKPOINTS, ensure_checkpoint, get_checkpoint

    if not names:
        known = ", ".join(CHECKPOINTS)
        raise ValueError(f"pass at least one checkpoint name; known: {known}")
    for name in names:
        get_checkpoint(name)
    for name in names:
        print(ensure_checkpoint(name, cache_dir=cache_dir))


@app.command
def export(
    *,
    config: str = _DEFAULT_CONFIG,
    out: str | None = None,
    device: str = "cuda:0",
    attention_engine: str | None = None,
):
    """Export one visual block via torch.export + AOTInductor (.pt2).

    Regional AOT: compile ``visual_transformer_blocks[0]`` once; at runtime the same package
    is bound onto every block with that block's Python weights. Compatible with
    future per-block offload (Python loop + ``user_managed`` constants).

    AOTI Inductor knobs come from ``compile.aoti`` in the YAML (default:
    max-autotune-no-cudagraphs). First export is slow; reuses
    ``compile.aoti.cache_dir`` or ``$KANDINSKY_HOME/export/inductor``.

    ``--out`` defaults to ``$KANDINSKY_HOME/export/dit_block_aoti.pt2``.
    Point ``paths.dit_export`` at that file. Skips ``compile.strategy`` / CacheDiT
    during export. Loads the bare DiT on ``--device``;
    only ``visual_transformer_blocks[0]`` is captured.
    """
    import os

    import torch

    from kandinsky.pipeline.config import load_config
    from kandinsky.pipeline.factory import bind_checkpoint_paths, create_bare_dit
    from kandinsky.runtime.aoti import (
        build_example_kwargs,
        compile_aoti,
        export_visual_block,
        save_exported_program,
    )
    from kandinsky.runtime.kernels import bind_attention

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = bind_checkpoint_paths(load_config(config))
    aoti = cfg.compile.aoti
    destination = Path(out) if out is not None else export_dir() / "dit_block_aoti.pt2"
    inductor_cache = aoti.cache_dir or str(export_dir() / "inductor")
    requested = attention_engine if attention_engine is not None else cfg.attention.engine
    engine = bind_attention(requested)
    target = torch.device(device)
    logger.info("attention_engine=%s", engine)
    logger.info(
        "compile.aoti: max_autotune=%s coord_descent=%s cudagraphs=%s",
        aoti.max_autotune,
        aoti.coordinate_descent_tuning,
        aoti.cudagraphs,
    )

    logger.info("load DiT on %s", target)
    dit = create_bare_dit(cfg, target, engine)
    gen = cfg.generation
    latent_frames = gen.latent_frames if gen.latent_frames is not None else 32
    example_kwargs = build_example_kwargs(
        dit,
        height=gen.height,
        width=gen.width,
        latent_frames=latent_frames,
        scale_factor=tuple(gen.scale_factor),
        text_max_length=cfg.text_embedder.max_length,
        in_text_dim=cfg.dit.in_text_dim,
        in_text_dim2=cfg.dit.in_text_dim2,
        device=target,
    )
    if not cfg.dit.text_token_padding:
        logger.warning(
            "dit.text_token_padding is false, so cond and uncond text lengths differ at runtime; "
            "set text_token_padding=true (K5 pad-to-max) for AOTI"
        )
    kind = "FusedTransformerDecoderBlock" if dit.is_multimodal else "TransformerDecoderBlock"
    logger.info(
        "torch.export %s (visual_transformer_blocks[0]) on %s (text_token_padding=%s)",
        kind,
        target,
        cfg.dit.text_token_padding,
    )
    ep = export_visual_block(
        dit,
        example_kwargs,
        device=target,
        text_max_length=cfg.text_embedder.max_length,
    )
    ep_path = destination.with_name(destination.name.removesuffix(".pt2") + ".ep.pt2")
    logger.info("torch.export.save → %s", ep_path)
    save_exported_program(ep, ep_path)
    del dit
    if target.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("AOTInductor → %s", destination)
    path = compile_aoti(
        ep,
        destination,
        inductor_configs=aoti.inductor_configs(),
        cache_dir=inductor_cache,
    )
    print(f"[export] wrote {path}")
    print(f"[export] wrote {ep_path}")


def configure_logging() -> None:
    """Send ``kandinsky`` logs to stderr and ``$KANDINSKY_HOME/logs/kandinsky.log``."""
    log_dir = logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "kandinsky.log").resolve()
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    kandinsky_logger = logging.getLogger("kandinsky")
    kandinsky_logger.setLevel(logging.INFO)
    kandinsky_logger.propagate = False
    has_stream = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in kandinsky_logger.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        kandinsky_logger.addHandler(stream)
    has_file = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in kandinsky_logger.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        kandinsky_logger.addHandler(file_handler)


def main() -> None:
    configure_logging()
    app()
