"""Twelve t2v timings on one GPU.

Lite and Pro, distilled and not, at SD, HD, and Full HD. Each checkpoint is
loaded once and reused for its three resolutions. Compile is torch, every cell
warms up, and non-distilled checkpoints use the MagCache block from their
checkpoint file. Profile mode is ``time_mem_module``. JSON files go to
``profiles/`` inside the run folder.

HD and Full HD match the performance report: the base clip is super-resolved
with the distilled 2-step VSR model. t2v here is text to video with audio and
no reference image. Distilled checkpoints use PiFlow, which requires audio.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import yaml

from kandinsky.pipeline.config import load_config
from kandinsky.runtime.home import kandinsky_home

_SR = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
_SR_STEPS = 2
DEFAULT_PROMPT = (
    "A red dragon sits in a cave on a pile of gold. "
    "Light fantasy style, realism. "
    "It tilts its head back and exhales a stream of flame."
)
DEFAULT_SEED = 0
_FREE_GPU_THRESHOLD_MIB = 1000
CHECKPOINTS = ("lite", "lite-distill", "pro", "pro-distill")


@dataclass(frozen=True, slots=True)
class Resolution:
    name: str
    height: int
    width: int
    scale: int | float | None


RESOLUTIONS = (
    Resolution("sd", 480, 864, None),
    Resolution("hd", 432, 768, 2),
    Resolution("fhd", 480, 864, 2.25),
)


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    checkpoint: str
    resolution: str
    config_path: Path
    sr: bool


@dataclass(frozen=True, slots=True)
class Group:
    checkpoint: str
    jobs: tuple[Job, ...]


@dataclass
class RunResult:
    name: str
    success: bool
    elapsed: float | None
    log_path: Path
    profile_path: Path | None = None
    error: str = ""


def bind_run_profiles(run: Path) -> Path:
    """Write ``time_mem_module`` JSON files into this run's ``profiles/`` folder."""
    directory = run / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["KANDINSKY_PROFILE_DIR"] = str(directory)
    return directory


def cell_document(checkpoint: str, resolution: Resolution) -> dict:
    """Device preset for one cell. The checkpoint file supplies steps and PiFlow."""
    document = {
        "checkpoint": checkpoint,
        "generation": {"height": resolution.height, "width": resolution.width},
        "cache": {"mode": "none"},
        "offload": {"strategy": "module"},
        "compile": {"strategy": "torch"},
        "attention": {"engine": "flash_attention_3"},
        "text_embedder": {"quantized_qwen": False},
        "beautifier": {"name": "qwen25"},
        "profile": {"mode": "time_mem_module"},
        "sr": {"enabled": False},
    }
    if not checkpoint.endswith("-distill"):
        # Ratios, thresh, K, and retention stay in the checkpoint YAML.
        document["cache"] = {"mode": "magcache"}
    if resolution.scale is not None:
        document["sr"] = {
            "enabled": True,
            "checkpoint_path": _SR,
            "resolution_scale": resolution.scale,
            "num_steps": _SR_STEPS,
        }
    return document


def write_plan(directory: Path) -> list[Group]:
    """Write one YAML per cell and group them by checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    groups: list[Group] = []
    for checkpoint in CHECKPOINTS:
        jobs: list[Job] = []
        for resolution in RESOLUTIONS:
            name = f"{checkpoint}-{resolution.name}"
            path = directory / f"{name}.yaml"
            body = yaml.safe_dump(cell_document(checkpoint, resolution), sort_keys=False)
            path.write_text(f"# {name} t2v\n{body}", encoding="utf-8")
            jobs.append(
                Job(
                    name=name,
                    checkpoint=checkpoint,
                    resolution=resolution.name,
                    config_path=path,
                    sr=resolution.scale is not None,
                )
            )
        groups.append(Group(checkpoint=checkpoint, jobs=tuple(jobs)))
    return groups


def _free_gpus(threshold_mib: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            index, used = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if used < threshold_mib:
            result.append(index)
    return result


def _one_gpu(requested: int | None, threshold_mib: int) -> int | None:
    if requested is not None:
        return requested
    free = _free_gpus(threshold_mib)
    if not free:
        print(
            f"No free GPUs found (memory.used < {threshold_mib} MiB). Pass --gpu.",
            file=sys.stderr,
        )
        return None
    print(f"Free GPUs detected: {free}; using cuda:{free[0]}")
    return free[0]


def _open_run(home: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root = home / "outputs" / f"bench_{stamp}"
    if root.exists():
        root = home / "outputs" / f"bench_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')}"
    for name in ("generations", "logs", "configs", "profiles"):
        (root / name).mkdir(parents=True)
    return root


def _write_launch(root: Path, groups: list[Group], prompt: str, seed: int) -> None:
    recorded = [
        {
            "name": job.name,
            "checkpoint": job.checkpoint,
            "resolution": job.resolution,
            "config": f"configs/{job.config_path.name}",
        }
        for group in groups
        for job in group.jobs
    ]
    payload = {
        "name": "bench",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt": prompt,
        "seed": seed,
        "profile_dir": str(root / "profiles"),
        "jobs": recorded,
    }
    (root / "launch.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _generate(pipe, prompt: str, seed: int, save_path: Path | None):
    return pipe(
        text=prompt,
        seed=seed,
        save_path=save_path,
        sample_audio=True,
        show_progress=True,
    )


def _warmup_then_generate(pipe, prompt: str, seed: int, save_path: Path):
    from kandinsky.runtime.profile import NoOpProfile

    timed = pipe.profile
    pipe.profile = NoOpProfile()
    try:
        print("Warmup", flush=True)
        _generate(pipe, prompt, seed, None)
    finally:
        pipe.profile = timed
    return _generate(pipe, prompt, seed, save_path)


def _synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


@torch.no_grad()
def _super_resolve(pipe, result, out: Path) -> None:
    """Saved SR pass. The base pipeline stays loaded for the next resolution."""
    import torch

    from kandinsky.pipeline.sr import load_sr_pipeline

    if not pipe.config.sr.enabled:
        return
    device = str(pipe.device)
    torch.cuda.empty_cache()
    sr_pipe = load_sr_pipeline(pipe.config, device)
    if sr_pipe is None:
        return
    try:
        from kandinsky_sr.pipeline.components import scale_factor_for
        from kandinsky_sr.pipeline.warmup import clip_base_resolution, compile_vae_decode
        from kandinsky_sr.pipeline.warmup import warmup as warmup_sr

        sr_path = out.with_name(f"{out.stem}_sr{out.suffix}")
        frames_tchw = result.frames[0].permute(1, 0, 2, 3).contiguous().cpu()
        with sr_pipe.offload.use("dit"):
            warmup_sr(sr_pipe.dit, scale_factor_for(sr_pipe), device)
        with sr_pipe.offload.use("vae"):
            compile_vae_decode(
                sr_pipe,
                device,
                bases=[clip_base_resolution(sr_pipe, tuple(frames_tchw.shape[-2:]))],
            )
        sr_kwargs = {
            "video": frames_tchw,
            "audio": result.audio,
            "fps": int(pipe.fps),
            "audio_sample_rate": pipe.audio_fps,
            "show_progress": True,
        }
        print("Warmup SR", flush=True)
        sr_pipe(**sr_kwargs, save_path=None)
        _synchronize(device)
        started = time.perf_counter()
        sr_result = sr_pipe(**sr_kwargs, save_path=str(sr_path))
        _synchronize(device)
        pipe.profile.note_sr(time.perf_counter() - started)
        shape = tuple(sr_result.frames.shape[-2:])
        print(f"Saved SR x{sr_pipe.resolution_scale} {shape} → {sr_result.path}", flush=True)
    finally:
        del sr_pipe
        torch.cuda.empty_cache()


def _second_pass_time(path: Path | None) -> float | None:
    """Profiled generation plus SR. Warmup is not in the JSON."""
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        return None
    total = measurements.get("total_time")
    if not isinstance(total, (int, float)):
        return None
    sr = measurements.get("sr_time")
    sr_time = float(sr) if isinstance(sr, (int, float)) else 0.0
    return float(total) + sr_time


def _release(pipe) -> None:
    import torch

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_cell(pipe, job: Job, prompt: str, seed: int, out_path: Path) -> tuple[float, Path]:
    result = _warmup_then_generate(pipe, prompt, seed, out_path)
    kind = "frames + audio" if result.audio is not None else "frames"
    print(f"Saved {result.frames.shape[2]} {kind} → {result.path}", flush=True)
    _super_resolve(pipe, result, out_path)
    profile_path = pipe.profile.path
    elapsed = _second_pass_time(profile_path)
    if profile_path is None or elapsed is None:
        raise RuntimeError("profile json was not written")
    if not out_path.is_file():
        raise RuntimeError(f"missing output {out_path}")
    if job.sr and not out_path.with_name(f"{out_path.stem}_sr{out_path.suffix}").is_file():
        raise RuntimeError(f"missing SR output for {job.name}")
    return elapsed, profile_path


def _execute(groups: list[Group], prompt: str, seed: int, run: Path, gpu_id: int) -> list[RunResult]:
    import torch

    from kandinsky.cli import configure_logging
    from kandinsky.pipeline.factory import get_pipeline

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["KANDINSKY_LOG_DIR"] = str(run / "logs")
    bind_run_profiles(run)
    configure_logging()
    results: list[RunResult] = []
    for group in groups:
        pipe = None
        for job in group.jobs:
            print(f"  [cuda:{gpu_id}] {job.name} starting …", flush=True)
            log_path = run / "logs" / f"{job.name}.log"
            out_path = run / "generations" / f"{job.name}.mp4"
            try:
                if pipe is None:
                    pipe = get_pipeline(str(job.config_path), device="cuda:0")
                else:
                    pipe.reconfigure(job.config_path)
                elapsed, profile_path = _run_cell(pipe, job, prompt, seed, out_path)
                result = RunResult(job.name, True, elapsed, log_path, profile_path)
                log_path.write_text(
                    f"OK gen+sr {elapsed:.3f}s\nprofile {profile_path}\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                result = RunResult(job.name, False, None, log_path, error=f"{type(exc).__name__}: {exc}")
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
                if pipe is not None:
                    _release(pipe)
                    pipe = None
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(result)
            tag = "OK" if result.success else f"FAIL ({result.error})"
            seconds = "n/a" if result.elapsed is None else f"{result.elapsed:.1f}s"
            print(f"  [cuda:{gpu_id}] {job.name}: {tag} in {seconds}", flush=True)
        if pipe is not None:
            _release(pipe)
    return results


def _print_summary(results: list[RunResult], wall: float, run: Path, profiles: Path) -> bool:
    print(f"\n{'Job':<20} {'Result':>6} {'Gen+SR':>8}")
    print("-" * 38)
    all_ok = True
    recorded = []
    for result in results:
        status = "OK" if result.success else "FAIL"
        shown = "n/a" if result.elapsed is None else f"{result.elapsed:7.1f}s"
        hint = "" if result.success else f"  <- {result.log_path}"
        print(f"{result.name:<20} {status:>6} {shown:>8}{hint}")
        all_ok = all_ok and result.success
        recorded.append(
            {
                "name": result.name,
                "success": result.success,
                "seconds": result.elapsed,
                "profile": None if result.profile_path is None else str(result.profile_path),
                "log": str(result.log_path),
                "error": result.error,
            }
        )
    print("-" * 38)
    verdict = "All passed." if all_ok else "Some jobs FAILED."
    print("Gen+SR is the profiled second pass: generation + SR. Warmup is excluded.")
    print(f"Wall time: {wall:.0f}s   {verdict}")
    print(f"Run: {run}")
    print(f"Profiles: {profiles}")
    (run / "results.json").write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    return all_ok


def run(
    *,
    gpu: int | None = None,
    prompt: str | None = None,
    seed: int = DEFAULT_SEED,
    mem_threshold: int = _FREE_GPU_THRESHOLD_MIB,
) -> int:
    """Run the twelve cells. Returns a process status code."""
    text = DEFAULT_PROMPT if prompt is None else prompt
    home = kandinsky_home()
    home.mkdir(parents=True, exist_ok=True)
    gpu_id = _one_gpu(gpu, mem_threshold)
    if gpu_id is None:
        return 1

    run_dir = _open_run(home)
    profiles = bind_run_profiles(run_dir)
    groups = write_plan(run_dir / "configs")
    _write_launch(run_dir, groups, text, seed)

    for group in groups:
        for job in group.jobs:
            loaded = load_config(job.config_path)
            sr = "off" if not loaded.sr.enabled else f"x{loaded.sr.resolution_scale}"
            print(
                f"{job.name}: {loaded.checkpoint} {loaded.generation.height}x{loaded.generation.width} "
                f"steps={loaded.generation.num_steps} cache={loaded.cache.mode} "
                f"compile={loaded.compile.strategy} sr={sr}"
            )
    print(f"\n{sum(len(group.jobs) for group in groups)} jobs on cuda:{gpu_id}")
    print(f"Run: {run_dir}")
    print(f"KANDINSKY_HOME={home}")
    print(f"Profiles: {profiles}")
    print()

    wall_started = time.monotonic()
    results = _execute(groups, text, seed, run_dir, gpu_id)
    ok = _print_summary(results, time.monotonic() - wall_started, run_dir, profiles)
    return 0 if ok else 1
