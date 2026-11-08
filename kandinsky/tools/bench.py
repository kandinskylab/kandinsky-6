"""Eight t2v timings on one GPU.

Lite and Pro, distilled and not, at SD and Full HD. Each checkpoint is
loaded once and reused for both resolutions. Compile is torch, and each
cell is three generations. The reported time is their median, and the module
chart uses the median of each module across those three profiles. There is no
warmup pass. Non-distilled checkpoints use the MagCache block from their
checkpoint file. Profile mode is ``time_mem_module``. JSON files go to
``profiles/`` inside the run folder. The same folder receives
``module-time-report.html`` after the cells finish.

Offload starts at ``none`` when the resident modules fit in VRAM, at ``module``
when their sum does not, and at ``block`` when any one module is larger than
the card. A CUDA out-of-memory error drops that attempt and tries the next
strategy. Completed passes of one strategy stay in the result: a later
out-of-memory error does not clear them. Failed attempts are not part of the
median.

Full HD matches the performance report: the base clip is super-resolved
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

from collections.abc import Sequence

from kandinsky.tools.bench_report import write_module_time_report
from kandinsky.pipeline.config import _deep_merge, _read_mapping, _resolve_token, load_config
from kandinsky.runtime.home import kandinsky_home
from kandinsky.runtime.offload import OffloadStrategy

_SR = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
_SR_STEPS = 2
DEFAULT_PROMPT = (
    "A red dragon sits in a cave on a pile of gold. "
    "Light fantasy style, realism. "
    "It tilts its head back and exhales a stream of flame."
)
DEFAULT_SEED = 0
REPEATS = 3
_FREE_GPU_THRESHOLD_MIB = 1000
_OFFLOAD_LADDER: tuple[OffloadStrategy, ...] = ("none", "module", "block")
_RESIDENT_MODULES = ("text_embedder", "dit", "vae", "audio_vae", "vocoder")
CHECKPOINTS = ("lite", "lite-distill", "pro", "pro-distill")


@dataclass(frozen=True, slots=True)
class Resolution:
    name: str
    height: int
    width: int
    scale: int | float | None


RESOLUTIONS = (
    Resolution("sd", 480, 864, None),
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
    offload: str = ""
    profiles: tuple[Path, ...] = ()
    error: str = ""


def bind_run_profiles(run: Path) -> Path:
    """Write ``time_mem_module`` JSON files into this run's ``profiles/`` folder."""
    directory = run / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["KANDINSKY_PROFILE_DIR"] = str(directory)
    return directory


def initial_offload(module_bytes: dict[str, int], vram_bytes: int) -> OffloadStrategy:
    """First offload strategy whose weights can reside on the card.

    Unknown sizes start at ``none``. A module larger than the card starts at
    ``block``. A sum larger than the card starts at ``module``.
    """
    if vram_bytes <= 0 or not module_bytes:
        return "none"
    if max(module_bytes.values()) > vram_bytes:
        return "block"
    if sum(module_bytes.values()) > vram_bytes:
        return "module"
    return "none"


def resident_module_bytes(pipe) -> dict[str, int]:
    """Parameter and buffer bytes of each offload module, counting each storage once."""
    from torch import nn

    sizes: dict[str, int] = {}
    for name in _RESIDENT_MODULES:
        module = getattr(pipe, name, None)
        if module is None:
            continue
        sizes[name] = _unique_nbytes(module, nn.Module)
    return sizes


def _unique_nbytes(module, module_type: type) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in _resident_tensors(module, module_type):
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        if key == 0 or key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _resident_tensors(module, module_type: type):
    if isinstance(module, module_type):
        yield from module.parameters()
        yield from module.buffers()
        return
    for value in vars(module).values():
        if isinstance(value, module_type):
            yield from value.parameters()
            yield from value.buffers()


def _keep_completed(
    kept: tuple[list[_Sample], OffloadStrategy] | None,
    samples: list[_Sample],
    strategy: OffloadStrategy,
) -> tuple[list[_Sample], OffloadStrategy] | None:
    """Remember the largest finished set. A later OOM must not drop it."""
    if not samples:
        return kept
    if kept is None or len(samples) > len(kept[0]):
        return (list(samples), strategy)
    return kept


def _next_offload(strategy: OffloadStrategy) -> OffloadStrategy | None:
    index = _OFFLOAD_LADDER.index(strategy)
    if index + 1 == len(_OFFLOAD_LADDER):
        return None
    return _OFFLOAD_LADDER[index + 1]


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def cell_document(checkpoint: str, resolution: Resolution) -> dict:
    """Device preset for one cell. The checkpoint file supplies steps and PiFlow."""
    document = {
        "checkpoint": checkpoint,
        "generation": {"height": resolution.height, "width": resolution.width},
        "cache": {"mode": "none"},
        "offload": {"strategy": "none"},
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


def _accel_documents(accel: Sequence[str]) -> list[tuple[str, dict]]:
    """Resolve each acceleration token to its YAML mapping, left-to-right.

    ``configs/accelerations/<token>.yaml`` is deep-merged into every cell by
    :func:`write_plan`. Resolution raises the same error as token composition
    (missing / ambiguous / wrong folder).
    """
    resolved: list[tuple[str, dict]] = []
    for token in accel:
        path = _resolve_token(token)
        if path.parent.name != "accelerations":
            raise ValueError(f"--accel expects a token from configs/accelerations/, got {token!r} in {path.parent.name}/")
        resolved.append((token, _read_mapping(path)))
    return resolved


def _accel_cell_name(checkpoint: str, resolution_name: str, accel: Sequence[str]) -> str:
    if not accel:
        return f"{checkpoint}-{resolution_name}"
    tokens = "+".join(accel)
    return f"{checkpoint}-{tokens}-{resolution_name}"


def write_plan(
    directory: Path,
    *,
    only: Sequence[str] | None = None,
    accel: Sequence[str] | None = None,
) -> list[Group]:
    """Write one YAML per cell and group them by checkpoint.

    ``only`` keeps the cells whose base name (``<checkpoint>-<resolution>``)
    appears in the sequence. ``accel`` resolves one or more acceleration tokens
    from ``configs/accelerations/`` and deep-merges them into each cell;
    cells are renamed to include the acceleration tokens.
    """
    directory.mkdir(parents=True, exist_ok=True)
    only_set = set(only) if only else None
    overlays = _accel_documents(accel or ())
    groups: list[Group] = []
    for checkpoint in CHECKPOINTS:
        jobs: list[Job] = []
        for resolution in RESOLUTIONS:
            base_name = f"{checkpoint}-{resolution.name}"
            if only_set is not None and base_name not in only_set:
                continue
            document = cell_document(checkpoint, resolution)
            for _, overlay in overlays:
                document = _deep_merge(document, overlay)
            name = _accel_cell_name(checkpoint, resolution.name, [token for token, _ in overlays])
            path = directory / f"{name}.yaml"
            body = yaml.safe_dump(document, sort_keys=False)
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
        if jobs:
            groups.append(Group(checkpoint=checkpoint, jobs=tuple(jobs)))
    if only_set is not None:
        matched = {f"{group.checkpoint}-{job.resolution}" for group in groups for job in group.jobs}
        missing = only_set - matched
        if missing:
            raise ValueError(f"--only names not in the bench matrix: {sorted(missing)}")
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


@dataclass
class _Sample:
    elapsed: float
    profile: Path


def _median_sample(samples: list[_Sample]) -> _Sample:
    """Middle sample of an odd count. An even count averages the two central times."""
    ordered = sorted(samples, key=lambda item: item.elapsed)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    elapsed = (ordered[mid - 1].elapsed + ordered[mid].elapsed) / 2
    return _Sample(elapsed, ordered[mid].profile)


@torch.no_grad()
def _super_resolve(pipe, result, out: Path) -> None:
    """Saved SR pass. The base pipeline stays loaded for the next resolution."""
    from kandinsky.pipeline.sr import load_sr_pipeline

    if not pipe.config.sr.enabled:
        return
    device = str(pipe.device)
    torch.cuda.empty_cache()
    sr_pipe = None
    try:
        with pipe.profile.measure_sr(device) as sr_pass:
            sr_pipe = load_sr_pipeline(pipe.config, device)
            if sr_pipe is None:
                return
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
            started = time.perf_counter()
            sr_result = sr_pipe(**sr_kwargs, save_path=str(sr_path))
            sr_pass.elapsed = time.perf_counter() - started
            sr_pass.save = True
            shape = tuple(sr_result.frames.shape[-2:])
            print(f"Saved SR x{sr_pipe.resolution_scale} {shape} → {sr_result.path}", flush=True)
    finally:
        del sr_pipe
        torch.cuda.empty_cache()


def _second_pass_time(path: Path | None) -> float | None:
    """Profiled generation plus SR."""
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


def _vram_bytes(device) -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    cuda = torch.device(device)
    index = cuda.index if cuda.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).total_memory)


def _empty_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _one_pass(pipe, job: Job, prompt: str, seed: int, out_path: Path) -> _Sample:
    result = _generate(pipe, prompt, seed, out_path)
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
    return _Sample(elapsed, profile_path)


def _use_offload(pipe, job: Job, strategy: OffloadStrategy):
    if getattr(pipe, "offload_strategy", None) == strategy:
        return pipe
    return pipe.reconfigure(job.config_path, offload_strategy=strategy)


def _cell_result(
    samples: list[_Sample], strategy: OffloadStrategy
) -> tuple[float, Path, OffloadStrategy, tuple[Path, ...]]:
    chosen = _median_sample(samples)
    return chosen.elapsed, chosen.profile, strategy, tuple(sample.profile for sample in samples)


def _run_cell(
    pipe, job: Job, prompt: str, seed: int, out_path: Path
) -> tuple[float, Path, OffloadStrategy, tuple[Path, ...]]:
    """Up to three generations at one offload strategy. OOM steps up the ladder.

    A finished pass is kept. Switching strategy starts a new set, and the
    result is the largest set that completed. An empty ladder still raises.
    """
    strategy = initial_offload(resident_module_bytes(pipe), _vram_bytes(pipe.device))
    samples: list[_Sample] = []
    kept: tuple[list[_Sample], OffloadStrategy] | None = None
    while len(samples) < REPEATS:
        try:
            pipe = _use_offload(pipe, job, strategy)
            sample = _one_pass(pipe, job, prompt, seed, out_path)
        except Exception as exc:
            _empty_cache()
            if not _is_oom(exc):
                raise
            nxt = _next_offload(strategy)
            print(f"OOM with offload={strategy}" + ("" if nxt is None else f"; trying {nxt}"), flush=True)
            kept = _keep_completed(kept, samples, strategy)
            samples.clear()
            if nxt is None:
                if kept is None:
                    raise
                print(
                    f"keeping {len(kept[0])} completed pass(es) at offload={kept[1]}",
                    flush=True,
                )
                return _cell_result(kept[0], kept[1])
            strategy = nxt
            continue
        samples.append(sample)
        print(f"  rep {len(samples)}/{REPEATS} {sample.elapsed:.1f}s offload={strategy}", flush=True)
    return _cell_result(samples, strategy)


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
                # Load on CPU. The cell moves weights to the card only after the size check.
                if pipe is None:
                    pipe = get_pipeline(str(job.config_path), device="cuda:0", offload_strategy="module")
                else:
                    pipe.reconfigure(job.config_path, offload_strategy="module")
                elapsed, profile_path, strategy, profiles = _run_cell(pipe, job, prompt, seed, out_path)
                result = RunResult(job.name, True, elapsed, log_path, profile_path, strategy, profiles)
                log_path.write_text(
                    f"OK median {elapsed:.3f}s offload={strategy} repeats={len(profiles)}\nprofile {profile_path}\n",
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
    print(f"\n{'Job':<20} {'Result':>6} {'Offload':>8} {'Median':>8}")
    print("-" * 48)
    all_ok = True
    recorded = []
    for result in results:
        status = "OK" if result.success else "FAIL"
        shown = "n/a" if result.elapsed is None else f"{result.elapsed:7.1f}s"
        hint = "" if result.success else f"  <- {result.log_path}"
        offload = result.offload or "-"
        print(f"{result.name:<20} {status:>6} {offload:>8} {shown:>8}{hint}")
        all_ok = all_ok and result.success
        recorded.append(
            {
                "name": result.name,
                "success": result.success,
                "seconds": result.elapsed,
                "offload": result.offload,
                "repeats": len(result.profiles) if result.success else 0,
                "profile": None if result.profile_path is None else str(result.profile_path),
                "profiles": [str(path) for path in result.profiles],
                "log": str(result.log_path),
                "error": result.error,
            }
        )
    print("-" * 48)
    verdict = "All passed." if all_ok else "Some jobs FAILED."
    print("Median uses the completed generations of one offload strategy. Failed attempts are excluded.")
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
    only: Sequence[str] | None = None,
    accel: Sequence[str] | None = None,
) -> int:
    """Run the eight cells. Returns a process status code.

    ``only`` and ``accel`` narrow the matrix: ``--only pro-sd --accel spd``
    runs one cell with the SPEED acceleration deep-merged on top of the usual
    ``cell_document``.
    """
    text = DEFAULT_PROMPT if prompt is None else prompt
    home = kandinsky_home()
    home.mkdir(parents=True, exist_ok=True)
    gpu_id = _one_gpu(gpu, mem_threshold)
    if gpu_id is None:
        return 1

    run_dir = _open_run(home)
    profiles = bind_run_profiles(run_dir)
    groups = write_plan(run_dir / "configs", only=only, accel=accel)
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
    report = write_module_time_report(run_dir)
    print("Report: no profiles" if report is None else f"Report: {report}")
    return 0 if ok else 1
