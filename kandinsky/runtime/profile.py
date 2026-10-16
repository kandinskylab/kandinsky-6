"""Profiling policy and optional ``record_function`` instrumentation.

The policy is off unless ``profile.mode`` is set. ``time_mem_module`` times each
pipeline component and records its CUDA memory high-water mark, plus the same
totals for the whole ``pipeline.__call__``.

Bench / chrome profiling is separate: it installs temporary
``record_function`` wrappers via :func:`instrument_record_functions`.
Normal inference does not enter those regions.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from torch.profiler import record_function

from .base import ConfigModel
from .home import profile_dir

logger = logging.getLogger("kandinsky")

ProfileMode = Literal["none", "time_mem_module"]

# nn.Module.forward labels by class name (no import of dit — avoid cycles).
_FORWARD_LABELS: dict[str, str] = {
    "MultiheadSelfAttentionEnc": "k6/attention/sa",
    "MultiheadSelfAttentionDec": "k6/attention/sa",
    "MultiheadCrossAttention": "k6/attention/ca",
    "FeedForward": "k6/ffn",
}

_SR_FORWARD_LABELS: dict[str, str] = {
    "MultiheadSelfAttention": "sr/attention/sa",
    "MultiheadCrossAttention": "sr/attention/ca",
    "FeedForward": "sr/ffn",
}

# DiffusionTransformer3D stage helpers (bound methods on the root module).
_DIT_METHOD_LABELS: dict[str, str] = {
    "_encode_t2v": "k6/embed/text",
    "_embed_visual": "k6/embed/visual",
    "_embed_audio": "k6/embed/audio",
    "_run_visual_blocks_single": "k6/blocks/visual",
    "_run_visual_blocks_fused": "k6/blocks/visual",
}


def _wrap_callable(fn: Callable[..., Any], label: str) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with record_function(label):
            return fn(*args, **kwargs)

    return wrapped


def _wrap_encode_text(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Label includes modality prefix: k6/embed/text_video | text_audio."""

    def wrapped(prefix: str, *args: Any, **kwargs: Any) -> Any:
        with record_function(f"k6/embed/text_{prefix}"):
            return fn(prefix, *args, **kwargs)

    return wrapped


@contextmanager
def instrument_record_functions(root: Any | Sequence[Any]) -> Iterator[None]:  # noqa: PLR0912
    """Patch one or more DiTs with temporary ``record_function`` wrappers.

    Safe to nest around a chrome-profiler active window only. Does not touch
    MagCache / offload. If a submodule ``forward`` was already ``torch.compile``'d
    into a parent graph, nested wrappers may not fire — stage-method wrappers
    (``_run_visual_blocks_*``, embeds) still do.
    """
    # (obj, attr, previous_instance_value|None) — None ⇒ delete instance attr on restore
    restores: list[tuple[Any, str, Any | None]] = []

    def _patch(obj: Any, attr: str, new: Callable[..., Any]) -> None:
        prev = obj.__dict__.get(attr, None)
        restores.append((obj, attr, prev))
        setattr(obj, attr, new)

    roots = tuple(root) if isinstance(root, (list, tuple)) else (root,)
    for candidate in roots:
        dit = getattr(candidate, "module", candidate)
        is_sr = type(dit).__module__.startswith("kandinsky_sr.")
        if hasattr(dit, "forward"):
            _patch(dit, "forward", _wrap_callable(dit.forward, "sr/dit" if is_sr else "k6/dit"))
        if not is_sr:
            for method, label in _DIT_METHOD_LABELS.items():
                if hasattr(dit, method):
                    _patch(dit, method, _wrap_callable(getattr(dit, method), label))
            if hasattr(dit, "_encode_text"):
                _patch(dit, "_encode_text", _wrap_encode_text(dit._encode_text))

        labels = _SR_FORWARD_LABELS if is_sr else _FORWARD_LABELS
        modules = dit.modules() if hasattr(dit, "modules") else ()
        for mod in modules:
            label = labels.get(type(mod).__name__)
            if label is None:
                continue
            _patch(mod, "forward", _wrap_callable(mod.forward, label))

    try:
        yield
    finally:
        for obj, attr, prev in reversed(restores):
            if prev is None:
                with suppress(AttributeError):
                    delattr(obj, attr)
            else:
                setattr(obj, attr, prev)


class ProfileConfig(ConfigModel):
    """Which profiler to attach. ``none`` leaves the pipeline uninstrumented."""

    mode: ProfileMode = "none"


@dataclass(frozen=True, slots=True)
class ModuleProfile:
    """One component over a single ``pipeline.__call__``.

    ``time`` is the sum of that component's calls, in seconds.
    Peaks are the maximum CUDA high-water mark across those calls, in bytes.
    A component that did not run is absent from the report.
    """

    time: float
    peak_allocated_mem: int
    peak_reserved_mem: int


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Totals cover the whole call, including work outside any one component."""

    modules: dict[str, ModuleProfile]
    total_time: float
    total_peak_allocated_mem: int
    total_peak_reserved_mem: int
    sr_time: float | None = None


class ProfileHandle(Protocol):
    mode: ProfileMode

    def apply(self, pipe: object) -> None: ...

    def clear(self, pipe: object) -> None: ...

    def pipeline(self) -> Iterator[None]: ...

    def note_call(self, *, seed: int | None = None, **raw: Any) -> None: ...

    def note_prompt(self, prompt: str | list[str]) -> None: ...

    def note_sr(self, elapsed: float) -> None: ...

    @property
    def report(self) -> ProfileReport: ...


def build_profile(mode: ProfileMode) -> ProfileHandle:
    if mode == "none":
        return NoOpProfile()
    if mode == "time_mem_module":
        return TimeMemModuleProfile()
    raise ValueError(f"Unknown profile mode: {mode!r}")


def attach_profile(pipe: object, mode: ProfileMode) -> ProfileHandle:
    """Install ``mode`` on ``pipe``, replacing a profiler that was already attached."""
    current = getattr(pipe, "profile", None)
    if current is not None:
        current.clear(pipe)
    handle = build_profile(mode)
    handle.apply(pipe)
    pipe.profile = handle
    return handle


class NoOpProfile:
    mode: ProfileMode = "none"

    def apply(self, pipe: object) -> None:
        del pipe

    def clear(self, pipe: object) -> None:
        del pipe

    @contextmanager
    def pipeline(self) -> Iterator[None]:
        yield

    def note_call(self, *, seed: int | None = None, **raw: Any) -> None:
        del seed, raw

    def note_prompt(self, prompt: str | list[str]) -> None:
        del prompt

    def note_sr(self, elapsed: float) -> None:
        del elapsed

    @property
    def report(self) -> ProfileReport:
        return ProfileReport(
            modules={},
            total_time=0.0,
            total_peak_allocated_mem=0,
            total_peak_reserved_mem=0,
        )


@dataclass
class _Accum:
    time: float = 0.0
    peak_allocated_mem: int = 0
    peak_reserved_mem: int = 0


class TimeMemModuleProfile:
    """Time and CUDA peak memory for each pipeline component.

    Component entry points are wrapped in place. Repeated calls (denoising
    steps, the positive and negative text pass) add their durations and keep
    the larger peak. Peaks are bytes from ``max_memory_allocated`` /
    ``max_memory_reserved`` on ``pipe.device``. On CPU those peaks stay 0.
    """

    mode: ProfileMode = "time_mem_module"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._restores: list[tuple[Any, str, Any | None]] = []
        self._accums: dict[str, _Accum] = {}
        self._open: set[str] = set()
        self._enabled = False
        self._active = False
        self._t0 = 0.0
        self.total_time = 0.0
        self.total_peak_allocated_mem = 0
        self.total_peak_reserved_mem = 0
        self._last = _empty_report()
        self._seed: int | None = None
        self._raw_call: dict[str, Any] = {}
        self._written: Path | None = None

    def apply(self, pipe: object) -> None:
        self.clear(pipe)
        self._pipe = pipe
        self._enabled = True
        for obj, attr, name in _component_targets(pipe):
            self._patch(obj, attr, self._wrap_component(getattr(obj, attr), name))
        if getattr(pipe, "release_text_encoder", False):
            self._watch_text_lifecycle(pipe.text_embedder)

    def _watch_text_lifecycle(self, text: Any) -> None:
        """Do not let profiling retain NF4 weights after release, including warmup.

        The benchmark temporarily assigns a NoOpProfile during warmup, but
        this handle's installed forward wrappers are still present. Unpatch
        old weights on release; wrap the replacement models after reload.
        """
        ensure_loaded, release = text.ensure_loaded, text.release

        def ensure(*args: Any, **kwargs: Any) -> Any:
            result = ensure_loaded(*args, **kwargs)
            existing = {(id(obj), attr) for obj, attr, _prev in self._restores}
            for attr, name in (("qwen", "text_encoder"), ("clip", "text_encoder_2")):
                module = getattr(text, attr, None)
                if module is not None and (id(module), "forward") not in existing:
                    self._patch(module, "forward", self._wrap_component(module.forward, name))
            return result

        def drop(*args: Any, **kwargs: Any) -> Any:
            self._forget_text_weights(text)
            return release(*args, **kwargs)

        self._patch(text, "ensure_loaded", ensure)
        self._patch(text, "release", drop)

    def _forget_text_weights(self, text: Any) -> None:
        module_ids = {id(getattr(text, attr, None)) for attr in ("qwen", "clip")}
        kept = []
        for obj, attr, prev in self._restores:
            if id(obj) not in module_ids:
                kept.append((obj, attr, prev))
            elif prev is None:
                with suppress(AttributeError):
                    delattr(obj, attr)
            else:
                setattr(obj, attr, prev)
        self._restores = kept

    @contextmanager
    def pipeline(self) -> Iterator[None]:
        """Time one ``Kandinsky6Pipeline.__call__``. The pipeline enters this."""
        if not self._enabled:
            yield
            return
        self._begin_pipeline()
        try:
            yield
        finally:
            self._end_pipeline()

    def clear(self, pipe: object) -> None:
        del pipe
        for obj, attr, prev in reversed(self._restores):
            if prev is None:
                with suppress(AttributeError):
                    delattr(obj, attr)
            else:
                setattr(obj, attr, prev)
        self._restores.clear()
        self._pipe = None
        self._enabled = False
        self._active = False
        self._open.clear()

    @property
    def report(self) -> ProfileReport:
        return self._last

    @property
    def path(self) -> Path | None:
        """JSON written for the last call, if that call was profiled."""
        return self._written

    def note_call(self, *, seed: int | None = None, **raw: Any) -> None:
        """Record this call's arguments. Only values that differ from the pipeline are saved."""
        self._seed = seed
        self._raw_call = raw

    def note_prompt(self, prompt: str | list[str]) -> None:
        """Record the caption encoded for this call, after prompt expansion."""
        self._raw_call["prompt"] = prompt

    def note_sr(self, elapsed: float) -> None:
        """Record the saved super-resolution pass on the JSON of the last profiled call."""
        self._last = replace(self._last, sr_time=elapsed)
        path = self._written
        if path is None or not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        measurements = payload.get("measurements")
        if not isinstance(measurements, dict):
            return
        measurements["sr_time"] = elapsed
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        logger.info("profile sr_time=%.4fs path=%s", elapsed, path)

    def _patch(self, obj: Any, attr: str, new: Callable[..., Any]) -> None:
        prev = obj.__dict__.get(attr, None)
        self._restores.append((obj, attr, prev))
        setattr(obj, attr, new)

    def _wrap_component(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self._active or name in self._open:
                return fn(*args, **kwargs)
            self._open.add(name)
            started = self._begin_span()
            try:
                return fn(*args, **kwargs)
            finally:
                self._end_span(name, started)
                self._open.discard(name)

        return wrapped

    def _begin_pipeline(self) -> None:
        self._accums.clear()
        self._open.clear()
        self._seed = None
        self._raw_call = {}
        self._written = None
        self.total_time = 0.0
        self.total_peak_allocated_mem = 0
        self.total_peak_reserved_mem = 0
        self._active = True
        if self._use_cuda():
            device = self._device()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self._t0 = time.perf_counter()

    def _end_pipeline(self) -> None:
        if self._use_cuda():
            torch.cuda.synchronize(self._device())
            self._fold_peak()
        self.total_time = time.perf_counter() - self._t0
        self._active = False
        self._last = self._snapshot()
        self._written = _write_time_mem_module(self._pipe, self._last, self._seed, self._raw_call)
        _log_report(self._last, self._written)

    def _begin_span(self) -> float:
        if self._use_cuda():
            device = self._device()
            torch.cuda.synchronize(device)
            self._fold_peak()
            torch.cuda.reset_peak_memory_stats(device)
        return time.perf_counter()

    def _end_span(self, name: str, started: float) -> None:
        allocated = 0
        reserved = 0
        if self._use_cuda():
            device = self._device()
            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            self.total_peak_allocated_mem = max(self.total_peak_allocated_mem, allocated)
            self.total_peak_reserved_mem = max(self.total_peak_reserved_mem, reserved)
        acc = self._accums.setdefault(name, _Accum())
        acc.time += time.perf_counter() - started
        acc.peak_allocated_mem = max(acc.peak_allocated_mem, allocated)
        acc.peak_reserved_mem = max(acc.peak_reserved_mem, reserved)

    def _fold_peak(self) -> None:
        device = self._device()
        self.total_peak_allocated_mem = max(self.total_peak_allocated_mem, int(torch.cuda.max_memory_allocated(device)))
        self.total_peak_reserved_mem = max(self.total_peak_reserved_mem, int(torch.cuda.max_memory_reserved(device)))

    def _snapshot(self) -> ProfileReport:
        modules = {
            name: ModuleProfile(
                time=acc.time,
                peak_allocated_mem=acc.peak_allocated_mem,
                peak_reserved_mem=acc.peak_reserved_mem,
            )
            for name, acc in self._accums.items()
        }
        return ProfileReport(
            modules=modules,
            total_time=self.total_time,
            total_peak_allocated_mem=self.total_peak_allocated_mem,
            total_peak_reserved_mem=self.total_peak_reserved_mem,
        )

    def _device(self) -> torch.device:
        return torch.device(self._pipe.device)

    def _use_cuda(self) -> bool:
        return self._device().type == "cuda" and torch.cuda.is_available()


def _empty_report() -> ProfileReport:
    return ProfileReport(
        modules={},
        total_time=0.0,
        total_peak_allocated_mem=0,
        total_peak_reserved_mem=0,
    )


def _log_report(report: ProfileReport, path: Path) -> None:
    logger.info(
        "profile mode=time_mem_module total_time=%.4fs total_peak_allocated_mem=%d total_peak_reserved_mem=%d path=%s",
        report.total_time,
        report.total_peak_allocated_mem,
        report.total_peak_reserved_mem,
        path,
    )
    for name, module in report.modules.items():
        logger.info(
            "profile module=%s time=%.4fs peak_allocated_mem=%d peak_reserved_mem=%d",
            name,
            module.time,
            module.peak_allocated_mem,
            module.peak_reserved_mem,
        )


def diff_launch_overrides(
    cfg: Any,
    *,
    attention_engine: str | None,
    cache_mode: str | None,
    offload_strategy: str | None,
) -> dict[str, Any]:
    """Launch arguments that differ from the loaded config. Unset arguments are omitted."""
    overrides: dict[str, Any] = {}
    if attention_engine is not None and attention_engine != cfg.attention.engine:
        overrides["attention_engine"] = attention_engine
    if cache_mode is not None and cache_mode != cfg.cache.mode:
        overrides["cache_mode"] = cache_mode
    if offload_strategy is not None and offload_strategy != cfg.offload.strategy:
        overrides["offload_strategy"] = offload_strategy
    return overrides


def _call_overrides(pipe: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    """``__call__`` arguments that differ from the pipeline built from the config."""
    overrides: dict[str, Any] = {}
    compared = (
        ("height", "height"),
        ("width", "width"),
        ("latent_frames", "latent_frames"),
        ("num_steps", "num_steps"),
        ("guidance_weight", "guidance_weight"),
        ("scheduler_scale", "scheduler_scale"),
        ("visual_cond_scheme", "visual_cond_scheme"),
    )
    for key, attr in compared:
        value = raw.get(key)
        if value is not None and value != getattr(pipe, attr, None):
            overrides[key] = value
    if raw.get("time_length") is not None:
        overrides["time_length"] = raw["time_length"]
    if raw.get("sample_audio") is not None:
        overrides["sample_audio"] = raw["sample_audio"]
    image = _image_marker(raw.get("image"))
    if image is not None:
        overrides["image"] = image
    return overrides


def _image_marker(image: Any) -> Any:
    if image is None:
        return None
    if isinstance(image, str):
        return image
    if isinstance(image, (list, tuple)):
        return [_image_marker(item) for item in image]
    return True


def _effective_run(pipe: Any, raw: Mapping[str, Any], seed: int | None) -> dict[str, Any]:
    """Knobs of this call, including values taken from the config."""

    def chosen(key: str, attr: str) -> Any:
        value = raw.get(key)
        return getattr(pipe, attr, None) if value is None else value

    config = getattr(pipe, "config", None)
    run: dict[str, Any] = {
        "checkpoint": None if config is None else config.checkpoint,
        "device": str(getattr(pipe, "device", "")),
        "attention_engine": getattr(pipe, "attention_engine", None),
        "cache_mode": getattr(pipe, "cache_mode", None),
        "offload_strategy": getattr(pipe, "offload_strategy", None),
        "height": chosen("height", "height"),
        "width": chosen("width", "width"),
        "latent_frames": chosen("latent_frames", "latent_frames"),
        "num_steps": chosen("num_steps", "num_steps"),
        "guidance_weight": chosen("guidance_weight", "guidance_weight"),
        "scheduler_scale": chosen("scheduler_scale", "scheduler_scale"),
        "seed": seed,
    }
    beautifier = getattr(getattr(pipe, "beautifier", None), "name", None)
    if isinstance(beautifier, str):
        run["beautifier"] = beautifier
    prompt = raw.get("prompt")
    if isinstance(prompt, (str, list)):
        run["prompt"] = prompt
    if raw.get("time_length") is not None:
        run["time_length"] = raw["time_length"]
    if raw.get("sample_audio") is not None:
        run["sample_audio"] = raw["sample_audio"]
    return run


def _environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "hostname": platform.node(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = props.name
        info["gpu_total_mem"] = props.total_memory
        info["compute_capability"] = f"{props.major}.{props.minor}"
    return info


def _profile_output_dir() -> Path:
    """Directory for one ``time_mem_module`` JSON.

    A run sets ``KANDINSKY_PROFILE_DIR`` to its ``profiles/`` folder. Otherwise
    reports stay under ``$KANDINSKY_HOME/profile/time_mem_module``.
    """
    if os.environ.get("KANDINSKY_PROFILE_DIR"):
        return profile_dir()
    return profile_dir() / "time_mem_module"


def _write_time_mem_module(
    pipe: Any,
    report: ProfileReport,
    seed: int | None,
    raw_call: Mapping[str, Any],
) -> Path:
    config_path = getattr(pipe, "config_path", None)
    overrides = dict(getattr(pipe, "launch_overrides", {}) or {})
    overrides.update(_call_overrides(pipe, raw_call))
    payload = {
        "mode": "time_mem_module",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": None if not config_path else Path(config_path).name,
        "config_path": None if not config_path else str(config_path),
        "overrides": overrides,
        "run": _effective_run(pipe, raw_call, seed),
        "measurements": {
            "total_time": report.total_time,
            "total_peak_allocated_mem": report.total_peak_allocated_mem,
            "total_peak_reserved_mem": report.total_peak_reserved_mem,
            "modules": {
                name: {
                    "time": module.time,
                    "peak_allocated_mem": module.peak_allocated_mem,
                    "peak_reserved_mem": module.peak_reserved_mem,
                }
                for name, module in report.modules.items()
            },
        },
        "environment": _environment(torch.device(getattr(pipe, "device", "cpu"))),
    }
    directory = _profile_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    stem = "run" if not config_path else Path(config_path).stem
    target = directory / f"{stem}_{stamp}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _component_targets(pipe: object) -> list[tuple[Any, str, str]]:
    """Public entry points, outermost call only, so measured spans do not nest."""
    rows: list[tuple[Any, str, str]] = []
    text = getattr(pipe, "text_embedder", None)
    if text is not None:
        if getattr(text, "qwen", None) is not None:
            rows.append((text.qwen, "forward", "text_encoder"))
        if getattr(text, "clip", None) is not None:
            rows.append((text.clip, "forward", "text_encoder_2"))
    dit = getattr(pipe, "dit", None)
    if dit is not None:
        rows.append((dit, "forward", "dit"))
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "encode"):
            rows.append((vae, "encode", "vae_encoder"))
        if hasattr(vae, "decode"):
            rows.append((vae, "decode", "vae_decoder"))
    audio = getattr(pipe, "audio_vae", None)
    if audio is not None:
        if hasattr(audio, "encode_audio"):
            rows.append((audio, "encode_audio", "audio_vae_encoder"))
        if hasattr(audio, "decode"):
            rows.append((audio, "decode", "audio_vae_decoder"))
    vocoder = getattr(pipe, "vocoder", None)
    if vocoder is not None and hasattr(vocoder, "forward"):
        rows.append((vocoder, "forward", "vocoder"))
    return rows
