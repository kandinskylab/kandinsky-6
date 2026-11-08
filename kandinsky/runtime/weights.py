"""Published Diffusers checkpoints and loading them into a DiT.

The catalog lists the Kandinsky 6.0 collection
https://huggingface.co/collections/kandinskylab/kandinsky-60-diffusers.
``weight_bytes`` is the sum of ``.safetensors`` files in the repo: DiT, both
text encoders, video VAE, audio VAE, and vocoder. Tokenizer text is not counted.

A snapshot is stored under ``$KANDINSKY_HOME/weights`` (default
``~/.cache/kandinsky/weights``), in the Hugging Face hub layout.
``cache_dir`` puts that snapshot on another disk.

Snapshots are taken from ``main``. That branch stores the renamed DiT keys.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file
from torch import nn

from .home import weights_dir

logger = logging.getLogger("kandinsky")

# Catalog sizes count safetensors only. Leave room for configs, tokenizer files, and cache metadata.
_REPO_SLACK_BYTES = 64 << 20

# A partial snapshot_download still leaves a snapshot directory. These files are what the pipeline loads.
_COMPONENT_FILES = (
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder_2/model.safetensors",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "vocoder/diffusion_pytorch_model.safetensors",
)

# Collection kandinskylab/kandinsky-60-diffusers. Renamed DiT keys are on main.
CHECKPOINT_REVISION = "main"

# Safetensors bytes on that collection, 2026-10-02. main replaces the
# transformer file in place, so the download size stays the same.
_PRO = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
_PRO_DISTILL = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
_PRO_PRETRAIN = "kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers"
_LITE = "kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers"
_LITE_DISTILL = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
_LITE_PRETRAIN = "kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One Hub repo and the on-disk size of its weight files."""

    repo_id: str
    weight_bytes: int

    @property
    def weight_gb(self) -> float:
        return self.weight_bytes / 1e9


CHECKPOINTS: dict[str, Checkpoint] = {
    "pro": Checkpoint(_PRO, 89_997_268_172),  # 90.00 GB
    "pro-distill": Checkpoint(_PRO_DISTILL, 80_607_441_652),  # 80.61 GB
    "pro-pretrain": Checkpoint(_PRO_PRETRAIN, 89_997_268_172),  # 90.00 GB
    "lite": Checkpoint(_LITE, 27_738_988_100),  # 27.74 GB
    "lite-distill": Checkpoint(_LITE_DISTILL, 26_644_058_740),  # 26.64 GB
    "lite-pretrain": Checkpoint(_LITE_PRETRAIN, 27_738_988_108),  # 27.74 GB
}


def get_checkpoint(name: str) -> Checkpoint:
    try:
        return CHECKPOINTS[name]
    except KeyError:
        known = ", ".join(CHECKPOINTS)
        raise KeyError(f"unknown checkpoint {name!r}; known: {known}") from None


def _cache_root(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else weights_dir()


def _cached_bytes(cache_root: Path, repo_id: str) -> int:
    """Bytes already stored for this repo: finished blobs and ``*.incomplete`` resumes."""
    blobs = cache_root / ("models--" + repo_id.replace("/", "--")) / "blobs"
    if not blobs.is_dir():
        return 0
    return sum(entry.stat().st_size for entry in blobs.iterdir() if entry.is_file())


def _disk_free(path: Path) -> tuple[Path, int]:
    current = path
    while True:
        try:
            return current, shutil.disk_usage(current).free
        except OSError:
            parent = current.parent
            if parent == current:
                raise
            current = parent


def _require_download_space(spec: Checkpoint, cache_root: str | Path) -> None:
    root = Path(cache_root)
    needed = spec.weight_bytes + _REPO_SLACK_BYTES - _cached_bytes(root, spec.repo_id)
    if needed <= 0:
        return
    volume, free = _disk_free(root)
    if free < needed:
        raise OSError(
            f"not enough disk space to download {spec.repo_id}: "
            f"need {needed / 1e9:.2f} GB more, {free / 1e9:.2f} GB free on {volume}"
        )


def _text_encoder_ready(snapshot: Path) -> bool:
    root = snapshot / "text_encoder"
    if (root / "model.safetensors").is_file():
        return True
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return False
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    return bool(shards) and all((root / shard).is_file() for shard in shards)


def _snapshot_ready(snapshot: Path) -> bool:
    """True when every weight file the pipeline opens is in the snapshot."""
    return all((snapshot / name).is_file() for name in _COMPONENT_FILES) and _text_encoder_ready(snapshot)


def _local_snapshot(kwargs: dict[str, str]) -> Path | None:
    try:
        return Path(snapshot_download(**kwargs, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def _ref_snapshot(cache_root: Path, repo_id: str, revision: str | None) -> Path | None:
    """Snapshot directory named by ``refs/<revision>``, ignoring other Hub files.

    ``snapshot_download(local_files_only=True)`` raises when any repo file is
    absent, including logos the pipeline never opens. A revision whose weight
    files are already on disk is usable without a network call.
    """
    repo = cache_root / ("models--" + repo_id.replace("/", "--"))
    ref = repo / "refs" / ("main" if revision is None else revision)
    if not ref.is_file():
        return None
    snapshot = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
    return snapshot if snapshot.is_dir() else None


def _ensure_snapshot(
    spec: Checkpoint,
    cache_root: Path,
    *,
    revision: str | None,
    ready,
) -> Path:
    kwargs: dict[str, str] = {"repo_id": spec.repo_id, "cache_dir": str(cache_root)}
    if revision is not None:
        kwargs["revision"] = revision
    path = _local_snapshot(kwargs)
    if path is None:
        path = _ref_snapshot(cache_root, spec.repo_id, revision)
    if path is not None and ready(path):
        logger.info("%s is already on disk", spec.repo_id)
        return path
    _require_download_space(spec, cache_root)
    logger.info("downloading %s (%.2f GB of weights)", spec.repo_id, spec.weight_gb)
    return Path(snapshot_download(**kwargs))


def ensure_checkpoint(name: str, *, cache_dir: str | Path | None = None) -> Path:
    """Return the local snapshot, downloading the repo when it is absent or incomplete.

    ``local_files_only`` returns a snapshot directory even when an earlier partial
    download left weight files out. A snapshot counts only when the DiT, both text
    encoders, both VAEs, and the vocoder are present. The remaining files are
    downloaded only when the cache volume has room for them.
    """
    spec = get_checkpoint(name)
    path = _ensure_snapshot(
        spec,
        _cache_root(cache_dir),
        revision=CHECKPOINT_REVISION,
        ready=_snapshot_ready,
    )
    logger.info("%s → %s", name, path)
    return path


def materialize_meta_buffers(module: nn.Module, device: torch.device) -> None:
    """Allocate non-persistent buffers left on ``meta`` after ``load_state_dict(assign=True)``.

    RoPE / TimeEmbeddings register ``persistent=False`` tables that never appear in
    checkpoints; after a meta-device construct they stay meta until recomputed.
    """
    for mod in module.modules():
        touched = False
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                mod._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)
                touched = True
        if touched and hasattr(mod, "reset_parameters"):
            mod.reset_parameters()


def load_bf16_checkpoint(dit: nn.Module, path: str) -> None:
    """Read a local safetensors file and assign it into ``dit``."""
    state_dict = load_file(path, device="cpu")
    state_dict = {
        key: value.to(torch.bfloat16) if value.dtype == torch.float32 else value for key, value in state_dict.items()
    }
    missing, unexpected = dit.load_state_dict(state_dict, strict=True, assign=True)
    if missing:
        logger.warning("missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("unexpected keys (%d): %s", len(unexpected), unexpected[:5])
