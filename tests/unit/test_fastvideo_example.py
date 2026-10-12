from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "fastvideo_ti2va.py"
_spec = importlib.util.spec_from_file_location("fastvideo_ti2va", _EXAMPLE)
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def _write_snapshot(root: Path, commit: str) -> Path:
    snapshot = root / "snapshots" / commit
    transformer = snapshot / "transformer"
    transformer.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}\n", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    return snapshot


def test_ready_snapshot_prefers_refs_main(tmp_path: Path) -> None:
    repo = tmp_path / "weights" / "models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers"
    older = _write_snapshot(repo, "older")
    current = _write_snapshot(repo, "current")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("current\n", encoding="utf-8")

    found = example.ready_snapshot("kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", tmp_path)

    assert found == current
    assert found != older


def test_model_for_fastvideo_links_a_distilled_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    repo = tmp_path / "weights" / "models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers"
    snapshot = _write_snapshot(repo, "abc")
    link_dir = tmp_path / "outputs" / "fastvideo_run"

    path = Path(example.model_for_fastvideo("kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", link_dir))

    assert path.name == "Kandinsky-6.0-Pro-distill-5s-Diffusers"
    assert path.resolve() == snapshot.resolve()
    assert example.os.environ["HF_HUB_CACHE"] == str(tmp_path / "weights")


def test_model_for_fastvideo_keeps_the_hub_id_when_the_cache_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))

    repo_id = example.model_for_fastvideo("kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", tmp_path / "run")

    assert repo_id == "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
