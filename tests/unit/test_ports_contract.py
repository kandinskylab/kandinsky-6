"""Tests for the stable public K6 algorithm port contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kandinsky.core import ports
from kandinsky.core.algo.conditioning import (
    encode_audio_condition,
    encode_video_condition,
)
from kandinsky.core.algo.denoise_loop import denoise_loop
from kandinsky.core.algo.flow_matching import flow_match_timesteps
from kandinsky.core.algo.guidance import apply_cfg
from kandinsky.core.algo.postprocess_audio import postprocess_audio
from kandinsky.core.algo.postprocess_video import postprocess_video
from kandinsky.core.algo.prepare_latents import (
    audio_latent_duration,
    prepare_audio_latents,
    prepare_video_latents,
)
from kandinsky.core.algo.prepare_ropes import compute_rope1d, compute_visual_rope
from kandinsky.core.types import LatentBundle, TextEmbeds


def test_public_contract_exports_the_canonical_types_and_functions() -> None:
    expected = {
        "LatentBundle": LatentBundle,
        "TextEmbeds": TextEmbeds,
        "prepare_video_latents": prepare_video_latents,
        "prepare_audio_latents": prepare_audio_latents,
        "audio_latent_duration": audio_latent_duration,
        "compute_rope1d": compute_rope1d,
        "compute_visual_rope": compute_visual_rope,
        "flow_match_timesteps": flow_match_timesteps,
        "apply_cfg": apply_cfg,
        "denoise_loop": denoise_loop,
        "encode_audio_condition": encode_audio_condition,
        "encode_video_condition": encode_video_condition,
        "postprocess_video": postprocess_video,
        "postprocess_audio": postprocess_audio,
    }

    assert set(ports.__all__) == set(expected)
    for name, implementation in expected.items():
        assert getattr(ports, name) is implementation


def test_contract_imports_without_optional_acceleration_modules() -> None:
    script = """
import builtins

blocked = {"flash_attn", "flash_attn_interface", "magi_compiler", "cuda_python"}
original_import = builtins.__import__

def import_without_optional_acceleration(name, *args, **kwargs):
    if name in blocked or any(name.startswith(module + ".") for module in blocked):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_optional_acceleration
import kandinsky.core.ports
"""
    repo = Path(__file__).resolve().parents[2]
    source = repo
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(path for path in (str(source), environment.get("PYTHONPATH")) if path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
