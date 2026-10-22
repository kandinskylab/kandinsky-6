"""File decoding and resampling for native audio/video conditioning inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..core.algo.mux import _require_av, extract_audio_from_video


def load_audio_condition(
    path: str | Path,
    *,
    sample_rate: int,
    num_samples: int,
    fps: float,
) -> NDArray[np.float32]:
    """Read mono audio, trim to duration, and tolerate one frame of encoder padding."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    waveform = extract_audio_from_video(path, sample_rate)
    if waveform is None:
        raise ValueError(f"No audio stream in {path}")
    if len(waveform) + int(np.ceil(sample_rate / fps)) < num_samples:
        raise ValueError(f"Audio is too short: need {num_samples / sample_rate:.3f}s in {path}")
    return np.pad(waveform[:num_samples], (0, max(0, num_samples - len(waveform))))


def load_video_condition(
    path: str | Path,
    *,
    height: int,
    width: int,
    num_frames: int,
    fps: float,
) -> Tensor:
    """Read RGB uint8 frames as (1, 3, T, H, W), sampled at the requested FPS."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    av = _require_av()
    frames: list[NDArray[np.uint8]] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or fps)
        start = None
        last_time = 0.0
        last_image = None
        for index, frame in enumerate(container.decode(video=0)):
            timestamp = frame.time if frame.time is not None else index / source_fps
            if start is None:
                start = timestamp
            last_time = timestamp - start
            if len(frames) / fps <= last_time + 1e-6:
                last_image = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                while len(frames) < num_frames and len(frames) / fps <= last_time + 1e-6:
                    frames.append(last_image)
            if len(frames) == num_frames:
                break
        if len(frames) < num_frames:
            if last_image is None or (num_frames - 1) / fps >= last_time + 1 / source_fps + 1e-6:
                raise ValueError(f"Video is too short: need {num_frames} frames at {fps} FPS in {path}")
            frames.extend([last_image] * (num_frames - len(frames)))
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
