"""Adaptive cache for paired audio and video block residuals."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from pydantic import Field
from torch import Tensor

from ..base import ConfigModel


def _sample(value: Tensor) -> Tensor:
    stride = max(1, math.ceil(value.numel() / 4096))
    return value.detach().reshape(-1)[::stride].to(dtype=torch.float32, copy=True)


def _relative_change(current: Tensor, previous: Tensor) -> float:
    change = (current - previous).abs().mean()
    return float((change / previous.abs().mean().clamp_min(1e-6)).item())


class AdaptiveBlockConfig(ConfigModel):
    input_threshold: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    residual_threshold: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    payload_gib: float = Field(default=32.0, ge=0, allow_inf_nan=False)


class AdaptiveBlockState:
    """Block 0 is always computed. Blocks [1, 60) share one paired BF16 residual."""

    def __init__(self, config: AdaptiveBlockConfig, num_steps: int):
        self.config = config
        self.num_steps = num_steps
        self.step = -1
        self.anchor_step = -1
        self.payload_limit = int(config.payload_gib * 1024**3)
        self.peak_payload_bytes = 0
        self.hits: list[int] = []
        self.invalidations: list[int] = []
        self.frames = None
        self.disabled_reason = None
        self.clear()

    def clear(self) -> None:
        self.residual = self.input_sketch = self.residual_sketch = self.ratio = None
        self.anchor_step = -1
        self.signature = None

    def next_step(self, video: Tensor) -> bool:
        self.step += 1
        frames = video.shape[1]
        if self.frames is not None and self.frames != frames:
            self.invalidations.append(self.step)
            self.clear()
        self.frames = frames
        # Only full-grid calls can supply anchors. The final call always runs in full.
        if self.disabled_reason is not None or frames != 31 or self.step >= self.num_steps - 1:
            self.clear()
            return False
        return True

    def apply(
        self, video: Tensor, audio: Tensor, run_tail: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        values = (video, audio)
        signature = tuple((value.shape, value.dtype, value.device) for value in values)
        if self.signature is not None and self.signature != signature:
            self.invalidations.append(self.step)
            self.clear()
        self.signature = signature
        inputs = tuple(_sample(value) for value in values)
        eligible = 2 <= self.step < self.num_steps - 1 and self.anchor_step == self.step - 1
        if eligible and self.ratio is not None:
            drift = tuple(_relative_change(now, before) for now, before in zip(inputs, self.input_sketch, strict=True))
            estimated = tuple(ratio * delta for ratio, delta in zip(self.ratio, drift, strict=True))
            if all(math.isfinite(delta) and math.isfinite(error)
                   and delta <= self.config.input_threshold
                   and error <= self.config.residual_threshold
                   for delta, error in zip(drift, estimated, strict=True)):
                self.hits.append(self.step)
                video_residual, audio_residual = self.residual
                return video + video_residual, audio + audio_residual

        output = run_tail(video, audio)
        size = sum(value.numel() * value.element_size() for value in output)
        if any(value.dtype != torch.bfloat16 for value in (*values, *output)):
            self.disabled_reason = "non_bf16_payload"
        elif size > self.payload_limit:
            self.disabled_reason = "payload_budget"
        if self.disabled_reason is not None:
            self.clear()
            return output

        residual = tuple(after - before for before, after in zip(values, output, strict=True))
        output_sketch = tuple(_sample(value) for value in residual)
        if self.input_sketch is not None:
            deltas = tuple(_relative_change(now, before) for now, before in zip(inputs, self.input_sketch, strict=True))
            errors = tuple(_relative_change(now, before)
                           for now, before in zip(output_sketch, self.residual_sketch, strict=True))
            self.ratio = tuple(error / max(delta, 1e-6) for error, delta in zip(errors, deltas, strict=True))
        self.input_sketch, self.residual_sketch = inputs, output_sketch
        self.residual = residual
        self.anchor_step = self.step
        self.peak_payload_bytes = max(self.peak_payload_bytes, size)
        return output

    def summary(self) -> dict[str, int | list[int] | str | None]:
        return {
            "hits": len(self.hits),
            "hit_step_indices": self.hits,
            "skipped_blocks": 59 * len(self.hits),
            "invalidation_step_indices": self.invalidations,
            "peak_payload_bytes": self.peak_payload_bytes,
            "disabled_reason": self.disabled_reason,
        }
