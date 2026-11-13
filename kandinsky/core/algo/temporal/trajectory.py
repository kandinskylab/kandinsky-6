"""Fixed and adaptive temporal trajectories for PiFlow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..prepare_ropes import compute_visual_rope
from .cross_rope import global_cross_ropes
from .spectral import _contract_sigma, _temporal_contract, _temporal_expand, temporal_high_ratio

LOW_FRAMES = 16
FULL_FRAMES = 31
MAX_LOW_CALLS = 8


@dataclass(frozen=True)
class VideoStep:
    video: Tensor
    offsets: Tensor
    rope: Tensor | None
    cross_rope: tuple[Tensor, Tensor] | None
    sigma: Tensor
    tau_src: Tensor
    tau_dst: Tensor
    segment: Tensor | None


class TemporalTrajectory:
    """Track the video grid and aligned PiFlow times for one generation."""

    def __init__(
        self,
        video: Tensor,
        audio: Tensor,
        dit: nn.Module,
        scale_factor: tuple[float, float, float],
        *,
        switch_after_calls: int | None,
        threshold: float | None,
    ):
        if video.ndim != 5 or video.shape[:2] != (1, FULL_FRAMES) or audio.ndim != 3 or audio.shape[0] != 1:
            raise ValueError("Temporal requires one dense video with 31 latents and one dense audio sequence")
        if switch_after_calls is None and threshold is None:
            raise ValueError("Adaptive Temporal requires an energy threshold")
        self.master = video[0]
        _, patch_h, patch_w = dit.patch_size
        grid = (LOW_FRAMES, video.shape[2] // patch_h, video.shape[3] // patch_w)
        self.local_rope = compute_visual_rope(
            dit.visual_rope, grid, scale_factor, device=video.device,
        )
        self.cross_rope = global_cross_ropes(
            dit, grid, FULL_FRAMES, audio.shape[1], scale_factor,
        )
        self.low_offsets = torch.tensor([0, LOW_FRAMES], dtype=torch.int32, device=video.device)
        self.ratio = math.sqrt(FULL_FRAMES / LOW_FRAMES)
        self.chosen = switch_after_calls
        self.threshold = threshold
        self.energy = torch.zeros((), dtype=torch.float32, device=video.device)
        self.reason = "fixed" if switch_after_calls is not None else None

    def before_step(
        self,
        step: int,
        video: Tensor,
        full_offsets: Tensor,
        full_rope: Tensor | None,
        q_src: Tensor,
        q_dst: Tensor,
        tau_src: Tensor,
        tau_dst: Tensor,
        shift: float,
    ) -> VideoStep:
        if step == 0:
            video, _ = _temporal_contract(self.master, LOW_FRAMES, 1.0)
            video = video.unsqueeze(0)
        if self.chosen is not None and step >= self.chosen:
            if step == self.chosen:
                low_sigma = float(_contract_sigma(q_src, self.ratio).item())
                expanded, aligned = _temporal_expand(video[0], FULL_FRAMES, low_sigma, self.master)
                if not math.isclose(aligned, float(q_src.item()), rel_tol=2e-6, abs_tol=2e-6):
                    raise RuntimeError("Temporal expansion disagrees with the full-grid PiFlow schedule")
                video = expanded.unsqueeze(0)
            return VideoStep(video, full_offsets, full_rope, None, q_src, tau_src, tau_dst, None)

        video_sigma = _contract_sigma(q_src, self.ratio)
        next_sigma = _contract_sigma(q_dst, self.ratio)
        # Convert aligned video noise levels back to PiFlow's unshifted time.
        video_tau = video_sigma / (shift - (shift - 1) * video_sigma)
        next_tau = next_sigma / (shift - (shift - 1) * next_sigma)
        return VideoStep(
            video, self.low_offsets, self.local_rope, self.cross_rope,
            video_sigma, video_tau, next_tau, video_tau - next_tau,
        )

    def observe(self, step: int, video: Tensor, velocity: Tensor, video_sigma: Tensor, q_delta: Tensor) -> None:
        if self.chosen is not None or self.threshold is None:
            return
        clean = video[0].float() - float(video_sigma.item()) * velocity[0].float()
        # Integrate the clean-latent high-band ratio over the full-grid noise schedule.
        self.energy = self.energy + temporal_high_ratio(clean) * float(q_delta.item())
        energy = float(self.energy.item())
        if not math.isfinite(energy):
            raise RuntimeError("Non-finite A2 temporal signal")
        if energy >= self.threshold:
            self.chosen, self.reason = step + 1, "threshold"
        elif step + 1 == MAX_LOW_CALLS:
            self.chosen, self.reason = MAX_LOW_CALLS, "fallback_max_calls"

    def summary(self) -> dict[str, int | str | None]:
        return {"switch_after_calls": self.chosen, "reason": self.reason}
