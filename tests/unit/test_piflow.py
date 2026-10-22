from __future__ import annotations

import torch
from torch import Tensor, nn

from kandinsky.core.algo.piflow_sampler import piflow_denoise_loop
from kandinsky.core.types import LatentBundle, TextEmbeds


class _TinyPiFlowDiT(nn.Module):
    is_multimodal = True
    visual_cond = False

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        x_video: Tensor,
        x_audio: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        del kwargs
        self.calls += 1
        return (
            torch.zeros_like(x_video),
            torch.zeros_like(x_audio),
        )


def test_piflow_restores_both_tail_references_at_every_step() -> None:
    expected_calls = 3
    references = torch.tensor([[[[[10.0]]], [[[20.0]]]]])
    video = torch.zeros(1, 5, 1, 1, 1)
    video[:, -2:] = references

    class CheckingDiT(_TinyPiFlowDiT):
        def forward(self, x_video, x_audio, **kwargs):
            assert torch.equal(x_video[:, -2:], references)
            return super().forward(x_video, x_audio, **kwargs)

    dit = CheckingDiT()
    result = piflow_denoise_loop(
        bundle=LatentBundle(
            video=video,
            audio=torch.zeros(1, 2, 1),
            video_cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
            audio_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        ),
        dit=dit,
        text_embeds=TextEmbeds(text_embeds=torch.zeros(1, 1, 1), pooled_embed=torch.zeros(1, 1)),
        visual_rope=None,
        audio_rope=None,
        text_rope=[],
        num_steps=expected_calls,
        scheduler_scale=1.0,
        first_frames=references,
        visual_cond_scheme="tail_cond_first_frame",
        num_policy_substeps=2,
    )
    assert dit.calls == expected_calls
    assert torch.equal(result.video[:, -2:], references)
