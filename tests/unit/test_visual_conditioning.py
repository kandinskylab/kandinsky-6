"""Framework-independent reference packing and RoPE contract."""

from __future__ import annotations

import pytest
import torch

from kandinsky.core.algo.prepare_latents import append_i2va_tail_condition
from kandinsky.core.ports import (
    append_reference_rope,
    append_visual_references,
    build_tail_condition,
    compute_reference_rope,
    restore_tail_references,
)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_reference_layout_mask_and_restoration(packed: bool, count: int) -> None:
    video = torch.randn(2, 3, 1, 1, 2)
    references = torch.randn(2, count, 1, 1, 2)
    source = video.flatten(0, 1) if packed else video
    result, roles, generated = append_visual_references(source, references, 2, 3)
    assert torch.equal(result[generated], video.flatten(0, 1))
    assert (roles[~generated] == 1).all()
    offsets = torch.tensor([0, 3 + count, 2 * (3 + count)])
    model_input = build_tail_condition(result, references, offsets)
    assert (model_input[..., -1][generated] == 0).all()
    assert (model_input[..., -1][~generated] == 1).all()
    result.zero_()
    restore_tail_references(result, references, offsets)
    assert torch.equal(result[~generated], references.flatten(0, 1))
    if count == 1:
        legacy = append_i2va_tail_condition(source, references[:, 0], 2, 3)
        assert torch.equal(legacy[1], roles)
        assert torch.equal(legacy[2], generated)


@pytest.mark.parametrize("count, positions", [(1, [0, 1, 2, 0]), (2, [0, 1, 2, 0, 2])])
def test_reference_rope_uses_boundary_positions(count: int, positions: list[int]) -> None:
    rope = torch.arange(3).reshape(3, 1, 1, 1)
    assert append_reference_rope(rope, count).flatten().tolist() == positions


def test_invalid_reference_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="one or two"):
        append_visual_references(torch.zeros(1, 3, 1, 1, 2), torch.zeros(1, 3, 1, 1, 2), 1, 3)
    with pytest.raises(ValueError, match="one or two"):
        append_reference_rope(torch.zeros(3, 1, 1, 1), 3)


@pytest.mark.parametrize("count, expected", [(0, [0, 1, 2, 3, 4]), (2, [0, 1, 2, 0, 2])])
def test_recomputed_rope_preserves_reference_positions(count: int, expected: list[int]) -> None:
    class TemporalRope(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("anchor", torch.zeros(1))

        def forward(
            self,
            shape: tuple[int, int, int],
            positions: list[torch.Tensor],
            _scale: tuple[float, float, float],
        ) -> torch.Tensor:
            return positions[0][:, None, None].expand(shape)

    actual = compute_reference_rope(TemporalRope(), (5, 1, 1), (1.0, 1.0, 1.0), count)
    assert actual.flatten().tolist() == expected
