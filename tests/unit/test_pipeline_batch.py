from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kandinsky.core.algo.denoise_loop import _build_video_input
from kandinsky.core.ports import restore_tail_references
from kandinsky.pipeline import pipeline as pipeline_module
from kandinsky.pipeline.pipeline import Kandinsky6Pipeline

BATCH_SIZE = 2


class _TextEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.processor = object()
        self.qwen = object()
        self.max_length = 8

    def encode(self, texts: list[str]):
        self.calls.append(texts)
        batch_size = len(texts)
        return (
            {
                "text_embeds": torch.zeros(batch_size, 2, 4),
                "pooled_embed": torch.zeros(batch_size, 3),
            },
            torch.arange(batch_size + 1, dtype=torch.int32) * 2,
            torch.ones(batch_size, 2, dtype=torch.bool),
        )


def _pipeline(
    *,
    visual_cond: bool = False,
    tail_condition: bool = False,
    multimodal: bool = False,
) -> Kandinsky6Pipeline:
    dit = SimpleNamespace(
        in_visual_dim=2,
        in_audio_dim=3,
        patch_size=(1, 1, 1),
        visual_cond=visual_cond,
        is_multimodal=multimodal,
        visual_token_type_num_embeddings=2 if tail_condition else 0,
    )
    return Kandinsky6Pipeline(
        dit=dit,
        text_embedder=_TextEmbedder(),
        vae=object(),
        audio_vae=SimpleNamespace(downsample_factor=1024) if multimodal else None,
        vocoder=object() if multimodal else None,
        device=torch.device("cpu"),
        height=8,
        width=8,
        latent_frames=3,
    )


def _patch_batch_execution(monkeypatch, *, tail_condition: bool = False) -> dict[str, object]:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        Kandinsky6Pipeline,
        "_materialize_ropes",
        lambda *_args, **_kwargs: (torch.zeros(3, 1, 1), None, torch.zeros(2), None),
    )

    def denoise(**kwargs):
        bundle = kwargs["bundle"]
        seen["video_shape"] = tuple(bundle.video.shape)
        seen["text_shape"] = tuple(kwargs["text_embeds"]["text_embeds"].shape)
        seen["first_frames"] = kwargs["first_frames"]
        return bundle

    def postprocess(bundle, _vae, bs):
        seen["decode_batch_size"] = bs
        return torch.zeros((bs, 3, bundle.video.shape[1], 8, 8), dtype=torch.uint8)

    monkeypatch.setattr(pipeline_module, "denoise_loop", denoise)
    monkeypatch.setattr(pipeline_module, "postprocess_video", postprocess)
    if tail_condition:
        monkeypatch.setattr(
            pipeline_module,
            "encode_i2va_first_frame",
            lambda image, *_args, **_kwargs: (
                torch.full((1, 1, 1, 2), 1 if image == "first" else 2),
                8,
                8,
            ),
        )
    return seen


def test_native_pipeline_batches_tail_conditioned_images(monkeypatch) -> None:
    seen = _patch_batch_execution(monkeypatch, tail_condition=True)
    pipe = _pipeline(visual_cond=True, tail_condition=True)

    result = pipe(
        text=["first", "second"],
        image=["first", "second"],
        sample_audio=False,
        visual_cond_scheme="tail_cond_first_frame",
    )

    assert seen["video_shape"] == (BATCH_SIZE, 4, 1, 1, 2)
    assert seen["first_frames"].shape == (BATCH_SIZE, 1, 1, 2)
    assert result.frames.shape == (BATCH_SIZE, 3, 3, 8, 8)


@pytest.mark.parametrize('images', [
    ['first', ['first', 'last']],
    [[], []],
    [['a', 'b', 'c']],
    [[None]],
])
def test_reference_batch_rejects_invalid_inputs(images):
    with pytest.raises(ValueError):
        pipeline_module._reference_batch(images)


def test_native_pipeline_batches_two_references(monkeypatch):
    seen = _patch_batch_execution(monkeypatch, tail_condition=True)
    pipe = _pipeline(visual_cond=True, tail_condition=True)

    def denoise(**kwargs):
        seen['types'] = kwargs['visual_token_type_ids'].clone()
        seen['rope'] = kwargs['visual_rope'].clone()
        seen['references'] = kwargs['first_frames'].clone()
        assert kwargs['bundle'].video.shape == (2, 5, 1, 1, 2)
        return kwargs['bundle']

    monkeypatch.setattr(pipeline_module, 'denoise_loop', denoise)
    monkeypatch.setattr(
        Kandinsky6Pipeline, '_materialize_ropes',
        lambda *_args, **_kwargs: (torch.arange(3).reshape(3, 1, 1), None, torch.zeros(2), None),
    )
    result = pipe(
        text=['one', 'two'], image=[['first', 'last'], ['first', 'last']],
        sample_audio=False,
    )
    assert seen['types'].tolist() == [[0, 0, 0, 1, 1]] * 2
    assert seen['rope'].flatten().tolist() == [0, 1, 2, 0, 2]
    assert seen['references'].shape == (2, 2, 1, 1, 2)
    assert result.frames.shape == (2, 3, 3, 8, 8)


@pytest.mark.parametrize('packed', [False, True])
def test_two_reference_channel_mask_and_restore(packed: bool) -> None:
    video = torch.zeros(2, 5, 1, 1, 2)
    references = torch.tensor([1., 2., 3., 4.]).reshape(2, 2, 1, 1, 1).expand(-1, -1, -1, -1, 2)
    cu = torch.tensor([0, 5, 10], dtype=torch.int32)
    if packed:
        video = video.flatten(0, 1)
    model_input = _build_video_input(video, True, references, cu, 'tail_cond_first_frame')
    mask = model_input[..., -1].reshape(2, 5)
    assert mask.tolist() == [[0, 0, 0, 1, 1]] * 2
    noise_value = 10
    video.add_(noise_value)
    restore_tail_references(video, references, cu)
    batched = video.reshape(2, 5, 1, 1, 2)
    assert torch.equal(batched[:, -2:], references)
    assert (batched[:, :3] == noise_value).all()
