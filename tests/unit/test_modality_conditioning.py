from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kandinsky.core.algo.denoise_loop import denoise_loop
from kandinsky.core.algo.mux import mux_video_audio
from kandinsky.core.algo.piflow_sampler import piflow_denoise_loop
from kandinsky.core.ports import encode_audio_condition, encode_video_condition
from kandinsky.core.types import LatentBundle
from kandinsky.pipeline import pipeline as pipeline_module
from kandinsky.pipeline.media import load_audio_condition, load_video_condition
from kandinsky.pipeline.pipeline import Kandinsky6Pipeline


def _pipeline(multimodal: bool = True) -> Kandinsky6Pipeline:
    class TextEmbedder:
        def encode(self, texts: list[str]):
            return (
                {"text_embeds": torch.zeros(1, 2, 4), "pooled_embed": torch.zeros(1, 3)},
                torch.tensor([0, 2], dtype=torch.int32),
                torch.ones(1, 2, dtype=torch.bool),
            )

    return Kandinsky6Pipeline(
        dit=SimpleNamespace(
            in_visual_dim=2, in_audio_dim=3, patch_size=(1, 1, 1),
            visual_cond=False, is_multimodal=multimodal,
        ),
        text_embedder=TextEmbedder(),
        vae=object(),
        audio_vae=SimpleNamespace(downsample_factor=1024),
        vocoder=object(),
        device=torch.device("cpu"),
        height=8, width=8, latent_frames=3,
    )


def _patch_batch_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Kandinsky6Pipeline, "_materialize_ropes",
        lambda *a, **kw: (torch.zeros(3, 1, 1), None, torch.zeros(2), None),
    )
    monkeypatch.setattr(
        pipeline_module, "postprocess_video",
        lambda bundle, vae, bs: torch.zeros((bs, 3, bundle.video.shape[1], 8, 8), dtype=torch.uint8),
    )


@pytest.mark.parametrize("sampler", ["euler", "piflow"])
@pytest.mark.parametrize("condition", ["audio", "video"])
def test_clean_modality_stays_fixed_with_zero_timestep(sampler: str, condition: str) -> None:
    class DiT(torch.nn.Module):
        is_multimodal = True
        visual_cond = True

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x_video, x_audio, time, **kwargs):
            self.calls.append((x_video[..., :2].clone(), x_audio.clone(), time))
            return torch.ones_like(x_video[..., :2]), torch.ones_like(x_audio)

    video = torch.randn(1, 3, 1, 1, 2)
    audio = torch.randn(1, 5, 3)
    dit = DiT()
    kwargs = {
        "bundle": LatentBundle(video.clone(), audio.clone(), torch.tensor([0, 3]), torch.tensor([0, 5])),
        "dit": dit,
        "text_embeds": {"text_embeds": torch.zeros(1, 2, 4), "pooled_embed": torch.zeros(1, 3)},
        "visual_rope": None,
        "audio_rope": None,
        "text_rope": None,
        "num_steps": 2,
        "scheduler_scale": 5.0,
        "sample_audio": condition != "audio",
        "sample_video": condition != "video",
    }
    if sampler == "euler":
        result = denoise_loop(
            **kwargs, null_text_embeds=kwargs["text_embeds"], null_text_rope=None, guidance_weight=1.0
        )
    else:
        result = piflow_denoise_loop(**kwargs, num_policy_substeps=2)
    fixed_index = 0 if condition == "video" else 1
    original = video if condition == "video" else audio
    fixed = result.video if condition == "video" else result.audio
    assert torch.equal(fixed, original)
    generated = result.audio if condition == "video" else result.video
    assert not torch.equal(generated, audio if condition == "video" else video)
    for current_video, current_audio, times in dit.calls:
        assert torch.equal((current_video, current_audio)[fixed_index], original)
        assert (times[fixed_index] == 0).all()
        assert (times[1 - fixed_index] > 0).all()


@pytest.mark.parametrize("condition", ["audio", "video"])
def test_pipeline_routes_encoded_condition_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    _patch_batch_execution(monkeypatch)
    pipe = _pipeline(multimodal=True)
    source_audio = np.zeros(16538, dtype=np.float32)
    source_frames = torch.full((1, 3, 9, 8, 8), 123, dtype=torch.uint8)
    monkeypatch.setattr(pipeline_module, "load_audio_condition", lambda *a, **kw: source_audio)
    monkeypatch.setattr(pipeline_module, "load_video_condition", lambda *a, **kw: source_frames)
    monkeypatch.setattr(pipeline_module, "encode_audio_condition", lambda *a: torch.full((1, 17, 3), 7.0))
    monkeypatch.setattr(pipeline_module, "encode_video_condition", lambda *a: torch.full((1, 3, 1, 1, 2), 9.0))
    monkeypatch.setattr(pipeline_module, "postprocess_audio", lambda *a: ["generated audio"])

    def denoise(**kw):
        assert kw["sample_audio"] == (condition != "audio")
        assert kw["sample_video"] == (condition != "video")
        expected = kw["bundle"].audio if condition == "audio" else kw["bundle"].video
        assert (expected == (7 if condition == "audio" else 9)).all()
        return kw["bundle"]

    monkeypatch.setattr(pipeline_module, "denoise_loop", denoise)
    output = pipe(text="test", **{condition: "input.mp4"})
    if condition == "video":
        assert output.frames is source_frames
        assert output.audio == ["generated audio"]
    else:
        assert output.audio[0] is source_audio


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio": "a", "video": "v"},
        {"audio": "a", "image": "i"},
        {"video": "v", "text": ["one", "two"]},
        {"audio": "a", "sample_audio": False},
    ],
)
def test_invalid_media_combinations_fail_early(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _pipeline(multimodal=True)(**({"text": "test"} | kwargs))


def test_media_io_resamples_and_rejects_short_inputs(tmp_path: Path) -> None:
    path = tmp_path / "source.mp4"
    mux_video_audio(
        torch.full((3, 9, 16, 16), 123, dtype=torch.uint8),
        np.zeros(24000, dtype=np.float32),
        path,
        fps=24,
        audio_sample_rate=48000,
    )
    frames = load_video_condition(path, height=8, width=8, num_frames=9, fps=24)
    audio = load_audio_condition(path, sample_rate=44100, num_samples=16538, fps=24)
    assert frames.shape == (1, 3, 9, 8, 8)
    assert frames.dtype == torch.uint8
    assert audio.shape == (16538,)
    with pytest.raises(ValueError, match="too short"):
        load_video_condition(path, height=8, width=8, num_frames=40, fps=24)
    with pytest.raises(ValueError, match="too short"):
        load_audio_condition(path, sample_rate=44100, num_samples=441000, fps=24)


def test_tensor_encoders_are_deterministic_and_preserve_layout() -> None:
    class AudioVAE(torch.nn.Module):
        scaling_factor = 2.0

        def wrapped_encode(self, samples: torch.Tensor) -> torch.Tensor:
            return samples[:, None]

    class VideoVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = {"scaling_factor": 2.0}

        def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels[:, :1]))

    waveform = torch.tensor([1.0, 2.0], requires_grad=True)
    audio = encode_audio_condition(waveform, AudioVAE())
    assert audio.tolist() == [[[2.0], [4.0]]]
    assert not audio.requires_grad
    frames = torch.full((1, 3, 2, 1, 1), 255, dtype=torch.uint8)
    video = encode_video_condition(frames, VideoVAE())
    assert video.shape == (1, 2, 1, 1, 1)
    assert torch.equal(video, torch.full_like(video, 2.0))
    assert not video.requires_grad
    with pytest.raises(ValueError, match="mono"):
        encode_audio_condition(torch.zeros(2, 3), AudioVAE())
    with pytest.raises(ValueError, match="RGB uint8"):
        encode_video_condition(frames.float(), VideoVAE())
