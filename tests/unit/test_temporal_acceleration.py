"""CPU checks for the native PiFlow integration, without model weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kandinsky.core.algo.piflow_sampler import piflow_denoise_loop
from kandinsky.core.algo.prepare_ropes import compute_visual_rope
from kandinsky.core.algo.temporal.trajectory import TemporalTrajectory
from kandinsky.core.components.dit import DiffusionTransformer3D
from kandinsky.core.components.piflow_dit import PiFlowDiffusionTransformer3D
from kandinsky.core.components.rope import RoPE1D, RoPE3D
from kandinsky.core.types import LatentBundle, TextEmbeds
from kandinsky.pipeline.config import load_config
from kandinsky.pipeline.pipeline import Kandinsky6Pipeline
from kandinsky.pipeline.reconfigure import _apply_cache, _apply_generation, remember
from kandinsky.runtime.cache import CacheDiT
from kandinsky.runtime.cache.adaptive_block import AdaptiveBlockConfig
from kandinsky.runtime.temporal import validate_temporal_and_block_cache


class TinyBlock(nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add = add
        self.calls = 0

    def forward(self, video, audio, *_args, **_kwargs):
        self.calls += 1
        return video + self.add, audio + self.add


class TinyDiT(nn.Module):
    is_multimodal = True
    visual_cond = False
    patch_size = (1, 1, 1)
    _run_visual_blocks_fused = DiffusionTransformer3D._run_visual_blocks_fused
    _run_accelerated_fused_blocks = DiffusionTransformer3D._run_accelerated_fused_blocks

    def __init__(self):
        super().__init__()
        self.visual_rope = RoPE3D((2, 2, 2))
        self.audio_rope = RoPE1D(6)
        self.visual_transformer_blocks = nn.ModuleList([TinyBlock(1), TinyBlock(0.25)] + [TinyBlock(0) for _ in range(58)])
        self.steps = []
        self.fused_outputs = []

    def forward(self, *, x_video, x_audio, time, visual_rope, audio_rope, **kwargs):
        self.steps.append((x_video.shape[1], time, kwargs.get("av_cross_rope")))
        block_inputs = (
            x_video.bfloat16(), x_audio.bfloat16(), None, None, None, None, visual_rope, audio_rope, None,
        )
        if kwargs.get("av_cross_rope") is not None or kwargs.get("adaptive_block_tail") is not None:
            fused = self._run_accelerated_fused_blocks(
                *block_inputs, kwargs.get("av_cross_rope"), kwargs.get("adaptive_block_tail"),
            )
        else:
            fused = self._run_visual_blocks_fused(*block_inputs)
        self.fused_outputs.append(fused)
        return torch.zeros_like(x_video), torch.zeros_like(x_audio)


def run(*, fixed=None, threshold=None, cache=False, payload_gib=32.0):
    module = TinyDiT()
    dit = CacheDiT(module)
    video = torch.randn(1, 31, 2, 2, 2, generator=torch.Generator().manual_seed(18)).bfloat16()
    audio = torch.ones(1, 5, 2, dtype=torch.bfloat16)
    scale = (1.0, 2.0, 2.0)
    temporal = None
    if fixed is not None or threshold is not None:
        temporal = TemporalTrajectory(
            video, audio, module, scale, switch_after_calls=fixed, threshold=threshold,
        )
    if cache:
        dit.set_cache("adaptive_block", adaptive_block=AdaptiveBlockConfig(payload_gib=payload_gib))
    state = dit.new_adaptive_block_state(10)
    result = piflow_denoise_loop(
        bundle=LatentBundle(video, audio, torch.tensor([0, 31]), torch.tensor([0, 5])),
        dit=dit,
        text_embeds=TextEmbeds(text_embeds=torch.zeros(1, 1, 1), pooled_embed=torch.zeros(1, 1)),
        visual_rope=compute_visual_rope(module.visual_rope, (31, 2, 2), scale),
        audio_rope=module.audio_rope(torch.arange(5)),
        text_rope=[],
        num_steps=10,
        scheduler_scale=5.0,
        temporal=temporal,
        dit_kwargs={"adaptive_block_state": state} if state is not None else None,
    )
    return result, module, temporal, state


@pytest.mark.parametrize(("fixed", "threshold", "expected"), [(2, None, 2), (None, 1e-10, 1), (None, 2.0, 8)])
def test_temporal_uses_native_audio_and_restores_full_grid(fixed, threshold, expected):
    baseline, native, _, _ = run()
    output, module, temporal, _ = run(fixed=fixed, threshold=threshold)
    assert temporal.chosen == expected
    assert [frames for frames, _, _ in module.steps] == [16] * expected + [31] * (10 - expected)
    assert [rope is not None for _, _, rope in module.steps] == [True] * expected + [False] * (10 - expected)
    assert all(torch.equal(times[1], native_times[1])
               for (_, times, _), (_, native_times, _) in zip(module.steps, native.steps))
    assert not torch.equal(module.steps[1][1][0], module.steps[1][1][1]) or expected == 1
    assert torch.equal(output.audio, baseline.audio)
    assert output.video.shape == baseline.video.shape
    assert output.video_cu_seqlens.tolist() == [0, 31]


@pytest.mark.parametrize(("fixed", "hits", "invalidations"), [(None, [2, 4, 6, 8], []), (2, [4, 6, 8], [2]), (8, [], [8])])
def test_cache_uses_full_anchors_and_never_skips_final_step(fixed, hits, invalidations):
    output, module, _, state = run(fixed=fixed, cache=True)
    reference, native, _, _ = run(fixed=fixed)
    assert state.hits == hits
    assert state.invalidations == invalidations
    assert state.residual is None
    assert module.visual_transformer_blocks[0].calls == 10
    assert all(block.calls == 10 - len(hits) for block in module.visual_transformer_blocks[1:])
    assert torch.equal(output.video, reference.video)
    assert torch.equal(output.audio, reference.audio)
    assert all(torch.equal(pair[1], native_pair[1])
               for pair, native_pair in zip(module.fused_outputs, native.fused_outputs))


def test_cache_budget_keeps_full_execution():
    _, module, _, state = run(cache=True, payload_gib=0)
    assert state.disabled_reason == "payload_budget"
    assert not state.hits
    assert state.residual is None
    assert all(block.calls == 10 for block in module.visual_transformer_blocks)


def test_temporal_runs_with_fused_attention_and_piflow_output_heads():
    dit = PiFlowDiffusionTransformer3D(
        n_grid=2, in_visual_dim=2, out_visual_dim=2, in_audio_dim=2, out_audio_dim=2,
        in_text_dim=4, in_text_dim2=4, time_dim=8, patch_size=(1, 1, 1),
        model_dim=12, ff_dim=24, num_text_blocks=1, num_visual_blocks=1,
        axes_dims=(2, 2, 2), is_multimodal=True, ca_rope=True, attention_engine="sdpa",
    ).bfloat16().eval()
    video = torch.ones(1, 31, 2, 2, 2, dtype=torch.bfloat16)
    audio = torch.ones(1, 5, 2, dtype=torch.bfloat16)
    scale = (1.0, 2.0, 2.0)
    temporal = TemporalTrajectory(video, audio, dit, scale, switch_after_calls=2, threshold=None)
    result = piflow_denoise_loop(
        bundle=LatentBundle(video, audio, torch.tensor([0, 31]), torch.tensor([0, 5])),
        dit=dit,
        text_embeds=TextEmbeds(
            text_embeds=torch.zeros(1, 3, 4, dtype=torch.bfloat16),
            pooled_embed=torch.zeros(1, 4, dtype=torch.bfloat16),
        ),
        visual_rope=compute_visual_rope(dit.visual_rope, (31, 2, 2), scale),
        audio_rope=dit.audio_rope(torch.arange(5)),
        text_rope=[dit.video_text_rope(torch.arange(3)), dit.audio_text_rope(torch.arange(3))],
        num_steps=3,
        scheduler_scale=5.0,
        num_policy_substeps=2,
        temporal=temporal,
    )
    assert result.video.shape == video.shape
    assert result.audio.shape == audio.shape
    assert torch.isfinite(result.video).all()
    assert torch.isfinite(result.audio).all()


@pytest.mark.parametrize(("tokens", "temporal", "mode"), [
    ([], "off", "none"),
    (["temporal-fixed"], "fixed", "none"),
    (["temporal-a2"], "adaptive", "none"),
    (["adaptive-block-cache"], "off", "adaptive_block"),
    (["temporal-a2", "adaptive-block-cache"], "adaptive", "adaptive_block"),
])
def test_presets_use_native_composition(tokens, temporal, mode):
    config = load_config(["h100", "pro-distill", *tokens])
    assert config.temporal.mode == temporal
    assert config.cache.mode == mode
    validate_temporal_and_block_cache(config, mode, "module")


def test_incompatible_modes_rejected():
    config = load_config(["h100", "pro", "adaptive-block-cache"])
    with pytest.raises(ValueError, match="pro-distill"):
        validate_temporal_and_block_cache(config, config.cache.mode, "module")
    config = load_config(["h100", "pro-distill", "temporal-fixed"])
    with pytest.raises(ValueError, match="none/module offload"):
        validate_temporal_and_block_cache(config, config.cache.mode, "block")


def test_reconfigure_restores_adaptive_cache_after_a_manual_override():
    config = load_config(["h100", "pro-distill", "adaptive-block-cache"])
    pipe = Kandinsky6Pipeline(
        dit=CacheDiT(TinyDiT()), text_embedder=object(), vae=object(), device=torch.device("cpu"),
        piflow_conf=config.piflow,
    )
    _apply_cache(pipe, config.cache, config.cache.mode, force=True)
    remember(pipe, config, attention_engine="sdpa", cache_mode=config.cache.mode, offload_strategy="none")
    pipe.set_cache(None)
    pipe.reconfigure(config, attention_engine="sdpa", offload_strategy="none")
    assert pipe.dit.mode == "adaptive_block"


def test_pipeline_switches_presets_and_resets_cache_between_requests(monkeypatch):
    module = TinyDiT()
    config = load_config(["h100", "pro-distill", "temporal-fixed", "adaptive-block-cache"])
    pipe = Kandinsky6Pipeline(
        dit=CacheDiT(module), text_embedder=object(), vae=object(), device=torch.device("cpu"),
        audio_vae=object(), vocoder=object(), piflow_conf=config.piflow,
    )
    _apply_generation(pipe, config, None)
    _apply_cache(pipe, config.cache, config.cache.mode, force=True)
    video = torch.ones(1, 31, 2, 2, 2, dtype=torch.bfloat16)
    audio = torch.ones(1, 5, 2, dtype=torch.bfloat16)

    def prepare(request):
        return SimpleNamespace(
            request=request,
            bundle=LatentBundle(video, audio, torch.tensor([0, 31]), torch.tensor([0, 5])),
            text_embeds=TextEmbeds(text_embeds=torch.zeros(1, 1, 1), pooled_embed=torch.zeros(1, 1)),
            visual_rope=compute_visual_rope(module.visual_rope, (31, 2, 2), pipe.scale_factor),
            audio_rope=module.audio_rope(torch.arange(5)),
            text_rope=[],
            attention_mask=None,
            visual_token_type_ids=None,
        )

    monkeypatch.setattr(pipe, "_prepare", prepare)
    monkeypatch.setattr(pipe, "_decode", lambda _, bundle: bundle)
    for _ in range(2):
        pipe("prompt", seed=42, show_progress=False)
        assert pipe.last_acceleration_stats["temporal"]["switch_after_calls"] == 2
        assert pipe.last_acceleration_stats["adaptive_block"]["hit_step_indices"] == [4, 6, 8]

    baseline = load_config(["h100", "pro-distill"])
    _apply_generation(pipe, baseline, config)
    _apply_cache(pipe, baseline.cache, baseline.cache.mode, force=False)
    pipe("prompt", seed=42, show_progress=False)
    assert pipe.raw_dit is module
    assert pipe.last_acceleration_stats is None
    assert [frames for frames, _, _ in module.steps[-10:]] == [31] * 10
