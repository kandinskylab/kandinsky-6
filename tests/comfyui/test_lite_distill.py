"""CPU-only adapter regressions; set COMFYUI_PATH to a ComfyUI checkout.

Run: COMFYUI_PATH=/path/to/ComfyUI python -m unittest discover -s tests/comfyui -v
Optional: K6_LITE_DISTILL_WEIGHTS=/path/to/diffusion_pytorch_model.safetensors
checks every key and shape of a real checkpoint, without allocating its weights.
K6_PRO_DISTILL_WEIGHTS and K6_LITE_WEIGHTS enable the same regression checks.
"""

import copy
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

COMFY_ROOT = Path(os.environ.get("COMFYUI_PATH", "__not_configured__"))
if not (COMFY_ROOT / "comfy").is_dir():
    raise unittest.SkipTest("Set COMFYUI_PATH to run the ComfyUI adapter tests.")
sys.path[:0] = [str(COMFY_ROOT), str(Path(__file__).resolve().parents[2] / "comfyui")]

import torch
import comfy.cli_args

# These checks must not initialize CUDA or interfere with another GPU job.
comfy.cli_args.args.cpu = True

import comfy.model_detection
import comfy.ops
import comfy.supported_models
from kandinsky6 import nodes, register, sampling
from kandinsky6.core_contract import DIT_CONFIG
from kandinsky6.model_base import Kandinsky6NativeAVDiT
from kandinsky6.piflow_contract import PIFLOW_DEFAULTS


def checkpoint_shapes(dit, n_grid):
    """Released architecture fixture on meta; no multi-GB allocations."""
    def tensor(*shape):
        return torch.empty(shape, device="meta")

    state = {
        "visual_embeddings.in_layer.weight": tensor(dit["model_dim"], 33 * math.prod(dit["patch_size"])),
        "out_layer.out_layer.weight": tensor(16 * n_grid * math.prod(dit["patch_size"]), dit["model_dim"]),
        "audio_embeddings.in_layer.weight": tensor(dit["model_dim_a"], 40),
        "audio_outLayer.out_layer.weight": tensor(40 * n_grid, dit["model_dim_a"]),
        "video_time_embeddings.out_layer.weight": tensor(dit["time_dim"], dit["time_dim"]),
        "audio_time_embeddings.out_layer.weight": tensor(dit["time_dim_a"], dit["time_dim_a"]),
        "video_text_embeddings.in_layer.weight": tensor(dit["model_dim"], dit["in_text_dim"]),
        "video_pooled_text_embeddings.in_layer.weight": tensor(dit["time_dim"], dit["in_text_dim2"]),
        "visual_blocks.0.videoT.feed_forward.in_layer.weight": tensor(dit["ff_dim"], dit["model_dim"]),
        "visual_blocks.0.audioT.feed_forward.in_layer.weight": tensor(dit["ff_dim_a"], dit["model_dim_a"]),
        "visual_blocks.0.videoT.self_attention.query_norm.weight": tensor(sum(dit["axes_dims"])),
        "visual_blocks.0.audioT.self_attention.query_norm.weight": tensor(sum(dit["axes_dims_a"])),
        "visual_blocks.0.va_modulation.out_layer.weight": tensor(2 * dit["model_dim"] + dit["model_dim_a"], dit["time_dim"]),
        "visual_blocks.0.va_cross_attention.to_key.weight": tensor(dit["model_dim"], dit["model_dim_a"]),
        "visual_token_type_embeddings.weight": tensor(2, dit["model_dim"]),
    }
    for i in range(dit["num_visual_blocks"]):
        state[f"visual_blocks.{i}.videoT.self_attention.to_query.weight"] = tensor(dit["model_dim"], dit["model_dim"])
    for prefix in ("video_text_transformer_blocks", "audio_text_transformer_blocks"):
        for i in range(dit["num_text_blocks"]):
            state[f"{prefix}.{i}.self_attention.to_query.weight"] = tensor(1)
    return state


class DetectionTests(unittest.TestCase):
    def test_all_four_released_variants_through_registered_diffusers_detector(self):
        fallback = mock.Mock()
        with mock.patch.object(comfy.supported_models, "models", []), mock.patch.object(
            comfy.model_detection, "detect_unet_config", fallback
        ):
            register.register()
            for dit in (DIT_CONFIG, register._LITE_DIT_CONFIG):
                for grid in (1, 10):
                    for prefix in ("", "model.diffusion_model."):
                        with self.subTest(dim=dit["model_dim"], grid=grid, prefix=prefix):
                            native = checkpoint_shapes(dit, grid)
                            state = {prefix + register._diffusers_key(k): v for k, v in native.items()}
                            config = comfy.model_detection.detect_unet_config(state, prefix)
                            self.assertEqual(config["n_grid"], grid)
                            self.assertEqual(config["model_dim"], dit["model_dim"])
                            self.assertEqual(config["out_visual_dim"], 16 * grid)
                            self.assertEqual(config["out_audio_dim"], 40 * grid)
                            self.assertTrue(config["cross_gates"])
                            for key, value in native.items():
                                self.assertIs(state[prefix + key], value)
            fallback.assert_not_called()

    def test_lite_distill_still_validates_video_and_audio_heads(self):
        state = checkpoint_shapes(register._LITE_DIT_CONFIG, 10)
        for audio_head, message in ((None, "missing its audio"), (torch.empty(40, 896, device="meta"), "do not match")):
            with self.subTest(message=message):
                invalid = {register._diffusers_key(k): v for k, v in state.items()}
                key = "audio_out_layer.out_layer.weight"
                if audio_head is None:
                    invalid.pop(key)
                else:
                    invalid[key] = audio_head
                original = invalid.copy()
                with self.assertRaisesRegex(ValueError, message):
                    register._detect_diffusers_k6(invalid, "")
                self.assertEqual(invalid.keys(), original.keys())
                for key in invalid:
                    self.assertIs(invalid[key], original[key])

    @unittest.skipUnless(os.environ.get("K6_LITE_DISTILL_WEIGHTS"), "Optional real-checkpoint check")
    def test_real_lite_distill_checkpoint_matches_every_model_parameter(self):
        self._check_real_checkpoint("K6_LITE_DISTILL_WEIGHTS", 1792, 10)

    @unittest.skipUnless(os.environ.get("K6_PRO_DISTILL_WEIGHTS"), "Optional real-checkpoint check")
    def test_real_pro_distill_checkpoint_matches_every_model_parameter(self):
        self._check_real_checkpoint("K6_PRO_DISTILL_WEIGHTS", 4096, 10)

    @unittest.skipUnless(os.environ.get("K6_LITE_WEIGHTS"), "Optional real-checkpoint check")
    def test_real_base_lite_checkpoint_matches_every_model_parameter(self):
        self._check_real_checkpoint("K6_LITE_WEIGHTS", 1792, 1)

    def _check_real_checkpoint(self, variable, model_dim, n_grid):
        with Path(os.environ[variable]).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        state = {key: torch.empty(item["shape"], device="meta") for key, item in header.items() if key != "__metadata__"}
        config = register._detect_diffusers_k6(state, "")
        self.assertEqual((config["model_dim"], config["n_grid"]), (model_dim, n_grid))
        model = Kandinsky6NativeAVDiT(**config, device="meta", dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        expected = model.state_dict()
        self.assertEqual(expected.keys(), state.keys())
        for key in expected:
            self.assertEqual(expected[key].shape, state[key].shape, key)


class SamplingTests(unittest.TestCase):
    def test_lite_distill_selects_piflow_and_rejects_cfg_or_partial_denoising(self):
        model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(model_dim=1792, n_grid=10)))
        sampler = nodes.Kandinsky6Sampler()
        latent, positive, negative = {}, [["conditioning"]], []
        result = {"sampled": True}
        with mock.patch.object(nodes, "sample_piflow", return_value=result) as piflow:
            self.assertEqual(sampler.sample(model, 42, 10, 1.0, "euler", "simple", positive, negative, latent), (result,))
            piflow.assert_called_once_with(model, latent, positive, 42, 10)
        for cfg, denoise in ((5.0, 1.0), (1.0, 0.5)):
            with self.subTest(cfg=cfg, denoise=denoise), self.assertRaisesRegex(ValueError, "CFG=1 and denoise=1"):
                sampler.sample(model, 42, 10, cfg, "euler", "simple", positive, negative, latent, denoise)
        model.model.diffusion_model.n_grid = 1
        with mock.patch.object(nodes.comfy_nodes.KSampler, "sample", return_value=(result,)) as stock:
            self.assertEqual(sampler.sample(model, 42, 50, 5.0, "euler", "simple", positive, negative, latent), (result,))
            stock.assert_called_once()

    def test_magcache_is_bypassed_for_lite_distill_without_changing_original(self):
        options = {"transformer_options": {"k6_magcache": object(), "unrelated": True}}
        model = SimpleNamespace(
            model=SimpleNamespace(diffusion_model=SimpleNamespace(model_dim=1792, n_grid=10)),
            model_options=options,
        )
        clone = SimpleNamespace(model_options=copy.deepcopy(options))
        model.clone = mock.Mock(return_value=clone)
        (result,) = nodes.Kandinsky6MagCache().apply(model, 10, 0.1, 4, 0.2)
        self.assertIs(result, clone)
        self.assertNotIn("k6_magcache", result.model_options["transformer_options"])
        self.assertTrue(result.model_options["transformer_options"]["unrelated"])
        self.assertIn("k6_magcache", options["transformer_options"])

    def test_tiny_distilled_dit_runs_t2va_and_i2va_ten_step_rollouts(self):
        # Real model/adapter/PiFlow code, reduced hidden widths for CPU execution.
        model = Kandinsky6NativeAVDiT(
            in_visual_dim=2, out_visual_dim=20, in_audio_dim=3, out_audio_dim=30,
            n_grid=10, in_text_dim=8, in_text_dim2=4, time_dim=8, model_dim=24,
            ff_dim=48, visual_embed_dim=5, time_dim_a=8, model_dim_a=12,
            ff_dim_a=24, head_dim_a=6, patch_size=(1, 1, 1), num_text_blocks=1,
            num_visual_blocks=1, axes_dims=(2, 2, 2), device="cpu",
            dtype=torch.float32, operations=comfy.ops.manual_cast,
        ).eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(42)
            for parameter in model.parameters():
                parameter.normal_(std=0.02)
            context, pooled = torch.randn(1, 3, 8), torch.randn(1, 4)
            for use_reference in (False, True):
                with self.subTest(i2va=use_reference):
                    video, audio = torch.randn(1, 2, 2, 2, 2), torch.randn(1, 3, 3)
                    reference = video[:, :, -1:].clone() if use_reference else None
                    with mock.patch.object(model, "forward", wraps=model.forward) as forward:
                        sampled_video, sampled_audio = sampling.rollout(
                            model, video, audio, context, pooled,
                            steps=PIFLOW_DEFAULTS["num_steps"], dtype=torch.float32,
                            reference=reference, transformer_options={"k6_magcache": object()},
                        )
                    self.assertEqual(forward.call_count, 10)
                    self.assertEqual(sampled_video.shape, video.shape)
                    self.assertEqual(sampled_audio.shape, audio.shape)
                    self.assertTrue(torch.isfinite(sampled_video).all())
                    self.assertTrue(torch.isfinite(sampled_audio).all())
                    self.assertNotIn("k6_magcache", forward.call_args.kwargs["transformer_options"])
                    if reference is not None:
                        self.assertTrue(torch.equal(sampled_video[:, :, -1:], reference))


if __name__ == "__main__":
    unittest.main()
