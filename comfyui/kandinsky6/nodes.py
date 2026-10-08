"""Kandinsky 6 nodes for ComfyUI."""
import math
import os

import torch

import comfy.model_management as mm
import comfy.nested_tensor
import folder_paths
import node_helpers
import nodes as comfy_nodes

from .audio_vae import K6AudioVAE
from .beautifier import Kandinsky6BeautifyPrompt
from .core_contract import (
    AUDIO_DEFAULTS,
    DIT_CONFIG,
    GENERATION_DEFAULTS,
    LATENT_DEFAULTS,
    MAGCACHE_DEFAULTS,
    PROMPT_TEMPLATE,
    PROMPT_TOKENIZER_MAX_LENGTH,
    SCHEDULER_DEFAULTS,
)
from .magcache import K6MagCacheState
from .piflow_contract import PIFLOW_DEFAULTS
from .sampling import sample_piflow


AUDIO_VAE_DIR = os.path.join(folder_paths.models_dir, "audio_vae")
folder_paths.add_model_folder_path("kandinsky6_audio_vae", AUDIO_VAE_DIR)

_K6_PROMPT_TEMPLATE = PROMPT_TEMPLATE
_VIDEO_CHANNELS = int(LATENT_DEFAULTS["video_channels"])
_VIDEO_SPATIAL_FACTOR = int(LATENT_DEFAULTS["video_spatial_compression_factor"])
_VIDEO_TEMPORAL_FACTOR = int(LATENT_DEFAULTS["video_temporal_compression_factor"])
_AUDIO_CHANNELS = int(LATENT_DEFAULTS["audio_channels"])
_AUDIO_DOWNSAMPLE_FACTOR = int(LATENT_DEFAULTS["audio_downsample_factor"])
_AUDIO_SAMPLE_RATE = int(LATENT_DEFAULTS["audio_sample_rate"])
_DIT_PATCH_SIZE = tuple(int(value) for value in DIT_CONFIG["patch_size"])


def _list_tod_vae():
    try:
        files = folder_paths.get_filename_list("kandinsky6_audio_vae")
    except Exception:
        files = []
    files = [f for f in files if os.sep not in f and "/" not in f]
    return [f for f in files if f.endswith((".pth", ".safetensors", ".pt"))] or ["v1-44.pth"]


def _list_bigvgan_dirs():
    dirs = []
    if os.path.isdir(AUDIO_VAE_DIR):
        for d in sorted(os.listdir(AUDIO_VAE_DIR)):
            if os.path.isdir(os.path.join(AUDIO_VAE_DIR, d)):
                dirs.append(d)
    return dirs or ["bigvgan_vocoder"]


def _audio_latent_len(length, fps, downsample_factor):
    seconds = length / fps
    return int(math.ceil(seconds * _AUDIO_SAMPLE_RATE / downsample_factor))


def _validate_generation_shape(width, height, length, batch_size):
    divisibility = int(GENERATION_DEFAULTS["image_divisibility"])
    if width % divisibility or height % divisibility:
        raise ValueError(
            f"Kandinsky 6 requires width and height divisible by {divisibility}, got {width}x{height}."
        )
    if (length - 1) % _VIDEO_TEMPORAL_FACTOR:
        raise ValueError(
            "Kandinsky 6 requires a frame count of the form "
            f"{_VIDEO_TEMPORAL_FACTOR}*n+1, got {length}."
        )
    if batch_size != 1:
        raise ValueError("The initial Kandinsky 6 ComfyUI release supports batch_size=1 only.")

    latent_shape = (
        (length - 1) // _VIDEO_TEMPORAL_FACTOR + 1,
        height // _VIDEO_SPATIAL_FACTOR,
        width // _VIDEO_SPATIAL_FACTOR,
    )
    if any(size % patch for size, patch in zip(latent_shape, _DIT_PATCH_SIZE, strict=True)):
        raise ValueError(
            "Kandinsky 6 latent dimensions must be divisible by the DiT patch "
            f"size {_DIT_PATCH_SIZE}, got {latent_shape}."
        )


def _joint_streams(joint_latent):
    samples = joint_latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "Kandinsky 6 expects a joint LATENT containing video and audio streams."
        )
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError(
            f"Kandinsky 6 expects exactly two latent streams, got {len(streams)}."
        )
    return streams


def _validate_joint_latent(joint_latent):
    video, audio = _joint_streams(joint_latent)
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("Kandinsky 6 video latent must be a 5D BCHTW tensor.")
    if not torch.is_tensor(audio) or audio.ndim != 3:
        raise ValueError("Kandinsky 6 audio latent must be a 3D BTF tensor.")

    batch, channels, frames, height, width = video.shape
    if batch != 1 or audio.shape[0] != batch:
        raise ValueError("The initial Kandinsky 6 ComfyUI release requires matching batch_size=1 latents.")
    if channels != _VIDEO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_VIDEO_CHANNELS} video latent channels, got {channels}."
        )
    if audio.shape[-1] != _AUDIO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_AUDIO_CHANNELS} audio latent channels, got {audio.shape[-1]}."
        )
    latent_shape = (frames, height, width)
    if any(size < 1 for size in latent_shape) or any(
        size % patch for size, patch in zip(latent_shape, _DIT_PATCH_SIZE, strict=True)
    ):
        raise ValueError(
            "Kandinsky 6 video latent dimensions must be positive and divisible "
            f"by the DiT patch size {_DIT_PATCH_SIZE}."
        )
    if audio.shape[1] < 1:
        raise ValueError("Kandinsky 6 audio latent must contain at least one frame.")

    fps = joint_latent.get("frame_rate")
    if fps is not None:
        sample_frames = (frames - 1) * _VIDEO_TEMPORAL_FACTOR + 1
        expected_audio_frames = _audio_latent_len(
            sample_frames,
            float(fps),
            _AUDIO_DOWNSAMPLE_FACTOR,
        )
        if audio.shape[1] != expected_audio_frames:
            raise ValueError(
                "Kandinsky 6 audio/video duration mismatch: "
                f"expected {expected_audio_frames} audio latent frames, got {audio.shape[1]}."
            )


def _tokenize_k6(clip, text):
    # Pass the canonical K6 template explicitly instead of mutating ComfyUI's
    # shared Kandinsky 5 tokenizer. CLIP-L still receives the raw caption.
    tokens = clip.tokenize(text, llama_template=_K6_PROMPT_TEMPLATE)
    # Canonical K6 admits at most 1024 caption tokens after its 129-token
    # prompt prefix. ComfyUI's Qwen tokenizer is intentionally unbounded.
    if "qwen25_7b" in tokens:
        tokens["qwen25_7b"] = [
            row[:PROMPT_TOKENIZER_MAX_LENGTH] for row in tokens["qwen25_7b"]
        ]
    return tokens


class Kandinsky6EmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": int(GENERATION_DEFAULTS["width"]), "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": int(GENERATION_DEFAULTS["height"]), "min": 256, "max": 2048, "step": 16}),
                "length": ("INT", {"default": int(GENERATION_DEFAULTS["sample_frames"]), "min": 1, "max": 1001, "step": _VIDEO_TEMPORAL_FACTOR,
                                   "tooltip": f"Pixel frames. Kandinsky 6 requires {_VIDEO_TEMPORAL_FACTOR}*n+1 frames."}),
                "fps": ("FLOAT", {"default": float(GENERATION_DEFAULTS["fps"]), "min": 1.0, "max": 120.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("joint_latent", "fps")
    FUNCTION = "build"
    CATEGORY = "Kandinsky 6"

    def build(self, width, height, length, fps, batch_size):
        _validate_generation_shape(width, height, length, batch_size)
        t_lat = (length - 1) // _VIDEO_TEMPORAL_FACTOR + 1
        h_lat = height // _VIDEO_SPATIAL_FACTOR
        w_lat = width // _VIDEO_SPATIAL_FACTOR
        device = mm.intermediate_device()
        video = torch.zeros(
            batch_size, _VIDEO_CHANNELS, t_lat, h_lat, w_lat, device=device
        )
        t_a = _audio_latent_len(
            length,
            fps,
            downsample_factor=_AUDIO_DOWNSAMPLE_FACTOR,
        )
        audio = torch.zeros(
            batch_size,
            t_a,
            _AUDIO_CHANNELS,
            device=device,
        )
        joint = comfy.nested_tensor.NestedTensor((video, audio))
        latent = {
            "samples": joint,
            "frame_rate": float(fps),
            "sample_rate": _AUDIO_SAMPLE_RATE,
        }
        _validate_joint_latent(latent)
        return latent, float(fps)


class Kandinsky6ImageToVideoAudio:
    """Append the clean I2VA reference tail while keeping stock Comfy inputs.

    ``reference_latent`` is intentionally produced by ComfyUI's standard
    ``VAEEncode`` node.  The only K6-specific work here is the canonical
    ``tail_cond_first_frame`` layout and its denoise mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "empty_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "joint_latent")
    FUNCTION = "apply"
    CATEGORY = "Kandinsky 6"

    def apply(self, positive, negative, empty_latent, reference_latent):
        _validate_joint_latent(empty_latent)
        video, audio = _joint_streams(empty_latent)

        reference = reference_latent.get("samples")
        if getattr(reference, "is_nested", False) or not torch.is_tensor(reference):
            raise ValueError(
                "Kandinsky 6 I2VA expects a regular video LATENT from the standard VAEEncode node."
            )
        if reference.ndim == 4:
            reference = reference.unsqueeze(2)
        if reference.ndim != 5:
            raise ValueError(
                "Kandinsky 6 I2VA reference latent must be a 5D BCHTW tensor, "
                f"got {tuple(reference.shape)}."
            )
        if reference.shape[2] != 1:
            raise ValueError(
                "Kandinsky 6 I2VA requires exactly one encoded reference frame, "
                f"got {reference.shape[2]} latent frames."
            )
        expected = (video.shape[0], video.shape[1], video.shape[3], video.shape[4])
        actual = (reference.shape[0], reference.shape[1], reference.shape[3], reference.shape[4])
        if actual != expected:
            raise ValueError(
                "Kandinsky 6 I2VA reference latent must match the empty video latent: "
                f"expected B/C/H/W {expected}, got {actual}. Resize the input image "
                "to the generation width and height before VAEEncode."
            )

        reference = reference.to(device=video.device, dtype=video.dtype)
        video_with_reference = torch.cat((video, reference), dim=2)

        # Comfy's standard inpaint mask keeps the reference clean throughout
        # KSampler.  The model adapter additionally returns the clean latent
        # from scale_latent_inpaint, matching core's per-step tail reset.
        video_mask = torch.ones_like(video_with_reference)
        video_mask[:, :, -1] = 0
        audio_mask = torch.ones_like(audio)

        latent = empty_latent.copy()
        latent["samples"] = comfy.nested_tensor.NestedTensor(
            (video_with_reference, audio)
        )
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        )
        latent["k6_reference_tail"] = True
        latent["k6_generated_video_latent_frames"] = int(video.shape[2])

        values = {"k6_reference_tail": True}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        return positive, negative, latent


class Kandinsky6RemoveReferenceLatent:
    """Remove the I2VA-only reference tail before standard VAE decoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"joint_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("joint_latent",)
    FUNCTION = "remove"
    CATEGORY = "Kandinsky 6"

    def remove(self, joint_latent):
        video, audio = _joint_streams(joint_latent)
        generated_frames = joint_latent.get("k6_generated_video_latent_frames")
        if joint_latent.get("k6_reference_tail") is not True or not isinstance(
            generated_frames, int
        ):
            raise ValueError(
                "Kandinsky 6 reference-tail metadata is missing. Connect the output "
                "of Kandinsky6 Image to Video+Audio through KSampler first."
            )
        if generated_frames < 1 or video.shape[2] != generated_frames + 1:
            raise ValueError(
                "Kandinsky 6 sampled I2VA latent has an unexpected temporal shape: "
                f"metadata={generated_frames}, video={video.shape[2]}."
            )

        output = joint_latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor(
            (video[:, :, :generated_frames], audio)
        )
        output.pop("noise_mask", None)
        output.pop("k6_reference_tail", None)
        output.pop("k6_generated_video_latent_frames", None)
        _validate_joint_latent(output)
        return (output,)


class Kandinsky6AudioVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tod_vae": (_list_tod_vae(),),
                "bigvgan_dir": (_list_bigvgan_dirs(),),
                "mode": ([str(AUDIO_DEFAULTS["mode"])], {"default": str(AUDIO_DEFAULTS["mode"])}),
                "scaling_factor": ("FLOAT", {"default": float(AUDIO_DEFAULTS["scaling_factor"]), "min": 0.0001, "max": 10.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "Kandinsky 6"

    def load(self, tod_vae, bigvgan_dir, mode, scaling_factor):
        tod_vae_ckpt = folder_paths.get_full_path("kandinsky6_audio_vae", tod_vae) \
            or os.path.join(AUDIO_VAE_DIR, tod_vae)
        bigvgan_path = os.path.join(AUDIO_VAE_DIR, bigvgan_dir)
        vae = K6AudioVAE(tod_vae_ckpt, bigvgan_path, mode=mode,
                         scaling_factor=scaling_factor)
        return (vae,)


class Kandinsky6AudioVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("VAE",),
                "joint_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Kandinsky 6"

    def decode(self, audio_vae, joint_latent):
        samples = joint_latent["samples"]
        audio = samples.unbind()[-1] if getattr(samples, "is_nested", False) else samples
        if not torch.is_tensor(audio) or audio.ndim != 3 or audio.shape[-1] != _AUDIO_CHANNELS:
            raise ValueError(
                f"Kandinsky 6 audio latent must be shaped [B, T, {_AUDIO_CHANNELS}]."
            )
        wav = audio_vae.decode_canonical(audio)           # [B, N, C]
        waveform = wav.movedim(-1, 1).cpu().float()       # [B, C, N]
        return ({"waveform": waveform, "sample_rate": audio_vae.sample_rate},)


class Kandinsky6TextEncode:
    """Encode one Kandinsky 6 prompt for both the video and the audio branch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "video_caption": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": (
                                                 "What happens in the shot: scene, subjects, action, "
                                                 "camera. Spoken lines go here too, wrapped in "
                                                 "<S>...<E> at the moment they are said."
                                             )}),
                "audio_caption": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": (
                                                 "What the shot sounds like. Appended to the video "
                                                 "caption as <AUDCAP>...<ENDAUDCAP>; leave empty to "
                                                 "describe no sound. Not for spoken lines."
                                             )}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Kandinsky 6"
    DESCRIPTION = (
        "Builds the canonical Kandinsky 6 prompt — the video caption plus an "
        "<AUDCAP>...<ENDAUDCAP> block — and encodes it once. The video and audio "
        "branches of the model both receive that one conditioning."
    )

    def encode(self, clip, video_caption, audio_caption):
        text = video_caption
        if audio_caption and audio_caption.strip():
            text = f"{video_caption} <AUDCAP>{audio_caption}<ENDAUDCAP>"

        # Kandinsky5's Qwen defaults to hidden[-1] (before final RMSNorm), while
        # canonical K6 consumes the transformer's final normalized state.
        encode_clip = clip.clone()
        encode_clip.clip_layer(-9999)
        cond = encode_clip.encode_from_tokens_scheduled(_tokenize_k6(clip, text))

        # The canonical pipeline encodes one caption and feeds it to both
        # branches. Carrying it under the audio keys as well keeps the same
        # condition-key contract on both CFG branches, so ComfyUI can batch them.
        for context, extra in cond:
            extra["k6_audio_context"] = context
            extra["k6_audio_pooled_output"] = extra["pooled_output"]
        return (cond,)


class Kandinsky6Sampler(comfy_nodes.KSampler):
    """Use PiFlow for distilled Pro/Lite, or ComfyUI's sampler for base models."""

    CATEGORY = "Kandinsky 6"
    DESCRIPTION = (
        "Automatically runs the DX PiFlow policy for distilled Pro/Lite (CFG=1). "
        "For non-distilled models, uses the normal ComfyUI sampler and scheduler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["steps"][1]["default"] = int(PIFLOW_DEFAULTS["num_steps"])
        inputs["required"]["cfg"][1]["default"] = 1.0
        return inputs

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        if model.model.diffusion_model.n_grid == 1:
            return super().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
        if cfg != 1.0 or denoise != 1.0:
            raise ValueError(
                "Distilled K6 PiFlow requires CFG=1 and denoise=1; "
                "sampler/scheduler selectors apply only to base models."
            )
        return (sample_piflow(model, latent_image, positive, seed, steps),)


class Kandinsky6MagCache:
    """Attach the canonical K6 Pro MagCache policy to a Comfy MODEL clone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": (
                    "INT",
                    {
                        "default": int(SCHEDULER_DEFAULTS["num_steps"]),
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": float(MAGCACHE_DEFAULTS["thresh"]),
                        "min": 0.0001,
                        "max": 1.0,
                        "step": 0.0001,
                    },
                ),
                "max_skip_steps": (
                    "INT",
                    {
                        "default": int(MAGCACHE_DEFAULTS["K"]),
                        "min": 1,
                        "max": 32,
                    },
                ),
                "retention_ratio": (
                    "FLOAT",
                    {
                        "default": float(MAGCACHE_DEFAULTS["retention_ratio"]),
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Kandinsky 6"
    DESCRIPTION = (
        "Native K6 Pro MagCache. Uses the released T2VA/I2VA calibration to "
        "skip about half of the expensive DiT block-stack evaluations. The "
        "steps value must match the sampler for base Pro (default 50). "
        "Automatically bypassed for distilled Pro/Lite. Not calibrated for base K6 Lite or VSR."
    )

    def apply(self, model, steps, threshold, max_skip_steps, retention_ratio):
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if getattr(diffusion_model, "n_grid", 1) > 1:
            uncached_model = model.clone()
            uncached_model.model_options.setdefault("transformer_options", {}).pop("k6_magcache", None)
            return (uncached_model,)
        expected_dim = int(DIT_CONFIG["model_dim"])
        actual_dim = getattr(diffusion_model, "model_dim", None)
        if actual_dim != expected_dim:
            raise ValueError(
                "Kandinsky6 MagCache is calibrated for K6 Pro only; "
                f"expected model_dim={expected_dim}, got {actual_dim!r}. "
                "Do not attach it to K6 Lite or the VSR model."
            )

        cached_model = model.clone()
        transformer_options = cached_model.model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options["k6_magcache"] = K6MagCacheState(
            num_steps=int(steps),
            threshold=float(threshold),
            max_skip_steps=int(max_skip_steps),
            retention_ratio=float(retention_ratio),
            mag_ratios=MAGCACHE_DEFAULTS["mag_ratios"],
        )
        return (cached_model,)


NODE_CLASS_MAPPINGS = {
    "Kandinsky6BeautifyPrompt": Kandinsky6BeautifyPrompt,
    "Kandinsky6Sampler": Kandinsky6Sampler,
    "Kandinsky6EmptyLatent": Kandinsky6EmptyLatent,
    "Kandinsky6ImageToVideoAudio": Kandinsky6ImageToVideoAudio,
    "Kandinsky6RemoveReferenceLatent": Kandinsky6RemoveReferenceLatent,
    "Kandinsky6AudioVAELoader": Kandinsky6AudioVAELoader,
    "Kandinsky6AudioVAEDecode": Kandinsky6AudioVAEDecode,
    "Kandinsky6TextEncode": Kandinsky6TextEncode,
    "Kandinsky6MagCache": Kandinsky6MagCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky6BeautifyPrompt": "Kandinsky 6 Beautify Prompt (Qwen3.5-9B)",
    "Kandinsky6Sampler": "Kandinsky 6 Sampler",
    "Kandinsky6EmptyLatent": "Kandinsky 6 Empty Latent (video+audio)",
    "Kandinsky6ImageToVideoAudio": "Kandinsky 6 Image to Video+Audio",
    "Kandinsky6RemoveReferenceLatent": "Kandinsky 6 Remove I2VA Reference",
    "Kandinsky6AudioVAELoader": "Kandinsky 6 Audio VAE Loader",
    "Kandinsky6AudioVAEDecode": "Kandinsky 6 Audio VAE Decode",
    "Kandinsky6TextEncode": "Kandinsky 6 Text Encode (video+audio)",
    "Kandinsky6MagCache": "Kandinsky 6 MagCache (Pro)",
}
