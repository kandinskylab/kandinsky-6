"""Stable algorithm contract for a framework port.

An agent writes a Diffusers, vLLM, SGLang, or ComfyUI shell against these
signatures.  This module re-exports the implementations from
``kandinsky.core.algo`` instead of wrapping or copying them, so the native
pipeline and a port share the same behavior and signatures.

Only device-agnostic K6 data preparation, denoising, and post-processing
primitives belong here.  Target-specific concerns (model loading, device
placement, callbacks, progress reporting, and serialization) stay outside
this module.  Importing the contract does not require Flash-Attention,
MagiCompiler, or another optional acceleration dependency.
"""

from .algo.denoise_loop import denoise_loop
from .algo.flow_matching import flow_match_timesteps
from .algo.guidance import apply_cfg
from .algo.postprocess_audio import postprocess_audio
from .algo.postprocess_video import postprocess_video
from .algo.prepare_latents import (
    audio_latent_duration,
    prepare_audio_latents,
    prepare_video_latents,
)
from .algo.prepare_ropes import compute_rope1d, compute_visual_rope
from .algo.visual_conditioning import (
    append_reference_rope,
    append_visual_references,
    build_tail_condition,
    compute_reference_rope,
    restore_tail_references,
)
from .types import LatentBundle, TextEmbeds

__all__ = (
    "LatentBundle",
    "TextEmbeds",
    "append_reference_rope",
    "append_visual_references",
    "apply_cfg",
    "audio_latent_duration",
    "build_tail_condition",
    "compute_reference_rope",
    "compute_rope1d",
    "compute_visual_rope",
    "denoise_loop",
    "flow_match_timesteps",
    "postprocess_audio",
    "postprocess_video",
    "prepare_audio_latents",
    "prepare_video_latents",
    "restore_tail_references",
)
