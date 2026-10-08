"""Native ComfyUI adapter for the joint Kandinsky 6 video/audio model."""

from __future__ import annotations

import math

import torch

import comfy.conds
import comfy.model_base
import comfy.nested_tensor

from .core_contract import DIT_CONFIG
from .ldm.model import Kandinsky6 as Kandinsky6DiT


class Kandinsky6NativeAVDiT(Kandinsky6DiT):
    """Adapt ComfyUI's generic multimodal call to the canonical K6 call.

    ComfyUI unpacks a nested ``LATENT`` before calling the diffusion model, so
    ``x`` is a two-item sequence containing the video and audio streams. The
    The released K6 T2VA checkpoint uses the core contract's noisy video
    channels and, when enabled, appends a zero visual condition plus mask.
    """

    def _legacy_forward(self, *args, **kwargs):
        return super()._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        timestep,
        context,
        y=None,
        k6_audio_context=None,
        k6_audio_pooled_output=None,
        control=None,
        transformer_options=None,
        freqs_scaling=None,
        k6_reference_tail=False,
        visual_token_type_ids=None,
        **kwargs,
    ):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "Kandinsky 6 expects a joint LATENT with video and audio streams."
            )

        video, audio = x
        if video.ndim != 5 or video.shape[1] != self.in_visual_dim:
            raise ValueError(
                "Kandinsky 6 expects video latents shaped "
                f"[B, {self.in_visual_dim}, T, H, W], got {tuple(video.shape)}."
            )
        if audio.ndim != 3 or audio.shape[-1] != self.in_audio_dim:
            raise ValueError(
                "Kandinsky 6 expects audio latents shaped "
                f"[B, T, {self.in_audio_dim}], got {tuple(audio.shape)}."
            )
        if video.shape[0] != audio.shape[0]:
            raise ValueError("Kandinsky 6 video and audio batch sizes must match.")
        if y is None:
            raise ValueError("Kandinsky 6 requires CLIP-L pooled conditioning.")

        reference_tail = bool(k6_reference_tail)
        if reference_tail:
            if video.shape[2] < 2:
                raise ValueError(
                    "Kandinsky 6 I2VA requires generated frames plus one reference tail."
                )
            if self.visual_token_type_num_embeddings < 2:
                raise ValueError(
                    "Kandinsky 6 I2VA requires two visual token-type embeddings."
                )
            visual_token_type_ids = torch.zeros(
                video.shape[2], dtype=torch.long, device=video.device
            )
            visual_token_type_ids[-1] = 1

        if bool(DIT_CONFIG["visual_cond"]):
            visual_cond = torch.zeros_like(video)
            visual_mask = torch.zeros_like(video[:, :1])
            if reference_tail:
                visual_mask[:, :, -1] = 1
            video_input = torch.cat((video, visual_cond, visual_mask), dim=1)
        else:
            video_input = video
        transformer_options = {} if transformer_options is None else transformer_options
        audio_context = context if k6_audio_context is None else k6_audio_context
        audio_pooled = y if k6_audio_pooled_output is None else k6_audio_pooled_output

        return self._legacy_forward(
            video_input,
            audio,
            (timestep, timestep),
            context,
            y,
            context_a=audio_context,
            pooled_a=audio_pooled,
            freqs_scaling=freqs_scaling,
            visual_token_type_ids=visual_token_type_ids,
            visual_reference_tail=reference_tail,
            transformer_options=transformer_options,
            **kwargs,
        )


class Kandinsky6(comfy.model_base.Kandinsky5):
    """ComfyUI ``BaseModel`` for a single packed video/audio flow latent."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        comfy.model_base.BaseModel.__init__(
            self,
            model_config,
            model_type,
            device=device,
            unet_model=Kandinsky6NativeAVDiT,
        )

    def concat_cond(self, **kwargs):
        # T2VA's zero visual condition and mask are model-input details. They
        # are added by Kandinsky6NativeAVDiT after ComfyUI unpacks the streams.
        return None

    def _apply_model(self, *args, **kwargs):
        if self.diffusion_model.n_grid > 1:
            raise ValueError(
                "Use Kandinsky6Sampler for distilled Pro/Lite PiFlow; "
                "a stock Euler KSampler cannot integrate DX grids."
            )
        return super()._apply_model(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn")
        if cross_attn is not None:
            # Kandinsky5 uses CONDRegular here. K6 prompts have independent
            # positive/negative token lengths, so use Comfy's standard cross-
            # attention wrapper to keep CFG batching correct and available.
            out["c_crossattn"] = comfy.conds.CONDCrossAttn(cross_attn)
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is not None:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)
        audio_context = kwargs.get("k6_audio_context")
        if audio_context is not None:
            out["k6_audio_context"] = comfy.conds.CONDCrossAttn(audio_context)
        audio_pooled = kwargs.get("k6_audio_pooled_output")
        if audio_pooled is not None:
            out["k6_audio_pooled_output"] = comfy.conds.CONDRegular(audio_pooled)
        reference_tail = kwargs.get("k6_reference_tail")
        if reference_tail is not None:
            out["k6_reference_tail"] = comfy.conds.CONDConstant(
                bool(reference_tail)
            )
        return out

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        """Keep I2VA's masked reference tail clean at every sampler step."""
        return latent_image

    def _process_video_stream(self, latent, process):
        """Apply the Hunyuan-video latent format to video, never to audio."""
        if getattr(latent, "is_nested", False):
            streams = list(latent.unbind())
            if not streams:
                return latent
            streams[0] = process(streams[0])
            return comfy.nested_tensor.NestedTensor(streams)

        shapes = self.latent_shapes
        if shapes is not None and len(shapes) > 1:
            video_elements = math.prod(shapes[0][1:])
            if latent.shape[-1] < video_elements:
                raise ValueError("Packed Kandinsky 6 latent is shorter than its video shape.")
            output = latent.clone()
            video = output[..., :video_elements].reshape(
                [output.shape[0]] + list(shapes[0])[1:]
            )
            video = process(video)
            output[..., :video_elements] = video.reshape(output.shape[0], 1, -1)
            return output

        # A non-packed latent is the video stream (for example when a caller
        # uses BaseModel's conversion helpers outside a sampling run).
        return process(latent)

    def process_latent_in(self, latent):
        return self._process_video_stream(latent, self.latent_format.process_in)

    def process_latent_out(self, latent):
        return self._process_video_stream(latent, self.latent_format.process_out)
