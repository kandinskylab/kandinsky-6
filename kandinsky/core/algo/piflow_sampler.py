"""Multimodal π-Flow sampler for distilled base Kandinsky checkpoints."""

from __future__ import annotations

import torch

from ..types import LatentBundle, TextEmbeds
from .generation import ImageCondition
from .piflow_math import MultimodalDXPolicy, policy_rollout_fm, shift_timesteps
from .temporal.trajectory import TemporalTrajectory


def _normalize_grid_prediction(prediction: torch.Tensor, state: torch.Tensor, name: str) -> torch.Tensor:
    """Return predictions as ``[B, N, ...state-shape[1:]]``."""
    if tuple(prediction.shape) == tuple(state.shape):
        return prediction.unsqueeze(1)
    if prediction.ndim == state.ndim:
        raise ValueError(
            f"{name} π-Flow prediction must preserve the state shape when no grid is present: "
            f"prediction={tuple(prediction.shape)}, state={tuple(state.shape)}"
        )
    if prediction.ndim != state.ndim + 1 or prediction.shape[0] != state.shape[0]:
        raise ValueError(
            f"{name} π-Flow prediction must have shape [B,N,*state.shape[1:]], "
            f"got prediction={tuple(prediction.shape)}, state={tuple(state.shape)}"
        )
    if tuple(prediction.shape[2:]) == tuple(state.shape[1:]):
        return prediction

    # Accept the legacy grid-last layout emitted by older distilled wrappers.
    if (
        prediction.shape[-1] == state.shape[-1]
        and tuple(prediction.shape[1:-2]) == tuple(state.shape[1:-1])
    ):
        return prediction.movedim(-2, 1)
    raise ValueError(
        f"{name} π-Flow prediction has incompatible grid layout: "
        f"prediction={tuple(prediction.shape)}, state={tuple(state.shape)}"
    )


@torch.no_grad()
def piflow_denoise_loop(  # noqa: PLR0913, PLR0915
    bundle: LatentBundle,
    dit: torch.nn.Module,
    text_embeds: TextEmbeds,
    visual_rope: torch.Tensor | None,
    audio_rope: torch.Tensor | None,
    text_rope: torch.Tensor | list[torch.Tensor],
    num_steps: int,
    scheduler_scale: float,
    image: ImageCondition | None = None,
    sample_video: bool = True,
    sample_audio: bool = True,
    *,
    attention_mask: torch.Tensor | None = None,
    visual_token_type_ids: torch.Tensor | None = None,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = 1e-6,
    final_step_size_scale: float = 0.5,
    shift: float | None = None,
    num_policy_substeps: int = 128,
    progress_callback=None,
    temporal: TemporalTrajectory | None = None,
    dit_kwargs: dict[str, object] | None = None,
) -> LatentBundle:
    """Run the reference multimodal DX/π-Flow schedule without CFG."""
    del scale_factor
    video = bundle.video
    audio = bundle.audio
    if video is None or audio is None:
        raise ValueError("base-model PiFlow requires both video and audio latents")
    if not getattr(dit, "is_multimodal", False):
        raise ValueError("base-model PiFlow requires a multimodal transformer")
    if bundle.video_cu_seqlens is None or bundle.audio_cu_seqlens is None:
        raise ValueError("base-model PiFlow requires video and audio sequence offsets")
    if num_steps < 1:
        raise ValueError(f"piflow_denoise_loop requires num_steps >= 1, got {num_steps}")
    if num_policy_substeps < 1:
        raise ValueError(f"piflow_denoise_loop requires num_policy_substeps >= 1, got {num_policy_substeps}")

    device = video.device
    video_cu_seqlens = bundle.video_cu_seqlens.to(device=device)
    audio_cu_seqlens = bundle.audio_cu_seqlens.to(device=device)
    shift = scheduler_scale if shift is None else shift
    final_step_size_scale = max(float(final_step_size_scale), eps)
    one_minus_final = 1.0 - final_step_size_scale
    base_segment_size = 1.0 / (float(num_steps) - one_minus_final)
    batch_size = video_cu_seqlens.shape[0] - 1
    raw_t_src = torch.ones(batch_size, device=device, dtype=torch.float32)
    image = image if image is not None else ImageCondition.parse("pretrain")

    video_time_scale = 1000.0 if sample_video else 0.0
    audio_time_scale = 1000.0 if sample_audio else 0.0

    full_video_offsets = video_cu_seqlens
    full_visual_rope = visual_rope
    dit_kwargs = dit_kwargs or {}
    for step_id in range(num_steps):
        is_final_step = step_id == num_steps - 1
        segment_size = base_segment_size * (final_step_size_scale if is_final_step else 1.0)
        tau_src = raw_t_src
        tau_dst = (tau_src - segment_size).clamp(min=eps)
        sigma_t_src = shift_timesteps(tau_src, shift)
        extra_dit_args = dit_kwargs
        video_step = None
        if temporal is not None:
            q_dst = shift_timesteps(tau_dst, shift)
            video_step = temporal.before_step(
                step_id, video, full_video_offsets, full_visual_rope, sigma_t_src, q_dst,
                tau_src, tau_dst, shift,
            )
            video, video_cu_seqlens = video_step.video, video_step.offsets
            visual_rope = video_step.rope
            if video_step.cross_rope is not None:
                extra_dit_args = {**dit_kwargs, "av_cross_rope": video_step.cross_rope}
        model_input_v = image.apply(
            video,
            has_visual_cond=bool(dit.visual_cond),
            cu_seqlens=video_cu_seqlens,
        )
        pred_video, pred_audio = dit(
            x_video=model_input_v,
            x_audio=audio,
            text_embed=text_embeds["text_embeds"],
            pooled_text_embed=text_embeds["pooled_embed"],
            time=[
                (sigma_t_src if video_step is None else video_step.sigma) * video_time_scale,
                sigma_t_src * audio_time_scale,
            ],
            visual_rope=visual_rope,
            audio_rope=audio_rope,
            text_rope=text_rope,
            attention_mask=attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            **extra_dit_args,
        )
        pred_video = _normalize_grid_prediction(pred_video, video, "video")
        pred_audio = _normalize_grid_prediction(pred_audio, audio, "audio")

        video_dim = pred_video.shape[-1]
        audio_dim = pred_audio.shape[-1]
        video_lengths = torch.diff(video_cu_seqlens)
        audio_lengths = torch.diff(audio_cu_seqlens)
        if video.ndim == 5:
            video_sigma = sigma_t_src.reshape(batch_size, *((video.dim() - 1) * [1]))
            audio_sigma = sigma_t_src.reshape(batch_size, *((audio.dim() - 1) * [1]))
            video_tau_src, video_tau_dst = tau_src, tau_dst
            audio_tau_src, audio_tau_dst = tau_src, tau_dst
        else:
            video_sigma = sigma_t_src.repeat_interleave(video_lengths).reshape(video.shape[0], *((video.dim() - 1) * [1]))
            audio_sigma = sigma_t_src.repeat_interleave(audio_lengths).reshape(audio.shape[0], *((audio.dim() - 1) * [1]))
            video_tau_src = tau_src.repeat_interleave(video_lengths)
            video_tau_dst = tau_dst.repeat_interleave(video_lengths)
            audio_tau_src = tau_src.repeat_interleave(audio_lengths)
            audio_tau_dst = tau_dst.repeat_interleave(audio_lengths)
        video_segment = torch.full(
            (batch_size if video.ndim == 5 else video.shape[0],),
            float(segment_size), device=device, dtype=sigma_t_src.dtype
        ).reshape(video.shape[0], *((video.dim() - 1) * [1]))
        audio_segment = torch.full(
            (batch_size if audio.ndim == 3 else audio.shape[0],),
            float(segment_size), device=device, dtype=sigma_t_src.dtype
        ).reshape(audio.shape[0] if audio.ndim == 2 else batch_size, *((audio.dim() - 1) * [1]))

        if video_step is not None and video_step.segment is not None:
            video_sigma = video_step.sigma.reshape_as(video_sigma)
            video_tau_src, video_tau_dst = video_step.tau_src, video_step.tau_dst
            video_segment = video_step.segment.reshape_as(video_segment)

        policy = MultimodalDXPolicy(
            denoising_output_v=pred_video,
            x_t_src_v=video[..., :video_dim],
            denoising_output_a=pred_audio,
            x_t_src_a=audio[..., :audio_dim],
            sigma_t_v_src=video_sigma,
            segment_v_size=video_segment,
            sigma_t_a_exp=audio_sigma,
            segment_a_size=audio_segment,
            shift=shift,
            mode="grid",
            eps=eps,
        ).detach()
        if temporal is not None and temporal.chosen is None:
            video_state = video[..., :video_dim]
            velocity = policy.policy_V.pi(video_state, video_sigma)
            temporal.observe(step_id, video_state, velocity, video_sigma, sigma_t_src - q_dst)
        if sample_video:
            video, _, _ = policy_rollout_fm(
                video[..., :video_dim],
                video_sigma,
                video_tau_src,
                video_tau_dst,
                num_policy_substeps,
                policy.policy_V,
            )
        if sample_audio:
            audio, _, _ = policy_rollout_fm(
                audio[..., :audio_dim],
                audio_sigma,
                audio_tau_src,
                audio_tau_dst,
                num_policy_substeps,
                policy.policy_A,
            )
        raw_t_src = tau_dst
        image.restore_after_step(video, video_cu_seqlens)
        if progress_callback is not None:
            progress_callback()

    image.pin(video, video_cu_seqlens)
    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=video_cu_seqlens,
        audio_cu_seqlens=audio_cu_seqlens,
    )
