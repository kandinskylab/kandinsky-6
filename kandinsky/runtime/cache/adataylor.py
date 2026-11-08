"""AdaTaylorCache: adaptive-threshold residual skip with optional first-order Taylor extrapolation.

Implements the EasyCacheController strategy from TeleFuser, lifted to the whole-DiT
level (same abstraction layer as MagCache) rather than the per-block level used in the
original patch.  Each denoising forward is treated as one "block" call — the skip
decision compares the adaptive accumulated-error signal to a threshold.

Algorithm (per CFG lane):
  On every forward call:
    acc += k * mean_abs(x_in - prev_in) / max(mean_abs(last_out), 1e-8)
    prev_in ← x_in
    if not must_compute AND residual is not None AND acc < thresh AND hits < max_hits:
        hits += 1
        return x_in + residual  [+ linear extrapolation if extrap=True]
    else:
        k ← mean_abs(output - last_out) / max(mean_abs(x_in - last_in), 1e-8)
        last_in ← x_in ; last_out ← output
        prev_residual ← residual ; residual ← output − x_in
        acc = 0 ; hits = 0

  must_compute = cnt < retain  OR  cnt >= num_steps − cooldown

  Linear extrapolation (extrap=True):
    extra = (residual − prev_residual) * (cnt − last_full_step) / (last_full_step − prev_full_step)
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import ConfigModel


class AdaTaylorConfig(ConfigModel):
    thresh: float = 0.3
    retain: int = 15     # leading DiT forward calls that always compute
    cooldown: int = 2    # trailing DiT forward calls that always compute
    max_hits: int = 2    # maximum consecutive skips
    extrap: bool = True  # enable first-order Taylor extrapolation of the cached residual


def _mean_abs(tensors: list[Tensor]) -> float:
    """Scalar mean absolute value averaged across all tensors in the list."""
    return sum(float(t.abs().mean(dtype=torch.float32)) for t in tensors) / len(tensors)


class AdaTaylorCache:
    """AdaTaylor skip mixin used by :class:`~kandinsky.runtime.cache.CacheDiT`."""

    def _reset_ada_state(self) -> None:
        if self._ada is None:
            return
        self._ada.update(
            {
                "cnt": 0,
                "k": [1.0, 1.0],
                "acc": [0.0, 0.0],
                "hits": [0, 0],
                "prev_in": [None, None],
                "last_in": [None, None],
                "last_out": [None, None],
                "residual": [None, None],
                "prev_residual": [None, None],
                "full_steps": [[], []],
                "computed_forwards": 0,
                "skipped_forwards": 0,
                "last_generation_stats": None,
            }
        )

    # ------------------------------------------------------------------
    # Skip logic
    # ------------------------------------------------------------------

    def _ada_should_skip(self, x_in_list: list[Tensor]) -> tuple[bool, list[Tensor] | None]:
        """Accumulate adaptive error and decide whether to skip.

        Updates ``acc`` and ``prev_in`` unconditionally.  Returns ``(skip, residual)``.
        ``residual`` is a list of tensors parallel to ``x_in_list``, or ``None``.
        """
        st = self._ada
        assert st is not None
        cnt = st["cnt"]
        lane = cnt % 2
        total = st["num_steps"]

        must_compute = cnt < st["retain"] or cnt >= total - st["cooldown"]
        residual = st["residual"][lane]
        prev_in = st["prev_in"][lane]
        last_out = st["last_out"][lane]

        if prev_in is not None and last_out is not None:
            delta_in = _mean_abs([xi - xp for xi, xp in zip(x_in_list, prev_in, strict=True)])
            denom = max(_mean_abs(last_out), 1e-8)
            st["acc"][lane] += st["k"][lane] * delta_in / denom

        st["prev_in"][lane] = [xi.clone() for xi in x_in_list]

        can_skip = (
            not must_compute
            and residual is not None
            and st["acc"][lane] < st["thresh"]
            and st["hits"][lane] < st["max_hits"]
        )
        return can_skip, residual

    def _ada_on_skip(self, lane: int) -> None:
        self._ada["hits"][lane] += 1

    def _ada_on_compute(
        self,
        lane: int,
        x_in_list: list[Tensor],
        out_list: list[Tensor],
    ) -> None:
        st = self._ada
        assert st is not None
        last_in = st["last_in"][lane]
        last_out = st["last_out"][lane]
        if last_in is not None and last_out is not None:
            d_out = _mean_abs([o - lo for o, lo in zip(out_list, last_out, strict=True)])
            d_in = max(_mean_abs([xi - li for xi, li in zip(x_in_list, last_in, strict=True)]), 1e-8)
            st["k"][lane] = d_out / d_in
        st["prev_residual"][lane] = st["residual"][lane]
        st["residual"][lane] = [o - xi for o, xi in zip(out_list, x_in_list, strict=True)]
        st["last_in"][lane] = [xi.clone() for xi in x_in_list]
        st["last_out"][lane] = [o.clone() for o in out_list]
        st["full_steps"][lane].append(st["cnt"])
        st["acc"][lane] = 0.0
        st["hits"][lane] = 0

    def _ada_extrapolate(self, lane: int, x_in_list: list[Tensor]) -> list[Tensor]:
        st = self._ada
        assert st is not None
        residual = st["residual"][lane]
        prev_residual = st["prev_residual"][lane]
        full_steps = st["full_steps"][lane]
        base = [xi + r for xi, r in zip(x_in_list, residual, strict=True)]
        if not st["extrap"] or prev_residual is None or len(full_steps) < 2:
            return base
        i1 = full_steps[-1]
        i0 = full_steps[-2]
        scale = (st["cnt"] - i1) / max(i1 - i0, 1)
        return [
            b + scale * (r - pr)
            for b, r, pr in zip(base, residual, prev_residual, strict=True)
        ]

    def _ada_advance(self) -> None:
        st = self._ada
        assert st is not None
        st["cnt"] += 2 if st["no_cfg"] else 1
        if st["cnt"] >= st["num_steps"]:
            st["last_generation_stats"] = {
                "computed_forwards": st["computed_forwards"],
                "skipped_forwards": st["skipped_forwards"],
                "total_forwards": st["computed_forwards"] + st["skipped_forwards"],
            }
            st["cnt"] = 0
            st["k"] = [1.0, 1.0]
            st["acc"] = [0.0, 0.0]
            st["hits"] = [0, 0]
            st["prev_in"] = [None, None]
            st["last_in"] = [None, None]
            st["last_out"] = [None, None]
            st["residual"] = [None, None]
            st["prev_residual"] = [None, None]
            st["full_steps"] = [[], []]
            st["computed_forwards"] = 0
            st["skipped_forwards"] = 0

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _adataylor_forward(
        self,
        x_video: Tensor | None,
        x_audio: Tensor | None,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        dit = self.module

        # Audio-only pass: no skip (same parity as MagCache).
        if x_video is None:
            return dit(
                x_video=None,
                x_audio=x_audio,
                text_embed=text_embed,
                pooled_text_embed=pooled_text_embed,
                time=time,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                attention_mask=attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        both = x_audio is not None and bool(getattr(dit, "is_multimodal", False))
        if both:
            return self._adataylor_forward_fused(
                x_video, x_audio, text_embed, pooled_text_embed, time,
                visual_rope, audio_rope, text_rope, attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )
        return self._adataylor_forward_video(
            x_video, text_embed, pooled_text_embed, time,
            visual_rope, text_rope, attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )

    def _adataylor_forward_fused(
        self,
        x_video: Tensor,
        x_audio: Tensor,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        """T2VA: skip text+visual blocks on cache hit; project via ``_time_only``."""
        dit = self.module
        st = self._ada
        assert st is not None
        attn_mask = dit._normalize_attn_mask(attention_mask) if hasattr(dit, "_normalize_attn_mask") else attention_mask

        te_v, pe_v = (
            (text_embed[0], pooled_text_embed[0]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        te_a, pe_a = (
            (text_embed[1], pooled_text_embed[1]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        rope_v, rope_a = (text_rope[0], text_rope[1]) if isinstance(text_rope, list) else (text_rope, text_rope)
        t_v, t_a = (time[0], time[1]) if isinstance(time, list) else (time, time)

        vis_embed, vis_shape, vis_rope = dit._embed_visual(
            x_video, visual_rope, visual_token_type_ids=visual_token_type_ids,
        )
        aud_embed, aud_rope = dit._embed_audio(x_audio, audio_rope)

        lane = st["cnt"] % 2
        x_in_list = [vis_embed, aud_embed]
        skip, _ = self._ada_should_skip(x_in_list)

        if skip:
            st["skipped_forwards"] += 1
            self._ada_on_skip(lane)
            out_list = self._ada_extrapolate(lane, x_in_list)
            video_tm = dit._time_only("video", pe_v, t_v)
            audio_tm = dit._time_only("audio", pe_a, t_a)
            self._ada_advance()
            return dit._project_fused(out_list[0], out_list[1], vis_shape, video_tm, audio_tm)

        st["computed_forwards"] += 1
        video_te, video_tm = dit._encode_text("video", te_v, pe_v, t_v, rope_v, attn_mask)
        audio_te, audio_tm = dit._encode_text("audio", te_a, pe_a, t_a, rope_a, attn_mask)
        vis_out, aud_out = dit._run_visual_blocks_fused(
            vis_embed, aud_embed, video_te, audio_te, video_tm, audio_tm, vis_rope, aud_rope, attn_mask,
        )
        self._ada_on_compute(lane, x_in_list, [vis_out, aud_out])
        self._ada_advance()
        return dit._project_fused(vis_out, aud_out, vis_shape, video_tm, audio_tm)

    def _adataylor_forward_video(
        self,
        x_video: Tensor,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        """T2V / multimodal video-only: skip text+visual blocks on cache hit."""
        dit = self.module
        st = self._ada
        assert st is not None
        attn_mask = dit._normalize_attn_mask(attention_mask) if hasattr(dit, "_normalize_attn_mask") else attention_mask

        te_in = text_embed[0] if isinstance(text_embed, list) else text_embed
        pe_in = pooled_text_embed[0] if isinstance(pooled_text_embed, list) else pooled_text_embed
        rope_in = text_rope[0] if isinstance(text_rope, list) else text_rope
        t_in = time[0] if isinstance(time, list) else time
        prefix = "video" if getattr(dit, "is_multimodal", False) else None

        vis_embed, vis_shape, vis_rope = dit._embed_visual(
            x_video, visual_rope, visual_token_type_ids=visual_token_type_ids,
        )

        lane = st["cnt"] % 2
        x_in_list = [vis_embed]
        skip, _ = self._ada_should_skip(x_in_list)

        if skip:
            st["skipped_forwards"] += 1
            self._ada_on_skip(lane)
            out_list = self._ada_extrapolate(lane, x_in_list)
            tm = dit._time_only(prefix, pe_in, t_in)
            self._ada_advance()
            return dit._project_video(out_list[0], vis_shape, tm)

        st["computed_forwards"] += 1
        if prefix is not None:
            te, tm = dit._encode_text(prefix, te_in, pe_in, t_in, rope_in, attn_mask)
        else:
            te, tm = dit._encode_t2v(te_in, pe_in, t_in, rope_in, attn_mask)
        vis_out, _ = dit._run_visual_blocks_single(vis_embed, None, te, tm, vis_rope, None, attn_mask)
        self._ada_on_compute(lane, x_in_list, [vis_out])
        self._ada_advance()
        return dit._project_video(vis_out, vis_shape, tm)
