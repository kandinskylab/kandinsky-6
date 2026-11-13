"""DiT cache policy. Each strategy lives in its own module.

Never monkey-patches ``DiffusionTransformer3D.forward``. The bare module stays
exportable and compilable via ``CacheDiT.module``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from torch import nn

from ..base import ConfigModel
from .adaptive_block import AdaptiveBlockConfig, AdaptiveBlockState
from .adataylor import AdaTaylorCache, AdaTaylorConfig
from .magcache import MagCache, MagCacheConfig, prepare_mag_ratios, select_mag_ratios
from .navicache import NaviCache, NaviCacheConfig

logger = logging.getLogger(__name__)

CacheMode = Literal["none", "magcache", "navicache", "adataylor", "adaptive_block"]
CacheModeName = CacheMode


class CacheConfig(ConfigModel):
    mode: CacheModeName = "none"
    magcache: MagCacheConfig | None = None
    navicache: NaviCacheConfig | None = None
    adataylor: AdaTaylorConfig | None = None
    adaptive_block: AdaptiveBlockConfig | None = None


class CacheDiT(MagCache, NaviCache, AdaTaylorCache, nn.Module):
    """Facade around a bare DiT with switchable MagCache / NaviCache / AdaTaylorCache / none."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.module = dit
        self.mode: CacheMode = "none"
        self._mag: dict | None = None
        self._navi: dict | None = None
        self._ada: dict | None = None
        self._adaptive_block: AdaptiveBlockConfig | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_cache(
        self,
        mode: CacheMode | None,
        *,
        num_steps: int | None = None,
        no_cfg: bool = False,
        # MagCache
        mag_ratios: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        mode_name: str | None = None,
        thresh: float | None = None,
        K: int | None = None,  # noqa: N803  # MagCache paper hyperparameter name
        retention_ratio: float | None = None,
        # NaviCache
        align_steps: int | None = None,
        process_noise: float | None = None,
        measurement_noise: float | None = None,
        # AdaTaylorCache
        retain: int | None = None,
        cooldown: int | None = None,
        max_hits: int | None = None,
        extrap: bool | None = None,
        adaptive_block: AdaptiveBlockConfig | None = None,
    ) -> None:
        if mode is None:
            mode = "none"
        if mode not in ("none", "magcache", "navicache", "adataylor", "adaptive_block"):
            raise ValueError(f"Unknown cache mode: {mode!r}")

        if mode == "adaptive_block":
            if not getattr(self.module, "is_multimodal", False) or len(self.module.visual_transformer_blocks) != 60:
                raise ValueError("Adaptive Block Cache requires 60 fused video/audio blocks")
            self.mode = mode
            self._mag = self._navi = self._ada = None
            self._adaptive_block = adaptive_block if adaptive_block is not None else AdaptiveBlockConfig()
            return

        if mode == "none":
            self._adaptive_block = None
            self.mode = "none"
            self._mag = None
            self._navi = None
            self._ada = None
            return

        if num_steps is None:
            raise ValueError(f"set_cache({mode!r}) requires num_steps")

        if mode == "magcache":
            if mag_ratios is None:
                raise ValueError("set_cache('magcache') requires mag_ratios")
            mag_ratios, mode_name = select_mag_ratios(mag_ratios, mode_name)
            self._navi = None
            self._ada = None
            self._mag = {
                "num_steps": num_steps * 2,
                "gen_mode": mode_name,
                "no_cfg": no_cfg,
                "thresh": 0.12 if thresh is None else thresh,
                "K": 2 if K is None else K,
                "retention_ratio": 0.2 if retention_ratio is None else retention_ratio,
                "mag_ratios": prepare_mag_ratios(mag_ratios, num_steps),
            }
            self.mode = "magcache"
            self._reset_mag_state()
            return

        if mode == "adataylor":
            self._mag = None
            self._navi = None
            self._ada = {
                "num_steps": num_steps * 2,
                "no_cfg": no_cfg,
                "thresh": 0.3 if thresh is None else thresh,
                "retain": 15 if retain is None else retain,
                "cooldown": 2 if cooldown is None else cooldown,
                "max_hits": 2 if max_hits is None else max_hits,
                "extrap": True if extrap is None else extrap,
            }
            self.mode = "adataylor"
            self._reset_ada_state()
            return

        # navicache
        if not getattr(self.module, "is_multimodal", False):
            raise NotImplementedError("NaviCache is only implemented for the multimodal (video+audio) T2VA DiT.")
        self._mag = None
        self._ada = None
        self._navi = {
            "num_forwards": num_steps * 2,
            "num_steps": num_steps,
            "no_cfg": no_cfg,
            "thresh": 0.05 if thresh is None else thresh,
            "align_forwards": (10 if align_steps is None else align_steps) * 2,
            "cutoff_forwards": num_steps * 2 - 2,
            "process_noise": 0.05 if process_noise is None else process_noise,
            "measurement_noise": 0.05 if measurement_noise is None else measurement_noise,
        }
        self.mode = "navicache"
        self._reset_navi_state()

    def align_schedule(self, num_steps: int, generation_mode: str, *, apply: Callable[..., None]) -> None:
        """Rebuild MagCache or NaviCache when this call's step count or mode differs.

        ``apply`` is the pipeline ``set_cache``. The doubled MagCache counter stays
        inside this class. ``num_steps`` passed to ``apply`` is the schedule length.
        """
        if self.mode == "magcache" and self._mag is not None:
            steps_changed = self._mag["num_steps"] != num_steps * 2
            mode_changed = self._mag["gen_mode"] is not None and generation_mode != self._mag["gen_mode"]
            if mode_changed:
                logger.warning(
                    "MagCache uses different ratios per mode for this model. Generation mode changed: %s",
                    generation_mode,
                )
            if steps_changed or mode_changed:
                apply(
                    "magcache",
                    num_steps=num_steps,
                    mag_ratios=self._mag["mag_ratios"],
                    mode_name=generation_mode,
                    thresh=self._mag["thresh"],
                    K=self._mag["K"],
                    retention_ratio=self._mag["retention_ratio"],
                )
        elif self.mode == "navicache" and self._navi is not None:
            if self._navi["num_steps"] != num_steps:
                apply(
                    "navicache",
                    num_steps=num_steps,
                    thresh=self._navi["thresh"],
                    align_steps=self._navi["align_forwards"] // 2,
                    process_noise=self._navi["process_noise"],
                    measurement_noise=self._navi["measurement_noise"],
                )
        elif self.mode == "adataylor" and self._ada is not None:
            if self._ada["num_steps"] != num_steps * 2:
                apply(
                    "adataylor",
                    num_steps=num_steps,
                    thresh=self._ada["thresh"],
                    retain=self._ada["retain"],
                    cooldown=self._ada["cooldown"],
                    max_hits=self._ada["max_hits"],
                    extrap=self._ada["extrap"],
                )

    def reset_cache_state(self) -> None:
        if self.mode == "magcache":
            self._reset_mag_state()
        elif self.mode == "navicache":
            self._reset_navi_state()
        elif self.mode == "adataylor":
            self._reset_ada_state()

    def invalidate_shape_state(self) -> None:
        """Drop shape-dependent cached tensors on a mid-sampling grid change.

        Used by SPEED between stages: residuals, last-in/out buffers, and any
        other tensor shaped by the previous grid are reset so the next forward
        recomputes from scratch at the new spatial size. Step counters and
        scalar statistics are preserved to keep accounting coherent across
        stages.
        """
        if self.mode == "magcache":
            self._invalidate_mag_shape_state()
        elif self.mode == "navicache":
            self._invalidate_navi_shape_state()
        elif self.mode == "adataylor":
            self._invalidate_ada_shape_state()

    @property
    def last_generation_stats(self) -> dict | None:
        if self.mode == "magcache" and self._mag is not None:
            return self._mag.get("last_generation_stats")
        if self.mode == "navicache" and self._navi is not None:
            return self._navi.get("last_generation_stats")
        if self.mode == "adataylor" and self._ada is not None:
            return self._ada.get("last_generation_stats")
        return None

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def new_adaptive_block_state(self, num_steps: int) -> AdaptiveBlockState | None:
        if self.mode == "adaptive_block" and self._adaptive_block is not None:
            return AdaptiveBlockState(self._adaptive_block, num_steps)
        return None

    def forward(self, *args, **kwargs):
        if self.mode == "adaptive_block":
            adaptive_block_state = kwargs.pop("adaptive_block_state", None)
            if adaptive_block_state is None:
                raise ValueError("Adaptive Block Cache requires a per-generation PiFlow state")
            if adaptive_block_state.next_step(kwargs["x_video"]):
                kwargs["adaptive_block_tail"] = adaptive_block_state.apply
            return self.module(*args, **kwargs)
        if self.mode == "none":
            return self.module(*args, **kwargs)
        if self.mode == "magcache":
            return self._magcache_forward(*args, **kwargs)
        if self.mode == "adataylor":
            return self._adataylor_forward(*args, **kwargs)
        return self._navicache_forward(*args, **kwargs)

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ raises AttributeError for missing attrs;
        # forward attribute lookups to the wrapped DiT (visual_cond, patch_size, …).
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self._modules.get("module") if "_modules" in self.__dict__ else None
            if module is None:
                raise
            return getattr(module, name)

__all__ = (
    "AdaptiveBlockConfig",
    "AdaTaylorConfig",
    "CacheConfig",
    "CacheDiT",
    "CacheMode",
    "CacheModeName",
    "MagCacheConfig",
    "NaviCacheConfig",
    "prepare_mag_ratios",
    "select_mag_ratios",
)
