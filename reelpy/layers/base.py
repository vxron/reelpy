"""
File: base.py
Classes: BaseLayer (RGBA compositing foundation, alpha blending w/ diff blend modes)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self
import numpy as np

# Auto-Naming Counter - Shared by all BaseLayer instances (Global)
# tracks layer count pre class name to auto-generate names like SolidLayer_0, ShapeLayer_1
_layer_counts: dict[str, int] = {} 

class BaseLayer(ABC):

    def __init__(
        self,
        name: str | None = None,
        opacity: float | Callable[[float], float] = 1.0, # 0.0-1.0 or callable of t. scales layer alpha before compositing
        effects: list | None = None,                     # per-layer effects on RGBA temp canvas
        blend_mode: str = "normal",                      # "normal", "add", "multiply", "screen"
        t_start: float | None = None,                    # visibilitiy window start (None=always visible)
        t_end: float | None = None,                      # visibility window end (None=always visible)
    ):
        # determine layer subclass name
        cls_name = type(self).__name__
        # auto-index
        curr_count = _layer_counts.get(cls_name, 0)
        _layer_counts[cls_name] = curr_count + 1 # increment 
        self.name = name if name is not None else f"{cls_name}_{curr_count}"

        self.opacity = opacity
        self.effects = list(effects) if effects is not None else []
        if blend_mode not in ["normal", "add", "multiply", "screen"]:
            raise ValueError("invalid blend_mode argument passed to layer")
        self.blend_mode = blend_mode
        self.t_start = t_start
        self.t_end = t_end

    @abstractmethod
    def _copy(self, **overrides) -> Self:
        pass

    @abstractmethod
    def _draw(self, canvas: np.ndarray, t: float) -> np.ndarray:
        pass

    def apply(self, effect) -> Self:
        # Appends an effect to this layer's per-layer pipeline & returns a new layer via _copy
        new_effects = list(self.effects)
        new_effects.append(effect)
        return self._copy(effects=new_effects)

    def rename(self, name: str) -> Self:
        # Rename layer
        return self._copy(name=name)

    def visible(self, t_start: float, t_end: float) -> Self:
        return self._copy(t_start=t_start, t_end=t_end)

    def render(self, canvas: np.ndarray, t:float) -> np.ndarray:
        # non-abstract orchestrator.

        # early return if outside visibility window
        if (self.t_start is not None and self.t_start > t) or (self.t_end is not None and self.t_end <= t):
            return canvas

        # 1) blank RGBA temp canvas, same H and W as main, but RGBA (4 channels)
        temp = np.zeros((*canvas.shape[:2], 4), dtype=np.uint8)

        # 2) subclass draws into temp, sets alpha=255 on drawn pixels
        temp = self._draw(temp, t)

        # 3) run per-layer effects on RGBA temp canvas
        for effect in self.effects:
            temp = effect.apply_frame(temp,t)

        # 4) resolve opacity - callable or fixed float 
        op = self.opacity(t) if callable(self.opacity) else self.opacity 

        # 5) extract alpha and src RGB (floats needed for blending math)
        alpha = temp[:, :, 3:4].astype(np.float32) / 255.0 * op
        src = temp[:, :, :3].astype(np.float32)
        dest = canvas.astype(np.float32)

        # 6) blend onto main canvas using blend_mode
        if self.blend_mode == "normal":
            out = alpha * src + (1.0 - alpha)*dest
        elif self.blend_mode == "add":
            out = np.clip(dest + alpha*src, 0, 255)
        elif self.blend_mode == "multiply":
            out = alpha * (dest * src / 255.0) + (1.0 - alpha) * dest
        elif self.blend_mode == "screen":
            out = alpha * (255.0 - (255.0 - dest) * (255.0 - src) / 255.0) + (1.0 - alpha) * dest

        return np.clip(out, 0, 255).astype(np.uint8)