"""
File: Base.py
Class: BaseEffect
Description: Abstract base class for effect types 
Child Classes: FadeInEffect, FadeOutEffect, ColorGradeEffect, ResizeEffect
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import copy

class BaseEffect(ABC):
    def __init__(self):
        # no shared state at the base level
        pass 
    
    # main API: shared apply method (CONCRETE) - handles both RGB and RGBA input
    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        if frame.shape[2] == 4:
            rgb = self._apply_rgb_frame(frame[:, :, :3], t) # if RGBA (LAYER LEVEL), only pass rgb part to subclass apply
            return np.concatenate([rgb, frame[:, :, 3:4]], axis=2) # return transformed rgb with original alpha channel
        
        return self._apply_rgb_frame(frame,t) # if rgb (CLIP LEVEL), just pass frame directly

    # acc apply rgb frame handled differently by effect subclasses
    @abstractmethod
    def _apply_rgb_frame(self, frame: np.ndarray, t:float) -> np.ndarray:
        pass

    # method returning a copy of the effect, needed for _copy on clips (for indep copies of its effects instead of shared refs) 
    # DEFAULT: SHALLOW COPY (creates a new object but shares the same internal attribute refs, like duration, etc for FadeIn)
    # NOTE: needs to be overwritten to be a DEEP COPY by subclasses w mutable state that shouldn't share internal attributes (e.g. MotionTrailEffect which keeps a buffer attribute of recent frames, i.e. not stateless)
    def clone(self) -> BaseEffect:
        return copy.copy(self) # shallow
