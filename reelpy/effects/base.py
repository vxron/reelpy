"""
File: Base.py
Class: BaseEffect
Description: This file defines two classes:
    1) BaseEffect: The root contract every effect must satisfy
    2) RGBEffect: A ready-made template for Category A effects so they don't have to reimplement the RGB/RGBA each time
Child Classes: FadeInEffect, FadeOutEffect, ColorGradeEffect, ResizeEffect
NOTE: Two Categories of Effects classes
    -Category A: RGB-only (ColorGradeEffect, ResizeEffect, etc)
        - do not touch alpha channel (same implementation at clip & layer levels)
        - these effect subclasses inherit from RGBEffect
    -Category B: channel-assymetric (Fade effects)
        - darken RGB at clip-level, but fade alpha channel at layer-level (when rgba available)
        - these effect subclasses inherit from BaseEffect
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import copy

class BaseEffect(ABC):
    """
    Root abstract base for every effect in the system.
    apply_frame is the ONLY required method. 
    """
    def __init__(self):
        pass # no shared state at the base level
    
    @abstractmethod
    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        """
        Contract: Transform a single frame at timestamp t.
        frame is RGB (H, W, 3) at clip-level or RGBA (H, W, 4) at layer-level.
        Must return an array of the SAME shape and dtype as the input.
        """
        pass

    # method returning a copy of the effect, needed for _copy on clips (for indep copies of its effects instead of shared refs) 
    # DEFAULT: SHALLOW COPY (creates a new object but shares the same internal attribute refs, like duration, etc for FadeIn)
    # NOTE: needs to be overwritten to be a DEEP COPY by subclasses w mutable state that shouldn't share internal attributes (e.g. MotionTrailEffect which keeps a buffer attribute of recent frames, i.e. not stateless)
    def clone(self) -> BaseEffect:
        return copy.copy(self) # shallow

class RGBEffect(BaseEffect):
    """
    Template for "Category A" effects that only ever transform RGB and 
    want alpha passed through untouched, automatically.
    HOW TO USE: subclass this instead of BaseEffect, then write _process_rgb()
    function implementation in the subclass, instead of apply_frame(). 
    We'll get splitting in apply_frame for free!
    """
    @abstractmethod
    def _process_rgb(self, rgb: np.ndarray, t:float) -> np.ndarray:
        """
        Transform the RGB channels only. 
        (RGBEffect.apply_frame has alr stripped lpha and will reattach after)
        """
        pass

    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        # concrete - handles the split once, so subclasses never have to
        if frame.shape[2] == 4:
            # layer-level: split off alpha, transform RGB only, reattach
            rgb_out = self._process_rgb(frame[:,:,:3], t)
            return np.concatenate([rgb_out, frame[:,:,3:4]], axis=2)
        # clip-level: no alpha channel to worry about
        return self._process_rgb(frame, t)
