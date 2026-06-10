"""
File: fades.py
Classes: FadeInEffect, FadeOutEffect
"""

import numpy as np
from reelpy.effects.base import BaseEffect

class FadeInEffect(BaseEffect):
    """
    Clip-level: Fades the clip in from black over duration seconds using linear interpolation.
    Layer-level: Fades the layer in from transparent to fully opaque over duration seconds.
    """
    def __init__(self, duration: float):
         self.duration = duration

    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        # early stop if fade is already complete
        if t > self.duration:
            return frame
        
        # check shape of frame (RGB for global effects and RGBA for per-layer effects)
        has_alpha = frame.shape[2] == 4  # frame.shape gives (H,W,#channels)
        alpha_factor = min(t / self.duration, 1.0) # linear time step over duration (t normalized from 0 to 1 for interp)
        if has_alpha:
            # layer context - fade by scaling alpha, leave RGB untouched
            result = frame.copy() 
            result[:, :, 3] = (frame[:, :, 3] * alpha_factor).astype(np.uint8) # scales in alpha channel [3rd in RGBA]
            return result
        else:
            # clip context - fade in by brightening RGB with time
            rgb = (frame * alpha_factor).astype(np.uint8)
            return rgb


class FadeOutEffect(BaseEffect):
    """
    Clip-level: Fades the clip out to black over duration seconds using linear interpolation.
    Layer-level: Fades the layer out from fully opaque to transparent on background over duration seconds.
    """
    def __init__(self, duration: float, clip_duration: float = 0.0):
         self.duration = duration
         self.clip_duration = clip_duration # injected by export() before rendering

    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        t_fade_start = self.clip_duration - self.duration
        # early stop if fade is not yet happening or already complete, past last frame
        if t < t_fade_start or t > t_fade_start + self.duration:
            return frame
        
        # check shape of frame (RGB for global effects and RGBA for per-layer effects)
        has_alpha = frame.shape[2] == 4  # frame.shape gives (H,W,#channels)
        alpha_factor = min((t-t_fade_start) / self.duration, 1.0) # linear time step over duration (t normalized from 0 to 1 for interp)
        if has_alpha:
            # layer context - fade by scaling alpha, leave RGB untouched
            result = frame.copy() 
            result[:, :, 3] = (frame[:, :, 3] * (1-alpha_factor)).astype(np.uint8) # scales in alpha channel [3rd in RGBA]
            return result
        else:
            # clip context - fade out by darkening RGB with time
            rgb = (frame * (1-alpha_factor)).astype(np.uint8)
            return rgb

