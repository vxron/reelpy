"""
File: fades.py
Classes: FadeInEffect, FadeOutEffect
Description: "Category B" effects (see base.py) - these subclass BaseEffect directly
"""

import numpy as np
from reelpy.effects.base import BaseEffect

class FadeInEffect(BaseEffect):
    """
    Clip-level (RGB): Fades the clip in from black over duration seconds using linear interpolation.
    Layer-level (RGBA): Fades the layer in from transparent to fully opaque over duration seconds.
    """
    def __init__(self, duration: float):
         self.duration = duration

    def _fade_rgb(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Clip-level path: darken RGB pixel values toward black at t=0"""
        alpha_factor = min(t / self.duration, 1.0)
        return (frame*alpha_factor).astype(np.uint8)
    
    def _fade_alpha(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Layer-level path: scale the alpha channel only, RGB untouched."""
        result = frame.copy()
        alpha_factor = min(t/self.duration, 1.0)
        result[:,:,3] = (frame[:,:,3] * alpha_factor).astype(np.uint8)
        return result
    
    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        if t > self.duration:
            return frame # fade already complete
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            return self._fade_alpha(frame, t)
        else:
            return self._fade_rgb(frame, t)


class FadeOutEffect(BaseEffect):
    """
    Clip-level: Fades the clip out to black over duration seconds using linear interpolation.
    Layer-level: Fades the layer out from fully opaque to transparent on background over duration seconds.
    """
    def __init__(self, duration: float, clip_duration: float = 0.0):
         self.duration = duration
         self.clip_duration = clip_duration # injected by export() before rendering

    def _fade_rgb(self, frame, t, t_fade_start):
        alpha_factor = min((t-t_fade_start) / self.duration, 1.0)
        return (frame * (1-alpha_factor)).astype(np.uint8)

    def _fade_alpha(self, frame: np.ndarray, t: float, t_fade_start) -> np.ndarray:
        result = frame.copy()
        alpha_factor = min((t-t_fade_start) / self.duration, 1.0)
        result[:, :, 3] = (frame[:, :, 3] * (1 - alpha_factor)).astype(np.uint8)
        return result

    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        t_fade_start = self.clip_duration - self.duration 
        if t < t_fade_start or t > self.clip_duration:
            return frame # outside the fade window - pass through
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            return self._fade_alpha(frame, t, t_fade_start)
        else:
            return self._fade_rgb(frame, t, t_fade_start)

