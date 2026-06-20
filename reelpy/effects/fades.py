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

    def apply_frame(self, frame, t):
        if frame.shape[2] == 4:
            if t > self.duration:
                return frame
            result = frame.copy()
            alpha_factor = min(t / self.duration, 1.0)
            result[:, :, 3] = (frame[:, :, 3] * alpha_factor).astype(np.uint8)
            return result
        return self._apply_rgb_frame(frame, t)

    def _apply_rgb_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        # early stop if fade is already complete
        if t > self.duration:
            return frame
        # clip context - fade in by brightening RGB with time
        alpha_factor = min(t / self.duration, 1.0)
        return (frame * alpha_factor).astype(np.uint8)


class FadeOutEffect(BaseEffect):
    """
    Clip-level: Fades the clip out to black over duration seconds using linear interpolation.
    Layer-level: Fades the layer out from fully opaque to transparent on background over duration seconds.
    """
    def __init__(self, duration: float, clip_duration: float = 0.0):
         self.duration = duration
         self.clip_duration = clip_duration # injected by export() before rendering

    def apply_frame(self, frame, t):
        t_fade_start = self.clip_duration - self.duration
        if frame.shape[2] == 4:
            if t < t_fade_start or t > t_fade_start + self.duration:
                return frame
            result = frame.copy()
            alpha_factor = min((t - t_fade_start) / self.duration, 1.0)
            result[:, :, 3] = (frame[:, :, 3] * (1 - alpha_factor)).astype(np.uint8)
            return result
        return self._apply_rgb_frame(frame, t)

    def _apply_rgb_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        t_fade_start = self.clip_duration - self.duration
        # early stop if fade is not yet happening or already complete, past last frame
        if t < t_fade_start or t > t_fade_start + self.duration:
            return frame
        alpha_factor = min((t-t_fade_start) / self.duration, 1.0) # linear time step over duration (t normalized from 0 to 1 for interp)
        # clip context - fade out by darkening RGB with time
        rgb = (frame * (1-alpha_factor)).astype(np.uint8)
        return rgb

