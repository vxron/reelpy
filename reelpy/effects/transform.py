"""
File: transform.py
Classes: ResizeEffect
         - Mode ABSOLUTE warps; ResizeEffect(width=1920, height=1080)
         - Mode SCALE maintains aspect ratio; ResizeEffect(scale=0.5) 
         - Mode FIT fits within certain box while maintaining aspect ratio; ResizeEffect(w=,h=,mode="fit")
"""

import cv2
import numpy as np
from reelpy.effects.base import BaseEffect

class ResizeEffect(BaseEffect):
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        scale: float | None = None, 
        interpolation: int = cv2.INTER_LINEAR # default to bilinear interpolation
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.interpolation = interpolation

        # make sure exactly one of (width,height) OR scale is provided
        self.absolute_mode = (width is not None or height is not None) # can infer from one dim by maintaining aspct ratio
        self.scale_mode = (scale is not None)
        if self.absolute_mode == self.scale_mode:
            # both on or both off = not allowed!
            raise ValueError("Invalid parameters passed to ResizeEffect. Mode undetermined.")
        if self.absolute_mode and (
            (width is not None and width <= 0) or 
            (height is not None and height <= 0)
        ):
            raise ValueError("Invalid width or height parameter in ResizeEffect.")
        if self.scale_mode and scale <= 0:
            raise ValueError("Invalid scale passed to ResizeEffect.")
        
    def apply_frame(self, frame, t):
        # (1) compute targt dimensions
        h, w = frame.shape[:2] # current dimensions
        if self.scale_mode:
            target_w = round(w*self.scale)
            target_h = round(h*self.scale)
        elif self.absolute_mode: # maintain aspect ratio if only one param given
            target_w = self.width if self.width is not None else round(w * (self.height/h))
            target_h = self.height if self.height is not None else round(h * (self.width/w))
        # (2) early return if no change needed
        if target_w == w and target_h == h:
            return frame
        # (3) resize
        resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=self.interpolation)
        return resized_frame

