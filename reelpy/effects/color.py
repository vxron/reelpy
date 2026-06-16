"""
File: color.py
Classes: ColorGradeEffect (brightness, contrast, saturation, hue, temperature)
        - Brightness: Default 1.0
        - Contrast: Default 1.0
        - Saturation: Default 1.0
        - Hue: Default 0.0 (180.0 = invert hue)
        - Temperature: Default 0.0 (positive = warmer, negative = cooler)

"""

import numpy as np
from reelpy.effects.base import BaseEffect
import cv2

class ColorGradeEffect(BaseEffect):
    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0, temperature=0.0):
        if brightness < 0 or contrast < 0 or saturation < 0:
            raise ValueError(f"Invalid negative input, brightness={brightness}, contrast={contrast}, sat={saturation}")
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.temperature = temperature 
    
    def apply_frame(self, frame, t):
        # split alpha is present, work on RGB only
        if frame.shape[2] == 4:
            has_alpha = True
            alpha = frame[:,:,3:4] # keep as (H,W,1 not (H,W))
            rgb = frame[:,:,:3].astype(np.float32)
        else:
            has_alpha = False
            rgb = frame.astype(np.float32)
        
        # brightness - multiply rgb channels by brightness factor
        if self.brightness != 1.0:
            rgb = np.multiply(rgb, self.brightness)
        
        # contrast - scale around midpoint 128
        if self.contrast != 1.0:
            rgb = (rgb - 128)*self.contrast + 128 
        
        # saturation
        if self.saturation != 1.0:
            # use ITU-R BT.601 perceptual weights (accounts for human eye sensitivity to diff colors)
            luminance = (0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2])
            luminance = luminance[:,:,np.newaxis] # broadcast from (H,W) to (H,W,1) so it works against rgb (H,W,3)
            # saturation = 0.0 --> pure greyscale (luminance+0), 2.0 --> color deviation from grey is doubled 
            rgb = luminance + (rgb - luminance) * self.saturation
        
        # hue
        if self.hue != 0.0:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8) # clamp to 0-255 before converting in case prior effects brought values outside 0-255 range
            hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) # convert to HSV
            hsv[:, :, 0] = (hsv[:, :, 0] + self.hue / 2.0) % 180.0 # wrap around 
            rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) # back to rgb
        
        # temp - simple channel bias
        if self.temperature != 0.0:
            # reasonable scale_factor is 15 so temperature=1.0 shifts R by +15 and B by -15
            scale_factor = 15
            rgb[:,:,0] = np.clip(rgb[:,:,0] + self.temperature * scale_factor, 0.0, 255.0) # R channel
            rgb[:,:,2] = np.clip(rgb[:,:,2] - self.temperature * scale_factor, 0.0, 255.0) # B channel

        # bring back to type uint8 after float operations
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        # reattach alpha if it was there 
        if has_alpha:
            rgb = np.concatenate([rgb, alpha], axis=2)

        return rgb
