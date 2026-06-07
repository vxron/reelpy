"""
File: synthetic.py
Class: SyntheticClip
Description: User-facing implementation of BaseClip that is initialized
to a solid color background. The user then adds whatever layers they
desire on top. 
Use case: When the primary source should not be a video file. 
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import Dict
import numpy as np
from reelpy.clip.base import BaseClip
from reelpy.io.writer import VideoWriter
from reelpy.config import config

class SyntheticClip(BaseClip):
    # Default is a 1080p 10s video at 30fps with black background
    def __init__(
        self, 
        width: int = 1920, 
        height: int = 1080, 
        fps: float = 30.0, 
        duration: float = 10.0, 
        background: tuple[int,int,int] = (0,0,0), 
        audio_source: str | None = None,
        mute: bool = False,
    ):
        super().__init__() # parent class sets start, end, layers and effects
        # validate inputs
        if (
            7680 < width or width <= 0 or 
            4320 < height or height <= 0 or # maximum 8k UHD res (7680x4320)
            70.0 < fps or fps <= 0.0 or # maximum 70fps
            7200.0 < duration or duration <= 0.0 # maximum 2hours
        ):
            raise ValueError(f"Invalid request for synthetic clip configuration: width={width}, height={height}, fps={fps}, duration={duration}")
        if len(background) != 3:
            raise ValueError(f"background must be an RGB tuple of 3 integers, got {background}")
        if any([
            background[0] < 0, background[0] > 255,
            background[1] < 0, background[1] > 255,
            background[2] < 0, background[2] > 255,
        ]):
            raise ValueError(f"Invalid request for synthetic clip background (must be from 0-255): {background}")
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.background = background
        self.audio_source = audio_source
        self.mute = mute

    def _copy(self, **overrides) -> SyntheticClip:
        # create a copy of obj w/o touching original (non-destructive)
        width = overrides.get("width", self.width)
        height = overrides.get("height", self.height)
        fps = overrides.get("fps", self.fps)
        duration = overrides.get("duration", self.duration)
        background = overrides.get("background", self.background)
        result = SyntheticClip(width, height, fps, duration, background)
        result.start = overrides.get("start", self.start)
        result.end = overrides.get("end", self.end)
        result.layers = overrides.get("layers",list(self.layers)) # new obj (not same list ref)
        result.effects = overrides.get("effects",list(self.effects))
        result.audio_source = overrides.get("audio_source", self.audio_source)
        result.mute = overrides.get("mute", self.mute)
        return result
    
    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        # Generates frames from scratch rather than decoding them from source
        effective_end = self.end if self.end is not None else self.duration
        # total frames needed
        total_frames = round((effective_end - self.start) * self.fps)
        for i in range(total_frames):
            # create entirely new canvas for each frame so that layers & effects can draw diff content (e.g. shape moves) and/or transform differently on diff frames (e.g. fadeout 10% vs. 50%)
            canvas = np.full((self.height,self.width,3), self.background, dtype=np.uint8) # fill canvas w background color
            # compute timestamp rel to start
            t = i/self.fps
            # TODO: apply layer stack & effect chain
             # for layer in self.layers:
             #     canvas = layer.render(canvas, t)
             # for effect in self.effects:
             #     canvas = effect.apply_frame(canvas, t)
            yield (canvas, t)

    def preview(self) -> None:
        # TODO: implement PreviewPlayer
        pass
    
    def export(self, out_path: str, bitrate: int = 4_000_000, audio_mode: str | None = None) -> None:
        audio_mode = audio_mode or config.audio_mode
        resolved_audio = self._resolve_audio_source()
        
        with VideoWriter(
            out_path, fps=self.fps, width=self.width, height=self.height, bitrate=bitrate, audio_source=self.audio_source
        ) as writer:
            for (arr, t) in self.frames(): #internally generated frames
                # todo: apply any effect chain 
                writer.write_frame(arr)
            
            # add audio if there is some
            if resolved_audio is not None:
                effective_end = self.end if self.end is not None else self.duration
                effective_duration = effective_end - self.start
                audio_end = self._resolve_audio_end(audio_mode, effective_duration)
                writer.copy_audio(start=0.0, end=audio_end) # copy audio based on mode; start is always 0.0 i.e. beginning of ext audio source

    def metadata(self) -> Dict:
        return {
            "background": self.background,
            "audio_source": self.audio_source,
            "mute": self.mute,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "end": self.end,
            "layer_count": len(self.layers),
            "effect_count": len(self.effects),
    }
            