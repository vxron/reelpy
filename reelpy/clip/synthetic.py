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
from reelpy.preview.player import PreviewPlayer

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
        eff_dur = self.effective_duration()
        for effect in self.effects:
            effect.prepare(clip_duration=eff_dur)
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
            for effect in self.effects:
                canvas = effect.apply_frame(canvas, t)
            yield (canvas, t)

    def preview(self) -> None:
        PreviewPlayer(self).run()
    
    def export(self, out_path: str, bitrate: int = 4_000_000, audio_mode: str | None = None) -> None:
        audio_mode = audio_mode or config.audio_mode
        resolved_audio = self._resolve_audio_source()
        frames_gen = self.frames()
        first_arr, first_t = next(frames_gen)
        out_h, out_w = first_arr.shape[:2]
        
        with VideoWriter(
            out_path, fps=self.fps, width=out_w, height=out_h, bitrate=bitrate, audio_source=resolved_audio
        ) as writer:
            writer.write_frame(first_arr)       # write the first frame we already pulled
            for (arr, t) in frames_gen:         # continue w internally generated frames w pre-built layers/effects
                writer.write_frame(arr)
            # add audio if there is some
            if resolved_audio is not None:
                audio_end = self._resolve_audio_end(audio_mode, self.effective_duration())
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

    # ALTERNATIVE TO FRAMES() FOR SCRUBBING (NON EXPORT)
    def seek_frames(self, t: float) -> Iterator[tuple[np.ndarray, float]]:
        # (1) determine the true editorial end of this clip
        end = self.end if self.end is not None else self.duration

        # (2) clamp t into valid range within the clip's editorial bounds
        t = max(self.start, min(t, end - 0.01))

        # (3) prepare effects using the ORIGINAL effective_duration — same as frames(),
        # never derived from seek position
        eff_dur = self.effective_duration()
        for effect in self.effects:
            effect.prepare(clip_duration=eff_dur)

        # (4) compute how many frames exist from t to the editorial end
        total_frames = round((end - t) * self.fps)

        for i in range(total_frames):
            # ABSOLUTE timestamp — t_abs is the true position in the original timeline,
            # never reset to 0 the way frames() does with its relative timestamps
            t_abs = t + i / self.fps
            t_trim = t_abs - self.start # what layers/effects expect

            # build a fresh canvas for this frame (same as frames())
            canvas = np.full((self.height, self.width, 3), self.background, dtype=np.uint8)

            # apply layer compositing stack 
            for layer in self.layers:
                canvas = layer.render(canvas, t_trim)
            # apply effect chain 
            for effect in self.effects:
                canvas = effect.apply_frame(canvas, t_trim)

            # yield ABSOLUTE timestamp — PreviewPlayer.self.t stays in sync with
            # the original timeline without any offset arithmetic
            yield (canvas, t_abs)

            