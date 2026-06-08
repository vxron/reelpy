"""
File: video.py
Class: Clip 
Description: User-facing implementation of BaseClip that uses
VideoReader & VideoWriter to edit a source clip from disk
Use case: When the primary source should be a video file.
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import Self, Dict
import numpy as np
from reelpy.clip.base import BaseClip
from reelpy.io.reader import VideoReader
from reelpy.io.writer import VideoWriter
from reelpy.config import config
from reelpy.effects.fades import FadeOutEffect

class Clip(BaseClip):
    def __init__(self, path: str, audio_source: str | None = None, mute: bool = False):
        super().__init__() # parent class sets start, end, layers and effects
        self.path = path
        # open videoreader to read metadata
        with VideoReader(path) as reader:
            self.fps = reader.fps
            self.height = reader.height
            self.width = reader.width
            self.duration = reader.duration_s
            self.has_audio = len(reader._container.streams.audio) > 0
        self.audio_source = audio_source # overrides file's own audio if set
        self.mute = mute

    def _copy(self, **overrides) -> Self:
        # new instance
        result = Clip(self.path)
        # set all abstract attributes from overrides or fall back to current values
        # recall overrides is a dict containing kwargs
        result.start = overrides.get("start", self.start)
        result.end = overrides.get("end", self.end)
        result.effects = overrides.get("effects", list(self.effects))
        result.layers = overrides.get("layers", list(self.layers))
        result.mute = overrides.get("mute", self.mute)
        result.audio_source = overrides.get("audio_source", self.audio_source)
        return result
    
    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        eff_dur = self.effective_duration()
        with VideoReader(self.path) as reader:
            # iterate thru frames from start to end
            for (arr, t) in reader.frames(start=self.start, end=self.end):
                # apply layer compositing stack
                for layer in self.layers:
                    # TODO: layer.render(arr, t)
                    pass
                # pass frames thru effect pipe
                for effect in self.effects:
                    if isinstance(effect, FadeOutEffect):
                        effect.clip_duration = eff_dur
                    arr = effect.apply_frame(arr, t)
                yield (arr, t)

    def preview(self) -> None:
        # TODO: implement PreviewPlayer
        pass

    def export(self, path: str, bitrate: int = 4_000_000, audio_mode: str | None = None) -> None:
        audio_mode = audio_mode or config.audio_mode # fall back to global config if mode not specifid in arg
        resolved_audio = self._resolve_audio_source() # mute, override, clip's own audio, etc...
        
        with VideoWriter(
            path=path, fps=self.fps, width=self.width, height=self.height, 
            bitrate=bitrate, 
            audio_source=resolved_audio # only attach audio if this clip has an audio stream
        ) as writer:
            for (arr, t) in self.frames(): # effects & layers applied inside frames
                writer.write_frame(arr)
            if resolved_audio is not None: # add audio 
                # if using file's own audio, pass trim bounds
                # if using external audio source, start from start=0, resolve end
                audio_end = self._resolve_audio_end(audio_mode, self.effective_duration()) # trim vs extend vs full
                audio_start = self.start if resolved_audio == self.path else 0.0
                writer.copy_audio(start=audio_start, end=audio_end)

    def metadata(self) -> Dict:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "end": self.end,
            "has_audio": self.has_audio,
            "audio_source": self.audio_source,
            "mute": self.mute,
            "layer_count": len(self.layers),
            "effect_count": len(self.effects)
        }

