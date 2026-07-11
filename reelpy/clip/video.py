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
from reelpy.preview.player import PreviewPlayer

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
        for effect in self.effects:
            effect.prepare(clip_duration=eff_dur) # no-op for most, meaningful for FadeOut bcuz it needs to consume the same clip_duration throughout read (SET ONCE OUTSIDE PER-FRAME LOOP)
        with VideoReader(self.path) as reader:
            # iterate thru frames from start to end
            for (arr, t) in reader.frames(start=self.start, end=self.end):
                # apply layer compositing stack
                for layer in self.layers:
                    # TODO: layer.render(arr, t)
                    pass
                # pass frames thru effect pipe
                for effect in self.effects:
                    arr = effect.apply_frame(arr, t)
                yield (arr, t)

    def preview(self) -> None:
        PreviewPlayer(self).run()

    def export(self, path: str, bitrate: int = 4_000_000, audio_mode: str | None = None) -> None:
        audio_mode = audio_mode or config.audio_mode # fall back to global config if mode not specifid in arg
        resolved_audio = self._resolve_audio_source() # mute, override, clip's own audio, etc...
        frames_gen = self.frames() # in case there were any size transformations, we need to know before export
        first_arr, first_t = next(frames_gen)
        out_h, out_w = first_arr.shape[:2]

        with VideoWriter(
            path=path, fps=self.fps, width=out_w, height=out_h, 
            bitrate=bitrate, 
            audio_source=resolved_audio # only attach audio if this clip has an audio stream
        ) as writer:
            writer.write_frame(first_arr)
            for (arr, t) in frames_gen: # effects & layers applied inside frames
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
    
    def seek_frames(self, t: float) -> Iterator[tuple[np.ndarray, float]]:
        # (1) determine the true editorial end of this clip
        end = self.end if self.end is not None else self.duration

        # (2) clamp t into valid range within the clip's editorial bounds
        # never seek before the clip's own start, never seek past the end
        t = max(self.start, min(t, end - 0.01))

        # (3) prepare effects using the ORIGINAL effective_duration — this never
        # changes based on seek position, so FadeOut timing is always correct
        # regardless of where we scrub to
        eff_dur = self.effective_duration()
        for effect in self.effects:
            effect.prepare(clip_duration=eff_dur)

        # (4) open reader, seek directly to t (jumps to nearest keyframe at or before t)
        with VideoReader(self.path) as reader:
            # reader.frames() yields relative timestamps (t_abs - start), starting at 0.0
            # adding t back converts them to absolute timestamps in the original timeline
            for arr, t_rel_to_seek in reader.frames(start=t, end=end):
                t_abs = t + t_rel_to_seek  # absolute position - what we YIELD for preview playing
                t_trim = t_abs - self.start # trim-relative - what frames() computes, and effects expect
                for layer in self.layers:
                    arr = layer.render(arr, t_trim)
                for effect in self.effects:
                    arr = effect.apply_frame(arr, t_trim)
                yield (arr, t_abs)

