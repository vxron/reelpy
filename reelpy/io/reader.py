"""
File: reader.py
Class: VideoReader 
Description: Class wraps PyAV containers, reading VIDEO stream only,
& provides a clean iterator over decoded frames as Numpy arrays.
Notes: All other classes obtain frames EXCLUSIVELY through VideoReader. 
"""
import av
from av.error import FFmpegError
import numpy as np
from fractions import Fraction 
from typing import Any
from collections.abc import Iterator
from reelpy.exceptions import InvalidVideoError, StreamNotFoundError

class VideoReader():
    def __init__(self, path: str, stream_index: int = 0):
        self.path = path
        try:
            self._container: Any = av.open(path)
        except FFmpegError as e:
            raise InvalidVideoError(f"Could not open video file: {path}") from e # use from to chain underlying error as __cause__
        if not self._container.streams.video:
            raise StreamNotFoundError(f"No video stream found in {path}")
        # Stream_index can be passed for user to select a stream other than default 0; however, 99% of cases will only have stream 0
        self._stream: Any = self._container.streams.video[stream_index] 
        self._timebase: Fraction = self._stream.time_base 

        # Metadata
        self.fps = float(self._stream.average_rate)
        self.width = int(self._stream.width)
        self.height = int(self._stream.height)
        self.duration_s = float(self._container.duration * av.time_base) # in seconds: since the container's duration is in units of av.time_base, and av.time_base is 1/1000000 s
        # NOTE: Duration from container is not always accurate for VBR files.

    def seek(self, t_sec: float) -> None:
        # seeks the container to the nearest KEYFRAME at or before t: caller is still responsible for decoding & discarding frames until exact target is reached from nearest keyframe
        # (1) convert t to stream integer timebase units
        timestamp = int(t_sec / self._timebase)
        # (2) seek container to expected loc in timebase units (backward=True guarantees we land at or before timestamp, never after)
        self._container.seek(timestamp, stream=self._stream, backward=True)

    # Inclusive of start, exclusive of end
    def frames(self, start: float = 0.0, end: float | None = None) -> Iterator[tuple[np.ndarray,float]]:
        """
        A generator that yields (frame_array, t) tuples where t is seconds elapsed since start
        """
        # (1) jump to nearest keyframe at or before start
        self.seek(start)
        # (2) decode & discard up until start
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            # Timestamp in seconds
            t_abs = float(frame.pts * self._timebase)
            if t_abs < start:
                continue # discard frame
            if end is not None and t_abs >= end:
                break
            # Otherwise, we decod frame and convert to NumPy
            arr = frame.to_ndarray(format="rgb24")
            yield (arr, t_abs - start) # start timestamp from 0

    def close(self) -> None:
        self._container.close()
    
    # Make VideoReader usable as context manager with WITH blocks
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    



