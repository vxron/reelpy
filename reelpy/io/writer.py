"""
File: writer.py
Class: VideoWriter 
Description: Class accepts a stream of NumPy RGB frames, encodes them
to H.264, muxes into an .mp4 container, and optionally copies an audio
track from a source file.
Notes:
-Bitrate is total data per second across the whole video (30 frames if 30fps)
-so FPS determines how many frames share the bitrate budget (higher FPS for same bitrate = lower res per frame)
"""
import av
import numpy as np
from av.error import FFmpegError
from reelpy.exceptions import InvalidVideoError, StreamNotFoundError
from typing import Any

class VideoWriter():
    def __init__(self, path: str, fps: float, width: int, height: int, bitrate: int = 4_000_000):
        # Open the output container
        self.path = path
        try:
            self._container: Any = av.open(path, "w")
        except FFmpegError as e:
            raise InvalidVideoError(f"VideoWriter could not open video file, check path: {path}") from e
        
        try:
            # Add + configure the H.264 stream
            self._stream = self._container.add_stream("h264", rate=fps)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p" # required by H.264 codec
            self._stream.bit_rate = bitrate
        except FFmpegError as e:
            self._container.close()
            raise InvalidVideoError(f"VideoWriter could not create video output: {path}") from e

        self.fps = fps
        self.width = width
        self.height = height
        self.bitrate = bitrate

    def write_frame(self, frame: np.ndarray) -> None:
        # (1) create PyAV frame from array
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        # (2) reformat to yuv420p for H.264 codec
        av_frame = av_frame.reformat(format="yuv420p")
        # (3) encode frames as packets into stream: PyAV handles timing automatically from stream configs
        packets = self._stream.encode(av_frame)
        # (4) mux resulting packets
        for packet in packets: 
            self._container.mux(packet)
        
    def copy_audio(
        self, source_path: str, start: float = 0.0, end: float | None = None,
        offset: float = 0.0, loop: bool = False
    ) -> None:
        # Stream-copies the audio track from source-path (start->end) into the output without re-encoding
        # offset represents where the audio should get placed in the timeline (e.g., clip b starting at 5s in timeline will have offset=5.0)
        # TODO: loop functionality 
        # if source_path is None (e.g., synthetic clip), return immediately
        if source_path is None:
            print("No audio to copy: This is a synthetic clip.")
            return
        try:
            src = av.open(source_path)
        except FFmpegError as e:
            raise InvalidVideoError(f"VideoWriter could not open video for audio copying: {source_path}") from e
        if not src.streams.audio: #empty list
            src.close()
            raise StreamNotFoundError(f"VideoWriter could not find audio stream in: {source_path}")
        
        # (1) Add audio stream to output container by copying the codec context from src
        src_audio_stream = src.streams.audio[0] #1st audio stream
        # new container with added audio stream (don't self-mutate) -> no data in it yet
        out_audio_stream = self._container.add_stream(template=src_audio_stream)
        
        # (2) demux audio packets from source in the [start, end] range
        if src_audio_stream.time_base is None:
            raise InvalidVideoError("Audio stream has no time_base")
        start_pos_time_base_units = int(start / src_audio_stream.time_base)
        output_timeline_offset_time_base_units = int(offset / src_audio_stream.time_base)
        # loop thru audio frames in SRC audio!
        for packet in src.demux(src_audio_stream): 
            # guard
            if packet.pts is None or packet.dts is None:
                continue
            t_abs = float(packet.pts * src_audio_stream.time_base) # timestamp in seconds
            # only collect from tabs = start to tabs = end
            if t_abs < start:
                continue 
            if end is not None and t_abs > end: 
                break
            # for the t_abs btwn start and end... fix the dts and pts so they start at 0 (remove source offset)
            # if an offset is given: add that as well because thats where it needs to be in the output timeline to be properly synced
            packet.dts = packet.dts - start_pos_time_base_units + output_timeline_offset_time_base_units
            packet.pts = packet.pts - start_pos_time_base_units + output_timeline_offset_time_base_units
            packet.stream = out_audio_stream # reassign stream to output stream before muxing so PyAV knows appropriate stream to MUX
            # (3) mux each new audio packet into self._container
            self._container.mux(packet)

        # close src container
        src.close()

    def close(self):
        # make sure attributes have alr been set in __init__ before force closing to avoid errors
        if hasattr(self, '_stream') and hasattr(self, '_container'):
            # (1) FLUSH THE ENCODER: forces any buffered frames out (otherwise vid would be corrupted)
            for packet in self._stream.encode():
                self._container.mux(packet) # flush any remaining packets
            # (2) close container
            self._container.close()

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

        