"""
File: writer.py
Class: VideoWriter 
Description: Class accepts a stream of NumPy RGB frames, encodes them
to H.264, muxes into an .mp4 container, and optionally copies an audio
track from a source file.
Notes:
-Bitrate is total data per second across the whole video (30 frames if 30fps)
-so FPS determines how many frames share the bitrate budget (higher FPS for same bitrate = lower res per frame)
-in copy_audio, path_override only works if called before any write_frame calls,
 due to MP4 container constraints. For normal use via export() this is ok.
"""
import av
import numpy as np
from av.error import FFmpegError
from reelpy.exceptions import InvalidVideoError, StreamNotFoundError
from typing import Any

class VideoWriter():
    def __init__(self, path: str, fps: float, width: int, height: int, bitrate: int = 4_000_000, audio_source: str | None = None):
        # Open the output container
        self.path = path
        try:
            self._container: Any = av.open(path, "w")
        except FFmpegError as e:
            raise InvalidVideoError(f"VideoWriter could not open video file, check path: {path}") from e
        
        try:
            # Add + configure the H.264 video stream
            self._stream = self._container.add_stream("h264", rate=int(fps))
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

        # Add + configure optional audio stream from src
        try:
            if audio_source is not None:
                src = av.open(audio_source)
                if src.streams.audio:
                    src_audio = src.streams.audio[0]
                    # always use aac for mp4 compatibility
                    self._audio_stream = self._container.add_stream(
                        "aac",
                        rate=src_audio.codec_context.sample_rate
                    )
                    self._audio_stream.codec_context.layout = src_audio.codec_context.layout
                    if src_audio.time_base is None:
                        raise InvalidVideoError("Audio stream has no time_base")
                    self._audio_stream.time_base = src_audio.time_base
                else: # no audio streams found
                    raise StreamNotFoundError(f"VideoWriter failed to source audio at {audio_source}")
                self._audio_source = audio_source
                src.close()
            else:
                self._audio_stream = None
                self._audio_source = None
        except FFmpegError as e:
            raise StreamNotFoundError(f"VideoWriter failed to source audio at {audio_source}")

    def write_frame(self, frame: np.ndarray) -> None:
        try:
            # (1) create PyAV frame from array
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            # (2) reformat to yuv420p for H.264 codec
            av_frame = av_frame.reformat(format="yuv420p")
            # (3) encode frames as packets into stream: PyAV handles timing automatically from stream configs
            packets = self._stream.encode(av_frame)
            # (4) mux resulting packets
            for packet in packets: 
                self._container.mux(packet)
        except FFmpegError as e:
            raise InvalidVideoError("Failed to write frame") from e
    
    def _copy_audio_reencode(self, src, src_audio_stream, start, end, offset):
        """Re-encodes audio to AAC: used for non-MP4-compatible sources like MP3."""
        resampler = av.AudioResampler(
            format="fltp",
            layout=src_audio_stream.codec_context.layout,
            rate=src_audio_stream.codec_context.sample_rate
        )
        offset_pts = int(offset * self._audio_stream.time_base.denominator)
        for frame in src.decode(src_audio_stream):
            if frame.pts is None:
                continue
            t_abs = float(frame.pts * src_audio_stream.time_base)
            if t_abs < start:
                continue
            if end is not None and t_abs > end:
                break
            # resample to fltp format which AAC encoder expects
            for resampled in resampler.resample(frame):
                # encode resampled frame to AAC packets
                for packet in self._audio_stream.encode(resampled):
                    # shift pts/dts by offset
                    packet.pts = (packet.pts or 0) + offset_pts
                    packet.dts = (packet.dts or 0) + offset_pts
                    self._container.mux(packet)
        # flush encoder
        for packet in self._audio_stream.encode(None):
            self._container.mux(packet)
        
    def copy_audio(
        self, start: float = 0.0, end: float | None = None,
        offset: float = 0.0, loop: bool = False
    ) -> None:
        # STREAM COPY PATH (AAC already): Stream-copies the audio track from source-path (start->end) into the output without re-encoding
        # RE-ENCODE PATH (MP3 --> AAC): If source is Mp3 it needs to be fully decoded so we can reencode to mp4-compatible format
        # TODO: loop functionality 
        # NOTE: audio source must be declared at VideoWriter.__init__ time.
        # MP4 containers require all streams to be declared before encoding begins.
        # NOTE: offset represents where the audio should get placed in the timeline (e.g., clip b starting at 5s in timeline will have offset=5.0)
        
        if self._audio_stream is None:
            return
        
        # open src audio so we can demux from it
        src = av.open(self._audio_source)
        src_audio_stream = src.streams.audio[0]
        src_codec = src_audio_stream.codec_context.name
        needs_reencode = src_codec not in ("aac", "ac3", "opus")  # codecs that stream copy cleanly into mp4

        if not needs_reencode: # stream copy path
            # demux audio packets from source in the [start, end] range
            start_pos_time_base_units = int(start / self._audio_stream.time_base)
            output_timeline_offset_time_base_units = int(offset / self._audio_stream.time_base)
            # loop thru audio frames in SRC audio!
            for packet in src.demux(src_audio_stream): 
                # guard
                if packet.pts is None or packet.dts is None or src_audio_stream.time_base is None:
                    continue
                t_abs = float(packet.pts * src_audio_stream.time_base) # timestamp in seconds
                # only collect from tabs = start to tabs = end
                if t_abs < start:
                    continue 
                if end is not None and t_abs > end: 
                    break
                # for the t_abs btwn start and end... fix the dts and pts so they start at 0 (remove source offset)
                # if an offset is given: add that as well because thats where it needs to be in the output timeline to be properly synced
                packet.stream = self._audio_stream # reassign stream to output stream before muxing 
                packet.dts = (packet.dts - start_pos_time_base_units + output_timeline_offset_time_base_units)
                packet.pts = (packet.pts - start_pos_time_base_units + output_timeline_offset_time_base_units)
                # (3) mux each new audio packet into self._container
                self._container.mux(packet)
        
        else: # full reencode path
            self._copy_audio_reencode(src, src_audio_stream, start, end, offset)

        # close src container
        src.close()

    def close(self):
        # make sure attributes have alr been set in __init__ before force closing to avoid errors
        if hasattr(self, '_stream') and hasattr(self, '_container'):
            try:
                # (1) FLUSH THE ENCODER: forces any buffered frames out (otherwise vid would be corrupted)
                for packet in self._stream.encode():
                    self._container.mux(packet) # flush any remaining packets
                # (2) close container
                self._container.close()
            except FFmpegError as e:
                raise InvalidVideoError(f"Failed to finalize video output at {self.path}") from e

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

        