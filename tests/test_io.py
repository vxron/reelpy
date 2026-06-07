"""
File: test_io.py
Tests: video reading & writing
Source files: reader.py, writer.py
"""
# TODO: test StreamNotFoundError (requires fixture with no video/audio stream)
import pytest
import numpy as np
from reelpy.io.reader import VideoReader
from reelpy.io.writer import VideoWriter
from reelpy.exceptions import InvalidVideoError
import av
import itertools
from tests.conftest import (
    VIDEO_FIXTURES, SAMPLE_3S_320x240_30FPS, SAMPLE_VERT_AUDIO, 
    assert_frame_valid, assert_timestamps_valid
)

def test_reader_nonexistent_file():
    with pytest.raises(InvalidVideoError):
        VideoReader("tests/fake_path.mp4")

def test_writer_nonexistent_directory():
    # NOTE: PyAV lazily opens output containers without raising even if the path is invalid.
    # So we need to simulate a write for error to raise.
    with pytest.raises(InvalidVideoError):
        with VideoWriter("/nonexistent/dir/fake_path.mp4", 30.0, 320, 240) as writer:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write_frame(frame)

@pytest.mark.parametrize(
    "file_path,exp_width,exp_height,exp_fps,exp_dur",
    VIDEO_FIXTURES
)
def test_reader_open_file_and_get_metadata(
    file_path: str, exp_width: int, exp_height: int, exp_fps: float, exp_dur: float
):
    with VideoReader(file_path) as reader:
        assert reader.width == exp_width
        assert reader.height == exp_height
        assert reader.fps == exp_fps
        assert reader.duration_s == pytest.approx(exp_dur, abs=0.1) # 0.1s handles any PyAV/ffprobe discrepancies

@pytest.mark.parametrize(
    "file_path,exp_width,exp_height,exp_fps,exp_dur",
    VIDEO_FIXTURES
)
def test_reader_frames(
    file_path: str, exp_width: int, exp_height: int, exp_fps: float, exp_dur: float
):
    # Expected frame count (verify none are missing)
    exp_frames = round(exp_fps*exp_dur)
    # Generate frames 
    with VideoReader(file_path) as reader:
        all_frames = []
        for frame in reader.frames():
            arr, t = frame #unpacking
            assert_frame_valid(arr, exp_width, exp_height)
            all_frames.append(frame)
        # codec schemes can sometimes drop last frame or two, but shouldn't hallucinate
        assert exp_frames-2 <= len(all_frames) <= exp_frames

    # Validate timestamps
    timestamps = [t for arr,t in all_frames]
    assert_timestamps_valid(timestamps, exp_dur)

@pytest.mark.parametrize(
    "file_path,exp_width,exp_height,exp_fps,exp_dur",
    VIDEO_FIXTURES
)
def test_reader_trim_bounds(
    file_path: str, exp_width: int, exp_height: int, exp_fps: float, exp_dur: float
):
    with VideoReader(file_path) as reader:
        all_frames = []
        for frame in reader.frames(start=1.0, end=2.0): # trim from 1 to 2s
            all_frames.append(frame)

        # verify timestamps: t is relative to start, so should range from 1 to 2s
        assert all_frames[0][1] == pytest.approx(0.0, abs=0.1)
        assert all_frames[-1][1] == pytest.approx(1.0, abs=0.1)

def test_reader_context_manager_closes():
    with VideoReader(SAMPLE_3S_320x240_30FPS) as reader:
        assert reader._container is not None
    # after exiting, container should be closed meaning decoding should fail
    with pytest.raises(Exception):
        next(reader.frames())

def test_writer_context_manager_closes(tmp_path):
    output_path = str(tmp_path / "output.mp4")
    with VideoWriter(output_path, 30.0, 320, 240) as writer:
        pass
    # after exiting, container should be closed meaning encoding frames should fail
    with pytest.raises(Exception):
        tmp_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write_frame(tmp_frame)

# tmp_path is built-in pytest fixture that injects temp directory + auto-cleans after test
def test_writer_creates_valid_output_file(tmp_path):
    """
    Creates an output file by reading from a sample file
    And copying all frames to the output path
    This test ensures 
    -reading/writing frame count is preserved
    -metadata is preserved
    -valid file paths & artifact creation
    """
    exp_output_path = str(tmp_path / "output.mp4")
    # Write frames from sample reader to output path
    frames_written = 0
    frames_read = 0
    with VideoReader(SAMPLE_3S_320x240_30FPS) as reader: 
        exp_fps = reader.fps
        exp_width = reader.width
        exp_height = reader.height
        exp_dur = reader.duration_s
        with VideoWriter(
            exp_output_path, exp_fps, exp_width, exp_height
        ) as writer:
            for (arr, t) in reader.frames():
                writer.write_frame(arr)
                frames_written += 1
    # Both are closed automatically exiting with blocks
    # Verify/validate output file was written correctly
    with VideoReader(exp_output_path) as output_reader:
        assert output_reader.width == exp_width
        assert output_reader.height == exp_height
        assert output_reader.fps == exp_fps
        assert output_reader.duration_s == pytest.approx(exp_dur)
        # read frames and ensure we haven't dropped any
        # tolerance for difference of 2 for encoder flush behavior
        for frame in output_reader.frames():
            frames_read += 1
        assert abs(frames_read - frames_written) <= 2

def test_writer_copy_audio(tmp_path):
    output_path = str(tmp_path / "output.mp4")
    audio_src = SAMPLE_VERT_AUDIO
    with VideoReader(audio_src) as reader:
        exp_fps = reader.fps
        exp_width = reader.width
        exp_height = reader.height
        with VideoWriter(output_path, exp_fps, exp_width, exp_height, audio_source=audio_src) as writer:
            for (arr,t) in reader.frames():
                writer.write_frame(arr) # write vid frames
            # copy audio
            writer.copy_audio()
    
    # validate copied audio stream
    with av.open(output_path) as container:
        # stream exists with valid properties
        assert len(container.streams.audio) > 0
        audio_stream = container.streams.audio[0]
        assert audio_stream.channels > 0
        
        # duration is roughly expected 
        # very rough bcuz h.264 encoder buffering from scratch leads to variations in dur
        assert audio_stream.time_base is not None and audio_stream.duration is not None
        audio_dur = float(audio_stream.duration * audio_stream.time_base)
        assert audio_dur == pytest.approx(4.131678, abs=0.3)
        
        # packets are decodable and non empty
        # take first 5 packets of demuxed audio stream & make into iterable list
        first_packets = list(itertools.islice(container.demux(audio_stream), 5))
        assert len(first_packets) > 0
        for packet in first_packets:
            assert packet.pts is not None # valid timestamp
            assert packet.size > 0 # bytes of compressed audio data

