"""
File: test_io.py
Tests: video reading & writing
Source files: reader.py
"""
import pytest
import numpy as np
from reelpy.io.reader import VideoReader
from reelpy.io.writer import VideoWriter
from reelpy.exceptions import InvalidVideoError

SAMPLE_3S_320x240_30FPS = "tests/fixtures/Sample_320x240_30fps.mp4"
SAMPLE_NOISE = "tests/fixtures/Sample_Noise.mp4"

# Parameter order: file_path,exp_width,exp_height,exp_fps,exp_dur
VIDEO_FIXTURES = [
    (SAMPLE_3S_320x240_30FPS, 320, 240, 30.0, 3.0),
    (SAMPLE_NOISE, 1920, 1080, 30.0, 5.333333) # and 160 frames
]

def test_reader_nonexistent_file():
    with pytest.raises(InvalidVideoError):
        VideoReader("tests/fake_path.mp4")

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
        assert reader.duration_s == pytest.approx(exp_dur)

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
            assert arr.shape == (exp_height, exp_width, 3)
            assert arr.dtype == np.uint8
            all_frames.append(frame)
        # codec schemes can sometimes drop last frame or two, but shouldn't hallucinate
        assert exp_frames-2 <= len(all_frames) <= exp_frames

    # Validate timestamps
    timestamps = [t for arr,t in all_frames]
    # starts near 0
    assert timestamps[0] == pytest.approx(0.0, abs=0.1) # allowed diff of 0.1s given finite fps
    # increases monotonically 
    assert all(timestamps[i+1] > timestamps[i] for i in range(0,len(timestamps)-1)) == True
    # last frame near expected dur
    assert timestamps[-1] == pytest.approx(exp_dur, abs=0.1)

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

def test_writer_creates_valid_file