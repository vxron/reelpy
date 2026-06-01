"""
File: test_io.py
Tests: video reading & writing
Source files: reader.py
"""
import pytest
from reelpy.io.reader import VideoReader
from reelpy.exceptions import InvalidVideoError
SAMPLE_3S_320x240_30FPS = "tests/fixtures/Sample_320x240_30fps.mp4"

def test_reader_nonexistent_file():
    with pytest.raises(InvalidVideoError):
        VideoReader("tests/fake_path.mp4")

@pytest.mark.parametrize("file_path,exp_width,exp_height,exp_fps,exp_dur", [
    (SAMPLE_3S_320x240_30FPS, 320, 240, 30.0, 3.0)
])
def test_reader_open_file_and_get_metadata(
    file_path: str, exp_width: int, exp_height: int, exp_fps: float, exp_dur: float
):
    with VideoReader(file_path) as reader:
        assert reader.width == exp_width
        assert reader.height == exp_height
        assert reader.fps == exp_fps
        assert reader.duration_s == pytest.approx(exp_dur)
    

def test_reader_frame_shape(reader):

def test_reader_timestamps(reader):

def test_reader_trim_bounds(reader):

def test_reader_ctx_manager_closes(reader):