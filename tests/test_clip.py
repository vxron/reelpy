"""
File: test_clip.py
Tests: BaseClip, Clip, SyntheticClip
Source files: base.py, synthetic.py, video.py
"""
import pytest
import numpy as np
import av
from reelpy.io.reader import VideoReader
from reelpy.io.writer import VideoWriter
from reelpy.clip.base import BaseClip
from reelpy.clip.video import Clip
from reelpy.exceptions import InvalidVideoError
from tests.conftest import (
    VIDEO_FIXTURES, SAMPLE_3S_320x240_30FPS, SAMPLE_VERT_AUDIO,
    assert_frame_valid, assert_timestamps_valid
)

# Fixture that returns both clip types (Real & Synthetic)
@pytest.fixture(params=["clip", "synthetic"])
def any_clip(request, tmp_path):
    if request.param == "clip":
        return Clip(SAMPLE_3S_320x240_30FPS)
    else:
        # TODO: return synthetic clip 
        pass


def test_clip_nonexistent_file():
    # source of this raise should be VideoReader on attempting to use av.open(path)
    with pytest.raises(InvalidVideoError):
        Clip("tests/fake_path.mp4")

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_init(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    assert clip.fps == fps
    assert clip.width == width
    assert clip.height == height
    assert clip.duration == pytest.approx(dur, abs=0.1)
    assert clip.start == 0.0 # default
    assert clip.end is None # default
    assert clip.layers == [] # default
    assert clip.effects == [] # default

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_trim(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    trimmed = clip.trim (1.0, 2.0)
    assert trimmed.start == 1.0
    assert trimmed.end == 2.0
    assert trimmed.path == clip.path # source path preserved
    assert trimmed.fps == clip.fps # metadata preserved
    # non-destructive checks (original unchanged)
    assert clip.start == 0.0 
    assert clip.end is None
    assert clip.layers is not trimmed.layers # lists should be indep copies, not same ref
    assert clip.effects is not trimmed.effects

def test_clip_invalid_trims():
    clip = Clip(SAMPLE_3S_320x240_30FPS)
    with pytest.raises(ValueError):
        clip.trim(-1.0, 2.0)
    with pytest.raises(ValueError):
        clip.trim(2.0, 1.0)
    with pytest.raises(ValueError):
        clip.trim(1.0, 1.0)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_frames(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    frames = clip.frames()
    # get first frame to validate metadata
    frame1 = next(frames)
    assert_frame_valid(frame1[0], width, height)
    # check all frames & timestamps
    all_frames = [frame1] + list(frames)
    timestamps = [t for (arr,t) in all_frames]
    assert_timestamps_valid(timestamps, dur)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_frames_with_trim(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    trimmed = clip.trim(0.8, 1.65)
    all_frames_trimmed = list(trimmed.frames())
    timestamps_trimmed = [t for (arr,t) in all_frames_trimmed]
    assert timestamps_trimmed[0] == pytest.approx(0.0, abs=0.1) # t is relative to trim start
    assert timestamps_trimmed[-1] == pytest.approx(1.65-0.8, abs=0.1)
    assert all(t<(1.65-0.8) for t in timestamps_trimmed) # nothn outside trim window

def test_clip_export(tmp_path):
    clip = Clip(SAMPLE_3S_320x240_30FPS).trim(0.5,2.5)
    output = str(tmp_path / "output.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        assert reader.width == 320
        assert reader.height == 240
        assert reader.fps == 30.0
        assert reader.duration_s == pytest.approx(2.0, abs=0.2)
        frames = list(reader.frames())
        # make sure frame count adds up with fps and duration_s
        assert len(frames) == pytest.approx(round(30.0*2.0), abs=1)

def test_clip_export_preserves_audio(tmp_path):
    clip = Clip(SAMPLE_VERT_AUDIO)
    output = str ( tmp_path / "output.mp4" )
    clip.export(output)
    with av.open(output) as container:
        # audio stream exists with valid properties
        assert len(container.streams.audio) > 0
        audio_stream = container.streams.audio[0]
        assert audio_stream.channels > 0
        # audio duration is roughly expected 
        assert audio_stream.time_base is not None and audio_stream.duration is not None
        audio_dur = float(audio_stream.duration * audio_stream.time_base)
        assert audio_dur == pytest.approx(4.131678, abs=0.3)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_copy(file_path: str, width: int, height: int, fps: float, dur: float):
    clip1 = Clip(file_path)
    clip2 = clip1._copy()
    clip3 = clip2._copy(start=1.0) # test overrides
    assert clip1 is not clip2 and clip1 is not clip3 and clip2 is not clip3
    assert clip1.width == clip2.width == clip3.width
    assert clip1.height == clip2.height == clip3.height
    assert clip1.fps == clip2.fps == clip3.fps
    assert clip1.duration == clip2.duration == clip3.duration # ALL SAME DURATION, because _copy ONLY CHANGES START
    assert clip1.start == clip2.start == 0.0 == clip3.start-1
    assert clip1.layers == clip2.layers == clip3.layers
    assert clip1.effects == clip2.effects == clip3.effects 

@pytest.mark.parametrize("file_path,width,height,fps,dur", VIDEO_FIXTURES)
def test_clip_metadata(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    meta = clip.metadata()
    # all keys present
    for key in ["duration", "fps", "width", "height", "start", "end", "layer_count", "effect_count"]:
        assert key in meta
    # values correct
    assert meta["fps"] == fps
    assert meta["width"] == width
    assert meta["height"] == height
    assert meta["duration"] == pytest.approx(dur, abs=0.1)
    assert meta["start"] == 0.0
    assert meta["end"] is None
    assert meta["layer_count"] == 0
    assert meta["effect_count"] == 0