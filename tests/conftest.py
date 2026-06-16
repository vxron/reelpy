"""
File: conftest.py
Description: fixtures, constants, helper functions, made automatically available to all test files
"""
import pytest 
import numpy as np
from reelpy.config import config
from reelpy.clip.synthetic import SyntheticClip
from reelpy.clip.video import Clip
from reelpy.effects.base import BaseEffect
from reelpy.effects.fades import FadeInEffect, FadeOutEffect


# ── CLI Arg Parsing ────────────────────────────────────────────────────────
def pytest_addoption(parser):
    """Add custom CLI flags to pytest."""
    parser.addoption(
        "--routine",
        action="store_true", # store as true if this arg is present in CLI
        default=False,
        help="only run tests marked as routine"
    )
    parser.addoption(
        "--exhaustive",
        action="store_true",
        default=False,
        help="run ALL tests including slow high-res ones"
    )

def pytest_collection_modifyitems(config, items):
    """
    Control which tests run based on the CLI flags.
    Default (no flags): run everything
    --routine: run only @pytest.mark.routine tests
    --exhaustive: run everything including slow tests
    """
    routine_only = config.getoption("--routine")
    exhaustive = config.getoption("--exhaustive")

    if exhaustive: # exhaustive overrides everything: run all w/o filtering
        return
    
    if routine_only:
        # skip anything not marked as routine
        skip = pytest.mark.skip(reason="not marked as routine")
        for item in items: # iterate thru tests 
            if "routine" not in item.keywords:
                item.add_marker(skip)


# ── Configs ────────────────────────────────────────────────────────
# Automatically reset global configs to defaults after every test 
@pytest.fixture(autouse=True)
def reset_config():
    yield
    config.audio_mode = "trim"
    config.default_bitrate = 4_000_000
    config.default_fps = 30.0


# ── Ready-To-Go Fixtures ────────────────────────────────────────────────────────

SAMPLE_3S_320x240_30FPS = "tests/fixtures/Sample_320x240_30fps.mp4"
SAMPLE_NOISE = "tests/fixtures/Sample_Noise.mp4"
SAMPLE_VERT_AUDIO = "tests/fixtures/Sample_Vertical_Audio.mp4"
SAMPLE_MP3 = "tests/fixtures/Sample_Audio_440hz.mp3"

# Parameter order: file_path,exp_width,exp_height,exp_fps,exp_dur
VIDEO_FIXTURES = [
    pytest.param(SAMPLE_3S_320x240_30FPS, 320, 240, 30.0, 3.0, marks=pytest.mark.routine),
    pytest.param(SAMPLE_NOISE, 1920, 1080, 30.0, 5.333333, marks=pytest.mark.exhaustive), # and 160 vid frames
    pytest.param(SAMPLE_VERT_AUDIO, 1280, 720, 30.0, 4.131678, marks=pytest.mark.routine)
]

SYNTHETIC_FIXTURES = [
    # (width, height, fps, duration, background, audio_source, start, end, test_level)
    pytest.param(320, 240, 30.0, 3.0, (255, 255, 255), None, None, None, marks=pytest.mark.routine),          # no audio, custom white bg
    pytest.param(1920, 1080, 30.0, 10.0, (0, 0, 0), None, None, None, marks=pytest.mark.exhaustive),          # DEFAULT: 1080p, 10s, 30fps
    pytest.param(1280, 720, 24.0, 2.0, (100, 150, 200), None, None, None, marks=pytest.mark.exhaustive),      # 720p, custom bg, 24fps
    pytest.param(320, 240, 30.0, 3.0, (56, 47, 8), SAMPLE_VERT_AUDIO, None, None, marks=pytest.mark.routine), # with video audio source
    pytest.param(320, 240, 30.0, 3.0, (0, 0, 0), SAMPLE_MP3, None, None, marks=pytest.mark.routine),          # mp3 audio source
    pytest.param(320, 240, 30.0, 3.0, (0, 0, 0), SAMPLE_VERT_AUDIO, 0.5, 2.0, marks=pytest.mark.routine),     # trimmed
    pytest.param(1920, 1080, 30.0, 10.0, (0, 0, 0), None, 1.0, 2.0, marks=pytest.mark.exhaustive),            # DEFAULT trimmed
]

AUDIO_FIXTURES = [
    (None, 0),                     # no audio source
    (SAMPLE_VERT_AUDIO, 4.131678), # path, audio dur based on ffprobe
    (SAMPLE_MP3, 5.041633),
]

# Fixture for testing effects on individual frames 
colored_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
colored_rgb[:,:,0] = 200 # R channel
colored_rgb[:,:,1] = 100 # G
colored_rgb[:,:,2] = 50  # B
TEST_FRAMES = {
    "black_rgb":   np.zeros((240, 320, 3), dtype=np.uint8),
    "white_rgb":   np.full((240, 320, 3), 255, dtype=np.uint8),
    "grey_rgb":    np.full((240, 320, 3), 128, dtype=np.uint8),
    "colored_rgb": colored_rgb,
    "black_rgba":  np.zeros((240, 320, 4), dtype=np.uint8),
    "white_rgba":  np.full((240, 320, 4), 255, dtype=np.uint8),
    "semi_rgba":   np.full((240, 320, 4), 128, dtype=np.uint8),
}

# ── Fixture Factories ────────────────────────────────────────────────────────
    
def make_synthetic_from_fixture(width, height, fps, duration, background=(0,0,0), audio_source=None, start=None, end=None, **kwargs):
    """Build a SyntheticClip from a SYNTHETIC_FIXTURES entry."""
    clip = SyntheticClip(width, height, fps, duration, background, audio_source, kwargs.get("mute", False))
    if start is not None and end is not None:
        clip = clip.trim(start, end)
    return clip

def make_video_from_fixture(file_path, *args, **kwargs):
    """Build a Clip from a VIDEO_FIXTURES entry. Extra positional args (width,neight,etc) are ignored. Keyword args go to kwargs"""
    return Clip(file_path, kwargs.get("audio_source", None), kwargs.get("mute", False))

SYNTHETIC_CLIP_FACTORY = [
    (lambda f=fixture: make_synthetic_from_fixture(*f.values), f"synthetic_{i}") # pytest.param stors its args in.values as a tuple!
    for i, fixture in enumerate(SYNTHETIC_FIXTURES)
]

VIDEO_CLIP_FACTORY = [ # builds series of Clip instances from make_video_from_fixture
    (lambda f=fixture: make_video_from_fixture(*f.values), f"video_{i}") # f=fixture so each individual lambda builder gets a diff fixture value (don't share last instance)
    for i, fixture in enumerate(VIDEO_FIXTURES)
]

ALL_CLIP_FACTORY = SYNTHETIC_CLIP_FACTORY + VIDEO_CLIP_FACTORY # elements are (clip: BaseClip, clip_ID: str)

# ── Test configs ────────────────────────────────────────────────────────

INVALID_CLIP_CONFIGS = [
    (0, 240, 30.0, 5.0, (0, 0, 0)),        # width=0
    (-1, 240, 30.0, 5.0, (0, 0, 0)),       # width negative
    (7681, 240, 30.0, 5.0, (0, 0, 0)),     # width over max
    (320, 0, 30.0, 5.0, (0, 0, 0)),        # height=0
    (320, 240, 0.0, 5.0, (0, 0, 0)),       # fps=0
    (320, 240, -1.0, 5.0, (0, 0, 0)),      # fps negative
    (320, 240, 71.0, 5.0, (0, 0, 0)),      # fps over max
    (320, 240, 30.0, 0.0, (0, 0, 0)),      # duration=0
    (320, 240, 30.0, -1.0, (0, 0, 0)),     # duration negative
    (320, 240, 30.0, 7201.0, (0, 0, 0)),   # duration over max
    (320, 240, 71.0, 5.0, (2, 50)),        # background out of range 1
    (320, 240, 30.0, 0.0, (-1, 0, 90)),    # background out of range 2
    (320, 240, 30.0, -1.0, (80, 84, 256)), # background out of range 3
]

# ── Helper functions ────────────────────────────────────────────────────────

def assert_frame_valid(arr, width, height):
    assert arr.shape == (height, width, 3)
    assert arr.dtype == np.uint8

def assert_timestamps_valid(timestamps, exp_dur):
    assert timestamps[0] == pytest.approx(0.0, abs=0.1)
    assert all(timestamps[i+1] > timestamps[i] for i in range(len(timestamps)-1)) # monotonic increasing timestamps
    assert timestamps[-1] == pytest.approx(exp_dur, abs=0.1)

def frame_list_from_clip(clip):
    return list(clip.frames())

def timestamp_list_from_frames(frames):
    return [t for arr, t in frames]