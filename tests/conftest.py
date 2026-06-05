"""
File: conftest.py
Description: fixtures, constants, helper functions, made automatically available to all test files
"""
import pytest 
import numpy as np

# ── Video Fixtures ────────────────────────────────────────────────────────

SAMPLE_3S_320x240_30FPS = "tests/fixtures/Sample_320x240_30fps.mp4"
SAMPLE_NOISE = "tests/fixtures/Sample_Noise.mp4"
SAMPLE_VERT_AUDIO = "tests/fixtures/Sample_Vertical_Audio.mp4"
SAMPLE_MP3 = "tests/fixtures/Sample_Audio_440hz.mp3"

# Parameter order: file_path,exp_width,exp_height,exp_fps,exp_dur
VIDEO_FIXTURES = [
    (SAMPLE_3S_320x240_30FPS, 320, 240, 30.0, 3.0),
    (SAMPLE_NOISE, 1920, 1080, 30.0, 5.333333), # and 160 vid frames
    (SAMPLE_VERT_AUDIO, 1280, 720, 30.0, 4.131678)
]

SYNTHETIC_FIXTURES = [
    # (width, height, fps, duration, background, audio_source)
    (320, 240, 30.0, 3.0, (255, 255, 255), None, None, None),          # no audio, custom white bg
    (1920, 1080, 30.0, 10.0, (0, 0, 0), None, None, None),             # DEFAULT: 1080p, 10s, 30fps
    (1280, 720, 24.0, 2.0, (100, 150, 200), None, None, None),         # 720p, custom bg, 24fps
    (320, 240, 30.0, 3.0, (0, 0, 0), SAMPLE_VERT_AUDIO, None, None),   # with video audio source
    (320, 240, 30.0, 3.0, (0, 0, 0), SAMPLE_MP3, None, None),          # mp3 audio source
    (320, 240, 30.0, 3.0, (0, 0, 0), SAMPLE_VERT_AUDIO, 0.5, 2.0),     # trimmed
    (1920, 1080, 30.0, 10.0, (0, 0, 0), None, 1.0, 2.0),               # DEFAULT trimmed
]

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