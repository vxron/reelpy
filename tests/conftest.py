"""
File: conftest.py
Description: fixtures, constants, helper functions, made automatically available to all test files
"""
import pytest 
import numpy as np

# ── Fixture file paths ────────────────────────────────────────────────────────

SAMPLE_3S_320x240_30FPS = "tests/fixtures/Sample_320x240_30fps.mp4"
SAMPLE_NOISE = "tests/fixtures/Sample_Noise.mp4"
SAMPLE_VERT_AUDIO = "tests/fixtures/Sample_Vertical_Audio.mp4"

# Parameter order: file_path,exp_width,exp_height,exp_fps,exp_dur
VIDEO_FIXTURES = [
    (SAMPLE_3S_320x240_30FPS, 320, 240, 30.0, 3.0),
    (SAMPLE_NOISE, 1920, 1080, 30.0, 5.333333), # and 160 vid frames
    (SAMPLE_VERT_AUDIO, 1280, 720, 30.0, 4.131678)
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