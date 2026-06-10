"""
File: test_effects.py
Tests: BaseEffect, FadeInEffect, FadeOutEffect
Source files: effects/
Test Setup:
 1) Unit tests: test apply_frame directly on NumPy array (test frame fixtures)
 2) Integration tests: test apply_frame on a real clip, export, verify output
"""
import pytest
import warnings
from reelpy.io.reader import VideoReader
from reelpy.effects.base import BaseEffect
from reelpy.effects.fades import FadeInEffect, FadeOutEffect
import numpy as np
from typing import Callable
from tests.conftest import (
    TEST_FRAMES, ALL_CLIP_FACTORY
)

# ── Expectation Functions ────────────────────────────────────────────────────────────────

def expect_valid_shape_dtype_range(inp, out, effect, t):
    assert out.shape == inp.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255

def assert_fadein_rgb_correct(inp, out, effect, t):
    """Verify pixel transformations for RGB fadein."""
    if inp.shape[2] == 4:
        raise ValueError("Mismatched channel configuration. Expected RGB, got RGBA.")
    alpha_factor = min(t / effect.duration, 1.0) 
    expected = (inp.astype(np.float32) * alpha_factor).astype(np.uint8) 
    assert np.allclose(out, expected, atol=1) # checks that every pixel in out is within atol=1 of the pixel in expected (handles rounding discrepancies)

def assert_fadein_alpha_correct(inp, out, effect, t):
    """Verify pixel transformations for RGBA fadein."""
    if inp.shape[2] != 4:
        raise ValueError("Mismatched channel configuration. Expected RGBA, got RGB.")
    alpha_factor = min(t / effect.duration, 1.0)
    assert np.array_equal(out[:,:,:3], inp[:,:,:3]), \
            f"RGB channels should be completely untouched for RGBA fadein"
    expected_alpha = (inp[:,:,3].astype(np.float32) * alpha_factor).astype(np.uint8) # what alpha channel should see for fade-in
    assert np.allclose(out[:,:,3], expected_alpha, atol=1)

def assert_all_black_frame(out):
    assert out.max() == 0

def assert_all_transparent_frame(out):
    assert out[:,:,3].max() == 0

def assert_fadeout_rgb_correct(inp, out, effect, t):
    """Verify pixel transformations for RGB fadeout."""
    fade_start = effect.clip_duration - effect.duration
    if fade_start <= 0: 
        # must start from black
        assert_all_black_frame(out)
        return
    if t < fade_start: # passthrough effect: out=inp
        assert np.array_equal(out, inp), \
            f"FadeOut RGB at t={t}: should be passthrough before fade_start={fade_start}"
    else:
        alpha_factor = max(0.0, 1.0 - (t - fade_start) / effect.duration)
        expected = (inp.astype(np.float32) * alpha_factor).astype(np.uint8)
        assert np.allclose(out, expected, atol=1), \
            f"FadeOut RGB at t={t}: expected mean {expected.mean():.1f}, got {out.mean():.1f}"

def assert_fadeout_alpha_correct(inp, out, effect, t):
    """Verify pixel math for RGBA fadeout: RGB unchanged, alpha scaled."""
    fade_start = effect.clip_duration - effect.duration
    if fade_start <= 0:
        # must start from transparent
        assert_all_transparent_frame(out)
        return
    if t < fade_start:
        assert np.array_equal(out, inp), \
            f"FadeOut RGBA at t={t}: should be passthrough before fade_start={fade_start}"
    else:
        alpha_factor = max(0.0, 1.0 - (t - fade_start) / effect.duration)
        assert np.array_equal(out[:,:,:3], inp[:,:,:3]), \
            f"FadeOut RGBA at t={t}: RGB channels were modified"
        expected_alpha = (inp[:,:,3].astype(np.float32) * alpha_factor).astype(np.uint8)
        assert np.allclose(out[:,:,3], expected_alpha, atol=1)

def assert_first_darker_than_last(frames):
    """FADEIN BEHAVIOR: First frame should be darker than last.
       frames is from reader.frames()"""
    start_idx = 0; end_idx = -1
    # for useful test: need frames to not be already black 
    while frames[start_idx][0].mean()< 1.0: 
        if start_idx >= len(frames)-1:
            pytest.skip("This clip is not suited for being tested w/ fadein.")
        start_idx += 1
    while frames[end_idx][0].mean() < 1.0:
        if abs(end_idx) >= len(frames):
            pytest.skip("This clip is not suited for being tested w/ fadein.")
        end_idx -= 1
    first = frames[start_idx][0].mean() 
    last = frames[end_idx][0].mean()
    assert first < last, \
        f"Expected earlier frame to be darker than later for FadeIn: first_mean={first:.1f}, first_idx={start_idx}, last_mean={last:.1f}, last_idx={end_idx}"
    
def assert_last_darker_than_first(frames):
    """FADEOUT BEHAVIOR: Last frame should be darker than first."""
    start_idx = 0; end_idx = -1
    # iterate until we get frames that arent nearly black
    while frames[start_idx][0].mean() < 1.0: # essentially black end frame 
        if start_idx >= len(frames)-1:
            pytest.skip("This clip is not suited for being tested w/ fadeout.") # avoid index errors
        start_idx += 1
    while frames[end_idx][0].mean() < 1.0: # essentially black end frame and bwd indxing start from 1
        if abs(end_idx) >= len(frames):
            pytest.skip("This clip is not suited for being tested w/ fadeout.")
        end_idx -= 1
    first = frames[start_idx][0].mean() 
    last = frames[end_idx][0].mean()
    assert last < first, \
        f"Expected later frame to be darker than earlier for FadeOut: first_mean={first:.1f}, first_idx={start_idx}, last_mean={last:.1f}, last_idx={end_idx}"
    
def assert_middle_brighter_than_ends(frames):
    """FADEIN AND FADEOUT CHAIN: Middle frame brighter than first & last."""
    start_idx = 0; end_idx = -1
    mid_idx = len(frames)//2
    while frames[start_idx][0].mean() < 1.0:
        if start_idx >= mid_idx:
            pytest.skip("This clip is not suited for being tested w/ fadein and fadeout together.")
        start_idx += 1
    while frames[end_idx][0].mean() < 1.0:
        if abs(end_idx) > mid_idx:
            pytest.skip("This clip is not suited for being tested w/ fadein and fadeout together.")
        end_idx -= 1
    first = frames[start_idx][0].mean()
    # recompute mid_idx w/ new start and end
    mid_idx = (start_idx + len(frames) + end_idx + 1) // 2
    mid = frames[mid_idx][0].mean()
    last = frames[end_idx][0].mean()
    assert first < mid, f"Middle should be brighter than first: mid={mid:.1f}, first={first:.1f}"
    assert last < mid,  f"Middle should be brighter than last: mid={mid:.1f}, last={last:.1f}"


# ── Expectations per Effect ────────────────────────────────────────────────────────────────

UNIVERSAL_EXPECTATIONS = [ # universal across all effect types
    expect_valid_shape_dtype_range,
]

# FOR UNIT LEVEL
# Dictionary with effect class types mapping to list of their expected behaviors after apply_frame 
# (effect_type, frame_type)
EFFECT_EXPECTATIONS: dict[tuple[type[BaseEffect], str], list[Callable]] = {
    (FadeInEffect, "rgb"): [assert_fadein_rgb_correct],
    (FadeInEffect, "rgba"): [assert_fadein_alpha_correct],         
    (FadeOutEffect, "rgb"): [assert_fadeout_rgb_correct],
    (FadeOutEffect, "rgba"): [assert_fadeout_alpha_correct],
}

# FOR INTEGRATION LEVEL
# (effects_to_apply, assertions, description)
INTEGRATION_EFFECT_EXPECTATIONS = [
    (
        lambda dur: [FadeInEffect(min(1.0, dur * 0.3))], # fade duration = 30% of clip, at least 1s
        [assert_first_darker_than_last],
        "FadeIn"
    ),
    (
        lambda dur: [FadeOutEffect(min(1.0, dur * 0.3))],
        [assert_last_darker_than_first],
        "FadeOut"
    ),
    (
        lambda dur: [FadeInEffect(min(0.5, dur * 0.2)), FadeOutEffect(min(0.5, dur * 0.2))],
        [assert_middle_brighter_than_ends],
        "FadeIn_FadeOut_Composite"
    ),
]


# ──  Test cases (effect_instance, t, frame_name, frame_type) ────────────────────────────────────────────────────────────────

# assertions looked up from EFFECT_EXPECTATIONS at runtime
EFFECT_TEST_CASES = [
    # ── FadeIn RGB (frame level) ────────────────────────────────────────────────────────────
    # boundary: t=0 --> fully black
    (FadeInEffect(1.0), 0.0,   "white_rgb", "rgb"),
    # linear steps through fade
    (FadeInEffect(1.0), 0.1,   "white_rgb", "rgb"),
    (FadeInEffect(1.0), 0.25,  "white_rgb", "rgb"),
    (FadeInEffect(1.0), 0.5,   "white_rgb", "rgb"),
    (FadeInEffect(1.0), 0.75,  "white_rgb", "rgb"),
    (FadeInEffect(1.0), 0.9,   "white_rgb", "rgb"),
    # boundary: exactly at duration --> passthrough
    (FadeInEffect(1.0), 1.0,   "white_rgb", "rgb"),
    # after duration --> passthrough
    (FadeInEffect(1.0), 1.5,   "white_rgb", "rgb"),
    (FadeInEffect(1.0), 2.0,   "white_rgb", "rgb"),
    # test halfway point with different frames
    (FadeInEffect(1.0), 0.5,   "grey_rgb", "rgb"),
    (FadeInEffect(1.0), 0.5,   "colored_rgb", "rgb"),
    # test different durations
    (FadeInEffect(2.0), 1.0,   "white_rgb", "rgb"),   # halfway through 2s fade
    (FadeInEffect(0.5), 0.25,  "white_rgb", "rgb"),   # halfway through 0.5s fade

    # ── FadeIn RGBA (layer level) ───────────────────────────────────────────────────────────
    (FadeInEffect(1.0), 0.0,   "white_rgba", "rgba"),  # alpha=0, fully transparent
    (FadeInEffect(1.0), 0.5,   "white_rgba", "rgba"),  # alpha=127
    (FadeInEffect(1.0), 1.0,   "white_rgba", "rgba"),  # passthrough
    (FadeInEffect(1.0), 0.5,   "semi_rgba", "rgba"),   # semi-transparent input to begin with

    # ── FadeOut RGB ───────────────────────────────────────────────────────────
    # before fade --> passthrough
    (FadeOutEffect(1.0, clip_duration=5.0), 0.0,  "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 2.5,  "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 3.9,  "white_rgb", "rgb"),
    # exactly at fade start
    (FadeOutEffect(1.0, clip_duration=5.0), 4.0,  "white_rgb", "rgb"),
    # linear steps through fade
    (FadeOutEffect(1.0, clip_duration=5.0), 4.1,  "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.25, "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.5,  "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.75, "white_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.9,  "white_rgb", "rgb"),
    # boundary: at clip end --> fully black
    (FadeOutEffect(1.0, clip_duration=5.0), 5.0,  "white_rgb", "rgb"),
    # different frames (halway marks)
    (FadeOutEffect(1.0, clip_duration=5.0), 4.5,  "grey_rgb", "rgb"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.5,  "colored_rgb", "rgb"),
    # different durations
    (FadeOutEffect(2.0, clip_duration=5.0), 4.0,  "white_rgb", "rgb"),  # halfway through 2s fade
    (FadeOutEffect(0.5, clip_duration=5.0), 4.75, "white_rgb", "rgb"),  # halfway through 0.5s fade
    # no clip duration injected (default clip_duration=0.0) --> expect persistent black screen 
    (FadeOutEffect(1.0),                    0.0,  "colored_rgb", "rgb"), # since fade_start = 0-1 = -1 and factor = 1 - [min(1.0, (t-fade_start) / duration)] = 1-min(1.0,t+1), it's always gonna be 0 (black screen)

    # ── FadeOut RGBA ──────────────────────────────────────────────────────────
    (FadeOutEffect(1.0, clip_duration=5.0), 0.0,  "white_rgba", "rgba"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.5,  "white_rgba", "rgba"),
    (FadeOutEffect(1.0, clip_duration=5.0), 5.0,  "white_rgba", "rgba"),
    (FadeOutEffect(1.0, clip_duration=5.0), 4.5,  "semi_rgba", "rgba"),
    # no clip duration injected (default clip_duration=0.0) --> expect fully transparent alpha from first frame (0.0)
    (FadeOutEffect(1.0),                    0.0,  "white_rgba", "rgba"), 
]

# ──  Effect Instances ────────────────────────────────────────────────────────────────
"""Build a dict containing a single effect instance per effect subclass for tests that run on all subclasses.
   Then extract .values() from this dict to get list of single instances we can parametrize over.
   NOTE since dicts don't allow duplicate keys -> guaranteed one instance per effect type."""
EFFECT_INSTANCES = list({
    type(effect): effect
    for effect, *_ in EFFECT_TEST_CASES
}.values())


# ── Main runners ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("effect,t,frame_name,frame_type", EFFECT_TEST_CASES)
def test_effect_frame(effect, t, frame_name, frame_type):
    """Main runner for UNIT level tests on frames (no clips yet)."""
    effect_type = type(effect)

    # look up specific assertions from dict, key format (effect_type, frame_type)
    specific_assertions = EFFECT_EXPECTATIONS.get((effect_type, frame_type), [])
    
    # warn if nothing registered for this effect type
    if (effect_type, frame_type) not in EFFECT_EXPECTATIONS:
        warnings.warn(
            f"No expectations registered for {effect_type.__name__} — "
            f"only universal contracts tested",
            UserWarning
        )

    frame = TEST_FRAMES[frame_name].copy()
    inp_snapshot = frame.copy()

    out = effect.apply_frame(frame, t)

    # universal contracts
    for expectation in UNIVERSAL_EXPECTATIONS:
        expectation(inp_snapshot, out, effect, t)

    # check inp_snapshot not mutated
    assert np.array_equal(frame, inp_snapshot), \
        f"{effect_type.__name__} mutated input frame"

    # call effect-specific assertions from dict 
    for assertion in specific_assertions:
        assertion(inp_snapshot, out, effect, t)

@pytest.mark.parametrize("effect", EFFECT_INSTANCES)
def test_clone_independence(effect: BaseEffect):
    """Main runner for per-effect subclass tests on frames."""
    clone = effect.clone()
    assert clone is not effect
    # verify attribute VALUES are preserved
    for attr in vars(effect): # vars() returns dict of all instance attributes; for attr in dict iterates thru KEYS (attributes)
        assert getattr(clone, attr) == getattr(effect, attr), \
            f"clone missing attribute {attr}"
    # verify independence - modifying clone doesn't affect original
    # we'll test one numeric attribute for simplicity
    checked = False
    for attr in vars(effect):
        original_val = getattr(effect,attr)
        if isinstance(original_val, float):
            setattr(clone, attr, original_val + 1.0)
            assert getattr(effect, attr) == original_val, \
                f"Modifying clone.{attr} affected original"
            checked = True
            break
        else:
            continue
    if not checked:
        warnings.warn("this didn't test clone independence cuz we didn't have any numeric attributes", UserWarning)

@pytest.mark.parametrize("clip_lambda, clip_label", ALL_CLIP_FACTORY)
@pytest.mark.parametrize("effect_chain, assertions, effect_chain_label", INTEGRATION_EFFECT_EXPECTATIONS)
def test_effect_clip(tmp_path, clip_lambda, clip_label, effect_chain, assertions, effect_chain_label):
    """Main runner for INTEGRATION level tests on clips."""
    clip = clip_lambda() # iterates thru constructing all factory objects in ALL_CLIP_FACTORY
    dur = clip.effective_duration() # clip's eff duration
    effects = effect_chain(dur)
    for effect in effects:
        clip = clip.apply(effect)

    output = str(tmp_path / f"output_{clip_label}_{effect_chain_label}.mp4")
    clip.export(output)

    # verify frames by reading at export level
    with VideoReader(output) as reader:
        frames = list(reader.frames())
    assert len(frames) > 0
    for assertion in assertions:
        assertion(frames)

