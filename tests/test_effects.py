"""
File: test_effects.py
Tests: BaseEffect, FadeInEffect, FadeOutEffect
Source files: effects/
Test Setup:
 1) Unit tests: test apply_frame directly on NumPy array (test frame fixtures)
 2) Integration tests: test apply_frame on a real clip, export, verify output
"""
import pytest
import cv2
import random
import warnings
from reelpy.io.reader import VideoReader
from reelpy.effects.base import BaseEffect
from reelpy.effects.fades import FadeInEffect, FadeOutEffect
from reelpy.effects.color import ColorGradeEffect
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

def assert_first_darker_than_last(frames, **kwargs):
    """FADEIN BEHAVIOR: First frame should be darker than last.
       frames is from reader.frames()"""
    start_idx = 0; end_idx = -1
    # for useful test: need frames to not be already black 
    while frames[start_idx][0].mean() < 3.0: 
        if start_idx >= len(frames)-1:
            pytest.skip("This clip is not suited for being tested w/ fadein.")
        start_idx += 1
    while frames[end_idx][0].mean() < 3.0:
        if abs(end_idx) >= len(frames):
            pytest.skip("This clip is not suited for being tested w/ fadein.")
        end_idx -= 1
    first = frames[start_idx][0].mean() 
    last = frames[end_idx][0].mean()
    assert first < last, \
        f"Expected earlier frame to be darker than later for FadeIn: first_mean={first:.1f}, first_idx={start_idx}, last_mean={last:.1f}, last_idx={end_idx}"
    
def assert_last_darker_than_first(frames, **kwargs):
    """FADEOUT BEHAVIOR: Last frame should be darker than first."""
    start_idx = 0; end_idx = -1
    # iterate until we get frames that arent nearly black
    while frames[start_idx][0].mean() < 3.0: # essentially black end frame 
        if start_idx >= len(frames)-1:
            pytest.skip("This clip is not suited for being tested w/ fadeout.") # avoid index errors
        start_idx += 1
    while frames[end_idx][0].mean() < 3.0: # essentially black end frame and bwd indxing start from 1
        if abs(end_idx) >= len(frames):
            pytest.skip("This clip is not suited for being tested w/ fadeout.")
        end_idx -= 1
    first = frames[start_idx][0].mean() 
    last = frames[end_idx][0].mean()
    assert last < first, \
        f"Expected later frame to be darker than earlier for FadeOut: first_mean={first:.1f}, first_idx={start_idx}, last_mean={last:.1f}, last_idx={end_idx}"
    
def assert_middle_brighter_than_ends(frames, **kwargs):
    """FADEIN AND FADEOUT CHAIN: Middle frame brighter than first & last."""
    start_idx = 0; end_idx = -1
    mid_idx = len(frames)//2
    # skip frames that are already black or white since they are alr saturated, hard to compare before after transformation
    while frames[start_idx][0].mean() < 3.0 or frames[start_idx][0].mean() > 253.0:
        if start_idx >= mid_idx:
            pytest.skip("This clip is not suited for being tested w/ fadein and fadeout together.")
        start_idx += 1
    while frames[end_idx][0].mean() < 3.0 or frames[start_idx][0].mean() > 253.0:
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

def assert_colorgrade_rgb_correct(inp, out, effect, t):
    """Recompute expected ColorGrade output independently and compare."""
    rgb = (inp[:,:,:3] if inp.shape[2] == 4 else inp).astype(np.float32)
    # brightness
    rgb = rgb * effect.brightness
    # contrast
    rgb = (rgb - 128.0) * effect.contrast + 128.0
    # saturation
    luminance = (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2])[:,:,np.newaxis]
    rgb = luminance + (rgb - luminance) * effect.saturation
    # hue
    if effect.hue != 0.0:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + effect.hue / 2.0) % 180.0
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # temperature
    if effect.temperature != 0.0:
        shift = effect.temperature * 15.0
        rgb[:,:,0] = np.clip(rgb[:,:,0] + shift, 0, 255)
        rgb[:,:,2] = np.clip(rgb[:,:,2] - shift, 0, 255)
    # saturation=0 greyscale check
    if effect.saturation == 0.0:
        clipped = np.clip(rgb, 0, 255).astype(np.uint8)
        assert np.allclose(clipped[:,:,0], clipped[:,:,1], atol=1), "R != G at saturation=0"
        assert np.allclose(clipped[:,:,1], clipped[:,:,2], atol=1), "G != B at saturation=0"
    expected = np.clip(rgb, 0, 255).astype(np.uint8)
    result = out[:,:,:3] if out.shape[2] == 4 else out
    assert np.allclose(result, expected, atol=1), \
        f"ColorGrade pixel math incorrect: expected mean {expected.mean():.1f}, got {result.mean():.1f}"

def assert_colorgrade_alpha_preserved(inp, out, effect, t):
    """Alpha channel must be completely unchanged by ColorGrade."""
    if inp.shape[2] != 4:
        return
    assert np.array_equal(out[:,:,3], inp[:,:,3]), \
        "ColorGrade modified alpha channel"
    
def assert_output_brighter_than_input(frames_after, frames_before, **kwargs):
    """ColorGrade brightness>1 should produce brighter output overall."""
    before_mean = np.mean([f[0].mean() for f in frames_before])
    after_mean  = np.mean([f[0].mean() for f in frames_after])
    if before_mean > 250.0:
        pytest.skip("Input clip is already white - cannot verify brightening")
    
    # if fade in/out is part of effect chain, it can drag overall means down despite clip brightness technically going up
    fade_in_present = kwargs.get("fadein_flag", False)
    fade_out_present = kwargs.get("fadeout_flag", False)
    half = len(frames_after) // 2
    if fade_in_present and fade_out_present:
        pytest.skip("Not a good test. FadeIn and FadeOut together risk creating an overall darker mean despite brightness increasing.")
    elif fade_in_present:
        # use 2nd half only
        before_mean = np.mean([f[0].mean() for f in frames_before[half:]])
        after_mean  = np.mean([f[0].mean() for f in frames_after[half:]])
    elif fade_out_present:
        # use 1st half only
        before_mean = np.mean([f[0].mean() for f in frames_before[:half]])
        after_mean  = np.mean([f[0].mean() for f in frames_after[:half]])
    
    assert after_mean > before_mean, \
        f"Expected brighter output: before={before_mean:.1f}, after={after_mean:.1f}"

def assert_output_darker_than_input(frames_after, frames_before, **kwargs):
    before_mean = np.mean([f[0].mean() for f in frames_before])
    after_mean  = np.mean([f[0].mean() for f in frames_after])
    if before_mean < 5.0:
        pytest.skip("Input clip is already black - cannot verify darkening.")
    assert after_mean < before_mean, \
        f"Expected darker output: before={before_mean:.1f}, after={after_mean:.1f}"

def assert_output_greyscale(frames, **kwargs):
    """saturation=0 should produce R==G==B on every frame."""
    for arr, t in frames:
        assert np.allclose(arr[:,:,0], arr[:,:,1], atol=2), "R != G after saturation=0"
        assert np.allclose(arr[:,:,1], arr[:,:,2], atol=2), "G != B after saturation=0"


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
    (ColorGradeEffect, "rgb"): [assert_colorgrade_rgb_correct],
    (ColorGradeEffect, "rgba"): [assert_colorgrade_rgb_correct, assert_colorgrade_alpha_preserved],
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
    (
        lambda dur: [ColorGradeEffect(brightness=0.3, saturation=1.4, hue=40)],
        [assert_output_darker_than_input],
        "ColorGrade_Dark"
    ),
    (
        lambda dur: [FadeInEffect(min(0.8, dur * 0.2)), ColorGradeEffect(brightness=1.5, temperature=0.5)],
        [assert_first_darker_than_last, assert_output_brighter_than_input],
        "FadeIn_ColorGrade"
    ),
    (
        lambda dur: [ColorGradeEffect(saturation=0.0, contrast=0.8), FadeOutEffect(min(0.5, dur * 0.15))],
        [assert_last_darker_than_first, assert_output_greyscale],
        "ColorGrade_FadeOut"
    ),
]


# ──  Test cases (effect_instance, t, frame_name, frame_type) ────────────────────────────────────────────────────────────────

def make_colorgrade_test_cases(seeds, frame_names, t_values):
    """Generate ColorGradeEffect test cases with randomized parameter combinations."""
    cases = []
    for seed in seeds:
        rng = random.Random(seed)
        effect = ColorGradeEffect(
            brightness  = rng.uniform(0.0, 3.0),
            contrast    = rng.uniform(0.0, 3.0),
            saturation  = rng.uniform(0.0, 3.0),
            hue         = rng.uniform(-180.0, 180.0),
            temperature = rng.uniform(-6.0, 6.0)
        )
        for frame_name in frame_names: # test this combo on a bunch of frames
            if frame_name[-3:] == "rgb":
                frame_type = "rgb"
            else:
                frame_type = "rgba"
            for t in t_values: # at a bunch of diff timestamps
                cases.append((effect, t, frame_name, frame_type))
    return cases

COLORGRADE_TEST_CASES = make_colorgrade_test_cases(
    seeds = list(range(10)), # 10 random combos of filters
    frame_names=["grey_rgb", "colored_rgb", "white_rgba", "semi_rgba"],
    t_values = [0.0, 1.0] # t should not matter for ColorGrade
)

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

    # ── ColorGrade tests ──────────────────────────────────────────────────────────
    *COLORGRADE_TEST_CASES, # * is for unpacking list :,)
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
    raw_clip = clip_lambda() # iterates thru constructing all factory objects in ALL_CLIP_FACTORY
    dur = raw_clip.effective_duration() # clip's eff duration
    effects = effect_chain(dur)
    clip = raw_clip
    for effect in effects: # apply all effects to the raw clip in a chain, overriding each time
        clip = clip.apply(effect)

    output = str(tmp_path / f"output_{clip_label}_{effect_chain_label}.mp4")
    clip.export(output)
    # verify frames by reading at export level
    with VideoReader(output) as reader:
        frames_after = list(reader.frames())
    assert len(frames_after) > 0

    # read raw clip frames for before after effect verifications
    raw_output = str(tmp_path / f"raw_{clip_label}.mp4")
    raw_clip.export(raw_output)
    with VideoReader(raw_output) as reader:
        frames_before = list(reader.frames())
    
    for assertion in assertions:
        # handle special cases
        if assertion == assert_output_brighter_than_input and any(isinstance(effect, FadeInEffect) for effect in effects) :
            assertion(frames_after, frames_before=frames_before, fadein_flag=True)
        elif assertion == assert_output_brighter_than_input and any(isinstance(effect, FadeOutEffect) for effect in effects) :
            assertion(frames_after, frames_before=frames_before, fadeout_flag=True)
        # functions that only need frames_after ignore frames_before via **kwargs
        else:
            assertion(frames_after, frames_before=frames_before)

