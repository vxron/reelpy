"""
File: conftest.py
Description: fixtures, constants, helper functions, made automatically available to all test files
"""
import pytest 
import random
import math
import numpy as np
from PIL import Image
from reelpy.config import config
from reelpy.clip.synthetic import SyntheticClip
from reelpy.clip.video import Clip
from reelpy.effects.base import BaseEffect
from reelpy.effects.fades import FadeInEffect, FadeOutEffect
from reelpy.layers.shapes import SolidLayer, ShapeLayer
from reelpy.layers.media import ImageLayer, VideoLayer
from reelpy.layers.text import TextLayer


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
    parser.addoption(
        "--local",
        action="store_true",
        default=False,
        help="run ALL tests including memory-heavy ones only suited for local machines, NOT CI"
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
    local = config.getoption("--local")

    # always skip local-only tests unless --local flag is passed
    if not local: 
        skip_local = pytest.mark.skip(reason="too memory-heavy for CI runners, run locally only")
        for item in items:
            if "local" in item.keywords:
                item.add_marker(skip_local) # skip local on CI 

    if exhaustive: # run all w/o filtering (except obv can't run local on CI)
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


# ── Ready-To-Go Fixtures & Primitives ────────────────────────────────────────────────────────

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

FONT_FIXTURES = [
    "tests/fixtures/fonts/DejaVuSans.ttf",
    "tests/fixtures/fonts/DejaVuSerif.ttf",
    "tests/fixtures/fonts/LiberationSerif-Regular.ttf",
    "tests/fixtures/fonts/LiberationSerif-Italic.ttf",
    "tests/fixtures/fonts/LiberationSerif-Bold.ttf",
    "tests/fixtures/fonts/LiberationSerif-BoldItalic.ttf",
    "tests/fixtures/fonts/LiberationSans-Regular.ttf",
]

# FIXTURE IMAGES
def make_image_png_transparent(tmp_path, seed=0):
    """
    Generates a PNG with a colored rectangle on a transparent background.
    Returns (file_path, params_dict) where params_dict contains rect position & color & size.
    """
    rng = random.Random(seed)
    img_w, img_h = 100, 100 # small-ish for speed
    # pick a random rect color (avoid near-black/white for reliable detection)
    rect_color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
    # pick rect bounds
    rect_x = rng.randint(10, 30)
    rect_y = rng.randint(10, 30)
    rect_w = rng.randint(20, 40)
    rect_h = rng.randint(20, 40)
    
    # build RGBA array - starts transparent
    arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    # fill rectangle with solid color
    arr[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w, :3] = rect_color
    arr[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w,  3] = 255 # full opcity
    
    # save as PNG 
    path = str(tmp_path / f"test_transparent_{seed}.png")
    Image.fromarray(arr, mode="RGBA").save(path, format="PNG")
    
    params_dict = {
        "image_type":     "png_transparent",
        "rect_color":     rect_color,
        "rect_x":         rect_x,
        "rect_y":         rect_y,
        "rect_w":         rect_w,
        "rect_h":         rect_h,
        "size":           (100,100)
    }
    return path, params_dict


def make_image_png_opaque(tmp_path, seed=0):
    """
    Generates a fully opaque PNG with a checkerboard pattern - Two alternating colors in 20x20 blocks.
    Returns (file_path, params_dict).
    """
    rng = random.Random(seed)
    img_w, img_h = 100, 100
    block_size = 20
    
    # pick two visually distinct colors
    color_a = (rng.randint(150, 255), rng.randint(0,  80),  rng.randint(0,  80))   # reddish
    color_b = (rng.randint(0,  80),  rng.randint(150, 255), rng.randint(0,  80))   # greenish
    
    # build RGB array & fill in checkerboard blocks
    arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for row in range(img_h // block_size):
        for col in range(img_w // block_size):
            color = color_a if (row + col) % 2 == 0 else color_b
            arr[row*block_size:(row+1)*block_size,
                col*block_size:(col+1)*block_size] = color
    
    # save as PNG 
    path = str(tmp_path / f"test_opaque_{seed}.png")
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    
    params_dict = {
        "image_type":     "png_opaque",
        "color_a":        color_a,
        "color_b":        color_b,
        "block_size":     block_size,
        "size":           (100,100)
    }
    return path, params_dict


def make_image_jpg(tmp_path, seed=0):
    """
    Generates a JPG checkerboard with same pattern as make_test_png_opaque
    but saved as JPG (lossy). JPG has no alpha channel. 
    Uses large blocks (40x40) so block centers are far from compression
    artifacts at block edges since JPEG compresses in 8x8 macroblocks, so
    checking the center of a 40x40 block avoids boundary ringing artifacts.
    """
    rng = random.Random(seed)
    img_w, img_h = 100, 100
    block_size = 40  # larger than PNG checkerboard
    color_a = (rng.randint(150, 220), rng.randint(0,  60),  rng.randint(0,  60))   # reddish
    color_b = (rng.randint(0,  60),  rng.randint(150, 220), rng.randint(0,  60))   # greenish
    
    arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for row in range(img_h // block_size + 1):  # +1 to handle partial blocks at boundary
        for col in range(img_w // block_size + 1):
            color = color_a if (row + col) % 2 == 0 else color_b
            r0, r1 = row*block_size, min((row+1)*block_size, img_h)
            c0, c1 = col*block_size, min((col+1)*block_size, img_w)
            arr[r0:r1, c0:c1] = color
    
    # quality=95 keeps drift well within atol=15 for block centers
    path = str(tmp_path / f"test_checkerboard_{seed}.jpg")
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=95)
    params_dict = {
        "image_type":     "jpg",
        "size":           (100,100), 
        "color_a":        color_a,
        "color_b":        color_b,
        "block_size":     block_size,
    }
    return path, params_dict


# FIXTURE FRAMES
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

LAYER_SHAPE_FIXTURES = [ # tuples with (shape_label, constructor kwargs dict)
    ("circle_filled", {"shape": "circle", "position": (50, 50), "radius": 20, "size": None, "points": None, "color": (200,100,50), "thickness": -1}),
    ("circle_stroked", {"shape": "circle", "position": (50, 50), "radius": 20, "size": None, "points": None, "color": (200,100,50), "thickness": 2}),
    ("rectangle_filled", {"shape": "rectangle", "position": (10, 10), "size": (30, 20), "radius": None, "points": None, "color": (50,200,100), "thickness": -1}),
    ("rectangle_stroked", {"shape": "rectangle", "position": (10, 10), "size": (30, 20), "radius": None, "points": None, "color": (50,200,100), "thickness": 2}),
    ("line", {"shape": "line", "position": (10, 10), "size": (80, 80), "radius": None, "points": None, "color": (0,0,200), "thickness": 3}),
    ("polygon_filled", {"shape": "polygon", "position": None, "size": None, "radius": None, "points": [(20,20),(60,20),(40,60)], "color": (200,200,0), "thickness": -1}),
    ("polygon_stroked", {"shape": "polygon", "position": None, "size": None, "radius": None, "points": [(20,20),(60,20),(40,60)], "color": (200,200,0), "thickness": 2})
]

LAYER_SOLID_FIXTURES = [ # tuples with (solid_label, constructor kwargs dict)
    ("solid_fullcanvas", {"color": (255,0,0), "rect": None}),
    ("solid_rect", {"color": (0,255,0), "rect": (10, 10, 30, 20)})
]

BLANK_CANVASES = {
    "blank_rgba_small": np.zeros((100, 100, 4), dtype=np.uint8),
    "blank_rgba_medium": np.zeros((240, 320, 4), dtype=np.uint8),
}

# ── Random Generators of Fixtures ────────────────────────────────────────────────────────
    
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

def make_random_solid_layer(seed, width, height):
    params_dict = {}
    rng = random.Random(seed)
    color = (int(rng.uniform(0, 255)), int(rng.uniform(0, 255)), int(rng.uniform(0, 255)))
    # pick size first, then bound position
    w = int(rng.uniform(5, width // 3))
    h = int(rng.uniform(5, height // 3))
    random_x = int(rng.uniform(0, width - w))
    random_y = int(rng.uniform(0, height - h))
    rect = (random_x, random_y, w, h)
    solid_layer = SolidLayer(
        color = color,
        rect  = rect,
    )
    params_dict["color"] = color
    params_dict["rect"] = rect
    return solid_layer, params_dict

def make_random_shape_layer(seed, width, height, shape: str | None = None):
    rng = random.Random(seed)
    if shape is None:
        # need to randomly choose shape
        shape = rng.choice(["circle", "rectangle", "line", "polygon"])
    
    params_dict = {}

    if shape == "circle":
        # center must be at least radius away from every edge
        # pick radius first
        radius = rng.randint(5, min(width, height) // 4)
        # then bound center pos
        x = rng.randint(radius, width - radius)
        y = rng.randint(radius, height - radius)
        shape_layer = ShapeLayer(
            shape = "circle",
            position = (x,y),
            size = None,
            radius = radius,
            points = None,
        )
    elif shape == "rectangle":
        # pick size first (min 10, max 1/3 dim of canvas)
        w = rng.randint(10, width // 3) 
        h = rng.randint(10, height // 3)
        size = (w,h)
        # pick position according to size 
        x = rng.randint(0, width-w)
        y = rng.randint(0, height-h)
        shape_layer = ShapeLayer(
            shape = "rectangle",
            position = (x,y),
            size = size,
            radius = None,
            points = None,
        )

    elif shape == "line":
        x1, y1 = rng.randint(0, width-1), rng.randint(0, height-1)
        x2, y2 = rng.randint(0, width-1), rng.randint(0, height-1)
        shape_layer = ShapeLayer(
            shape = "line",
            position = (x1,y1),
            size = (x2,y2),
            radius = None,
            points = None,
        )

    elif shape == "polygon":
        n_points = rng.randint(3, 6) # triangle to hexagon range
        poly_radius = rng.randint(15, 40)
        margin = poly_radius + 5  # small buffer
        cx, cy = rng.randint(margin, width-margin), rng.randint(margin, height-margin) # center of an invisible circle that the polygon will be inscribed in
        # generate points within canvas bounds, around the circle at random angles to guarantee convex (non self-intersecting) shape
        angles = sorted(rng.uniform(0, 2*math.pi) for _ in range(n_points))
        points = [
            (int(cx + poly_radius * math.cos(a)), int(cy + poly_radius * math.sin(a)))
            for a in angles
        ]
        shape_layer = ShapeLayer(
            shape = "polygon",
            position = None,
            size = None,
            radius = None,
            points = points,
        )

    params_dict = vars(shape_layer)
    return shape_layer, params_dict

def make_random_text_layer(seed, width, height, **overrides):
    rng = random.Random(seed)
    font_path = rng.choice(FONT_FIXTURES + [None]) # none = default font if no path provided
    text = rng.choice(["Test", "hello! this is a longer message!", "Reelpy<3", "43"]) # make sure min length of msg is 2 here to avoid dividebyzero error
    msg_length = len(text)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    anchor = rng.choice(["la", "mm", "ls"])

    # font size bounded relative to canvas + text length + max width if specified in overrides
    font_size = overrides.get("font_size")
    if font_size is None:
        max_font_size = max(15, (min(width, height) // max(msg_length // 2, 1)) )
        font_size = rng.randint(10, max_font_size)
        if overrides.get("max_width") is not None:
            # keep font_size reasonable relative to max_width so it's not just one word per line
            font_size = min(font_size, overrides["max_width"] // 4)
    # floor at 30px for test reliability - smaller fonts have too few solid-color interior pixels to survive H.264 encode/decode round-trip detection.
    font_size = max(font_size, 25)

    # margin computed deterministically: how much space could this text occupy,
    # regardless of which anchor is used (anchor could place text in any direction
    # relative to position, so margin must cover the worst case on every side)
    text_extent_w = font_size * msg_length
    text_extent_h = font_size * 2
    margin_w = min(text_extent_w, width // 2 - 1)   # never exceed half the canvas
    margin_h = min(text_extent_h, height // 2 - 1)
    # randomize position WITHIN the safe remaining area
    x = rng.randint(margin_w, width - margin_w)
    y = rng.randint(margin_h, height - margin_h)
    
    text_layer = TextLayer(
        text = overrides.get("text", text),
        position = (x,y),
        font_path = overrides.get("font_path", font_path),
        font_size = font_size,
        color = color,
        anchor = anchor,
        max_width = overrides.get("max_width", None),
        line_spacing = overrides.get("line_spacing", 4)
    )
    params_dict = vars(text_layer)
    return text_layer, params_dict


def make_random_image_layer(seed, image_path, image_params, canvas_width, canvas_height) -> tuple[ImageLayer, dict]:
    rng = random.Random(seed)
    size = image_params.get("size", (100,100))
    x = rng.randint(0, canvas_width - size[0])
    y = rng.randint(0, canvas_height - size[1])
    layer = ImageLayer(image_path=image_path, position=(x,y), size=None)
    layer_params = {}
    layer_params["position"] = (x,y)
    return layer, layer_params


def make_random_video_layer(seed, video_params, canvas_width, canvas_height, video_path = None) -> tuple[ImageLayer, dict]:
    rng = random.Random(seed)
    layer_params = {}
    
    if video_path is None:
        # random synthetic clip of reasonable size
        bg_color = (rng.randint(50,200), rng.randint(50,200), rng.randint(50,200))
        duration = float(rng.randint(1,4))
        fps = 30.0
        loop = rng.choice([True, False])
        t_offset = rng.uniform(0.0, duration)
        size = (rng.randint(160,320), rng.randint(120,240)) # fixed small size
        x = rng.randint(0, canvas_width - size[0])
        y = rng.randint(0, canvas_height - size[1])
        source_clip = SyntheticClip(size[0], size[1], fps, duration, bg_color)
        layer = VideoLayer(source_clip, (x,y), size, loop, t_offset)
    
    else:
        size = video_params.get("size")
    
    
    layer_params["position"] = (x,y)
    layer_params["background"] = bg_color
    layer_params["duration"] = duration
    layer_params["fps"] = fps
    layer_params["loop"] = loop
    layer_params["size"] = size
    layer_params["t_offset"] = t_offset
    return layer, layer_params



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