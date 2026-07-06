import pytest
import numpy as np
from reelpy.layers.shapes import SolidLayer, ShapeLayer
from reelpy.layers.text import TextLayer
from reelpy.io.reader import VideoReader
from reelpy.layers.media import ImageLayer, VideoLayer
from reelpy.clip.synthetic import SyntheticClip
from PIL import Image, ImageDraw, ImageFont
from tests.conftest import (LAYER_SHAPE_FIXTURES, LAYER_SOLID_FIXTURES, ALL_CLIP_FACTORY, FONT_FIXTURES,
    make_random_shape_layer, make_random_solid_layer, make_random_text_layer, make_image_png_transparent, make_image_jpg, make_image_png_opaque, make_random_image_layer)

# ── Expectation Functions ──────────────────────────────────
def assert_circle_correct(result, kwargs):
    cx, cy = kwargs["position"]
    radius = kwargs["radius"]
    color = kwargs["color"]
    thickness = kwargs["thickness"]
    if thickness == -1:
        check_x, check_y = cx, cy  # filled: center is safe
    else:
        check_x, check_y = cx + radius, cy  # stroked: check directly on the boundary
    actual_color = result[check_y, check_x, :3]
    assert np.allclose(actual_color, color, atol=10) 
    # NOTE: why atol allowed? H.264 compression drift on a thin shape
    if kwargs.get("has_alpha", True) == True:
        assert result[check_y, check_x, 3] == 255

def assert_rectangle_correct(result, kwargs):
    x, y = kwargs["position"]
    w, h = kwargs["size"]
    color = kwargs["color"]
    thickness = kwargs["thickness"]
    if thickness == -1:
        check_x, check_y = x + w // 2, y + h // 2
    else:
        check_x, check_y = x + w // 2, y  # exactly on the top border line
    assert np.allclose(result[check_y, check_x, :3], color, atol=10)
    if kwargs.get("has_alpha", True) == True:
        assert result[check_y, check_x, 3] == 255

def assert_line_correct(result, kwargs):
    # check the midpoint between start and end
    x1, y1 = kwargs["position"]
    x2, y2 = kwargs["size"]
    mx, my = (x1+x2)//2, (y1+y2)//2
    if kwargs.get("has_alpha", True) == True:
        assert result[my, mx, 3] == 255

def assert_polygon_correct(result, kwargs):
    points = kwargs["points"]
    thickness = kwargs["thickness"]
    if thickness == -1:
        # filled: centroid is safe, interior is painted
        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
    else:
        # stroked: only the outline is painted, check midpoint of first edge
        x1, y1 = points[0]
        x2, y2 = points[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if kwargs.get("has_alpha", True) == True:
        assert result[cy, cx, 3] == 255

SHAPE_ASSERTIONS = {
    "circle": assert_circle_correct,
    "rectangle": assert_rectangle_correct,
    "line": assert_line_correct,
    "polygon": assert_polygon_correct,
}

def assert_solid_layer_correct(frame, kwargs):
    x, y, w, h = kwargs["rect"]
    new_kwargs = {}
    new_kwargs["position"] = (x,y)
    new_kwargs["size"] = (w,h)
    new_kwargs["color"] = kwargs["color"]
    new_kwargs["thickness"] = -1
    new_kwargs["has_alpha"] = False
    assert_rectangle_correct(frame, new_kwargs)

def assert_shape_layer_correct(frame, kwargs):
    shape = kwargs["shape"]
    kwargs["has_alpha"] = False # no alpha ch in exported video frames
    if shape == "circle":
        assert_circle_correct(frame, kwargs)
    elif shape == "rectangle":
        assert_rectangle_correct(frame, kwargs)
    elif shape == "line":
        assert_line_correct(frame, kwargs)
    elif shape == "polygon":
        assert_polygon_correct(frame, kwargs)
    else:
        raise ValueError("No matching shape found in assert_shape_layer_correct.")
    
def assert_text_layer_composited(frame, kwargs, atol=10):
    x, y = kwargs["position"]
    font_size = kwargs["font_size"]
    color = kwargs["color"]
    text = kwargs["text"]
    max_width = kwargs.get("max_width")
    anchor = kwargs.get("anchor", "la")
    if not text.strip():
        return
    msg_length = len(text)

    if max_width is None:
        # single-line path: width scales with character count, height is one line tall
        half_w = (font_size * msg_length) // 2 + 10
        half_h = font_size + 10
    else:
        # wrapped path: width is capped by max_width itself, height scales with estimated line count (rough estimate)
        estimated_lines = max(1, (font_size * msg_length) // max_width + 1) # matches calculations in make_random_text_layer's own margin-sizing logic
        half_w = max_width // 2 + 10
        half_h = (font_size * estimated_lines) + 10

    y0, y1 = max(0, y - half_h), min(frame.shape[0], y + half_h)
    x0, x1 = max(0, x - half_w), min(frame.shape[1], x + half_w)

    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        pytest.fail(f"Search region collapsed to empty — position {(x,y)} likely out of bounds")

    matches = np.all(np.abs(region.astype(int) - np.array(color)) <= atol, axis=-1)
    assert np.any(matches), (
        f"No pixels matching color {color} found near position {(x,y)} "
        f"(anchor={anchor}, max_width={max_width}, font_size={font_size}, text={text!r})"
    )

def assert_im_layer_correct(layer_drawn, image_params):
    assert layer_drawn is not None
    assert layer_drawn.shape == (200,200,4) #canvas dimensions unchanged from test (see test)
    assert np.all(layer_drawn[0:5, 0:5, 3] == 0) # corners should have alpha=0 given image started at (10,10), not true top-left corner

def assert_im_png_transparent_correct(layer_drawn, image_params):
    # compute absolute canvas position of the rectangle's center sinc layer is placed at (10,10) 
    # add image-relative rect coords on top
    check_x = 10 + image_params["rect_x"] + image_params["rect_w"] // 2
    check_y = 10 + image_params["rect_y"] + image_params["rect_h"] // 2
    # check rect center has correct color - PNG is lossless so exact equality expected
    assert tuple(layer_drawn[check_y, check_x, :3]) == image_params["rect_color"], \
        f"Expected rect color {image_params['rect_color']} at ({check_x},{check_y}), got {tuple(layer_drawn[check_y, check_x, :3])}"
    # rect interior must be fully opaque
    assert layer_drawn[check_y, check_x, 3] == 255 # center point
    # canvas pos (10,10) maps to im pixel (0,0) - should be transparent since rect_x and rect_y are always >= 10pixels from image edge based on current make_image_png_transparent
    assert layer_drawn[10,10,3] == 0, \
        "Expected alpha=0 at image top-left"
    
def assert_im_png_opaque_correct(layer_drawn, image_params):
    # compute check pixel: center of the checkerboard block (0,0) in canvas coords
    check_x = 10 + image_params["block_size"] // 2
    check_y = 10 + image_params["block_size"] // 2
    assert tuple(layer_drawn[check_y, check_x, :3]) == image_params["color_a"] # first block should be color a
    assert layer_drawn[check_y, check_x, 3] == 255 # fully opaque 
    assert layer_drawn[10, 10, 3] == 255 # entire im region should be opaque, not just block centroid

def assert_im_jpg_correct(layer_drawn, image_params):
    check_x = 10 + image_params["block_size"] // 2
    check_y = 10 + image_params["block_size"] // 2
    # first block should be color a - tolerance since JPG is lossy so exact equality won't hold
    assert np.allclose(layer_drawn[check_y, check_x, :3], image_params["color_a"], atol=8) 
    assert layer_drawn[check_y, check_x, 3] == 255 # no lpha in JPG
    assert layer_drawn[10, 10, 3] == 255

def assert_image_layer_composited(test_frame, image_params, layer_params, atol=15):
    # image_params = what color/pattern is in the image
    # layer_params = where the layer was plced on the canvas
    img_type = image_params["image_type"]
    # CHECK PIXELS GUARANTEED TO BE COLORED IN
    if img_type == "png_transparent": # solid rect
        check_x = layer_params["position"][0] + image_params["rect_x"] + image_params["rect_w"] // 2
        check_y = layer_params["position"][1] + image_params["rect_y"] + image_params["rect_h"] // 2
        expected_color = image_params["rect_color"]
    else: # both others are checkerboard
        check_x = layer_params["position"][0] + image_params["block_size"] // 2
        check_y = layer_params["position"][1] + image_params["block_size"] // 2
        expected_color = image_params["color_a"]
    assert np.allclose(test_frame[check_y, check_x, :3], expected_color, atol=atol)

# ── Unit level: _draw() in isolation ──────────────────────────

WRAP_TEST_CASES = [ # (text, max_width, description)
    ("Short", 200, "single word, definitely fits one line"),
    ("This is a longer sentence that should wrap onto multiple lines", 100, "normal wrapping"),
    ("Supercalifragilisticexpialidocious", 50, "single word wider than max_width — must not crash, must not split mid-word"),
    ("", 100, "empty string edge case"),
]

IMAGE_TEST_CASES = ["png_transparent", "png_opaque", "jpg"]

@pytest.mark.parametrize("label,kwargs_dict", LAYER_SOLID_FIXTURES)
def test_solid_layer_draw(kwargs_dict, label):
    canvas = np.zeros((100, 100, 4), dtype=np.uint8)
    layer = SolidLayer(**kwargs_dict)
    result = layer._draw(canvas, t=0.0)

    color = kwargs_dict["color"]
    rect = kwargs_dict["rect"]

    if rect is None:
        drawn = result
        untouched = None
    else:
        x, y, w, h = rect
        drawn = result[y:y+h, x:x+w]
        untouched = result[0:5, 0:5]  # assumes rect never starts at (0,0) in fixtures

    # drawn region: correct color, alpha=255
    assert np.all(drawn[:, :, 0] == color[0])
    assert np.all(drawn[:, :, 1] == color[1])
    assert np.all(drawn[:, :, 2] == color[2])
    assert np.all(drawn[:, :, 3] == 255)

    # untouched region (only checked when rect doesn't cover full canvas)
    if untouched is not None:
        assert np.all(untouched[:, :, 3] == 0)


@pytest.mark.parametrize("label,kwargs_dict", LAYER_SHAPE_FIXTURES)
def test_shape_layer_draw(kwargs_dict, label):
    canvas = np.zeros((100, 100, 4), dtype=np.uint8)
    layer = ShapeLayer(**kwargs_dict)
    result = layer._draw(canvas, t=0.0)
    # universal checks: apply to every shape
    assert result is not None
    assert np.any(result[:, :, 3] == 255) # something was drawn
    corner = result[0:5, 0:5]
    assert np.all(corner[:, :, 3] == 0) # corners stay untouched
    # shape specific
    SHAPE_ASSERTIONS[kwargs_dict["shape"]](result, kwargs_dict)


@pytest.mark.parametrize("text,max_width,description", WRAP_TEST_CASES)
def test_text_layer_draw(text, max_width, description):
    overrides = {}
    overrides["text"] = text
    overrides["font_size"] = 24
    overrides["max_width"] = max_width
    overrides["font_path"] = FONT_FIXTURES[0]
    seed = 0
    result, _ = make_random_text_layer(seed, 320, 240, **overrides)
    font = ImageFont.truetype(result.font_path, 24)
    wrapped = result._wrap_text(result.text, font) # must wrap text manually for unit test since this fxn call is only executed automatically at render time
    
    if text == "":
        # edge case: empty string passes through unchanged
        assert wrapped == ""
        return
    
    lines = wrapped.split("\n")
    # testing the diff wrapping strats based on max_width
    for line in lines:
        words_in_line = line.split()
        if len(words_in_line) <= 1:
            # single-word lines are allowed to exceed max_width (oversized-word fallback)
            continue
        assert font.getlength(line) <= max_width, \
            f"[{description}] line exceeded max_width: '{line}' ({font.getlength(line)} > {max_width})"
        
    # no word loss: rejoining all lines' words must reconstruct the original word list
    original_words = text.split()
    rejoined_words = wrapped.replace("\n", " ").split()
    assert rejoined_words == original_words, \
        f"[{description}] word content changed during wrapping: {original_words} != {rejoined_words}"
    

@pytest.mark.parametrize("im_type", IMAGE_TEST_CASES)
def test_image_layer_draw(tmp_path, im_type):
    canvas = np.zeros((200,200,4), dtype=np.uint8)
    if im_type == "png_transparent":
        im_path, params = make_image_png_transparent(tmp_path)
        im_layer = ImageLayer(image_path=im_path, position=(10,10), size=None) #fixed pos
        result = im_layer._draw(canvas,t=0.0)
        assert_im_png_transparent_correct(result, params)
    elif im_type == "png_opaque":
        im_path, params = make_image_png_opaque(tmp_path)
        im_layer = ImageLayer(image_path=im_path, position=(10,10), size=None) #fixed pos
        result = im_layer._draw(canvas,t=0.0)
        assert_im_png_opaque_correct(result, params)
    elif im_type == "jpg":
        im_path, params = make_image_jpg(tmp_path)
        im_layer = ImageLayer(image_path=im_path, position=(10,10), size=None) #fixed pos
        result = im_layer._draw(canvas,t=0.0)
        assert_im_jpg_correct(result, params)
    assert_im_layer_correct(result, params)
    


# ── Integration-level tests ──────────────────────────

# small-ish canvas size for efficiency, 320x240
RANDOM_SOLID_CASES = [make_random_solid_layer(seed, 320, 240) for seed in range(8)]
RANDOM_SHAPE_CASES = [make_random_shape_layer(seed, 320, 240, shape) 
                      for shape in ["circle", "rectangle", "line", "polygon"] # outer
                      for seed in range(3)] # 3 of each shape
RANDOM_TEXT_CASES = [make_random_text_layer(seed, 320, 240, max_width=mw) 
                     for seed in range(3)
                     for mw in [None, 80, 150]]

@pytest.mark.parametrize("clip_lambda, clip_label", ALL_CLIP_FACTORY)
@pytest.mark.parametrize("solid_layer, params_dict", RANDOM_SOLID_CASES)
def test_add_solid_layer(tmp_path, clip_lambda, clip_label, solid_layer, params_dict):
    """Main runner for INTEGRATION level tests w SOLID LAYERS."""
    clip = clip_lambda() # iterates thru constructing all factory objects in ALL_CLIP_FACTORY
    clip = clip.add_layer(solid_layer)
    output = str(tmp_path / f"output_{clip_label}.mp4")
    clip.export(output)
    # verify frames by reading at export level
    with VideoReader(output) as reader:
        frames_after = list(reader.frames())
    assert len(frames_after) > 0
    # choose random frame in the middle
    test_frame, test_t = frames_after[len(frames_after)//2]
    assert_solid_layer_correct(test_frame, params_dict)

@pytest.mark.parametrize("clip_lambda, clip_label", ALL_CLIP_FACTORY)
@pytest.mark.parametrize("shape_layer, params_dict", RANDOM_SHAPE_CASES)
def test_add_shape_layer(tmp_path, clip_lambda, clip_label, shape_layer, params_dict):
    """Main runner for INTEGRATION level tests w SHAPE LAYERS."""
    clip = clip_lambda()
    clip = clip.add_layer(shape_layer)
    output = str(tmp_path / f"output_{clip_label}.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        frames_after = list(reader.frames())
    assert len(frames_after) > 0
    test_frame, test_t = frames_after[len(frames_after)//2]
    assert_shape_layer_correct(test_frame, params_dict)

@pytest.mark.parametrize("clip_lambda, clip_label", ALL_CLIP_FACTORY)
@pytest.mark.parametrize("text_layer, params_dict", RANDOM_TEXT_CASES)
def test_add_text_layer(tmp_path, clip_lambda, clip_label, text_layer, params_dict):
    """Main runner for INTEGRATION level tests w SHAPE LAYERS."""
    clip = clip_lambda()
    clip = clip.add_layer(text_layer)
    output = str(tmp_path / f"output_{clip_label}.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        frames_after = list(reader.frames())
    assert len(frames_after) > 0
    test_frame, test_t = frames_after[len(frames_after)//2]
    assert_text_layer_composited(test_frame, params_dict)

@pytest.mark.parametrize("clip_lambda, clip_label", ALL_CLIP_FACTORY)
@pytest.mark.parametrize("im_type", IMAGE_TEST_CASES)
def test_add_image_layer(tmp_path, clip_lambda, clip_label, im_type):
    if im_type == "png_transparent":
        im_path, params = make_image_png_transparent(tmp_path)
    elif im_type == "png_opaque":
        im_path, params = make_image_png_opaque(tmp_path)
    elif im_type == "jpg":
        im_path, params = make_image_jpg(tmp_path)
        
    layer, layer_params = make_random_image_layer(seed=0, image_path=im_path, image_params=params, canvas_width=320, canvas_height=240)
    clip = clip_lambda()
    clip = clip.add_layer(layer)
    output = str(tmp_path / f"im_output_{clip_label}.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        frames_after = list(reader.frames())
    assert len(frames_after) > 0
    test_frame, test_t = frames_after[len(frames_after)//2]
    assert_image_layer_composited(test_frame, params, layer_params, atol=15)