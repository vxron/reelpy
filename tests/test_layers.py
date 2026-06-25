import pytest
import numpy as np
from reelpy.layers.shapes import SolidLayer, ShapeLayer
from reelpy.io.reader import VideoReader
from tests.conftest import LAYER_SHAPE_FIXTURES, LAYER_SOLID_FIXTURES, ALL_CLIP_FACTORY, make_random_shape_layer, make_random_solid_layer, make_random_text_layer

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
    assert np.allclose(actual_color, color, atol=6) # tolerance of 3 for real vid footage
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
    assert np.allclose(result[check_y, check_x, :3], color, atol=6)
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

# ── Unit level: _draw() in isolation ──────────────────────────

WRAP_TEST_CASES = [
    ("Short", 200, "single word, definitely fits one line"),
    ("This is a longer sentence that should wrap onto multiple lines", 100, "normal wrapping"),
    ("Supercalifragilisticexpialidocious", 50, "single word wider than max_width — must not crash, must not split mid-word"),
    ("", 100, "empty string edge case"),
]

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

# ── Integration-level tests ──────────────────────────

# small-ish canvas size for efficiency, 320x240
RANDOM_SOLID_CASES = [make_random_solid_layer(seed, 320, 240) for seed in range(10)]
RANDOM_SHAPE_CASES = [make_random_shape_layer(seed, 320, 240, shape) 
                      for shape in ["circle", "rectangle", "line", "polygon"] # outer
                      for seed in range(3)] # 3 of each shape
RANDOM_TEXT_CASES = [make_random_text_layer(seed, 320, 240) for seed in range(10)]

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