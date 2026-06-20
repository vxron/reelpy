import pytest
import numpy as np
from reelpy.layers.shapes import SolidLayer, ShapeLayer
from tests.conftest import LAYER_SHAPE_FIXTURES, LAYER_SOLID_FIXTURES

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
    assert tuple(result[check_y, check_x, :3]) == color
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
    assert tuple(result[check_y, check_x, :3]) == color
    assert result[check_y, check_x, 3] == 255

def assert_line_correct(result, kwargs):
    # check the midpoint between start and end
    x1, y1 = kwargs["position"]
    x2, y2 = kwargs["size"]
    mx, my = (x1+x2)//2, (y1+y2)//2
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
    assert result[cy, cx, 3] == 255

SHAPE_ASSERTIONS = {
    "circle": assert_circle_correct,
    "rectangle": assert_rectangle_correct,
    "line": assert_line_correct,
    "polygon": assert_polygon_correct,
}

# ── Unit level: _draw() in isolation ──────────────────────────

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