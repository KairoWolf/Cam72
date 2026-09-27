import numpy as np
import pytest

from puppycam.geometry import (
    autocrop_borders, box_gap, box_iou, clip_box, from_yolo_line, mask_to_polygon, polygon_area, polygon_box,
    rect_polygon, to_yolo_line,
)


def test_mask_to_polygon_rectangle():
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 30:130] = 1
    poly = mask_to_polygon(mask)
    assert polygon_box(poly) == pytest.approx([30, 20, 129, 59], abs=1)
    assert polygon_area(poly) == pytest.approx(99 * 39, rel=0.05)


def test_mask_to_polygon_merges_split_parts_and_drops_specks():
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 10:60] = 1  # part one
    mask[20:60, 100:150] = 1  # part two (a puppy with another one lying across it)
    mask[90:92, 190:192] = 1  # speck
    poly = mask_to_polygon(mask)
    x1, y1, x2, y2 = polygon_box(poly)
    assert x1 <= 11 and x2 >= 148  # both parts are in one outline
    assert y2 < 80  # the speck is gone


def test_mask_to_polygon_empty():
    assert mask_to_polygon(np.zeros((10, 10), np.uint8)) == []


def test_box_helpers():
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert box_iou(a, b) == pytest.approx(25 / 175)
    assert box_iou(a, [20, 20, 30, 30]) == 0
    assert box_gap(a, b) == 0
    assert box_gap(a, [13, 0, 20, 10]) == pytest.approx(3)
    assert box_gap(a, [13, 14, 20, 20]) == pytest.approx(5)
    assert clip_box([-5, 3, 500, 40], 100, 50) == [0, 3, 99, 40]


def test_yolo_line_roundtrip():
    poly = [[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]]
    line = to_yolo_line(1, poly, 200, 100)
    assert line.startswith("1 0.050000 0.200000")
    cls, back = from_yolo_line(line, 200, 100)
    assert cls == 1
    assert np.allclose(back, poly)


def test_yolo_detection_line_becomes_rectangle():
    cls, poly = from_yolo_line("0 0.5 0.5 0.2 0.4", 100, 100)
    assert cls == 0
    assert polygon_box(poly) == pytest.approx([40, 30, 60, 70])
    assert rect_polygon([40, 30, 60, 70]) == poly


def test_autocrop_borders_removes_black_bars():
    img = np.zeros((100, 300, 3), np.uint8)
    img[:, 50:250] = 200
    out = autocrop_borders(img)
    assert out.shape == (100, 200, 3)
    assert autocrop_borders(np.zeros((10, 10, 3), np.uint8)).shape == (10, 10, 3)
