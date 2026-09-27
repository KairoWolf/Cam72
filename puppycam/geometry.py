"""Small geometry helpers: masks, polygons, boxes and YOLO label lines."""

from __future__ import annotations

import math

import cv2
import numpy as np

Box = list[float]  # [x1, y1, x2, y2] in pixels
Polygon = list[list[float]]  # [[x, y], ...] in pixels


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0, min_part: float = 0.1) -> Polygon:
    """Outline of a binary mask as one polygon.

    Parts smaller than ``min_part`` of the largest part are dropped (SAM speckles). When a puppy is
    split into several visible parts (e.g. another puppy lies across it), the parts are joined with
    zero-width bridges, the same way Ultralytics converts COCO masks.
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    areas = [cv2.contourArea(c) for c in contours]
    largest = max(areas)
    if largest <= 0:
        return []
    parts = [c for c, a in zip(contours, areas) if a >= min_part * largest]
    parts = [cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2) for c in parts]
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return []
    if len(parts) == 1:
        points = parts[0]
    else:
        try:
            from ultralytics.data.converter import merge_multi_segment

            points = np.concatenate(merge_multi_segment([p.reshape(-1).tolist() for p in parts]), axis=0)
        except Exception:
            points = max(parts, key=lambda p: cv2.contourArea(p.astype(np.float32)))
    return [[float(x), float(y)] for x, y in points]


def polygon_box(poly: Polygon) -> Box:
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def polygon_area(poly: Polygon) -> float:
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return float(abs(cv2.contourArea(arr))) if len(arr) >= 3 else 0.0


def rect_polygon(box: Box) -> Polygon:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def clip_polygon(poly: Polygon, width: int, height: int) -> Polygon:
    return [[min(max(float(x), 0.0), width - 1.0), min(max(float(y), 0.0), height - 1.0)] for x, y in poly]


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0)))
    return [float(x1), float(y1), float(x2), float(y2)]


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_size(box: Box) -> float:
    """Characteristic length of a box (geometric mean of width and height)."""
    return math.sqrt(box_area(box))


def box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_gap(a: Box, b: Box) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def to_yolo_line(cls: int, poly: Polygon, width: int, height: int) -> str:
    coords = []
    for x, y in poly:
        coords.append(f"{min(max(x / width, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(y / height, 0.0), 1.0):.6f}")
    return f"{int(cls)} " + " ".join(coords)


def from_yolo_line(line: str, width: int, height: int) -> tuple[int, Polygon]:
    parts = line.split()
    cls = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:  # plain detection label: cx cy w h
        cx, cy, w, h = values
        box = [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
        return cls, rect_polygon(box)
    return cls, [[values[i] * width, values[i + 1] * height] for i in range(0, len(values) - 1, 2)]


def to_gray3(frame: np.ndarray) -> np.ndarray:
    """Grayscale image with 3 channels, so day (color) and night (IR) frames look alike to the model."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def autocrop_borders(image: np.ndarray, threshold: int = 10, min_keep: float = 0.5) -> np.ndarray:
    """Remove the black bars around phone screenshots of a camera app."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cols = np.where(gray.mean(axis=0) > threshold)[0]
    rows = np.where(gray.mean(axis=1) > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return image
    x1, x2, y1, y2 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    h, w = gray.shape[:2]
    if (x2 - x1) < min_keep * w or (y2 - y1) < min_keep * h:
        return image
    return image[y1:y2, x1:x2].copy()
