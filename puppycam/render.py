"""Draws the highlight overlay: a colored outline and a numbered badge on every puppy."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .analysis import CameraState

# Bright, easy to tell apart on a gray night-vision image (BGR).
# Red is reserved for a puppy that is away from the litter.
PALETTE = [
    (0, 200, 255), (80, 220, 80), (255, 160, 30), (230, 90, 230), (220, 220, 0),
    (0, 140, 255), (160, 100, 255), (60, 255, 190), (255, 90, 140), (40, 180, 180),
    (190, 190, 255), (255, 255, 120),
]
AWAY_COLOR = (40, 40, 255)
MOM_COLOR = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(number: int | None) -> tuple[int, int, int]:
    return PALETTE[((number or 1) - 1) % len(PALETTE)]


def _centroid(pts: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(pts.astype(np.float32))
    if m["m00"]:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def _text(img, text, org, size, color, thickness, bg=None, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, size, thickness)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x, y), FONT, size, color, thickness, cv2.LINE_AA)
    return tw, th


def _badge(img, center, text, color, radius, font_size, thickness):
    x, y = center
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT, font_size, thickness)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    ink = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    cv2.putText(img, text, (x - tw // 2, y + th // 2), FONT, font_size, ink, thickness, cv2.LINE_AA)


def render(
    frame: np.ndarray,
    state: CameraState | None,
    label: str,
    expected: int,
    message: str | None = None,
    max_width: int = 1280,
) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        img = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = frame.copy()
    H = img.shape[0]
    unit = max(1, round(H / 400))
    font = H / 900.0
    blink = int(time.time() * 2) % 2 == 0

    if state is not None:
        polys = []
        for det in state.puppies + ([state.mom] if state.mom else []):
            pts = (np.asarray(det.polygon, dtype=np.float32) * scale).astype(np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                polys.append((det, pts))

        # Translucent fill, then crisp outlines on top.
        overlay = img.copy()
        for det, pts in polys:
            if det.cls == 0:
                cv2.fillPoly(overlay, [pts], AWAY_COLOR if det.away else color_for(det.number))
        img = cv2.addWeighted(overlay, 0.38, img, 0.62, 0)
        for det, pts in polys:
            if det.cls == 0:
                color = AWAY_COLOR if det.away else color_for(det.number)
                thick = 2 * unit + (2 * unit if det.away and blink else 0)
                cv2.polylines(img, [pts], True, (0, 0, 0), thick + 2, cv2.LINE_AA)
                cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True, MOM_COLOR, unit, cv2.LINE_AA)
        for det, pts in polys:
            cx, cy = _centroid(pts)
            if det.cls == 0:
                color = AWAY_COLOR if det.away else color_for(det.number)
                radius = max(9, int(H * 0.021))
                _badge(img, (cx, cy), str(det.number or "?"), color, radius, radius / 24.0, max(1, unit))
                if det.away:
                    _text(img, "ALONE", (cx + radius + 6, cy + radius // 2), font * 0.8, (255, 255, 255),
                          max(1, unit), bg=AWAY_COLOR)
            else:
                _text(img, "Mom", (cx - int(20 * unit), cy), font * 0.8, (0, 0, 0), max(1, unit), bg=MOM_COLOR)

    # Header strip.
    bar_h = int(H * 0.075)
    strip = img[:bar_h].copy()
    cv2.rectangle(strip, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
    img[:bar_h] = cv2.addWeighted(strip, 0.55, img[:bar_h], 0.45, 0)
    baseline = int(bar_h * 0.72)
    tw, _ = _text(img, label, (int(12 * unit), baseline), font * 0.9, (255, 255, 255), max(1, unit))
    if state is not None:
        count_text = f"Puppies {state.smoothed}/{expected}"
        ok = state.smoothed >= expected
        _text(img, count_text, (int(12 * unit) + tw + int(24 * unit), baseline), font * 0.9,
              (120, 230, 120) if ok else (60, 190, 255), max(1, 2 * unit))
    stamp = time.strftime("%H:%M:%S")
    (sw, _), _ = cv2.getTextSize(stamp, FONT, font * 0.8, max(1, unit))
    _text(img, stamp, (img.shape[1] - sw - int(12 * unit), baseline), font * 0.8, (220, 220, 220), max(1, unit))

    if message:
        lines = message.split("\n")
        for i, line in enumerate(lines):
            (lw, lh), _ = cv2.getTextSize(line, FONT, font * 0.9, max(1, unit))
            y = H // 2 + int((i - len(lines) / 2) * lh * 2.0)
            _text(img, line, ((img.shape[1] - lw) // 2, y), font * 0.9, (255, 255, 255), max(1, unit), bg=(0, 0, 0), pad=8)
    return img


def placeholder(label: str, text: str, size=(1280, 720)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), 32, dtype=np.uint8)
    return render(img, None, label, 0, message=text, max_width=size[0])
