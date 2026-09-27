"""Box -> precise outline with SAM 2, used by the labeling page.

You draw a rough box around one puppy; SAM traces its outline. Image features are cached, so
every extra box on the same frame takes a few milliseconds.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .geometry import Box, Polygon, mask_to_polygon, rect_polygon

log = logging.getLogger(__name__)


class BoxSegmenter:
    def __init__(self, weights: str, device: str = "cpu"):
        self.weights = weights
        self.device = device
        self._predictor = None
        self._key: str | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def _load(self):
        from ultralytics.models.sam import SAM2Predictor

        overrides = dict(conf=0.0, task="segment", mode="predict", imgsz=1024, model=self.weights,
                         device=self.device, save=False, verbose=False)
        self._predictor = SAM2Predictor(overrides=overrides)

    def segment(self, key: str, image: np.ndarray, box: Box) -> tuple[Polygon, bool]:
        """Outline of the object inside ``box``. Returns (polygon, used_sam)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        if x2 - x1 < 4 or y2 - y1 < 4:
            return rect_polygon(box), False
        try:
            with self._lock:
                if self._predictor is None:
                    self._load()
                if key != self._key:
                    self._predictor.set_image(image)
                    self._key = key
                result = self._predictor(bboxes=[[x1, y1, x2, y2]])[0]
            if result.masks is None or not len(result.masks.data):
                return rect_polygon(box), False
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
        except Exception as exc:
            self.error = str(exc)
            log.exception("SAM failed, falling back to the plain box")
            self._key = None
            return rect_polygon(box), False

        # Keep the outline inside the drawn box (plus a small margin) so it cannot leak into mom.
        mx, my = 0.06 * (x2 - x1), 0.06 * (y2 - y1)
        keep = np.zeros_like(mask)
        keep[max(0, int(y1 - my)):min(h, int(y2 + my) + 1), max(0, int(x1 - mx)):min(w, int(x2 + mx) + 1)] = 1
        mask &= keep
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        area = int(mask.sum())
        if area < 0.08 * (x2 - x1) * (y2 - y1):  # SAM found nothing sensible
            return rect_polygon(box), False
        poly = mask_to_polygon(mask, epsilon=1.2)
        return (poly, True) if len(poly) >= 3 else (rect_polygon(box), False)
