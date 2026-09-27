"""The puppy model: a YOLO segmentation model fine-tuned on your own camera frames."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from .geometry import Box, Polygon, rect_polygon, to_gray3

log = logging.getLogger(__name__)

PUPPY, MOM = 0, 1
CLASS_NAMES = {PUPPY: "puppy", MOM: "mom"}


@dataclass
class Detection:
    cls: int
    conf: float
    box: Box
    polygon: Polygon = field(default_factory=list)
    number: int | None = None  # stable on-screen number, puppies only
    away: bool = False  # separated from the litter

    def to_dict(self, scale: float = 1.0) -> dict:
        return {
            "cls": self.cls,
            "label": CLASS_NAMES.get(self.cls, str(self.cls)),
            "conf": round(self.conf, 3),
            "box": [round(v * scale, 1) for v in self.box],
            "polygon": [[round(x * scale, 1), round(y * scale, 1)] for x, y in self.polygon],
            "number": self.number,
            "away": self.away,
        }


class Detector:
    """Thread-safe wrapper around an Ultralytics YOLO model (segmentation or detection)."""

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 1280, grayscale: bool = True):
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.grayscale = grayscale
        self.model = YOLO(weights)
        self._lock = threading.Lock()
        # Map the model's class names onto ours, so models with extra/other classes still work.
        self.class_map: dict[int, int] = {}
        for idx, name in self.model.names.items():
            lowered = str(name).lower()
            if lowered in ("puppy", "puppies", "pup"):
                self.class_map[int(idx)] = PUPPY
            elif lowered in ("mom", "mother", "dam", "dog"):
                self.class_map[int(idx)] = MOM
        if PUPPY not in self.class_map.values():
            log.warning("model %s has no 'puppy' class (classes: %s)", weights, self.model.names)

    @property
    def has_mom_class(self) -> bool:
        return MOM in self.class_map.values()

    def warmup(self) -> None:
        self.predict([np.zeros((360, 640, 3), dtype=np.uint8)], conf=0.5)

    def predict(self, frames: list[np.ndarray], conf: float) -> list[list[Detection]]:
        """Detect puppies (and mom) in a batch of frames. Returns one list per frame."""
        if not frames:
            return []
        images = [to_gray3(f) if self.grayscale else f for f in frames]
        with self._lock:
            results = self.model.predict(
                images, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False, max_det=60
            )
        out: list[list[Detection]] = []
        for res in results:
            dets: list[Detection] = []
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy().tolist()
                confs = res.boxes.conf.cpu().numpy().tolist()
                classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
                polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
                for box, score, c, poly in zip(boxes, confs, classes, polys):
                    if c not in self.class_map:
                        continue
                    polygon = [[float(x), float(y)] for x, y in poly] if poly is not None and len(poly) >= 3 else []
                    dets.append(
                        Detection(cls=self.class_map[c], conf=float(score), box=box, polygon=polygon or rect_polygon(box))
                    )
            out.append(dets)
        return out
