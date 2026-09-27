"""Frames saved for labeling, and their labels (JSON + YOLO segmentation .txt)."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import clip_polygon, polygon_area, polygon_box, rect_polygon, to_yolo_line

VALID_CLASSES = (0, 1)


class Dataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.meta = self.root / "meta"
        for d in (self.images, self.labels, self.meta):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # paths ---------------------------------------------------------------------------------
    def image_path(self, frame_id: str) -> Path:
        return self.images / f"{frame_id}.jpg"

    def label_path(self, frame_id: str) -> Path:
        return self.labels / f"{frame_id}.txt"

    def meta_path(self, frame_id: str) -> Path:
        return self.meta / f"{frame_id}.json"

    @staticmethod
    def valid_id(frame_id: str) -> bool:
        return bool(frame_id) and all(c.isalnum() or c in "-_" for c in frame_id)

    # writing -------------------------------------------------------------------------------
    def add(self, image: np.ndarray, camera: str, reason: str, created: float | None = None) -> str:
        created = created or time.time()
        stamp = datetime.fromtimestamp(created).strftime("%Y%m%d-%H%M%S")
        cam = "".join(c if c.isalnum() else "-" for c in camera)[:24] or "cam"
        with self._lock:
            frame_id, n = f"{stamp}-{cam}", 2
            while self.meta_path(frame_id).exists():
                frame_id, n = f"{stamp}-{cam}-{n}", n + 1
            h, w = image.shape[:2]
            cv2.imwrite(str(self.image_path(frame_id)), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta = {
                "id": frame_id,
                "camera": camera,
                "reason": reason,
                "created": created,
                "width": w,
                "height": h,
                "labeled": False,
                "objects": [],
            }
            self._write_meta(meta)
        return frame_id

    def _write_meta(self, meta: dict) -> None:
        tmp = self.meta_path(meta["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(self.meta_path(meta["id"]))

    def save_labels(self, frame_id: str, objects: list[dict]) -> dict:
        """Store the reviewed objects. An empty list means 'no puppies in this frame'."""
        meta = self.get(frame_id)
        if meta is None:
            raise KeyError(frame_id)
        w, h = meta["width"], meta["height"]
        clean = []
        for obj in objects:
            cls = int(obj.get("cls", 0))
            if cls not in VALID_CLASSES:
                raise ValueError(f"unknown class {cls}")
            poly = obj.get("polygon") or []
            if len(poly) < 3 and obj.get("box"):
                poly = rect_polygon([float(v) for v in obj["box"]])
            poly = clip_polygon([[float(x), float(y)] for x, y in poly], w, h)
            if len(poly) < 3 or polygon_area(poly) < 4:
                continue
            clean.append({"cls": cls, "polygon": [[round(x, 1), round(y, 1)] for x, y in poly],
                          "box": [round(v, 1) for v in polygon_box(poly)]})
        with self._lock:
            self.label_path(frame_id).write_text("".join(to_yolo_line(o["cls"], o["polygon"], w, h) + "\n" for o in clean))
            meta.update(labeled=True, objects=clean, labeled_at=time.time())
            self._write_meta(meta)
        return meta

    def delete(self, frame_id: str) -> None:
        with self._lock:
            for path in (self.image_path(frame_id), self.label_path(frame_id), self.meta_path(frame_id)):
                path.unlink(missing_ok=True)

    # reading -------------------------------------------------------------------------------
    def get(self, frame_id: str) -> dict | None:
        if not self.valid_id(frame_id):
            return None
        try:
            return json.loads(self.meta_path(frame_id).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def all(self) -> list[dict]:
        items = []
        for path in self.meta.glob("*.json"):
            try:
                items.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return items

    def list(self, status: str = "all") -> list[dict]:
        items = self.all()
        if status == "unlabeled":
            items = [m for m in items if not m.get("labeled")]
            # Frames the model found hard come first: they teach it the most.
            items.sort(key=lambda m: (not str(m.get("reason", "")).startswith("uncertain"), -m["created"]))
        elif status == "labeled":
            items = [m for m in items if m.get("labeled")]
            items.sort(key=lambda m: -m.get("labeled_at", m["created"]))
        else:
            items.sort(key=lambda m: -m["created"])
        return [
            {
                "id": m["id"],
                "camera": m.get("camera"),
                "reason": m.get("reason"),
                "created": m["created"],
                "labeled": m.get("labeled", False),
                "puppies": sum(1 for o in m.get("objects", []) if o["cls"] == 0),
                "mom": sum(1 for o in m.get("objects", []) if o["cls"] == 1),
            }
            for m in items
        ]

    def labeled(self) -> list[dict]:
        return [m for m in self.all() if m.get("labeled")]

    def stats(self) -> dict:
        items = self.all()
        labeled = [m for m in items if m.get("labeled")]
        cameras: dict[str, dict] = {}
        for m in items:
            c = cameras.setdefault(m.get("camera") or "?", {"frames": 0, "labeled": 0})
            c["frames"] += 1
            c["labeled"] += int(bool(m.get("labeled")))
        return {
            "frames": len(items),
            "labeled": len(labeled),
            "unlabeled": len(items) - len(labeled),
            "puppies": sum(1 for m in labeled for o in m.get("objects", []) if o["cls"] == 0),
            "moms": sum(1 for m in labeled for o in m.get("objects", []) if o["cls"] == 1),
            "cameras": cameras,
        }


class CapturePolicy:
    """Decides which live frames to keep for labeling.

    * one frame per camera every ``every_minutes`` (variety: day/night, sleeping/feeding), and
    * frames where the model is unsure (active learning): these improve accuracy the fastest.
    Near-identical frames are skipped, and capturing pauses when the unlabeled pile gets big.
    """

    def __init__(self, dataset: Dataset, every_minutes: float, uncertain: bool, max_per_hour: int,
                 max_unlabeled: int, min_gap_seconds: float = 300.0):
        self.dataset = dataset
        self.every = every_minutes * 60
        self.uncertain = uncertain
        self.max_per_hour = max_per_hour
        self.max_unlabeled = max_unlabeled
        self.min_gap = min_gap_seconds
        self._last_periodic: dict[str, float] = {}
        self._last_uncertain: dict[str, float] = {}
        self._uncertain_times: list[float] = []
        self._thumbs: dict[str, list[np.ndarray]] = {}
        self._unlabeled_cache = (0.0, 0)

    def _unlabeled(self, ts: float) -> int:
        if ts - self._unlabeled_cache[0] > 60:
            self._unlabeled_cache = (ts, sum(1 for m in self.dataset.all() if not m.get("labeled")))
        return self._unlabeled_cache[1]

    def _different(self, camera: str, frame: np.ndarray) -> bool:
        """False when the frame looks like one of the last few frames saved from this camera."""
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA)
        recent = self._thumbs.setdefault(camera, [])
        if any(float(np.mean(cv2.absdiff(thumb, prev))) < 2.5 for prev in recent):
            return False
        recent.append(thumb)
        del recent[:-12]
        return True

    def consider(self, camera: str, frame: np.ndarray, ts: float, reasons: list[str] | None) -> str | None:
        if self.every <= 0 and not self.uncertain:
            return None
        if self._unlabeled(ts) >= self.max_unlabeled:
            return None
        reason = None
        self._uncertain_times = [t for t in self._uncertain_times if ts - t < 3600]
        last_uncertain = self._last_uncertain.get(camera, float("-inf"))
        last_periodic = self._last_periodic.get(camera, float("-inf"))
        if (self.uncertain and reasons and len(self._uncertain_times) < self.max_per_hour
                and ts - last_uncertain >= self.min_gap):
            reason = "uncertain: " + ", ".join(reasons)
        elif self.every > 0 and ts - last_periodic >= self.every:
            reason = "periodic"
        if reason is None or not self._different(camera, frame):
            return None
        frame_id = self.dataset.add(frame, camera, reason, created=ts)
        if reason == "periodic":
            self._last_periodic[camera] = ts
        else:
            self._last_uncertain[camera] = ts
            self._uncertain_times.append(ts)
        self._unlabeled_cache = (self._unlabeled_cache[0], self._unlabeled_cache[1] + 1)
        return frame_id
