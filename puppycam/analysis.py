"""Turns raw detections into what matters: how many puppies, which one is away from the pile."""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field

from .detector import MOM, PUPPY, Detection
from .geometry import box_gap, box_iou, box_size
from .tracking import PuppyTracker


@dataclass
class CameraState:
    camera: str
    ts: float
    puppies: list[Detection] = field(default_factory=list)
    mom: Detection | None = None
    count: int = 0  # puppies in this frame
    smoothed: int = 0  # most common count over the last few seconds
    away: list[int] = field(default_factory=list)  # numbers of puppies away from the litter
    uncertain: list[str] = field(default_factory=list)  # why this frame is worth labeling

    def to_dict(self, scale: float = 1.0) -> dict:
        return {
            "camera": self.camera,
            "ts": self.ts,
            "count": self.count,
            "smoothed": self.smoothed,
            "mom": self.mom is not None,
            "away": self.away,
            "puppies": [p.to_dict(scale) for p in self.puppies],
            "mom_detection": self.mom.to_dict(scale) if self.mom else None,
            "uncertain": self.uncertain,
        }


def find_away(puppies: list[Detection], mom: Detection | None, factor: float) -> list[Detection]:
    """Puppies that are not part of the main litter group.

    Objects closer than ``factor`` x (median puppy size) are linked. The main group is the one
    with mom in it (or the biggest group of puppies when mom is not visible).
    """
    if len(puppies) < 2 and mom is None:
        return []
    objects = [p.box for p in puppies] + ([mom.box] if mom else [])
    size = statistics.median(box_size(p.box) for p in puppies) if puppies else 0.0
    limit = factor * size
    parent = list(range(len(objects)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if box_gap(objects[i], objects[j]) <= limit:
                parent[root(i)] = root(j)

    groups = Counter(root(i) for i in range(len(puppies)))
    if mom is not None:
        main = root(len(objects) - 1)
    else:
        main = max(groups, key=lambda g: groups[g]) if groups else None
    return [p for i, p in enumerate(puppies) if root(i) != main]


class CameraAnalyzer:
    """Per-camera: thresholds, litter-size clamp, stable numbers, smoothing and 'away' checks."""

    def __init__(
        self,
        camera: str,
        expected: int,
        clamp: bool = True,
        away_factor: float = 0.75,
        smooth_seconds: float = 5.0,
    ):
        self.camera = camera
        self.expected = expected
        self.clamp = clamp
        self.away_factor = away_factor
        self.smooth_seconds = smooth_seconds
        self.tracker = PuppyTracker()
        self.history: deque[tuple[float, int]] = deque()
        self._previous: list[Detection] = []

    def update(self, detections: list[Detection], ts: float, conf: float) -> CameraState:
        low = conf * 0.5
        candidates = [d for d in detections if d.cls == PUPPY and d.conf >= low]
        kept = [d for d in candidates if d.conf >= conf]
        dropped = []
        for det in (d for d in candidates if d.conf < conf):
            # Hysteresis: a puppy that was there a moment ago keeps counting while partly hidden,
            # unless it is just a weaker duplicate of a puppy that is already counted.
            was_there = any(box_iou(det.box, prev.box) > 0.5 for prev in self._previous)
            duplicate = any(box_iou(det.box, k.box) > 0.5 for k in kept)
            if was_there and not duplicate:
                kept.append(det)
            else:
                dropped.append(det)
        kept.sort(key=lambda d: d.conf, reverse=True)
        if self.clamp and self.expected > 0 and len(kept) > self.expected:
            dropped.extend(kept[self.expected :])
            kept = kept[: self.expected]

        moms = sorted((d for d in detections if d.cls == MOM and d.conf >= conf), key=lambda d: d.conf, reverse=True)
        mom = moms[0] if moms else None

        for det, number in zip(kept, self.tracker.update([d.box for d in kept], ts)):
            det.number = number
        kept.sort(key=lambda d: d.number or 0)
        away = find_away(kept, mom, self.away_factor)
        for det in away:
            det.away = True

        count = len(kept)
        self.history.append((ts, count))
        while self.history and ts - self.history[0][0] > self.smooth_seconds:
            self.history.popleft()
        counts = Counter(c for _, c in self.history)
        smoothed = max(counts, key=lambda c: (counts[c], c))

        uncertain = []
        if dropped:
            uncertain.append("low-confidence puppy")
        if len(counts) > 1:
            uncertain.append("count flickering")
        if any(d.conf < conf + 0.15 for d in kept):
            uncertain.append("borderline puppy")

        self._previous = kept
        return CameraState(
            camera=self.camera,
            ts=ts,
            puppies=kept,
            mom=mom,
            count=count,
            smoothed=smoothed,
            away=[d.number for d in away if d.number is not None],
            uncertain=uncertain,
        )


@dataclass
class FusedState:
    ts: float
    count: int  # best view (max over cameras): never counts a puppy twice
    best_camera: str | None
    mom_visible: bool
    away: dict[str, list[int]]  # camera -> away puppy numbers
    per_camera: dict[str, int]


def fuse(states: list[CameraState], ts: float) -> FusedState:
    live = [s for s in states if s is not None]
    best = max(live, key=lambda s: s.smoothed, default=None)
    return FusedState(
        ts=ts,
        count=best.smoothed if best else 0,
        best_camera=best.camera if best else None,
        mom_visible=any(s.mom is not None for s in live),
        away={s.camera: s.away for s in live if s.away},
        per_camera={s.camera: s.smoothed for s in live},
    )
