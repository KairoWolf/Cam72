"""Keeps the on-screen puppy numbers (#1..#9) steady from frame to frame.

Newborn puppies barely move between detection passes, so a greedy IoU match is enough. The numbers
are labels for the current view, not identities: a puppy that stays hidden for longer than
``max_age`` seconds may come back with a different number.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Box, box_center, box_iou, box_size


@dataclass
class Track:
    number: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1


class PuppyTracker:
    def __init__(self, max_age: float = 20.0, min_score: float = 0.01):
        self.max_age = max_age
        self.min_score = min_score
        self.tracks: list[Track] = []

    def _similarity(self, track: Track, box: Box) -> float:
        iou = box_iou(track.box, box)
        if iou > 0:
            return iou
        # No overlap: allow a small score for a nearby box (a puppy that shuffled a bit).
        (tx, ty), (bx, by) = box_center(track.box), box_center(box)
        dist = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
        size = max(box_size(track.box), 1.0)
        return max(0.0, 0.5 - dist / size) * 0.2

    def update(self, boxes: list[Box], ts: float) -> list[int]:
        """Assign a number to each box. Returns the numbers in the same order as ``boxes``."""
        self.tracks = [t for t in self.tracks if ts - t.last_seen <= self.max_age]
        pairs = []
        for ti, track in enumerate(self.tracks):
            for bi, box in enumerate(boxes):
                score = self._similarity(track, box)
                if score > 0:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)

        numbers: list[int | None] = [None] * len(boxes)
        used_tracks: set[int] = set()
        for score, ti, bi in pairs:
            if ti in used_tracks or numbers[bi] is not None or score < self.min_score:
                continue
            track = self.tracks[ti]
            track.box, track.last_seen, track.hits = boxes[bi], ts, track.hits + 1
            numbers[bi] = track.number
            used_tracks.add(ti)

        taken = {t.number for t in self.tracks}
        for bi, box in enumerate(boxes):
            if numbers[bi] is None:
                number = 1
                while number in taken:
                    number += 1
                taken.add(number)
                self.tracks.append(Track(number=number, box=box, first_seen=ts, last_seen=ts))
                numbers[bi] = number
        return [int(n) for n in numbers]

    def reset(self) -> None:
        self.tracks = []
