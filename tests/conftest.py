from __future__ import annotations

import time
from pathlib import Path

import pytest

from puppycam.config import CameraConfig, Settings
from puppycam.detector import MOM, PUPPY, Detection
from puppycam.geometry import rect_polygon

ASSETS = Path(__file__).parent / "assets"


def det(box, cls=PUPPY, conf=0.9):
    return Detection(cls=cls, conf=conf, box=list(map(float, box)), polygon=rect_polygon(list(map(float, box))))


class FakeDetector:
    """Stands in for the YOLO model: returns the same detections for every frame."""

    has_mom_class = True

    def __init__(self, detections=None):
        self.detections = detections if detections is not None else [
            det([100, 100, 160, 140]), det([165, 100, 225, 140]), det([230, 100, 290, 140], conf=0.6),
            det([50, 50, 400, 300], cls=MOM, conf=0.95),
        ]
        self.calls = 0

    def predict(self, frames, conf):
        self.calls += 1
        return [[Detection(**{**d.__dict__}) for d in self.detections if d.conf >= conf] for _ in frames]


class FakeSegmenter:
    def segment(self, key, image, box):
        return rect_polygon(box), True


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        cameras=[
            CameraConfig(slug="top", label="Top", url=str(ASSETS / "wyze_cam_1.jpg")),
            CameraConfig(slug="side", label="Side", url=str(ASSETS / "side_cam_2.jpg")),
        ],
        data_dir=tmp_path / "data",
        expected_puppies=3,
        device="cpu",
        capture_every_minutes=0,
        capture_uncertain=False,
    )


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False
