"""Background frame readers. Each keeps only the newest frame so detection never lags behind."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")

# RTSP over TCP survives Docker NAT and Wi-Fi packet loss far better than UDP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def source_kind(url: str) -> str:
    if url.lower().startswith(STREAM_PREFIXES):
        return "stream"
    if url.isdigit():
        return "webcam"
    path = Path(url)
    if path.is_dir():
        return "folder"
    if path.suffix.lower() in IMAGE_EXTS:
        return "image"
    return "video"


class FrameSource:
    """Reads a camera (or a test file) in a thread and exposes the latest frame."""

    def __init__(self, url: str, name: str, image_seconds: float = 5.0):
        self.url = url
        self.name = name
        self.kind = source_kind(url)
        self.image_seconds = image_seconds
        self.connected = False
        self.error: str | None = None
        self.fps = 0.0
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"source-{name}", daemon=True)

    # public API ---------------------------------------------------------------------------
    def start(self) -> FrameSource:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            return self._frame, self._ts, self._seq

    @property
    def last_frame_age(self) -> float | None:
        with self._lock:
            return None if self._seq == 0 else time.time() - self._ts

    # internals ----------------------------------------------------------------------------
    def _publish(self, frame: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            if self._seq and now > self._ts:
                inst = 1.0 / (now - self._ts)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self._frame, self._ts, self._seq = frame, now, self._seq + 1
        self.connected = True
        self.error = None

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                if self.kind == "image":
                    self._run_image()
                elif self.kind == "folder":
                    self._run_folder()
                else:
                    self._run_capture()
                backoff = 1.0
            except Exception as exc:  # keep the thread alive no matter what
                self.error = str(exc)
                log.warning("camera %s: %s", self.name, exc)
            self.connected = False
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def _run_image(self) -> None:
        frame = cv2.imread(self.url)
        if frame is None:
            raise RuntimeError(f"cannot read image {self.url}")
        while not self._stop.is_set():
            self._publish(frame)
            self._stop.wait(0.2)

    def _run_folder(self) -> None:
        files = sorted(p for p in Path(self.url).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise RuntimeError(f"no images in {self.url}")
        for path in files:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            end = time.time() + self.image_seconds
            while time.time() < end and not self._stop.is_set():
                self._publish(frame)
                self._stop.wait(0.2)
            if self._stop.is_set():
                return

    def _run_capture(self) -> None:
        if self.kind == "webcam":
            cap = cv2.VideoCapture(int(self.url))
        else:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000]
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self._safe_url()}")
        log.info("camera %s connected (%s)", self.name, self._safe_url())
        is_file = self.kind == "video"
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0) if is_file else 0.0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:  # loop test videos forever
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("stream ended or timed out")
                self._publish(frame)
                if delay:
                    time.sleep(delay)
        finally:
            cap.release()

    def _safe_url(self) -> str:
        """URL without the password, for logs and the UI."""
        return redact_url(self.url)


def redact_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    authority = rest.split("/", 1)[0]
    if "@" not in authority:
        return url
    creds, host = authority.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}{rest[len(authority):]}"
