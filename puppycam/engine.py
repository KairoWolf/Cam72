"""Ties everything together: cameras -> detection (one GPU batch) -> analysis -> alerts -> video."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .alerts import AlertManager, Notifier
from .analysis import CameraAnalyzer, CameraState, FusedState, fuse
from .config import CameraConfig, Settings
from .dataset import CapturePolicy, Dataset
from .detector import MOM, PUPPY, Detector
from .registry import ModelRegistry
from .render import placeholder, render
from .segmenter import BoxSegmenter
from .source import FrameSource, redact_url

log = logging.getLogger(__name__)

NO_MODEL_MESSAGE = "No puppy model yet\nOpen the Label page, label some frames, then Train"


@dataclass
class CameraRuntime:
    config: CameraConfig
    source: FrameSource
    analyzer: CameraAnalyzer
    state: CameraState | None = None
    state_seq: int = 0
    processed_seq: int = 0
    jpeg: bytes | None = None
    jpeg_seq: int = 0
    rendered_key: tuple = field(default_factory=tuple)


class Engine:
    def __init__(self, settings: Settings, registry: ModelRegistry, dataset: Dataset, notifier: Notifier | None = None):
        self.settings = settings
        self.registry = registry
        self.dataset = dataset
        self.device = settings.resolve_device()
        self.cameras: dict[str, CameraRuntime] = {
            c.slug: CameraRuntime(
                config=c,
                source=FrameSource(c.url, c.slug),
                analyzer=CameraAnalyzer(c.slug, settings.expected_puppies, settings.clamp_to_expected,
                                        settings.away_factor),
            )
            for c in settings.cameras
        }
        self.detector: Detector | None = None
        self.model_meta: dict | None = None
        self.model_error: str | None = None
        self._model_version: float | None = None
        self.conf = settings.conf_override or 0.4
        self.fused: FusedState | None = None
        self.detect_fps = 0.0
        self.capture = CapturePolicy(dataset, settings.capture_every_minutes, settings.capture_uncertain,
                                     settings.capture_max_per_hour, settings.max_unlabeled)
        notifier = notifier or Notifier(settings.ntfy_server, settings.ntfy_topic, settings.ntfy_token,
                                        settings.webhook_url)
        self.alerts = AlertManager(settings, notifier, snapshot=self.alert_snapshot)
        self._segmenter: BoxSegmenter | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # lifecycle -----------------------------------------------------------------------------
    def start(self) -> Engine:
        for cam in self.cameras.values():
            cam.source.start()
        threading.Thread(target=self._prefetch_weights, name="prefetch", daemon=True).start()
        for target, name in ((self._detect_loop, "detect"), (self._render_loop, "render")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _prefetch_weights(self) -> None:
        """Download the outline AI and the training base model now, not when you first need them."""
        for name in (self.settings.sam_model, self.settings.base_model):
            try:
                self.registry.ensure_weights(name)
            except Exception as exc:  # offline is fine, we retry on first use
                log.warning("could not download %s yet: %s", name, exc)

    def stop(self) -> None:
        self._stop.set()
        for cam in self.cameras.values():
            cam.source.stop()

    # model ---------------------------------------------------------------------------------
    def reload_model_if_changed(self) -> None:
        version = self.registry.version()
        if version == self._model_version:
            return
        self._model_version = version
        current = self.registry.current()
        if current is None:
            self.detector, self.model_meta = None, None
            return
        weights, meta = current
        try:
            detector = Detector(str(weights), device=self.device, imgsz=int(meta.get("imgsz", self.settings.imgsz)),
                                grayscale=bool(meta.get("grayscale", self.settings.grayscale)))
            detector.warmup()
        except Exception as exc:
            self.model_error = f"could not load {weights}: {exc}"
            log.exception("model load failed")
            return
        with self._lock:
            self.detector, self.model_meta, self.model_error = detector, meta, None
            self.conf = self.settings.conf_override or float(meta.get("conf", 0.4))
            for cam in self.cameras.values():
                cam.analyzer.tracker.reset()
        log.info("using model %s (threshold %.2f) on %s", meta.get("name"), self.conf, self.device)

    @property
    def mom_class(self) -> bool:
        """Mom alerts only make sense when the model was actually taught what mom looks like."""
        meta = self.model_meta or {}
        return bool(self.detector and self.detector.has_mom_class and meta.get("dataset", {}).get("train_moms", 0) > 0)

    # loops ---------------------------------------------------------------------------------
    def _detect_loop(self) -> None:
        period = 1.0 / max(0.1, self.settings.process_fps)
        while not self._stop.is_set():
            started = time.time()
            try:
                self.step()
            except Exception:
                log.exception("detection step failed")
            elapsed = time.time() - started
            if elapsed > 0:
                self.detect_fps = 0.8 * self.detect_fps + 0.2 * (1.0 / max(elapsed, period))
            self._stop.wait(max(0.0, period - elapsed))

    def step(self) -> None:
        """One detection pass over all cameras (public so tests can drive it)."""
        self.reload_model_if_changed()
        batch = []
        for cam in self.cameras.values():
            frame, _, seq = cam.source.latest()
            if frame is not None and seq != cam.processed_seq:
                batch.append((cam, frame, seq))
        now = time.time()
        detector = self.detector
        if batch:
            results = None
            if detector is not None:
                try:
                    results = detector.predict([f for _, f, _ in batch], conf=self.conf * 0.5)
                except Exception as exc:
                    self.model_error = f"detection failed: {exc}"
                    log.exception("detection failed")
            for i, (cam, frame, seq) in enumerate(batch):
                cam.processed_seq = seq
                state = None
                if results is not None:
                    state = cam.analyzer.update(results[i], now, self.conf)
                    with self._lock:
                        cam.state, cam.state_seq = state, cam.state_seq + 1
                try:
                    self.capture.consider(cam.config.slug, frame, now, state.uncertain if state else None)
                except Exception:
                    log.exception("capture failed")
        with self._lock:
            fresh = [c.state for c in self.cameras.values() if c.state is not None and now - c.state.ts < 10]
            self.fused = fuse(fresh, now) if detector is not None and fresh else None
        ages = {cam.config.label: cam.source.last_frame_age for cam in self.cameras.values()}
        self.alerts.update(now, self.fused, ages, model_ready=detector is not None, mom_class=self.mom_class)

    def _render_loop(self) -> None:
        period = 1.0 / max(0.5, self.settings.display_fps)
        while not self._stop.is_set():
            started = time.time()
            for cam in self.cameras.values():
                try:
                    self.render_camera(cam)
                except Exception:
                    log.exception("render failed for %s", cam.config.slug)
            self._stop.wait(max(0.0, period - (time.time() - started)))

    def render_camera(self, cam: CameraRuntime) -> None:
        frame, _, seq = cam.source.latest()
        state = cam.state if cam.state is not None and time.time() - cam.state.ts < 5 else None
        blink = int(time.time() * 2) if state is not None and state.away else 0
        key = (seq, cam.state_seq, blink, self.detector is not None, cam.source.error,
               int(time.time()) if frame is None else 0)
        if key == cam.rendered_key:
            return
        if frame is None:
            text = f"Connecting to {cam.config.label}..."
            if cam.source.error:
                text += f"\n{cam.source.error[:80]}"
            img = placeholder(cam.config.label, text)
        else:
            message = None if self.detector is not None else NO_MODEL_MESSAGE
            img = render(frame, state, cam.config.label, self.settings.expected_puppies, message,
                         self.settings.stream_max_width)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        if ok:
            with self._lock:
                cam.jpeg, cam.jpeg_seq = buf.tobytes(), cam.jpeg_seq + 1
        cam.rendered_key = key

    # used by the web server ----------------------------------------------------------------
    def jpeg(self, slug: str) -> tuple[bytes | None, int]:
        cam = self.cameras.get(slug)
        if cam is None:
            return None, 0
        with self._lock:
            return cam.jpeg, cam.jpeg_seq

    def alert_snapshot(self) -> bytes | None:
        cams = list(self.cameras.values())
        cams.sort(key=lambda c: (not (c.state and c.state.away), -(c.state.smoothed if c.state else -1)))
        for cam in cams:
            data, _ = self.jpeg(cam.config.slug)
            if data:
                return data
        return None

    def capture_now(self, slug: str) -> str:
        cam = self.cameras.get(slug)
        if cam is None:
            raise KeyError(slug)
        frame, _, _ = cam.source.latest()
        if frame is None:
            raise RuntimeError("no video from this camera yet")
        return self.dataset.add(frame, slug, "manual")

    def suggestions(self, image: np.ndarray) -> dict:
        """Model predictions used to pre-fill the labeling page."""
        detector = self.detector
        if detector is None:
            return {"objects": [], "maybe": [], "model": None}
        dets = detector.predict([image], conf=self.conf * 0.5)[0]
        puppies = sorted((d for d in dets if d.cls == PUPPY), key=lambda d: -d.conf)
        confident = [d for d in puppies if d.conf >= self.conf][: self.settings.expected_puppies or None]
        maybe = [d for d in puppies if d not in confident]
        moms = sorted((d for d in dets if d.cls == MOM and d.conf >= self.conf), key=lambda d: -d.conf)[:1]
        return {
            "objects": [d.to_dict() for d in confident + moms],
            "maybe": [d.to_dict() for d in maybe],
            "model": (self.model_meta or {}).get("name"),
        }

    def segment(self, key: str, image: np.ndarray, box: list[float]) -> tuple[list, bool]:
        if self._segmenter is None:
            weights = self.registry.ensure_weights(self.settings.sam_model)
            self._segmenter = BoxSegmenter(weights, self.device)
        return self._segmenter.segment(key, image, box)

    def status(self) -> dict:
        with self._lock:
            fused = self.fused
            cams = []
            for cam in self.cameras.values():
                state = cam.state if cam.state is not None and time.time() - cam.state.ts < 10 else None
                cams.append({
                    "slug": cam.config.slug,
                    "label": cam.config.label,
                    "url": redact_url(cam.config.url),
                    "connected": cam.source.connected,
                    "error": cam.source.error,
                    "fps": round(cam.source.fps, 1),
                    "frame_age": cam.source.last_frame_age,
                    "count": state.smoothed if state else None,
                    "mom": (state.mom is not None) if state else None,
                    "away": state.away if state else [],
                    "puppies": [{"number": p.number, "conf": round(p.conf, 2), "away": p.away}
                                for p in state.puppies] if state else [],
                })
            meta = self.model_meta or {}
            status = {
                "time": time.time(),
                "expected": self.settings.expected_puppies,
                "count": fused.count if fused else None,
                "best_camera": fused.best_camera if fused else None,
                "mom_visible": fused.mom_visible if fused else None,
                "cameras": cams,
                "model": {
                    "name": meta.get("name"),
                    "conf": round(self.conf, 3),
                    "eval": meta.get("eval"),
                    "error": self.model_error,
                    "device": self.device,
                    "mom_alerts": self.mom_class,
                },
                "detect_fps": round(self.detect_fps, 1),
            }
        status["alerts"] = self.alerts.to_dict()  # outside our lock (the alert manager has its own)
        return status
