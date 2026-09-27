"""Alert rules and phone notifications (ntfy.sh and/or a webhook)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import requests

from .analysis import FusedState

log = logging.getLogger(__name__)


class Sustained:
    """Becomes true once a condition has held for ``duration`` seconds.

    Drops shorter than ``grace`` seconds (a puppy flickering out of one frame) do not reset it.
    """

    def __init__(self, duration: float, grace: float = 5.0):
        self.duration = duration
        self.grace = grace
        self.since: float | None = None
        self.last_true: float | None = None

    def update(self, ts: float, active: bool) -> bool:
        if active:
            if self.since is None:
                self.since = ts
            self.last_true = ts
        elif self.last_true is not None and ts - self.last_true > self.grace:
            self.since = self.last_true = None
        return self.since is not None and ts - self.since >= self.duration


@dataclass
class Alert:
    key: str
    title: str
    message: str
    level: str  # "warning" or "critical"
    started: float
    last_sent: float | None = None
    record: dict | None = None  # its entry in the history list

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "message": self.message, "level": self.level, "started": self.started}


class Notifier:
    """Sends notifications in the background so a slow network never stalls detection."""

    def __init__(self, ntfy_server: str = "https://ntfy.sh", ntfy_topic: str | None = None,
                 ntfy_token: str | None = None, webhook_url: str | None = None):
        self.ntfy_url = f"{ntfy_server.rstrip('/')}/{ntfy_topic}" if ntfy_topic else None
        self.ntfy_token = ntfy_token
        self.webhook_url = webhook_url

    @property
    def enabled(self) -> bool:
        return bool(self.ntfy_url or self.webhook_url)

    def send(self, title: str, message: str, image: bytes | None = None, priority: str = "high") -> None:
        if self.enabled:
            threading.Thread(target=self._send, args=(title, message, image, priority), daemon=True).start()

    def _send(self, title: str, message: str, image: bytes | None, priority: str) -> None:
        if self.ntfy_url:
            headers = {
                "Title": _ascii(title),
                "Priority": priority,
                "Tags": "dog" if priority != "urgent" else "rotating_light,dog",
            }
            if self.ntfy_token:
                headers["Authorization"] = f"Bearer {self.ntfy_token}"
            try:
                if image:
                    headers.update({"Message": _ascii(message), "Filename": "puppies.jpg"})
                    requests.put(self.ntfy_url, data=image, headers=headers, timeout=20)
                else:
                    requests.post(self.ntfy_url, data=message.encode("utf-8"), headers=headers, timeout=20)
            except requests.RequestException as exc:
                log.warning("ntfy notification failed: %s", exc)
        if self.webhook_url:
            text = f"{title}: {message}"
            try:
                requests.post(self.webhook_url, json={"title": title, "message": message, "text": text, "content": text},
                              timeout=20)
            except requests.RequestException as exc:
                log.warning("webhook notification failed: %s", exc)


def _ascii(text: str) -> str:
    """HTTP headers must be ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


class AlertManager:
    def __init__(self, settings, notifier: Notifier, snapshot=lambda: None):
        self.s = settings
        self.notifier = notifier
        self.snapshot = snapshot  # returns JPEG bytes of the most useful view
        self.active: dict[str, Alert] = {}
        self.history: deque[dict] = deque(maxlen=200)
        self.started = time.time()
        self.last_all_visible: float | None = None
        self.last_mom_seen: float | None = None
        self._away = Sustained(settings.away_seconds)
        self._last_away: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def update(self, ts: float, fused: FusedState | None, camera_ages: dict[str, float | None],
               model_ready: bool, mom_class: bool) -> None:
        wanted: dict[str, tuple[str, str, str]] = {}

        # Cameras that stopped sending frames.
        for cam, age in camera_ages.items():
            offline_for = (ts - self.started) if age is None else age
            if offline_for >= self.s.offline_minutes * 60:
                wanted[f"offline:{cam}"] = ("warning", f"Camera {cam} is offline",
                                            f"No video from {cam} for {offline_for / 60:.0f} min.")

        if model_ready and fused is not None and fused.per_camera:
            expected = self.s.expected_puppies
            if self.last_all_visible is None or fused.count >= expected:
                self.last_all_visible = ts
            missing_for = ts - self.last_all_visible
            if expected and missing_for >= self.s.count_low_minutes * 60:
                wanted["count_low"] = (
                    "critical", f"Only {fused.count} of {expected} puppies visible",
                    f"Not all {expected} puppies have been visible for {missing_for / 60:.0f} min. "
                    "Check that no puppy is stuck under mom or outside the box.")

            if any(fused.away.values()):
                self._last_away = fused.away
            if self._away.update(ts, any(fused.away.values())):
                where = "; ".join(f"#{', #'.join(map(str, nums))} on {cam}" for cam, nums in self._last_away.items())
                wanted["away"] = ("critical", "Puppy away from the litter",
                                  f"Puppy {where} has been away from mom and the other puppies "
                                  f"for over {self.s.away_seconds:.0f} s. It may get cold.")

            if mom_class:
                if self.last_mom_seen is None or fused.mom_visible:
                    self.last_mom_seen = ts
                gone = ts - self.last_mom_seen
                if gone >= self.s.mom_away_minutes * 60:
                    wanted["mom_away"] = ("warning", "Mom is not with the puppies",
                                          f"Mom has not been seen for {gone / 60:.0f} min.")

        self._apply(ts, wanted)

    def _apply(self, ts: float, wanted: dict[str, tuple[str, str, str]]) -> None:
        outbox = []
        with self._lock:
            for key in list(self.active):
                if key not in wanted:
                    alert = self.active.pop(key)
                    if alert.record is not None:
                        alert.record["ended"] = ts
            for key, (level, title, message) in wanted.items():
                alert = self.active.get(key)
                if alert is None:
                    alert = self.active[key] = Alert(key, title, message, level, started=ts)
                    alert.record = {**alert.to_dict(), "ended": None}
                    self.history.appendleft(alert.record)
                else:
                    alert.title, alert.message = title, message
                    if alert.record is not None:
                        alert.record.update(title=title, message=message)
                if alert.last_sent is None or ts - alert.last_sent >= self.s.alert_repeat_minutes * 60:
                    alert.last_sent = ts
                    outbox.append((title, message, level))
        # Outside the lock: grabbing the snapshot takes the engine's lock.
        for title, message, level in outbox:
            image = None
            try:
                image = self.snapshot()
            except Exception:  # never let a snapshot problem block an alert
                log.exception("snapshot for alert failed")
            self.notifier.send(title, message, image, "urgent" if level == "critical" else "high")
            log.warning("ALERT %s: %s", title, message)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "active": [a.to_dict() for a in self.active.values()],
                "history": [dict(h) for h in list(self.history)[:50]],
                "last_all_visible": self.last_all_visible,
                "last_mom_seen": self.last_mom_seen,
                "notifications": self.notifier.enabled,
            }
