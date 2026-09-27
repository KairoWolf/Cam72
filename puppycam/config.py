"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CAMERAS = 8


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


def _int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(float(value))


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def prepare_environment() -> None:
    """Create YOLO_CONFIG_DIR before Ultralytics is imported, otherwise it falls back to /tmp."""
    config_dir = os.environ.get("YOLO_CONFIG_DIR")
    if config_dir:
        try:
            Path(config_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "camera"


@dataclass
class CameraConfig:
    slug: str  # used in URLs and file names
    label: str  # shown in the UI
    url: str  # rtsp://..., http://..., a video file, an image or a folder of images


def cameras_from_env() -> list[CameraConfig]:
    """Read CAM1_URL/CAM1_NAME ... CAM8_URL/CAM8_NAME (or a single CAMERA_URL)."""
    found: list[tuple[str, str]] = []
    for i in range(1, MAX_CAMERAS + 1):
        url = _env(f"CAM{i}_URL")
        if url:
            found.append((_env(f"CAM{i}_NAME", f"Camera {i}"), url))
    if not found and _env("CAMERA_URL"):
        found.append((_env("CAMERA_NAME", "Camera 1"), _env("CAMERA_URL")))

    cameras, used = [], set()
    for label, url in found:
        slug = base = slugify(label)
        n = 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        cameras.append(CameraConfig(slug=slug, label=label, url=url))
    return cameras


@dataclass
class Settings:
    cameras: list[CameraConfig] = field(default_factory=list)
    data_dir: Path = Path("data")

    # The litter
    expected_puppies: int = 9

    # Detection
    process_fps: float = 4.0  # detection passes per second (all cameras in one GPU batch)
    display_fps: float = 10.0  # frame rate of the highlighted live view
    imgsz: int = 1280  # model input size; puppies are small, so keep this high
    device: str = ""  # "" = auto (GPU if available), "cpu", "0", "1", ...
    grayscale: bool = True  # treat day (color) and night (IR) frames the same way
    conf_override: float | None = None  # otherwise the tuned threshold saved with the model is used
    clamp_to_expected: bool = True  # never report more puppies than the litter size

    # Alerts
    away_factor: float = 0.75  # gap (in puppy sizes) that separates a puppy from the pile
    away_seconds: float = 60.0
    count_low_minutes: float = 10.0
    mom_away_minutes: float = 20.0
    offline_minutes: float = 2.0
    alert_repeat_minutes: float = 15.0
    ntfy_server: str = "https://ntfy.sh"
    ntfy_topic: str | None = None
    ntfy_token: str | None = None
    webhook_url: str | None = None

    # Collecting training frames
    capture_every_minutes: float = 10.0
    capture_uncertain: bool = True
    capture_max_per_hour: int = 6
    max_unlabeled: int = 400

    # Models
    base_model: str = "yolo26l-seg.pt"
    sam_model: str = "sam2.1_l.pt"

    # Web
    host: str = "0.0.0.0"
    port: int = 8080
    web_password: str | None = None
    stream_max_width: int = 1280
    jpeg_quality: int = 80

    @classmethod
    def from_env(cls) -> Settings:
        conf = _env("CONF")
        return cls(
            cameras=cameras_from_env(),
            data_dir=Path(_env("DATA_DIR", "data")),
            expected_puppies=_int("EXPECTED_PUPPIES", 9),
            process_fps=_float("PROCESS_FPS", 4.0),
            display_fps=_float("DISPLAY_FPS", 10.0),
            imgsz=_int("IMGSZ", 1280),
            device=_env("DEVICE", "") or "",
            grayscale=_bool("GRAYSCALE", True),
            conf_override=float(conf) if conf else None,
            clamp_to_expected=_bool("CLAMP_TO_EXPECTED", True),
            away_factor=_float("AWAY_FACTOR", 0.75),
            away_seconds=_float("AWAY_SECONDS", 60.0),
            count_low_minutes=_float("COUNT_LOW_MINUTES", 10.0),
            mom_away_minutes=_float("MOM_AWAY_MINUTES", 20.0),
            offline_minutes=_float("OFFLINE_MINUTES", 2.0),
            alert_repeat_minutes=_float("ALERT_REPEAT_MINUTES", 15.0),
            ntfy_server=_env("NTFY_SERVER", "https://ntfy.sh"),
            ntfy_topic=_env("NTFY_TOPIC"),
            ntfy_token=_env("NTFY_TOKEN"),
            webhook_url=_env("WEBHOOK_URL"),
            capture_every_minutes=_float("CAPTURE_EVERY_MINUTES", 10.0),
            capture_uncertain=_bool("CAPTURE_UNCERTAIN", True),
            capture_max_per_hour=_int("CAPTURE_MAX_PER_HOUR", 6),
            max_unlabeled=_int("MAX_UNLABELED", 400),
            base_model=_env("BASE_MODEL", "yolo26l-seg.pt"),
            sam_model=_env("SAM_MODEL", "sam2.1_l.pt"),
            host=_env("HOST", "0.0.0.0"),
            port=_int("PORT", 8080),
            web_password=_env("WEB_PASSWORD"),
            stream_max_width=_int("STREAM_MAX_WIDTH", 1280),
            jpeg_quality=_int("JPEG_QUALITY", 80),
        )

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "0"
        except Exception:  # pragma: no cover - torch import problems
            pass
        return "cpu"
