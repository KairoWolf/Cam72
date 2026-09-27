"""Where models live: downloaded base weights and your trained puppy models."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path


VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ModelRegistry:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.weights_dir = self.data_dir / "weights"
        self.models_dir = self.data_dir / "models"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._download_lock = threading.Lock()

    # base weights --------------------------------------------------------------------------
    def ensure_weights(self, name: str) -> str:
        """Path to a base model (e.g. yolo26l-seg.pt, sam2.1_l.pt), downloading it once."""
        path = Path(name)
        if path.is_file():
            return str(path)
        target = self.weights_dir / path.name
        with self._download_lock:  # the prefetch thread and a web request may ask at the same time
            if not target.is_file():
                from ultralytics.utils.downloads import attempt_download_asset

                attempt_download_asset(target)
        if not target.is_file():
            raise FileNotFoundError(f"could not download {name}")
        return str(target)

    # trained models ------------------------------------------------------------------------
    @property
    def pointer(self) -> Path:
        return self.models_dir / "current.json"

    def register(self, name: str, weights: Path, meta: dict, activate: bool) -> Path:
        folder = self.models_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "best.pt"
        shutil.copy2(weights, target)
        (folder / "model.json").write_text(json.dumps({**meta, "name": name, "created": time.time()}, indent=2))
        if activate or self.current() is None:
            self.activate(name)
        return target

    def activate(self, name: str) -> None:
        if not VALID_NAME.match(name) or not (self.models_dir / name / "best.pt").is_file():
            raise FileNotFoundError(name)
        with self._lock:
            tmp = self.pointer.with_suffix(".tmp")
            tmp.write_text(json.dumps({"name": name, "activated": time.time()}))
            tmp.replace(self.pointer)

    def meta(self, name: str) -> dict | None:
        if not VALID_NAME.match(name):
            return None
        try:
            return json.loads((self.models_dir / name / "model.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def current_name(self) -> str | None:
        try:
            return json.loads(self.pointer.read_text())["name"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def current(self) -> tuple[Path, dict] | None:
        name = self.current_name()
        if not name:
            return None
        weights = self.models_dir / name / "best.pt"
        if not weights.is_file():
            return None
        return weights, self.meta(name) or {"name": name}

    def version(self) -> float:
        """Changes whenever a different model becomes active (the live view reloads then)."""
        try:
            return self.pointer.stat().st_mtime
        except OSError:
            return 0.0

    def list(self) -> list[dict]:
        current = self.current_name()
        out = []
        for folder in self.models_dir.iterdir():
            if folder.is_dir() and (folder / "best.pt").is_file():
                meta = self.meta(folder.name) or {"name": folder.name, "created": folder.stat().st_mtime}
                meta = {k: v for k, v in meta.items() if k != "mistakes"}
                meta["current"] = folder.name == current
                out.append(meta)
        out.sort(key=lambda m: -m.get("created", 0))
        return out
