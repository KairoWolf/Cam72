"""Web app: live highlighted video, labeling tool and training page."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .engine import Engine
from .geometry import autocrop_borders, clip_box

log = logging.getLogger(__name__)
WEB = Path(__file__).parent / "web"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class TrainingManager:
    """Runs ``python -m puppycam.training`` as a separate process and reports its progress."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runs = settings.data_dir / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.name: str | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, base: str, epochs: int, imgsz: int) -> str:
        if self.running or self.status().get("running"):
            raise RuntimeError("a training run is already in progress")
        self.name = datetime.now().strftime("puppies-%Y%m%d-%H%M%S")
        run_dir = self.runs / self.name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "puppycam.training", "--data-dir", str(self.settings.data_dir),
               "--base", base, "--epochs", str(epochs), "--imgsz", str(imgsz), "--name", self.name,
               "--expected", str(self.settings.expected_puppies)]
        if self.settings.device:
            cmd += ["--device", self.settings.device]
        if not self.settings.grayscale:
            cmd.append("--color")
        if not self.settings.clamp_to_expected:
            cmd.append("--no-clamp")
        logfile = open(run_dir / "train.log", "wb")
        self.proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                     cwd=str(Path(__file__).resolve().parent.parent))
        logfile.close()
        return self.name

    def stop(self) -> None:
        if self.running:
            self.proc.send_signal(signal.SIGINT)  # lets the trainer write a clean "stopped" state
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        state_path = self.runs / (self.name or "") / "state.json"
        if self.name and state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("phase") not in ("done", "failed", "stopped"):
                state.update(phase="stopped", message="Stopped", finished=time.time())
                state_path.write_text(json.dumps(state))

    def latest_name(self) -> str | None:
        runs = [p for p in self.runs.iterdir() if (p / "state.json").is_file()]
        if not runs:
            return self.name
        return max(runs, key=lambda p: (p / "state.json").stat().st_mtime).name

    @staticmethod
    def _alive(pid) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def status(self) -> dict:
        name = self.latest_name()
        if not name:
            return {"running": False, "name": None}
        run_dir = self.runs / name
        try:
            state = json.loads((run_dir / "state.json").read_text())
        except (OSError, json.JSONDecodeError):
            state = {"phase": "starting"}
        unfinished = state.get("phase") not in ("done", "failed", "stopped")
        # A run started from the command line (docker compose exec ...) counts as running too.
        running = self.running or (unfinished and self._alive(state.get("pid")))
        if unfinished and not running and (self.proc is not None or time.time() - state.get("started", 0) > 60):
            state.update(phase="failed", message=state.get("message") or "Training process exited unexpectedly")
        tail = ""
        log_path = run_dir / "train.log"
        if log_path.is_file():
            with open(log_path, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 16000))
                text = ANSI.sub("", fh.read().decode("utf-8", "replace"))
            lines = [line.split("\r")[-1] for line in text.split("\n")]
            tail = "\n".join(line for line in lines if line.strip())[-6000:]
        return {"running": running, "name": name, "state": state, "log": tail}


class LabelsIn(BaseModel):
    objects: list[dict]


class BoxIn(BaseModel):
    box: list[float]


class TrainIn(BaseModel):
    base: str = "yolo26l-seg.pt"
    epochs: int = 150
    imgsz: int = 1280


ALLOWED_BASES = {f"yolo26{s}-seg.pt" for s in "nsmlx"} | {f"yolo11{s}-seg.pt" for s in "nsmlx"}


def create_app(engine: Engine, settings: Settings) -> FastAPI:
    app = FastAPI(title="PuppyCam")
    trainer = TrainingManager(settings)
    dataset = engine.dataset

    if settings.web_password:
        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            if request.url.path != "/api/health":
                header = request.headers.get("authorization", "")
                ok = False
                if header.lower().startswith("basic "):
                    try:
                        _, _, password = base64.b64decode(header[6:]).decode().partition(":")
                        ok = secrets.compare_digest(password, settings.web_password)
                    except Exception:
                        ok = False
                if not ok:
                    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="PuppyCam"'})
            return await call_next(request)

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def live_page():
        return FileResponse(WEB / "index.html")

    @app.get("/label")
    def label_page():
        return FileResponse(WEB / "label.html")

    @app.get("/train")
    def train_page():
        return FileResponse(WEB / "train.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    # live ----------------------------------------------------------------------------------
    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/cameras/{slug}/stream.mjpg")
    async def stream(slug: str):
        if slug not in engine.cameras:
            raise HTTPException(404, "unknown camera")

        async def frames():
            last = -1
            while True:
                data, seq = engine.jpeg(slug)
                if data is not None and seq != last:
                    last = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(0.04)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{slug}/snapshot.jpg")
    def snapshot(slug: str):
        data, _ = engine.jpeg(slug)
        if data is None:
            raise HTTPException(404, "no image yet")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/cameras/{slug}/capture")
    def capture(slug: str):
        try:
            return {"id": engine.capture_now(slug)}
        except KeyError:
            raise HTTPException(404, "unknown camera")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    # labeling ------------------------------------------------------------------------------
    def _meta_or_404(frame_id: str) -> dict:
        meta = dataset.get(frame_id)
        if meta is None:
            raise HTTPException(404, "unknown frame")
        return meta

    def _image(frame_id: str) -> np.ndarray:
        img = cv2.imread(str(dataset.image_path(frame_id)))
        if img is None:
            raise HTTPException(404, "image missing")
        return img

    @app.get("/api/frames")
    def frames(status: str = "unlabeled"):
        return {"frames": dataset.list(status), "stats": dataset.stats(), "expected": settings.expected_puppies}

    @app.get("/api/frames/{frame_id}")
    def frame(frame_id: str, suggest: bool = True):
        meta = _meta_or_404(frame_id)
        out = {**meta, "suggestions": None}
        if suggest and not meta.get("labeled"):
            try:
                out["suggestions"] = engine.suggestions(_image(frame_id))
            except Exception as exc:
                log.exception("suggestions failed")
                out["suggestions_error"] = str(exc)
        return out

    @app.get("/api/frames/{frame_id}/image.jpg")
    def frame_image(frame_id: str):
        _meta_or_404(frame_id)
        return FileResponse(dataset.image_path(frame_id), media_type="image/jpeg")

    @app.put("/api/frames/{frame_id}")
    def save_frame(frame_id: str, body: LabelsIn):
        _meta_or_404(frame_id)
        try:
            meta = dataset.save_labels(frame_id, body.objects)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, "puppies": sum(1 for o in meta["objects"] if o["cls"] == 0)}

    @app.delete("/api/frames/{frame_id}")
    def delete_frame(frame_id: str):
        _meta_or_404(frame_id)
        dataset.delete(frame_id)
        return {"ok": True}

    @app.post("/api/frames/{frame_id}/segment")
    def segment(frame_id: str, body: BoxIn):
        meta = _meta_or_404(frame_id)
        if len(body.box) != 4:
            raise HTTPException(400, "box must be [x1, y1, x2, y2]")
        box = clip_box(body.box, meta["width"], meta["height"])
        polygon, used_sam = engine.segment(frame_id, _image(frame_id), box)
        return {"polygon": [[round(x, 1), round(y, 1)] for x, y in polygon], "sam": used_sam}

    @app.post("/api/frames/upload")
    async def upload(files: list[UploadFile] = File(...), camera: str = "upload", every: float = 2.0,
                     max_frames: int = 150):
        camera = "".join(c if c.isalnum() else "-" for c in camera)[:24] or "upload"
        ids = []
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            data = await upload.read()
            if suffix in VIDEO_EXTS:
                ids += await asyncio.to_thread(_import_video, data, suffix, camera, every, max_frames)
            else:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(400, f"{upload.filename}: not an image or video")
                ids.append(dataset.add(autocrop_borders(img), camera, "upload"))
        return {"ids": ids}

    def _import_video(data: bytes, suffix: str, camera: str, every: float, max_frames: int) -> list[str]:
        ids: list[str] = []
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            step = max(1, int(round(fps * max(every, 0.2))))
            index, last_thumb = 0, None
            while len(ids) < max_frames:
                ok = cap.grab()
                if not ok:
                    break
                if index % step == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54))
                        if last_thumb is None or float(np.mean(cv2.absdiff(thumb, last_thumb))) >= 2.5:
                            ids.append(dataset.add(frame, camera, "video", created=time.time() + index * 1e-4))
                            last_thumb = thumb
                index += 1
            cap.release()
        return ids

    # training ------------------------------------------------------------------------------
    @app.get("/api/train")
    def train_status():
        return {**trainer.status(), "dataset": dataset.stats(), "models": engine.registry.list(),
                "defaults": {"base": settings.base_model, "imgsz": settings.imgsz, "device": engine.device}}

    @app.post("/api/train")
    def train_start(body: TrainIn):
        base = body.base
        if base == "current":  # fine-tune the model in use on the bigger dataset: faster rounds
            current = engine.registry.current()
            if current is None:
                raise HTTPException(400, "there is no current model yet")
            base = str(current[0])
        elif base not in ALLOWED_BASES:
            raise HTTPException(400, "unknown base model")
        if dataset.stats()["labeled"] < 1:
            raise HTTPException(400, "label at least one frame first")
        try:
            name = trainer.start(base, max(1, min(body.epochs, 1000)), max(320, min(body.imgsz, 1920)))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"name": name}

    @app.post("/api/train/stop")
    def train_stop():
        trainer.stop()
        return {"ok": True}

    @app.get("/api/models")
    def models():
        return {"models": engine.registry.list()}

    @app.get("/api/models/{name}/mistakes")
    def model_mistakes(name: str):
        meta = engine.registry.meta(name)
        if meta is None:
            raise HTTPException(404, "unknown model")
        return {"mistakes": meta.get("mistakes", [])}

    @app.post("/api/models/{name}/activate")
    def activate(name: str):
        try:
            engine.registry.activate(name)
        except FileNotFoundError:
            raise HTTPException(404, "unknown model")
        return {"ok": True}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("request failed: %s", request.url.path)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app
