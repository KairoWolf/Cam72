import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from puppycam.dataset import Dataset
from puppycam.engine import Engine
from puppycam.registry import ModelRegistry
from puppycam.server import create_app

from .conftest import ASSETS, FakeDetector, FakeSegmenter, wait_for


@pytest.fixture
def engine(settings):
    registry = ModelRegistry(settings.data_dir)
    dataset = Dataset(settings.data_dir / "dataset")
    eng = Engine(settings, registry, dataset)
    eng.detector = FakeDetector()
    eng.model_meta = {"name": "fake", "conf": 0.4, "dataset": {"train_moms": 3}}
    eng._model_version = registry.version()  # keep the fake model
    eng._segmenter = FakeSegmenter()
    for cam in eng.cameras.values():
        cam.source.start()
    assert wait_for(lambda: all(c.source.latest()[0] is not None for c in eng.cameras.values()))
    yield eng
    eng.stop()


@pytest.fixture
def client(engine, settings):
    return TestClient(create_app(engine, settings))


def test_detection_step_counts_and_renders(engine):
    engine.step()
    status = engine.status()
    assert status["count"] == 3 and status["expected"] == 3
    assert status["mom_visible"] is True
    assert {c["slug"]: c["count"] for c in status["cameras"]} == {"top": 3, "side": 3}
    assert [p["number"] for p in status["cameras"][0]["puppies"]] == [1, 2, 3]
    for cam in engine.cameras.values():
        engine.render_camera(cam)
        data, seq = engine.jpeg(cam.config.slug)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert seq == 1 and img.shape[1] <= engine.settings.stream_max_width
    assert engine.alert_snapshot()


def test_stale_cameras_drop_out_of_the_count(engine):
    for cam in engine.cameras.values():
        cam.source.stop()  # the cameras go quiet
    time.sleep(0.3)
    engine.step()
    assert engine.fused.count == 3
    for cam in engine.cameras.values():
        cam.state.ts -= 60  # ...and a minute passes without new frames
    engine.step()
    assert engine.fused is None


def test_pages_and_status(client, engine):
    engine.step()
    for page in ("/", "/label", "/train", "/static/label.js", "/static/style.css"):
        assert client.get(page).status_code == 200
    s = client.get("/api/status").json()
    assert s["count"] == 3 and s["model"]["name"] == "fake" and len(s["cameras"]) == 2
    assert client.get("/api/cameras/nope/snapshot.jpg").status_code == 404


def test_labeling_roundtrip(client):
    fid = client.post("/api/cameras/side/capture").json()["id"]
    frames = client.get("/api/frames?status=unlabeled").json()
    assert frames["frames"][0]["id"] == fid and frames["expected"] == 3

    frame = client.get(f"/api/frames/{fid}").json()
    assert frame["width"] == 1212 and len(frame["suggestions"]["objects"]) == 4  # 3 puppies + mom
    assert client.get(f"/api/frames/{fid}/image.jpg").headers["content-type"] == "image/jpeg"

    seg = client.post(f"/api/frames/{fid}/segment", json={"box": [100, 100, 5000, 200]}).json()
    assert seg["sam"] and max(x for x, _ in seg["polygon"]) <= 1211  # clipped to the image

    objects = [{"cls": 0, "polygon": seg["polygon"]}, {"cls": 1, "polygon": [[0, 0], [50, 0], [50, 50]]}]
    assert client.put(f"/api/frames/{fid}", json={"objects": objects}).json() == {"ok": True, "puppies": 1}
    assert client.get("/api/frames?status=labeled").json()["stats"]["labeled"] == 1
    assert client.put(f"/api/frames/{fid}", json={"objects": [{"cls": 5, "polygon": [[0, 0], [9, 0], [9, 9]]}]}
                      ).status_code == 400
    assert client.delete(f"/api/frames/{fid}").json()["ok"]
    assert client.get(f"/api/frames/{fid}").status_code == 404
    assert client.get("/api/frames/..%2F..%2Fetc").status_code == 404


def test_upload_image_crops_borders(client):
    img = np.zeros((200, 400, 3), np.uint8)
    img[:, 100:300] = 180
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/frames/upload?camera=phone", files=[("files", ("shot.png", buf.tobytes(), "image/png"))])
    fid = r.json()["ids"][0]
    meta = client.get(f"/api/frames/{fid}?suggest=false").json()
    assert (meta["width"], meta["height"], meta["camera"]) == (200, 200, "phone")
    bad = client.post("/api/frames/upload", files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert bad.status_code == 400


def test_upload_video_extracts_distinct_frames(client, tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    for i in range(60):
        frame = np.full((90, 160, 3), 40, np.uint8)
        cv2.rectangle(frame, (i * 2, 20), (i * 2 + 30, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    r = client.post("/api/frames/upload?every=1", files=[("files", ("clip.avi", path.read_bytes(), "video/x-msvideo"))])
    assert r.status_code == 200 and 3 <= len(r.json()["ids"]) <= 6


def test_training_endpoints(client):
    t = client.get("/api/train").json()
    assert t["running"] is False and t["dataset"]["labeled"] == 0
    assert client.post("/api/train", json={"base": "yolo26l-seg.pt"}).status_code == 400  # nothing labeled
    assert client.post("/api/train", json={"base": "../../evil.pt"}).status_code == 400
    assert client.post("/api/train", json={"base": "current"}).status_code == 400  # no model yet
    assert client.get("/api/models").json() == {"models": []}
    assert client.post("/api/models/nope/activate").status_code == 404


def test_password_protection(engine, settings):
    settings.web_password = "secret"
    c = TestClient(create_app(engine, settings))
    assert c.get("/api/status").status_code == 401
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/status", auth=("me", "secret")).status_code == 200
    assert c.get("/api/status", auth=("me", "wrong")).status_code == 401


def test_camera_source_kinds(tmp_path):
    from puppycam.source import redact_url, source_kind

    assert source_kind("rtsp://u:p@1.2.3.4/live") == "stream"
    assert source_kind("0") == "webcam"
    assert source_kind(str(ASSETS)) == "folder"
    assert source_kind(str(ASSETS / "side_cam_1.jpg")) == "image"
    assert source_kind("clip.mp4") == "video"
    assert redact_url("rtsp://admin:p@ss@10.0.0.2:554/stream1") == "rtsp://admin:***@10.0.0.2:554/stream1"
    assert redact_url("rtsp://10.0.0.2/live") == "rtsp://10.0.0.2/live"


def test_training_status_sees_command_line_runs(settings):
    import json
    import os

    from puppycam.server import TrainingManager

    trainer = TrainingManager(settings)
    run = settings.data_dir / "runs" / "cli-run"
    run.mkdir(parents=True)
    state = {"phase": "training", "pid": os.getpid(), "started": time.time() - 300, "epoch": 3}
    (run / "state.json").write_text(json.dumps(state))
    status = trainer.status()
    assert status["name"] == "cli-run" and status["running"] and status["state"]["phase"] == "training"
    with pytest.raises(RuntimeError):
        trainer.start("yolo26n-seg.pt", 1, 320)  # one run at a time
    (run / "state.json").write_text(json.dumps({**state, "pid": 2**22 + 12345}))  # process is gone
    status = trainer.status()
    assert not status["running"] and status["state"]["phase"] == "failed"
