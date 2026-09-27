import json

import cv2
import numpy as np
import pytest

from puppycam.dataset import CapturePolicy, Dataset
from puppycam.evaluation import ImageResult, best_threshold, count_at, evaluate, match
from puppycam.training import build_dataset, split_items


def image(value=120, w=320, h=180):
    img = np.full((h, w, 3), value, np.uint8)
    cv2.circle(img, (value % w, 90), 30, (255, 255, 255), -1)
    return img


def test_add_label_list_delete(tmp_path):
    ds = Dataset(tmp_path)
    fid = ds.add(image(), "side cam", "manual", created=1_700_000_000)
    assert ds.valid_id(fid) and "side-cam" in fid
    assert ds.list("unlabeled")[0]["id"] == fid
    meta = ds.save_labels(fid, [
        {"cls": 0, "polygon": [[10, 10], [60, 10], [60, 40], [10, 40]]},
        {"cls": 1, "box": [100, 20, 300, 170]},
        {"cls": 0, "polygon": [[1, 1], [2, 1]]},  # degenerate: dropped
    ])
    assert meta["labeled"] and len(meta["objects"]) == 2
    lines = ds.label_path(fid).read_text().splitlines()
    assert lines[0].startswith("0 0.031250 0.055556") and lines[1].startswith("1 ")
    assert ds.stats()["puppies"] == 1 and ds.stats()["moms"] == 1
    assert ds.list("labeled")[0]["puppies"] == 1
    ds.delete(fid)
    assert ds.get(fid) is None and not ds.image_path(fid).exists()


def test_save_empty_frame_and_reject_bad_class(tmp_path):
    ds = Dataset(tmp_path)
    fid = ds.add(image(), "cam", "manual")
    ds.save_labels(fid, [])
    assert ds.get(fid)["labeled"] and ds.label_path(fid).read_text() == ""
    with pytest.raises(ValueError):
        ds.save_labels(fid, [{"cls": 7, "polygon": [[0, 0], [5, 0], [5, 5]]}])


def test_ids_are_unique_and_safe(tmp_path):
    ds = Dataset(tmp_path)
    ids = {ds.add(image(), "../../etc", "manual", created=1_700_000_000) for _ in range(3)}
    assert len(ids) == 3 and all(ds.valid_id(i) and "/" not in i for i in ids)
    assert ds.get("../secret") is None


def test_capture_policy_periodic_uncertain_and_dedupe(tmp_path):
    ds = Dataset(tmp_path)
    policy = CapturePolicy(ds, every_minutes=10, uncertain=True, max_per_hour=2, max_unlabeled=100)
    assert policy.consider("cam", image(10), 1000.0, None)  # first periodic frame
    assert policy.consider("cam", image(10), 1100.0, ["low-confidence puppy"]) is None  # same picture
    assert policy.consider("cam", image(90), 1400.0, ["low-confidence puppy"])  # uncertain + different
    assert policy.consider("cam", image(150), 1500.0, None) is None  # periodic not due yet
    assert policy.consider("cam", image(200), 1700.0, None)  # periodic due (10 min)
    reasons = sorted(m["reason"] for m in ds.all())
    assert reasons == ["periodic", "periodic", "uncertain: low-confidence puppy"]


def test_capture_policy_stops_when_backlog_is_full(tmp_path):
    ds = Dataset(tmp_path)
    policy = CapturePolicy(ds, every_minutes=1, uncertain=False, max_per_hour=0, max_unlabeled=1)
    assert policy.consider("cam", image(10), 0.0, None)
    policy._unlabeled_cache = (0.0, 0)  # force a recount
    assert policy.consider("cam", image(100), 120.0, None) is None


def test_split_keeps_sessions_together():
    items = [{"id": f"f{i}", "camera": "side" if i % 2 else "top", "created": 1_700_000_000 + i * 900}
             for i in range(80)]
    train, val, overlap = split_items(items, 0.15)
    assert not overlap and len(train) + len(val) == 80 and 0 < len(val) < 30
    hour = lambda m: (m["camera"], (m["created"] - 1_700_000_000) // 3600)  # noqa: E731
    assert not {hour(m) for m in val} & {hour(m) for m in train}
    assert split_items(items[:1], 0.15)[2] is True
    assert split_items(items, 0)[2] is True


def test_build_dataset_writes_yolo_layout(tmp_path):
    ds = Dataset(tmp_path / "ds")
    for i in range(6):
        fid = ds.add(image(20 * i), "cam", "manual", created=1_700_000_000 + i * 4000)
        ds.save_labels(fid, [{"cls": 0, "polygon": [[10, 10], [60, 10], [60, 40]]}])
    info = build_dataset(ds, tmp_path / "out", grayscale=True, val_fraction=0.2)
    assert info["train_frames"] + info["val_frames"] == 6 and info["val_frames"] >= 1
    assert len(list((tmp_path / "out/images/train").glob("*.jpg"))) == info["train_frames"]
    assert len(list((tmp_path / "out/labels/val").glob("*.txt"))) == info["val_frames"]
    img = cv2.imread(str(next((tmp_path / "out/images/train").glob("*.jpg"))))
    assert np.allclose(img[..., 0], img[..., 2], atol=2)  # grayscale
    assert "names" in (tmp_path / "out/data.yaml").read_text()


def test_count_metrics_and_threshold():
    truth = [[0, 0, 10, 10], [20, 0, 30, 10]]
    good = ImageResult("a", truth, [(0.9, [0, 0, 10, 10]), (0.8, [20, 0, 30, 10]), (0.2, [50, 50, 60, 60])])
    hard = ImageResult("b", truth, [(0.9, [0, 0, 10, 10]), (0.35, [20, 0, 30, 10])])
    assert count_at(good.preds, 0.5, 9, True) == 2
    assert count_at(good.preds, 0.1, 2, True) == 2  # clamped to the litter size
    assert match(good.preds, truth, 0.5) == (2, 0, 0)
    thr = best_threshold([good, hard], expected=9)
    assert 0.2 < thr <= 0.35
    report = evaluate([good, hard], thr, expected=9)
    assert report["count_accuracy"] == 1.0 and report["mistakes"] == []
    report = evaluate([good, hard], 0.5, expected=9)
    assert report["count_accuracy"] == 0.5 and report["mistakes"] == [{"id": "b", "true": 2, "predicted": 1}]
    assert json.dumps(evaluate([], 0.5, 9))
