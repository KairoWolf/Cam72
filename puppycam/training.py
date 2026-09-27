"""Train a puppy model on the labeled frames, tune it for exact counting, and activate it.

Run from the web page (Train tab) or by hand:
    python -m puppycam.training --data-dir /data --epochs 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from .config import prepare_environment
from .dataset import Dataset
from .evaluation import ImageResult, best_threshold, evaluate
from .geometry import to_gray3
from .registry import ModelRegistry

log = logging.getLogger("puppycam.training")

NAMES = {0: "puppy", 1: "mom"}


class State:
    """Progress file the web page polls."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"phase": "starting", "started": time.time(), "pid": os.getpid()}
        self.write()

    def update(self, **kwargs) -> None:
        self.data.update(kwargs)
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, default=float))
        tmp.replace(self.path)


def _hash(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def split_items(items: list[dict], val_fraction: float = 0.15) -> tuple[list[dict], list[dict], bool]:
    """Split labeled frames into train/val. Returns (train, val, overlap).

    Frames taken minutes apart look almost identical, so frames are grouped by camera and hour and
    whole groups go to validation. Otherwise the validation score would be far too optimistic.
    """
    if len(items) < 2 or val_fraction <= 0:  # nothing to hold out: test on the training frames
        return list(items), list(items), True
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in items:
        hour = datetime.fromtimestamp(m["created"]).strftime("%Y%m%d%H")
        groups[f"{m.get('camera')}-{hour}"].append(m)
    target = max(1, round(val_fraction * len(items)))
    val: list[dict] = []
    if len(groups) >= 4:
        for key in sorted(groups, key=_hash):
            if len(val) >= target:
                break
            if len(val) + len(groups[key]) <= max(target * 2, 1):
                val.extend(groups[key])
    if not val:  # few sessions: fall back to a per-frame split
        val = sorted(items, key=lambda m: _hash(m["id"]))[:target]
    val_ids = {m["id"] for m in val}
    train = [m for m in items if m["id"] not in val_ids]
    if not train:
        return list(items), val, True
    return train, val, False


def build_dataset(dataset: Dataset, out: Path, grayscale: bool, val_fraction: float) -> dict:
    items = dataset.labeled()
    if not items:
        raise RuntimeError("No labeled frames yet. Label some frames on the Label page first.")
    train, val, overlap = split_items(items, val_fraction)
    for split, subset in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for m in subset:
            src = dataset.image_path(m["id"])
            dst = out / "images" / split / src.name
            if grayscale:
                img = cv2.imread(str(src))
                cv2.imwrite(str(dst), to_gray3(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(src, dst)
            shutil.copy2(dataset.label_path(m["id"]), out / "labels" / split / f"{m['id']}.txt")
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": NAMES,
    }))
    count = lambda subset, cls: sum(1 for m in subset for o in m["objects"] if o["cls"] == cls)  # noqa: E731
    return {
        "train_frames": len(train), "val_frames": len(val), "val_overlaps_train": overlap,
        "train_puppies": count(train, 0), "train_moms": count(train, 1),
        "val_ids": [m["id"] for m in val],
    }


def predict_frames(weights: str, items: list[dict], dataset: Dataset, imgsz: int, device: str,
                   grayscale: bool) -> list[ImageResult]:
    from .detector import PUPPY, Detector

    det = Detector(weights, device=device, imgsz=imgsz, grayscale=grayscale)
    results = []
    for m in items:
        img = cv2.imread(str(dataset.image_path(m["id"])))
        if img is None:
            continue
        preds = det.predict([img], conf=0.05)[0]
        results.append(ImageResult(
            id=m["id"],
            truth=[o["box"] for o in m["objects"] if o["cls"] == 0],
            preds=[(d.conf, d.box) for d in preds if d.cls == PUPPY],
        ))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the puppy model on your labeled frames.")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--base", default=os.environ.get("BASE_MODEL", "yolo26l-seg.pt"),
                   help="starting model: yolo26n/s/m/l/x-seg.pt, or a previous best.pt to continue from")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=int(os.environ.get("IMGSZ", 1280)))
    p.add_argument("--batch", type=float, default=-1, help="-1 = as much as fits in GPU memory")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=os.environ.get("DEVICE", ""))
    p.add_argument("--name", default=None)
    p.add_argument("--expected", type=int, default=int(os.environ.get("EXPECTED_PUPPIES", 9)))
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--color", action="store_true", help="train on color instead of grayscale")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="share of frames held out for testing (0 = test on the training frames)")
    p.add_argument("--activate", choices=["auto", "always", "never"], default="auto",
                   help="auto = only if it counts at least as well as the current model")
    args = p.parse_args(argv)
    prepare_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    name = args.name or datetime.now().strftime("puppies-%Y%m%d-%H%M%S")
    run_dir = data_dir / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = State(run_dir / "state.json")
    state.update(name=name, epochs=args.epochs, epoch=0, base=args.base, imgsz=args.imgsz)
    grayscale = not args.color
    clamp = not args.no_clamp

    try:
        import torch

        device = args.device or ("0" if torch.cuda.is_available() else "cpu")
        batch = args.batch if device != "cpu" else (args.batch if args.batch > 0 else 2)
        batch = int(batch) if float(batch).is_integer() else batch

        dataset = Dataset(data_dir / "dataset")
        registry = ModelRegistry(data_dir)

        state.update(phase="preparing", message="Building the training set")
        info = build_dataset(dataset, run_dir / "dataset", grayscale, args.val_fraction)
        state.update(dataset=info)
        log.info("dataset: %s", {k: v for k, v in info.items() if k != "val_ids"})

        from ultralytics import YOLO

        base = args.base if Path(args.base).is_file() else registry.ensure_weights(args.base)
        model = YOLO(base)

        def on_epoch_end(trainer):
            metrics = {k: round(float(v), 4) for k, v in (trainer.metrics or {}).items()}
            state.update(phase="training", epoch=trainer.epoch + 1, epochs=trainer.epochs, metrics=metrics,
                         message=f"Epoch {trainer.epoch + 1}/{trainer.epochs}")

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        state.update(phase="training", message="Training (the first epoch includes a warm-up)")
        model.train(
            data=str(run_dir / "dataset" / "data.yaml"),
            epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=batch, device=device,
            workers=args.workers, project=str(run_dir), name="train", exist_ok=True, seed=0,
            # Top-down camera, sleeping puppies: any orientation is realistic.
            fliplr=0.5, flipud=0.5, degrees=10.0, scale=0.3, translate=0.1,
            # Night vision is grayscale; only brightness varies.
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,
            mosaic=1.0, close_mosaic=15, copy_paste=0.2,
            plots=True, verbose=True,
        )
        best = run_dir / "train" / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("training finished without producing best.pt")

        state.update(phase="evaluating", message="Measuring counting accuracy on held-out frames")
        val_items = [m for m in dataset.labeled() if m["id"] in set(info["val_ids"])]
        results = predict_frames(str(best), val_items, dataset, args.imgsz, device, grayscale)
        thr = best_threshold(results, args.expected, clamp)
        report = evaluate(results, thr, args.expected, clamp)

        previous = registry.current()
        previous_report = None
        if previous is not None:
            prev_meta = previous[1]
            prev_results = predict_frames(str(previous[0]), val_items, dataset, prev_meta.get("imgsz", args.imgsz),
                                          device, prev_meta.get("grayscale", True))
            previous_report = evaluate(prev_results, prev_meta.get("conf", 0.4), args.expected, clamp)
            previous_report = {"name": prev_meta.get("name"), **{k: v for k, v in previous_report.items()
                                                                  if k != "mistakes"}}

        better = previous_report is None or (
            (report["count_accuracy"], -report["count_mae"])
            >= (previous_report["count_accuracy"], -previous_report["count_mae"]))
        activate = args.activate == "always" or (args.activate == "auto" and better)
        meta = {
            "conf": thr,
            "imgsz": args.imgsz,
            "grayscale": grayscale,
            "base": args.base,
            "epochs": args.epochs,
            "classes": NAMES,
            "dataset": {k: v for k, v in info.items() if k != "val_ids"},
            "eval": {k: v for k, v in report.items() if k != "mistakes"},
            "mistakes": report.get("mistakes", []),
            "previous": previous_report,
        }
        registry.register(name, best, meta, activate=activate)
        verdict = "activated" if activate else "kept the current model (it counts better)"
        state.update(phase="done", finished=time.time(), eval=meta["eval"], previous=previous_report,
                     activated=activate,
                     message=f"Done: exact count on {report['frames']} held-out frames = "
                             f"{report['count_accuracy'] * 100:.1f}% (threshold {thr:.2f}); {verdict}.")
        log.info(state.data["message"])
        return 0
    except KeyboardInterrupt:
        state.update(phase="stopped", finished=time.time(), message="Stopped")
        return 1
    except Exception as exc:
        traceback.print_exc()
        state.update(phase="failed", finished=time.time(), message=f"Failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
