"""How good is a model at the thing we care about: counting the puppies exactly?"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .geometry import Box, box_iou


@dataclass
class ImageResult:
    id: str
    truth: list[Box]  # labeled puppy boxes
    preds: list[tuple[float, Box]] = field(default_factory=list)  # (confidence, box), low threshold


def count_at(preds: list[tuple[float, Box]], thr: float, expected: int, clamp: bool) -> int:
    n = sum(1 for conf, _ in preds if conf >= thr)
    return min(n, expected) if clamp and expected > 0 else n


def match(preds: list[tuple[float, Box]], truth: list[Box], thr: float, iou: float = 0.5) -> tuple[int, int, int]:
    """Greedy matching by confidence. Returns (true positives, false positives, misses)."""
    used: set[int] = set()
    tp = fp = 0
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        if conf < thr:
            continue
        best, best_iou = None, iou
        for i, t in enumerate(truth):
            if i not in used and (v := box_iou(box, t)) >= best_iou:
                best, best_iou = i, v
        if best is None:
            fp += 1
        else:
            used.add(best)
            tp += 1
    return tp, fp, len(truth) - len(used)


def evaluate(images: list[ImageResult], thr: float, expected: int, clamp: bool = True) -> dict:
    if not images:
        return {"frames": 0, "threshold": round(thr, 3), "count_accuracy": 0.0, "count_mae": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "mistakes": []}
    exact = 0
    errors = []
    tp = fp = fn = 0
    mistakes = []
    for img in images:
        predicted = count_at(img.preds, thr, expected, clamp)
        true = len(img.truth)
        exact += int(predicted == true)
        errors.append(abs(predicted - true))
        if predicted != true:
            mistakes.append({"id": img.id, "true": true, "predicted": predicted})
        # With the clamp, only the most confident `expected` detections count.
        kept = sorted(img.preds, key=lambda p: -p[0])
        kept = [p for p in kept if p[0] >= thr][: expected if clamp and expected > 0 else None]
        a, b, c = match(kept, img.truth, 0.0)
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "frames": len(images),
        "threshold": round(thr, 3),
        "count_accuracy": round(exact / len(images), 4),
        "count_mae": round(sum(errors) / len(images), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "mistakes": mistakes,
    }


def best_threshold(images: list[ImageResult], expected: int, clamp: bool = True,
                   lo: float = 0.1, hi: float = 0.9, step: float = 0.01) -> float:
    """Confidence threshold that gets the most frames' puppy count exactly right.

    Many thresholds usually tie; the middle of the best plateau is the most robust choice.
    """
    if not images:
        return 0.4
    grid = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    scored = []
    for thr in grid:
        exact = sum(count_at(img.preds, thr, expected, clamp) == len(img.truth) for img in images)
        mae = sum(abs(count_at(img.preds, thr, expected, clamp) - len(img.truth)) for img in images)
        scored.append((exact, -mae, thr))
    best_exact, best_mae, _ = max(scored)
    runs: list[list[float]] = []
    previous_best = False
    for exact, mae, thr in scored:
        is_best = exact == best_exact and mae == best_mae
        if is_best:
            if previous_best:
                runs[-1].append(thr)
            else:
                runs.append([thr])
        previous_best = is_best
    longest = max(runs, key=len)
    return float(statistics.median(longest))
