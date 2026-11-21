"""Output-level parity metrics between a reference and a candidate backend.

Cross-backend risk is not "it crashes" but "it runs and the boxes drift": a small
IoU change alters tracking ids, which alters rally boundaries and scores. These
helpers compare detection / keypoint outputs directly, so a new backend is only
accepted when it reproduces the Torch reference on the golden sample.

Boxes are ``(x1, y1, x2, y2)`` in pixels (both backends work on the same frames)
and carry a confidence. Matching is greedy one-to-one by descending IoU, which is
what "same object" means for a detector without an identity model.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def _split(item: Any) -> tuple[tuple[float, float, float, float], float]:
    """Accept ``(xyxy, conf)`` tuples or ``{"xyxy": [...], "conf": ...}`` dicts."""
    if isinstance(item, dict):
        box = item.get("xyxy") or item.get("box") or (0, 0, 0, 0)
        return tuple(float(v) for v in box[:4]), float(item.get("conf", 1.0))
    box, conf = item
    return tuple(float(v) for v in box[:4]), float(conf)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    ref: Iterable[Any], test: Iterable[Any], iou_thr: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one matching of one frame's detections."""
    ref_items = [_split(item) for item in ref]
    test_items = [_split(item) for item in test]
    pairs: list[tuple[int, int, float]] = []
    for ri, (rbox, _rc) in enumerate(ref_items):
        for ti, (tbox, _tc) in enumerate(test_items):
            score = iou(rbox, tbox)
            if score >= iou_thr:
                pairs.append((ri, ti, score))
    pairs.sort(key=lambda p: -p[2])
    used_ref: set[int] = set()
    used_test: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for ri, ti, score in pairs:
        if ri in used_ref or ti in used_test:
            continue
        used_ref.add(ri)
        used_test.add(ti)
        matched.append((ri, ti, score))
    diffs = [test_items[ti][1] - ref_items[ri][1] for ri, ti, _s in matched]
    # Many-to-one coverage: "is there any box on the other side?" A detector that keeps
    # two near-duplicate boxes for one person is still seeing the same object, and the
    # tracker deduplicates them; the 1:1 numbers above punish that as a miss. Both are
    # reported because the 1:1 view catches shifted boxes and coverage catches real misses.
    cov_ref = sum(1 for rbox, _rc in ref_items
                  if any(iou(rbox, tbox) >= iou_thr for tbox, _tc in test_items))
    cov_test = sum(1 for tbox, _tc in test_items
                   if any(iou(tbox, rbox) >= iou_thr for rbox, _rc in ref_items))
    return {
        "ref": len(ref_items),
        "test": len(test_items),
        "matched": len(matched),
        "mean_iou": round(float(np.mean([s for _r, _t, s in matched])), 4) if matched else 0.0,
        "conf_delta": round(float(np.mean(diffs)), 4) if diffs else 0.0,
        "covered_ref": cov_ref,
        "covered_test": cov_test,
    }


def match_frames(
    ref_frames: Sequence[Sequence[Any]], test_frames: Sequence[Sequence[Any]],
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    """Aggregate parity over a frame sequence (same length expected)."""
    n = min(len(ref_frames), len(test_frames))
    ref_total = test_total = matched_total = 0
    cov_ref_total = cov_test_total = 0
    ious: list[float] = []
    deltas: list[float] = []
    frame_recall: list[float] = []
    for fi in range(n):
        m = match_boxes(ref_frames[fi], test_frames[fi], iou_thr)
        ref_total += m["ref"]
        test_total += m["test"]
        matched_total += m["matched"]
        cov_ref_total += m["covered_ref"]
        cov_test_total += m["covered_test"]
        if m["matched"]:
            ious.append(m["mean_iou"])
            deltas.append(m["conf_delta"])
        frame_recall.append(1.0 if m["ref"] == 0 else m["matched"] / m["ref"])
    recall = 1.0 if ref_total == 0 else matched_total / ref_total
    precision = 1.0 if test_total == 0 else matched_total / test_total
    return {
        "frames": n,
        "ref_boxes": ref_total,
        "test_boxes": test_total,
        "matched": matched_total,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "coverage_recall": round(1.0 if ref_total == 0 else cov_ref_total / ref_total, 4),
        "coverage_precision": round(1.0 if test_total == 0 else cov_test_total / test_total, 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "conf_delta": round(float(np.mean(deltas)), 4) if deltas else 0.0,
        "worst_frame_recall": round(min(frame_recall), 4) if frame_recall else 1.0,
        "frames_below_1": int(sum(1 for r in frame_recall if r < 1.0)),
    }


def keypoint_l2(
    ref_frames: Sequence[Sequence[dict]], test_frames: Sequence[Sequence[dict]],
    crop_size: float = 192.0,
) -> dict[str, Any]:
    """Mean / p90 keypoint error in patch pixels, matched by ``slot`` within a frame.

    Both sides are ``analyze_pose`` crops resized to ``crop_size``, so the error is
    comparable across frames and does not depend on how big the player was.
    """
    errors: list[float] = []
    slots = 0
    for fi in range(min(len(ref_frames), len(test_frames))):
        by_slot = {int(t.get("slot", i)): t for i, t in enumerate(test_frames[fi])}
        for ri, ref_entry in enumerate(ref_frames[fi]):
            test_entry = by_slot.get(int(ref_entry.get("slot", ri)))
            if test_entry is None:
                continue
            ref_kps = np.asarray(ref_entry.get("kps") or [], dtype=np.float32)
            test_kps = np.asarray(test_entry.get("kps") or [], dtype=np.float32)
            if ref_kps.shape != test_kps.shape or ref_kps.ndim != 2 or ref_kps.shape[1] < 2:
                continue
            dist = np.linalg.norm(ref_kps[:, :2] - test_kps[:, :2], axis=1) / max(1.0, crop_size)
            errors.extend(float(v) for v in dist)
            slots += 1
    if not errors:
        return {"slots": 0, "mean": 0.0, "p90": 0.0}
    arr = np.asarray(errors, dtype=np.float32)
    return {
        "slots": slots,
        "mean": round(float(arr.mean()), 5),
        "p90": round(float(np.percentile(arr, 90)), 5),
    }
