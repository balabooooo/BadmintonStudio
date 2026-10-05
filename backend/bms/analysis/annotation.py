"""Reading/writing manual annotations, and "using annotations to automatically optimize segmentation parameters".

Why it is needed
----------------
Segmentation parameters (quiet-segment scale, minimum rally length, etc.) used to be tunable
only by eyeballing the results and tuning by feel — HANDOVER section 8.6 explicitly says "without
ground truth you cannot tell whether tuning is correct". With manual annotations this becomes a
quantifiable problem:

1. treat the annotations as ground truth;
2. re-run segmentation over a parameter grid (reusing the stored signals, in milliseconds, without re-running AI);
3. use IoU matching to compute Precision / Recall / F1, and pick the set with the highest F1;
4. write the best parameters back into ``AnalysisParams``, then one more resegment applies them to the project.

This module only does "read annotations / save annotations / evaluate / search parameters" and
does not touch HTTP; for routes see :mod:`bms.api.annotations`.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import ANNOTATIONS_DIR
from ..i18n import tr
from . import pipeline as P
from . import rally as RA
from . import rally_vision as RV
from ..core.models import AnalysisParams, AnalysisResult, MediaInfo

#: Parameter search grid. Each item is (field name, candidate values).
#: Only the quantities that truly affect segmentation are searched: the four quiet-segment scales
#: + minimum rally length. ``pre_roll`` / ``post_roll`` / ``hit_tail_seconds`` affect boundary
#: snapping and also affect IoU, but more dimensions cause a combinatorial explosion, so they are
#: fixed to the project's current values for now.
SEARCH_GRID: list[tuple[str, list[float]]] = [
    ("seg_prominence", [0.10, 0.15, 0.22, 0.30]),
    ("seg_min_core", [0.8, 1.2, 1.8, 2.5]),
    ("seg_min_rest", [0.6, 0.8, 1.2]),
    ("seg_min_quiet", [0.6, 0.8, 1.0]),
    ("min_rally_seconds", [2.0, 2.5, 3.0]),
]


# ------------------------------------------------------------------ Annotation I/O


def annotation_path(media: MediaInfo) -> Path:
    """Annotation files are named after the "proxy video name" (kept consistent with the old standalone annotation tool).

    Use the proxy name rather than the original name: the analysis actually processes the proxy,
    and annotations are made against the proxy frames; moreover the old tool has already saved a
    batch of annotations under the proxy name, so reusing the name lets them be read directly.

    ``proxy_path`` may be cleared together with the cache, so reconstruct the (deterministic)
    proxy name from the media instead of falling back to the source name -- otherwise existing
    annotations would suddenly appear missing.
    """
    if media.proxy_path:
        return ANNOTATIONS_DIR / f"{Path(media.proxy_path).stem}.anno.json"
    if media.path:
        from ..core.media import proxy_stem

        return ANNOTATIONS_DIR / f"{proxy_stem(media)}.anno.json"
    return ANNOTATIONS_DIR / f"{media.id}.anno.json"


def load_annotation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rallies": [], "hits": [], "focus": None, "note": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"rallies": [], "hits": [], "focus": None, "note": ""}
    if not isinstance(data, dict):
        return {"rallies": [], "hits": [], "focus": None, "note": ""}
    data.setdefault("rallies", [])
    data.setdefault("hits", [])
    return data


def normalize_rallies(raw: Iterable[dict]) -> list[dict[str, Any]]:
    """Sanitize annotated rallies: drop items with illegal / inverted start-end, sort by start."""
    out: list[dict[str, Any]] = []
    for r in raw or []:
        try:
            start = max(0.0, float(r["start"]))
            end = float(r["end"])
        except Exception:
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "note": str(r.get("note") or ""),
            "source": str(r.get("source") or "manual"),
        })
    out.sort(key=lambda x: x["start"])
    return out


def normalize_hits(raw: Iterable[dict]) -> list[dict[str, Any]]:
    """Sanitize hit-level labels ``{t, ours}`` (``ours=True`` = our court, ``False`` = neighboring court)."""
    out: list[dict[str, Any]] = []
    for h in raw or []:
        try:
            t = float(h["t"])
        except Exception:
            continue
        if not np.isfinite(t) or t < 0:
            continue
        out.append({"t": round(t, 3), "ours": bool(h.get("ours", True))})
    out.sort(key=lambda x: x["t"])
    return out


def save_annotation(path: Path, media: MediaInfo, payload: dict, duration: float,
                    fps: float) -> dict[str, Any]:
    rallies = normalize_rallies(payload.get("rallies") or [])
    hits = normalize_hits(payload.get("hits") or [])
    focus = payload.get("focus")
    doc = {
        "video": str(media.proxy_path or media.path),
        "video_name": Path(media.proxy_path or media.path).name,
        "duration": round(float(duration), 3),
        "fps": round(float(fps), 4),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rallies),
        "hit_count": len(hits),
        "focus": focus,
        "note": str(payload.get("note") or ""),
        "rallies": rallies,
        "hits": hits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return doc


def annotation_to_csv(doc: dict, fps: float) -> str:
    lines = ["index,start,end,duration,start_frame,end_frame,source,note"]
    for i, r in enumerate(normalize_rallies(doc.get("rallies") or []), 1):
        s, e = float(r["start"]), float(r["end"])
        sf = int(round(s * fps)) if fps else ""
        ef = int(round(e * fps)) if fps else ""
        note = str(r.get("note") or "").replace(",", " ")
        lines.append(f"{i},{s:.3f},{e:.3f},{e - s:.3f},{sf},{ef},{r.get('source','')},{note}")
    return "\n".join(lines)


# ------------------------------------------------------------------ Evaluation


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_pairs(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
                thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy IoU matching (pair from highest to lowest IoU, one-to-one)."""
    cand: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for v, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, v))
    return pairs


def metrics(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
            thr: float = 0.5) -> dict[str, float]:
    pairs = match_pairs(preds, gt, thr)
    tp = len(pairs)
    fp = len(preds) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"iou": thr, "n": len(preds), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def _nearest_label(t: float, labels: list[tuple[float, bool]], tol: float) -> bool | None:
    best: bool | None = None
    bd = tol
    for lt, ours in labels:
        d = abs(lt - t)
        if d <= bd:
            bd, best = d, ours
    return best


def hit_metrics(preds: Iterable[float], labels: Iterable[tuple[float, bool]],
                tol: float = 0.12) -> dict[str, float]:
    """Precision / recall / F1 of the *hit attribution gate* against hit-level labels.

    ``labels`` carry ``(time, ours)`` where ``ours=False`` marks a neighboring-court sound. A kept
    hit that matches a "neighbor" label (or matches nothing) is a false positive; an "ours" label
    with no kept hit near it is a false negative.
    """
    lab = [(float(t), bool(o)) for t, o in labels]
    if not lab:
        return {}
    ours = [t for t, o in lab if o]
    preds = [float(t) for t in preds]
    tp = fp = fn = 0
    for t in preds:
        if _nearest_label(t, lab, tol) is True:
            tp += 1
        else:
            fp += 1
    for t in ours:
        if not any(abs(p - t) <= tol for p in preds):
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ------------------------------------------------------------------ Parameter search


@dataclass
class _Context:
    fused: RA.FusedSignal
    sig: dict[str, Any]
    hits_raw: Any
    pose: Any
    has_raw: bool
    player_motion: np.ndarray | None
    player_coverage: np.ndarray | None
    fps: float
    duration: float
    lo: float
    hi: float


def _build_context(res: AnalysisResult, lo: float, hi: float) -> _Context:
    sig = res.signals or {}
    act = np.asarray(sig.get("activity_full") or [], dtype=np.float32)
    if act.size == 0:
        raise ValueError(tr("analysis.annotation.no_activity"))
    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or (act.size / fps)
    med = float(np.median(act))
    fused = RA.FusedSignal(
        fps=fps, duration=duration, activity=act,
        threshold_hi=max(float(np.percentile(act, 78)), med * 1.25),
        threshold_lo=max(float(np.percentile(act, 55)) * 0.92, med * 1.05),
        audio_reliability=float(res.stats.get("audio_reliability", 1.0) or 1.0),
    )
    has_raw = bool(sig.get("hit_times_raw"))
    hits_raw = P._rebuild_hits_raw(sig) if has_raw else P._rebuild_hits(sig)
    pose = P._rebuild_pose_signal(sig)
    pfps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    pm_raw = sig.get("player_motion_full") or []
    cov_raw = sig.get("player_coverage_full") or []
    player_motion = np.asarray(pm_raw, dtype=np.float32) if pm_raw and pfps > 0 else None
    player_coverage = (np.asarray(cov_raw, dtype=np.float32)
                       if cov_raw and len(cov_raw) == len(pm_raw) else None)
    return _Context(fused=fused, sig=sig, hits_raw=hits_raw, pose=pose, has_raw=has_raw,
                    player_motion=player_motion, player_coverage=player_coverage,
                    fps=fps, duration=duration, lo=lo, hi=hi)


def _gated_hits(ctx: _Context, params: AnalysisParams):
    """Apply the (possibly re-tuned) hit attribution gate to the raw candidates."""
    hits, _trace = P.regate_hits(ctx.sig, params, hits_raw=ctx.hits_raw, pose=ctx.pose)
    return hits


def _predict(ctx: _Context, params: AnalysisParams, hits=None) -> list[tuple[float, float]]:
    """Run one segmentation + finishing pass with a set of parameters, returning rallies that fall within the annotation window.

    ``hits`` may be supplied when the caller already gated the raw candidates (e.g. the segmentation
    stage, where the gate parameters are constant), avoiding a redundant re-gate for every combo.
    """
    if hits is None:
        hits = _gated_hits(ctx, params)
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds),
        max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll), post_roll=float(params.post_roll),
        min_quiet=float(params.seg_min_quiet),
        prominence_ratio=float(params.seg_prominence),
        min_rest=float(params.seg_min_rest),
        min_core=float(params.seg_min_core),
    )
    intervals, trace = P._segment_rallies(
        ctx.fused, params, ctx.duration,
        player_motion=ctx.player_motion, player_coverage=ctx.player_coverage,
        hits=hits, opt_override=opt)
    method = str(trace.get("method") or "")
    if hits is not None and getattr(hits, "times", np.zeros(0)).size:
        intervals = RA.refine_with_hits(
            intervals, hits, pre_roll=params.pre_roll, post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds,
            trim_start=method != "player_motion")
    intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=ctx.fps,
                                   activity=ctx.fused.activity)
    intervals = [iv for iv in intervals if iv.end - iv.start >= params.min_rally_seconds]
    intervals.sort(key=lambda v: v.start)
    intervals = P._join_abutting(intervals)
    return [(iv.start, iv.end) for iv in intervals
            if iv.end > ctx.lo and iv.start < ctx.hi]


def _predict_hit_times(ctx: _Context, params: AnalysisParams, hits=None) -> list[float]:
    if hits is None:
        hits = _gated_hits(ctx, params)
    if hits is None or hits.times.size == 0:
        return []
    # Only hits inside the evaluation window are scoreable: hit labels cover just the annotated
    # region, so a kept hit outside it would count as a false positive it does not deserve.
    return [float(t) for t in hits.times if ctx.lo <= t <= ctx.hi]


def _grid_search(ctx: _Context, gt: list[tuple[float, float]], base: AnalysisParams,
                 grid: list[tuple[str, list[float]]], iou_thr: float,
                 hit_labels: list[tuple[float, bool]] | None,
                 fixed_hits=None) -> list[dict[str, Any]]:
    names = [n for n, _ in grid]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*[vals for _, vals in grid]):
        patch = dict(zip(names, combo))
        params = base.model_copy(update=patch)
        try:
            preds = _predict(ctx, params, hits=fixed_hits)
        except Exception:  # noqa: BLE001
            continue
        m = metrics(preds, gt, iou_thr)
        m["params"] = {k: float(v) for k, v in patch.items()}
        if hit_labels:
            m["hit"] = hit_metrics(_predict_hit_times(ctx, params, hits=fixed_hits), hit_labels)
        out.append(m)
    return out


def _pick_best(results: list[dict[str, Any]], hit_labels: bool) -> dict[str, Any] | None:
    if not results:
        return None
    if hit_labels:
        results = sorted(
            results,
            key=lambda m: ((m.get("hit") or {}).get("f1", 0.0),
                           (m.get("hit") or {}).get("recall", 0.0),
                           m.get("f1", 0.0)),
            reverse=True)
    else:
        results = sorted(results,
                         key=lambda m: (m["f1"], m["recall"], m["precision"]),
                         reverse=True)
    return results[0]


#: Gate-parameter grid (stage B). The "cross-court suppression threshold" is the quantity most worth
#: calibrating from annotations: too low keeps neighboring-court sounds (rallies get glued / extended),
#: too high drops real hits.
GATE_GRID: list[tuple[str, list[float]]] = [
    ("pose_gate_threshold", [0.12, 0.18, 0.22, 0.28, 0.35, 0.45]),
]

#: Hit-detection sensitivity grid (stage C). Requires the cached audio via ``sensitivity_fn``.
SENS_GRID: list[tuple[str, list[float]]] = [
    ("hit_sensitivity", [0.3, 0.4, 0.5, 0.65, 0.8]),
]


def optimize(res: AnalysisResult, gt: list[tuple[float, float]],
             focus: tuple[float, float] | None = None,
             grid: list[tuple[str, list[float]]] | None = None,
             iou_thr: float = 0.5,
             hit_labels: list[tuple[float, bool]] | None = None,
             sensitivity_fn=None) -> dict[str, Any]:
    """Search for segmentation / gate / hit-sensitivity parameters with the highest F1.

    Staged coordinate descent (avoids the combinatorial explosion of a joint grid):

    * **stage A — segmentation**: the quiet-valley scales + minimum rally length (rally IoU F1);
    * **stage B — hit attribution gate**: ``pose_gate_threshold``, scored by hit-level F1 when
      ``hit_labels`` are given, otherwise by rally F1 (re-gating from the raw candidates);
    * **stage C — hit sensitivity**: ``hit_sensitivity`` via ``sensitivity_fn(sensitivity)`` which
      re-detects hits from the cached audio; only meaningful with ``hit_labels``.

    Scoring happens inside an evaluation window = ``focus`` ∩ annotation bounding box. Anything
    outside is unscored (neither positive nor negative evidence), so annotating only part of the
    video is safe.

    Returns a dict with ``baseline``, ``best`` (combined parameters), ``results`` and per-stage
    ``stages``. This is **only a search**; applying it still requires writing ``best`` back into params
    and resegmenting.
    """
    gt = [(float(a), float(b)) for a, b in gt if b > a]
    if not gt:
        raise ValueError(tr("analysis.annotation.empty_gt"))
    hit_labels = [(float(t), bool(o)) for t, o in (hit_labels or [])]
    # Evaluation window = saved focus ∩ annotation bounding box. Regions without annotations are
    # **not** negative evidence ("no rally here") — they are simply unscored: a partial annotation
    # (say 0:10~1:00 of a 12min video) must not count correctly detected rallies in the rest of the
    # video as false positives, or the search would be biased toward over-suppressed segmentation.
    bbox_lo = min(a for a, _ in gt)
    bbox_hi = max(b for _, b in gt)
    lo, hi = bbox_lo, bbox_hi
    if focus:
        lo, hi = max(bbox_lo, float(focus[0])), min(bbox_hi, float(focus[1]))
        if hi <= lo:  # focus does not overlap the annotations at all: fall back to the bbox
            lo, hi = bbox_lo, bbox_hi
    ctx = _build_context(res, lo, hi)
    base = res.params
    grid = grid or SEARCH_GRID

    baseline_preds = _predict(ctx, base)
    baseline = metrics(baseline_preds, gt, iou_thr)
    baseline["params"] = {n: float(getattr(base, n)) for n, _ in grid}
    if hit_labels:
        baseline["hit"] = hit_metrics(_predict_hit_times(ctx, base), hit_labels)

    out: list[dict[str, Any]] = []
    stages: dict[str, Any] = {}

    # ---- Stage A: segmentation structure (gate fixed -> gate once, reuse across combos)
    hits_a = _gated_hits(ctx, base)
    res_a = _grid_search(ctx, gt, base, grid, iou_thr, hit_labels, fixed_hits=hits_a)
    out.extend(res_a)
    best_a = _pick_best(res_a, hit_labels=False)
    params_a = base
    if best_a is not None:
        params_a = base.model_copy(update=best_a["params"])
    stages["segment"] = {
        "search_fields": [n for n, _ in grid],
        "tried": len(res_a),
        "best": best_a,
    }

    # ---- Stage B: hit attribution gate (needs candidate hits + pose)
    params_b = params_a
    stage_b_best = None
    if ctx.pose is not None and ctx.hits_raw is not None \
            and getattr(ctx.hits_raw, "times", np.zeros(0)).size:
        res_b = _grid_search(ctx, gt, params_a, GATE_GRID, iou_thr, hit_labels)
        out.extend(res_b)
        stage_b_best = _pick_best(res_b, hit_labels=bool(hit_labels))
        if stage_b_best is not None:
            # Only accept the gate change when it actually improves the chosen objective.
            base_key = "hit" if hit_labels else None
            if base_key:
                cur = (baseline.get("hit") or {}).get("f1", 0.0)
                new = (stage_b_best.get("hit") or {}).get("f1", 0.0)
                if new >= cur:
                    params_b = params_a.model_copy(update=stage_b_best["params"])
            else:
                if stage_b_best["f1"] >= (best_a["f1"] if best_a else baseline["f1"]):
                    params_b = params_a.model_copy(update=stage_b_best["params"])
        stages["gate"] = {
            "search_fields": [n for n, _ in GATE_GRID],
            "tried": len(res_b),
            "best": stage_b_best,
        }

    # ---- Stage C: hit sensitivity (needs cached audio to re-detect)
    params_c = params_b
    if sensitivity_fn is not None and hit_labels:
        res_c: list[dict[str, Any]] = []
        for _n, vals in SENS_GRID:
            for sens in vals:
                patch = {"hit_sensitivity": float(sens)}
                try:
                    raw = sensitivity_fn(float(sens))
                except Exception:  # noqa: BLE001
                    continue
                if raw is None:
                    continue
                cctx = replace(ctx, hits_raw=raw, has_raw=True)
                p = params_b.model_copy(update=patch)
                try:
                    preds = _predict(cctx, p)
                except Exception:  # noqa: BLE001
                    continue
                m = metrics(preds, gt, iou_thr)
                m["params"] = patch
                m["hit"] = hit_metrics(_predict_hit_times(cctx, p), hit_labels)
                res_c.append(m)
                out.append(m)
        best_c = _pick_best(res_c, hit_labels=True)
        if best_c is not None:
            cur = ((baseline.get("hit") or {}).get("f1", 0.0)
                   if not stage_b_best else (stage_b_best.get("hit") or {}).get("f1", 0.0))
            if (best_c.get("hit") or {}).get("f1", 0.0) >= cur:
                params_c = params_b.model_copy(update=best_c["params"])
        stages["sensitivity"] = {
            "search_fields": [n for n, _ in SENS_GRID],
            "tried": len(res_c),
            "best": best_c,
        }

    # Combined best parameters re-evaluated end to end.
    final_preds = _predict(ctx, params_c)
    best = metrics(final_preds, gt, iou_thr)
    patch_names = ("seg_prominence", "seg_min_core", "seg_min_rest", "seg_min_quiet",
                   "min_rally_seconds", "pose_gate_threshold", "hit_sensitivity")
    best["params"] = {n: float(getattr(params_c, n)) for n in patch_names
                      if hasattr(params_c, n)}
    if hit_labels:
        best["hit"] = hit_metrics(_predict_hit_times(ctx, params_c), hit_labels)

    out.sort(key=lambda m: (m["f1"], m["recall"], m["precision"]), reverse=True)
    return {
        "gt_count": len(gt),
        "focus": [lo, hi],
        "iou_threshold": iou_thr,
        "baseline": baseline,
        "best": best,
        "results": out[:40],
        "tried": len(out),
        "search_fields": [n for n, _ in grid],
        "stages": stages,
        "hit_label_count": len(hit_labels),
    }


# ------------------------------------------------------------------ Suggestions (defaults)

#: Objective suggestions computable from the annotations that do not depend on parameter search, used to explain "why these parameters are more suitable".
def suggest(gt: list[tuple[float, float]]) -> dict[str, float]:
    gt = [(float(a), float(b)) for a, b in gt if b > a]
    if not gt:
        return {}
    durs = np.asarray([b - a for a, b in gt], dtype=np.float64)
    gaps = np.asarray([gt[i + 1][0] - gt[i][1] for i in range(len(gt) - 1)], dtype=np.float64)
    gaps = gaps[gaps > 0]
    out = {
        "count": float(len(gt)),
        "duration_min": round(float(durs.min()), 2),
        "duration_median": round(float(np.median(durs)), 2),
        "duration_max": round(float(durs.max()), 2),
    }
    if gaps.size:
        out["gap_median"] = round(float(np.median(gaps)), 2)
        out["gap_min"] = round(float(gaps.min()), 2)
    return out
