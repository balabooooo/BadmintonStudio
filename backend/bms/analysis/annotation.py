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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import ANNOTATIONS_DIR
from ..i18n import tr
from . import pipeline as P
from . import boundary as BD
from . import rally as RA
from . import rally_vision as RV
from ..core.models import AnalysisParams, AnalysisResult, MediaInfo

#: Parameter search grid. Each item is (field name, candidate values).
#: Only the quantities that truly affect segmentation are searched: the four quiet-segment scales
#: + minimum rally length. ``pre_roll`` / ``post_roll`` / ``hit_tail_seconds`` affect boundary
#: snapping and also affect IoU, but more dimensions cause a combinatorial explosion, so they are
#: fixed to the project's current values for now.
EPS = 1e-9

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
            thr: float = 0.5, *, include_pairs: bool = False) -> dict[str, Any]:
    pairs = match_pairs(preds, gt, thr)
    tp = len(pairs)
    fp = len(preds) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out: dict[str, Any] = {"iou": thr, "n": len(preds), "tp": tp, "fp": fp, "fn": fn,
                           "precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(f1, 4)}
    if include_pairs:
        out["pairs"] = [[i, j, round(v, 4)] for i, j, v in pairs]
    return out


#: Tolerance bands (seconds) used by boundary-localization scoring: a predicted boundary whose
#: distance to its matched GT boundary is within the band counts as correct.
DEFAULT_BOUNDARY_BANDS: tuple[float, ...] = (0.5, 1.0, 1.5)


def error_quantiles(errors: list[float]) -> dict[str, float | None]:
    """Absolute/signed summaries of signed boundary errors (``pred - gt`` seconds)."""
    if not errors:
        return {"mean": None, "median": None, "p90": None, "signed_median": None}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "mean": round(float(np.mean(np.abs(arr))), 3),
        "median": round(float(np.median(np.abs(arr))), 3),
        "p90": round(float(np.percentile(np.abs(arr), 90)), 3),
        "signed_median": round(float(np.median(arr)), 3),
    }


def boundary_metrics(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
                     match_thr: float = 0.3,
                     bands: tuple[float, ...] = DEFAULT_BOUNDARY_BANDS) -> dict[str, Any]:
    """Boundary-localization quality, pairing predictions to GT at a LOOSE IoU threshold.

    Strict IoU@0.5 hides useful information (a rally found with boundaries 0.6s off still fails it),
    so pairs are formed at ``match_thr`` (0.3 by default) and the boundary distance is scored
    separately:

    * ``start`` / ``end`` signed errors ``pred - gt`` per matched pair, plus abs quantiles;
    * ``bands[b].rate`` — fraction of matched pairs whose error is within ``b`` seconds;
    * ``bands_f1[b]`` — boundary-point P/R/F1 within ``b`` seconds: a matched pair inside the band is
      TP, a matched pair outside it is both FP and FN (the boundary "moved"), and every unmatched
      prediction / GT adds one FP / FN. Unmatched intervals are penalized so the band score cannot
      be gamed by matching one rally perfectly while missing all others.

    Empty inputs degrade to ``None`` rates rather than raising.
    """
    pairs = match_pairs(preds, gt, match_thr)
    n_pred, n_gt = len(preds), len(gt)
    matched = len(pairs)
    out: dict[str, Any] = {
        "match_iou": match_thr, "matched": matched,
        "n_pred": n_pred, "n_gt": n_gt,
    }
    for side, idx in (("start", 0), ("end", 1)):
        errs = [float(preds[i][idx] - gt[j][idx]) for i, j, _ in pairs]
        q = error_quantiles(errs)
        band_rates: dict[str, float | None] = {}
        band_f1: dict[str, dict[str, float | int]] = {}
        for b in bands:
            if matched:
                tp = sum(1 for e in errs if abs(e) <= b)
                moved = matched - tp
                fp = moved + (n_pred - matched)
                fn = moved + (n_gt - matched)
                f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
                band_rates[str(b)] = round(tp / matched, 3)
                band_f1[str(b)] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4)}
            else:
                band_rates[str(b)] = None
                band_f1[str(b)] = {"tp": 0, "fp": n_pred, "fn": n_gt, "f1": 0.0}
        out[side] = {**q, "matched": matched, "errors": errs,
                     "bands": band_rates, "bands_f1": band_f1}
    return out


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
    boundary_ev: BD.BoundaryEvidence | None = None
    #: Fuse-weight tuple the context's ``fused`` was built with.
    fused_weights: tuple = ()
    #: Lazily re-fused activity per non-default weight tuple (weights stage); millisecond cost.
    fused_cache: dict = field(default_factory=dict)


def _build_context(res: AnalysisResult, lo: float, hi: float,
                   params: AnalysisParams | None = None) -> _Context:
    sig = res.signals or {}
    act = np.asarray(sig.get("activity_full") or [], dtype=np.float32)
    if act.size == 0:
        raise ValueError(tr("analysis.annotation.no_activity"))
    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or (act.size / fps)
    params = params or res.params
    audio_rel = float(res.stats.get("audio_reliability", 1.0) or 1.0)
    # Same offline re-fusion path as resegment: stored activity is reused verbatim with
    # default weights; calibrated weight overrides re-fuse component_*_full (no AI rerun).
    fused = P._rebuild_fused(sig, params, act=act, fps=fps, duration=duration,
                             audio_rel=audio_rel)
    has_raw = bool(sig.get("hit_times_raw"))
    hits_raw = P._rebuild_hits_raw(sig) if has_raw else P._rebuild_hits(sig)
    pose = P._rebuild_pose_signal(sig)
    pfps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    pm_raw = sig.get("player_motion_full") or []
    cov_raw = sig.get("player_coverage_full") or []
    player_motion = np.asarray(pm_raw, dtype=np.float32) if pm_raw and pfps > 0 else None
    player_coverage = (np.asarray(cov_raw, dtype=np.float32)
                       if cov_raw and len(cov_raw) == len(pm_raw) else None)
    # P3: evidence curves are independent of the gate parameters, so build once per context;
    # _predict swaps in the current gated hit times (hit feature is computed live).
    boundary_ev = None
    try:
        boundary_ev = BD.evidence_from_signals(sig, None)
    except Exception:
        boundary_ev = None
    return _Context(fused=fused, sig=sig, hits_raw=hits_raw, pose=pose, has_raw=has_raw,
                    player_motion=player_motion, player_coverage=player_coverage,
                    fps=fps, duration=duration, lo=lo, hi=hi, boundary_ev=boundary_ev,
                    fused_weights=tuple(round(params.fuse_weight_base()[k], 6)
                                        for k in RA.FUSE_COMPONENT_KEYS))


def _weight_key(params: AnalysisParams) -> tuple:
    return tuple(round(params.fuse_weight_base()[k], 6)
                 for k in RA.FUSE_COMPONENT_KEYS)


def _fused_for(ctx: _Context, params: AnalysisParams) -> RA.FusedSignal:
    """Fused signal for candidate params; re-fuses stored components only when weights differ."""
    key = _weight_key(params)
    if key == ctx.fused_weights:
        return ctx.fused
    cached = ctx.fused_cache.get(key)
    if cached is None:
        act = np.asarray(ctx.sig.get("activity_full") or [], dtype=np.float32)
        cached = P._rebuild_fused(
            ctx.sig, params, act=act, fps=ctx.fps, duration=ctx.duration,
            audio_rel=ctx.fused.audio_reliability)
        ctx.fused_cache[key] = cached
    return cached


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
    fused = _fused_for(ctx, params)
    intervals, trace = P._segment_rallies(
        fused, params, ctx.duration,
        player_motion=ctx.player_motion, player_coverage=ctx.player_coverage,
        hits=hits, opt_override=opt)
    method = str(trace.get("method") or "")
    # The optimizer MUST finish intervals with the same canonical helper as production
    # (run_analysis / resegment): historically this copy skipped the no-hit padding
    # while production applied the rolls a second time on top of rally_vision's
    # internal ones, so the F1 seen during optimization did not match the F1 shipped.
    if ctx.boundary_ev is not None:
        ctx.boundary_ev.hit_times = np.asarray(
            hits.times if hits is not None else [], dtype=np.float32)
    intervals = P._finish_intervals(
        intervals, hits=hits, params=params, fused=fused,
        duration=ctx.duration, method=method,
        boundary_ev=ctx.boundary_ev)
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


#: Boundary tolerance band (seconds) blended into the optimizer objective.
OBJECTIVE_BAND = 1.0
#: Objective blend: 0.7 IoU F1 + 0.3 boundary-band F1 (mean of start/end at ``OBJECTIVE_BAND``).
#: IoU dominates, but two IoU-tied parameter sets are distinguished by boundary localization,
#: reducing the grip of per-label timing jitter on parameter selection.
OBJECTIVE_BAND_WEIGHT = 0.3
#: The blended objective must never sacrifice more than this much IoU F1 vs the incumbent.
OBJECTIVE_IOU_FLOOR = 0.005


def boundary_band_f1(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
                     band: float = OBJECTIVE_BAND) -> float:
    """Mean of start/end boundary-point F1 within ``band`` seconds (unmatched intervals included)."""
    bm = boundary_metrics(preds, gt, bands=(band,))
    vals = [bm[side]["bands_f1"][str(band)]["f1"] for side in ("start", "end")]
    return round(float(sum(vals) / len(vals)), 4)


def combo_score(iou_f1: float, band_f1: float,
                weight: float = OBJECTIVE_BAND_WEIGHT) -> float:
    """Blended optimizer objective (IoU F1 + boundary-band F1)."""
    return round((1.0 - weight) * float(iou_f1) + weight * float(band_f1), 4)


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
        m["boundary_band_f1"] = boundary_band_f1(preds, gt)
        m["score"] = combo_score(m["f1"], m["boundary_band_f1"])
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
                         key=lambda m: (m.get("score", m["f1"]), m["f1"],
                                        m["recall"], m["precision"]),
                         reverse=True)
    return results[0]


#: Safety clamps for annotation-derived grid candidates (never let label statistics propose
#: degenerate segmentation scales).
_GRID_CLAMP: dict[str, tuple[float, float]] = {
    "min_rally_seconds": (1.5, 4.0),
    "seg_min_core": (0.6, 3.0),
    "seg_min_rest": (0.4, 1.6),
    "seg_min_quiet": (0.4, 1.4),
}


def _clamp_grid(field: str, *vals: float, step: float = 0.1) -> list[float]:
    lo, hi = _GRID_CLAMP[field]
    out = sorted({round(float(min(hi, max(lo, v))) / step) * step for v in vals if np.isfinite(v)})
    return [round(v, 2) for v in out if lo <= v <= hi]


def dynamic_grid(gt: list[tuple[float, float]]) -> list[tuple[str, list[float]]]:
    """Data-driven search grid: static candidates UNION annotation-statistic-derived ones.

    * short-rally duration quantiles (p10/p25) keep ``min_rally`` / ``min_core`` below the shortest
      true rallies so recall of quick exchanges is not gated out;
    * inter-rally gap quantiles (p10/p25) scale ``min_rest`` / ``min_quiet`` to how tightly rallies
      actually follow each other in this footage.

    Derived values are safety-clamped (:data:`_GRID_CLAMP`) and unioned with the static grid, so the
    search can never be narrower than the hard-coded one.
    """
    durs = np.asarray([b - a for a, b in gt], dtype=np.float64)
    gaps = np.asarray([gt[i + 1][0] - gt[i][1] for i in range(len(gt) - 1)], dtype=np.float64)
    gaps = gaps[gaps > 0]
    extra: dict[str, list[float]] = {}
    if durs.size >= 3:
        p10, p25 = np.percentile(durs, 10), np.percentile(durs, 25)
        extra["min_rally_seconds"] = _clamp_grid("min_rally_seconds", 0.6 * p10, 0.8 * p25)
        extra["seg_min_core"] = _clamp_grid("seg_min_core", 0.4 * p10, 0.5 * p25)
    if gaps.size >= 3:
        g10, g25 = np.percentile(gaps, 10), np.percentile(gaps, 25)
        extra["seg_min_rest"] = _clamp_grid("seg_min_rest", 0.5 * g10, 0.7 * g25)
        extra["seg_min_quiet"] = _clamp_grid("seg_min_quiet", 0.5 * g10, 0.7 * g25)
    merged: list[tuple[str, list[float]]] = []
    for name, static_vals in SEARCH_GRID:
        vals = sorted(set(static_vals) | set(extra.get(name, [])))
        merged.append((name, vals))
    return merged


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

#: Fusion-weight stage (stage E): component key -> AnalysisParams field. Fusion is scale
#: invariant (weights are normalized by their sum inside ``combine_components``), so the
#: search walks multiplicative factors around the incumbent instead of absolute values.
WEIGHT_FIELDS: dict[str, str] = {
    "players": "fuse_weight_players",
    "motion": "fuse_weight_motion",
    "audio_hits": "fuse_weight_audio",
    "shuttle": "fuse_weight_shuttle",
    "roi": "fuse_weight_roi",
}
#: Per-coordinate candidate factors (1.0 = keep incumbent); coordinate descent repeats.
WEIGHT_FACTORS = (0.5, 0.75, 1.0, 1.3, 1.7)
#: Hard clamp for any component weight after applying a factor.
WEIGHT_VALUE_RANGE = (0.1, 3.0)
#: Minimum blended-score gain for a weight move to be accepted (fights grid noise / overfit).
WEIGHT_MIN_GAIN = 0.002
#: Coordinate-descent passes (each pass walks all five components once).
WEIGHT_MAX_PASSES = 2


def _components_available(sig: dict[str, Any]) -> bool:
    """Whether all five full-rate component curves are stored (P4-1+ projects)."""
    return all(sig.get(f"component_{k}_full") for k in WEIGHT_FIELDS)


def _weights_stage(ctx: _Context, gt: list[tuple[float, float]], params: AnalysisParams,
                   iou_thr: float) -> tuple[AnalysisParams, dict[str, Any], list[dict[str, Any]]]:
    """Coordinate-descent search over the five fusion base weights (stage E).

    Returns (accepted params, stage trace, all tried metrics). Rallies are re-fused from the
    stored full-rate component curves per candidate — milliseconds, no AI rerun. Acceptance
    requires a strict blended-score gain (>= ``WEIGHT_MIN_GAIN``) and no IoU-F1 regression
    beyond ``OBJECTIVE_IOU_FLOOR``; the gate is fixed, so gated hits are reused.
    """
    hits_e = _gated_hits(ctx, params)

    def score_of(p: AnalysisParams) -> dict[str, Any]:
        preds = _predict(ctx, p, hits=hits_e)
        m = metrics(preds, gt, iou_thr)
        m["boundary_band_f1"] = boundary_band_f1(preds, gt)
        m["score"] = combo_score(m["f1"], m["boundary_band_f1"])
        return m

    cur = params
    cur_m = score_of(cur)
    tried: list[dict[str, Any]] = []
    accepted_moves = 0
    for _pass in range(WEIGHT_MAX_PASSES):
        moved = False
        for comp_key, field_name in WEIGHT_FIELDS.items():
            cur_w = float(getattr(cur, field_name))
            candidates: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
            for factor in WEIGHT_FACTORS:
                new_w = round(min(WEIGHT_VALUE_RANGE[1],
                                  max(WEIGHT_VALUE_RANGE[0], cur_w * factor)), 4)
                if abs(new_w - cur_w) < 1e-9:
                    continue
                patch = {field_name: new_w}
                p = cur.model_copy(update=patch)
                try:
                    m = score_of(p)
                except Exception:  # noqa: BLE001
                    continue
                m["params"] = {field_name: new_w}
                tried.append(m)
                # Prefer larger score, then no-op factor on ties.
                candidates.append((m["score"], -abs(factor - 1.0), m, patch))
            if not candidates:
                continue
            score, _tie, m, patch = max(candidates, key=lambda x: (x[0], x[1]))
            if (score > cur_m["score"] + WEIGHT_MIN_GAIN
                    and m["f1"] >= cur_m["f1"] - OBJECTIVE_IOU_FLOOR):
                cur = cur.model_copy(update=patch)
                cur_m = m
                moved = True
                accepted_moves += 1
        if not moved:
            break

    best_patch = {f: float(getattr(cur, f)) for f in WEIGHT_FIELDS.values()}
    trace = {
        "search_fields": list(WEIGHT_FIELDS.values()),
        "factors": list(WEIGHT_FACTORS),
        "tried": len(tried),
        "accepted_moves": accepted_moves,
        "best": {**cur_m, "params": best_patch} if accepted_moves else None,
    }
    return cur, trace, tried



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
    * **stage E — fusion weights**: coordinate descent over the five ``fuse_weight_*`` base
      weights, re-fusing stored ``component_*_full`` curves (P4-1+ projects only); skipped on
      older archives.

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
    # Data-driven grid: annotation-derived candidates (short-rally / gap quantiles) union the
    # static grid; explicit caller grids (tests) bypass the augmentation.
    grid = grid or dynamic_grid(gt)

    baseline_preds = _predict(ctx, base)
    baseline = metrics(baseline_preds, gt, iou_thr)
    baseline["boundary_band_f1"] = boundary_band_f1(baseline_preds, gt)
    baseline["score"] = combo_score(baseline["f1"], baseline["boundary_band_f1"])
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
                cur_score = (best_a.get("score", baseline["f1"])
                             if best_a else baseline["score"])
                if stage_b_best.get("score", stage_b_best["f1"]) >= cur_score:
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
                m["boundary_band_f1"] = boundary_band_f1(preds, gt)
                m["score"] = combo_score(m["f1"], m["boundary_band_f1"])
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

    # ---- Stage D: boundary padding (pre_roll / post_roll from baseline residual medians)
    # Predicted boundaries that are systematically late/early are corrected by extending the pad
    # on that side. Candidates stay close to the incumbent (0..2s extra), and the blended score
    # must improve without costing more than OBJECTIVE_IOU_FLOOR IoU F1.
    params_d = params_c
    bm0 = boundary_metrics(baseline_preds, gt, bands=(OBJECTIVE_BAND,))
    ds = bm0["start"]["signed_median"]
    de = bm0["end"]["signed_median"]
    if ds is not None and de is not None and (abs(ds) >= 0.25 or abs(de) >= 0.25):
        pre_cand = sorted({float(params_c.pre_roll),
                           round(min(2.0, max(0.0, float(params_c.pre_roll) + float(ds))), 2)})
        post_cand = sorted({float(params_c.post_roll),
                            round(min(2.0, max(0.0, float(params_c.post_roll) - float(de))), 2)})
        pad_grid = [("pre_roll", pre_cand), ("post_roll", post_cand)]
        hits_d = _gated_hits(ctx, params_c)
        res_d = _grid_search(ctx, gt, params_c, pad_grid, iou_thr, hit_labels, fixed_hits=hits_d)
        out.extend(res_d)
        best_d = _pick_best([m for m in res_d if not hit_labels or "hit" in m],
                            hit_labels=bool(hit_labels))
        cur_preds = _predict(ctx, params_c, hits=hits_d)
        cur_iou = metrics(cur_preds, gt, iou_thr)["f1"]
        cur_score = combo_score(cur_iou, boundary_band_f1(cur_preds, gt))
        if best_d is not None and best_d.get("score", 0.0) > cur_score \
                and best_d["f1"] >= cur_iou - OBJECTIVE_IOU_FLOOR:
            params_d = params_c.model_copy(update=best_d["params"])
        stages["padding"] = {
            "search_fields": ["pre_roll", "post_roll"],
            "tried": len(res_d), "best": best_d,
            "start_residual": ds, "end_residual": de,
        }

    # ---- Stage E: fusion component weights (needs P4-1+ full-rate component curves)
    params_e = params_d
    if _components_available(ctx.sig):
        params_e, w_trace, res_e = _weights_stage(ctx, gt, params_d, iou_thr)
        out.extend(res_e)
        stages["weights"] = w_trace

    # Combined best parameters re-evaluated end to end.
    final_preds = _predict(ctx, params_e)
    best = metrics(final_preds, gt, iou_thr)
    best["boundary_band_f1"] = boundary_band_f1(final_preds, gt)
    best["score"] = combo_score(best["f1"], best["boundary_band_f1"])
    patch_names = ("seg_prominence", "seg_min_core", "seg_min_rest", "seg_min_quiet",
                   "min_rally_seconds", "pre_roll", "post_roll",
                   "pose_gate_threshold", "hit_sensitivity")
    best["params"] = {n: float(getattr(params_e, n)) for n in patch_names
                      if hasattr(params_e, n)}
    if "weights" in stages:
        # Persist the five fusion weights only for projects that store component_*_full;
        # on older archives they would silently no-op in resegment, so leave them implicit.
        best["params"].update({f: float(getattr(params_e, f))
                               for f in WEIGHT_FIELDS.values()})
    if hit_labels:
        best["hit"] = hit_metrics(_predict_hit_times(ctx, params_e), hit_labels)

    out.sort(key=lambda m: (m.get("score", m["f1"]), m["f1"], m["recall"], m["precision"]),
             reverse=True)
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
        "objective": {"band": OBJECTIVE_BAND, "band_weight": OBJECTIVE_BAND_WEIGHT},
    }


# ------------------------------------------------------------------ Label quality (P2)

#: A boundary farther than this (seconds) from the nearest quiet-valley bottom is flagged.
#: Calibrated from clip1-3 GT: pooled external-gap distance q75 ~= 2.05s (start) / 1.82s (end).
LQ_QUIET_WARN = 2.0
#: Search radius (seconds) for a snap candidate around a boundary.
LQ_SNAP_RADIUS = 3.0
#: A boundary farther than this (seconds) from the nearest gated hit gets an info-level note.
LQ_HIT_FAR = 2.5
#: Robust-z threshold (|0.6745 * (d - median) / MAD|) for a duration outlier warning.
LQ_DURATION_Z = 3.5
#: Minimum number of labeled rallies before the duration-MAD check is statistically meaningful.
LQ_DURATION_MIN_N = 5
#: Adjacent labels with a gap below this are "too close" (possible missed merge).
LQ_TOO_CLOSE = 0.5
#: Half-window (seconds) for the activity-direction contrast at a boundary.
LQ_CONTRAST_WIN = 1.0


def _valley_bottoms(ctx: _Context, params: AnalysisParams) -> list[tuple[float, float]]:
    """Quiet-valley bottoms as ``(time, depth)`` on the player-motion curve (activity fallback)."""
    m = ctx.player_motion
    if m is None or m.size == 0:
        m = ctx.fused.activity
    spans = RV.find_quiet_spans(
        np.asarray(m, dtype=np.float32), ctx.fps,
        min_quiet=float(params.seg_min_quiet),
        prominence_ratio=float(params.seg_prominence))
    # The span expands over the whole below-threshold floor (which can be seconds wide); the snap
    # target is the detector's minimum frame, not the span midpoint.
    return [(q.bottom / ctx.fps, q.depth) for q in spans]


def _nearest(values: list[float], t: float) -> float | None:
    best = None
    for v in values:
        d = abs(v - t)
        if best is None or d < best:
            best = d
    return best


def _snap_candidate(t: float, valleys: list[tuple[float, float]],
                    lo_guard: float, hi_guard: float,
                    radius: float = LQ_SNAP_RADIUS) -> tuple[float, float] | None:
    """Deepest quiet valley near ``t`` (prefer depth, break ties by distance), clamped by guards.

    Returns ``(snap_t, distance)`` or ``None`` when no valley lies within ``radius`` and the
    ``(lo_guard, hi_guard)`` legality interval (no crossing neighboring labels, min 0.5s length).
    """
    cand = [(d, dist, bt) for bt, d in valleys
            if (dist := abs(bt - t)) <= radius and lo_guard < bt < hi_guard]
    if not cand:
        return None
    _, dist, bt = min(cand, key=lambda x: (-x[0], x[1]))
    return round(float(bt), 3), round(float(dist), 3)


def label_quality(res: AnalysisResult, gt: list[tuple[float, float]],
                  focus: tuple[float, float] | None = None) -> dict[str, Any]:
    """Audit manual labels for likely imprecision; emit warnings + snap *suggestions* only.

    Labels are never rewritten here. Every warning carries a stable ASCII ``code`` (translated in
    the UI, like rally tags), an optional ``side`` (start/end), measured ``distance`` and, when a
    legal candidate exists, ``snap_t``/``snap_kind``. All checks are signal-only (no AI rerun):

    * ``overlap`` / ``too_close`` — structural problems between neighboring labels;
    * ``duration_outlier`` — robust-z outlier vs the same clip's duration MAD;
    * ``boundary_off_quiet`` — boundary far from the nearest quiet-valley bottom, with a suggested
      snap to the deepest nearby valley;
    * ``boundary_no_hit`` (info) — boundary far from the nearest gated hit as well;
    * ``evidence_contradiction`` — activity on the "rally" side of the boundary is not higher than
      on the outside (a start placed inside activity / an end placed outside it).

    Missing signals degrade silently to structural checks only.
    """
    gt = [(float(a), float(b)) for a, b in gt if b > a]
    n = len(gt)
    items: list[dict[str, Any]] = []
    sev_counts = {"warn": 0, "info": 0}

    durs = np.asarray([b - a for a, b in gt], dtype=np.float64) if gt else np.zeros(0)
    z_dur: dict[int, float] = {}
    if n >= LQ_DURATION_MIN_N:
        med = float(np.median(durs))
        mad = float(np.median(np.abs(durs - med)))
        if mad > EPS:
            for i, d in enumerate(durs):
                z_dur[i] = 0.6745 * abs(float(d) - med) / mad

    ctx: _Context | None = None
    valleys: list[tuple[float, float]] = []
    hit_times: list[float] = []
    if gt:
        bbox_lo, bbox_hi = gt[0][0], max(b for _, b in gt)
        lo, hi = bbox_lo, bbox_hi
        if focus:
            lo2, hi2 = max(bbox_lo, float(focus[0])), min(bbox_hi, float(focus[1]))
            if hi2 > lo2:
                lo, hi = lo2, hi2
        try:
            ctx = _build_context(res, lo - LQ_SNAP_RADIUS, hi + LQ_SNAP_RADIUS)
            valleys = _valley_bottoms(ctx, res.params)
            gated = _gated_hits(ctx, res.params)
            hit_times = [float(t) for t in gated.times] if gated is not None else []
        except Exception:  # noqa: BLE001
            ctx = None  # no usable signals: structural checks only

    def add(item: dict[str, Any], w: dict[str, Any]) -> None:
        item["warnings"].append(w)
        sev_counts[w["severity"]] = sev_counts.get(w["severity"], 0) + 1

    for i, (s, e) in enumerate(gt):
        item: dict[str, Any] = {"index": i, "start": round(s, 3), "end": round(e, 3),
                                "severity": "info", "warnings": []}
        prev_end = gt[i - 1][1] if i > 0 else None
        next_start = gt[i + 1][0] if i + 1 < n else None

        # ---- structural
        gap_before = s - prev_end if prev_end is not None else None
        if gap_before is not None and gap_before < -EPS:
            add(item, {"code": "overlap", "severity": "warn", "side": "start",
                       "distance": round(-gap_before, 3)})
        elif gap_before is not None and 0.0 <= gap_before < LQ_TOO_CLOSE:
            add(item, {"code": "too_close", "severity": "info", "side": "start",
                       "distance": round(gap_before, 3)})
        if i + 1 < n:
            gap_after = next_start - e
            if gap_after < -EPS:
                add(item, {"code": "overlap", "severity": "warn", "side": "end",
                           "distance": round(-gap_after, 3)})
            elif 0.0 <= gap_after < LQ_TOO_CLOSE:
                add(item, {"code": "too_close", "severity": "info", "side": "end",
                           "distance": round(gap_after, 3)})
        if i in z_dur and z_dur[i] > LQ_DURATION_Z:
            add(item, {"code": "duration_outlier", "severity": "warn",
                       "duration": round(float(durs[i]), 3), "z": round(z_dur[i], 2)})

        # ---- signal-based
        if ctx is not None:
            act = ctx.fused.activity
            fps = ctx.fps

            def activity_mean(t0: float, t1: float) -> float:
                a = max(0, int(round(t0 * fps)))
                b = min(act.size, int(round(t1 * fps)))
                return float(act[a:b].mean()) if b > a else float("nan")

            # start: outside=[s-1,s], inside=[s,s+1]; expect inside > outside.
            pre_s, post_s = activity_mean(s - LQ_CONTRAST_WIN, s), activity_mean(s, s + LQ_CONTRAST_WIN)
            pre_e, post_e = activity_mean(e - LQ_CONTRAST_WIN, e), activity_mean(e, e + LQ_CONTRAST_WIN)
            rng = max(float(np.percentile(act, 95) - np.percentile(act, 20)), EPS)
            if np.isfinite(pre_s) and np.isfinite(post_s) and post_s - pre_s < -0.1 * rng:
                add(item, {"code": "evidence_contradiction", "severity": "warn", "side": "start",
                           "contrast": round(float((post_s - pre_s) / rng), 3)})
            if np.isfinite(pre_e) and np.isfinite(post_e) and pre_e - post_e < -0.1 * rng:
                add(item, {"code": "evidence_contradiction", "severity": "warn", "side": "end",
                           "contrast": round(float((pre_e - post_e) / rng), 3)})

            # A boundary belongs in the GAP between rallies: only valley bottoms in the external
            # gap (before the start / after the end, never inside the labeled interval) are valid
            # references and snap targets — intra-rally valleys are P3's negatives, not edges.
            gap_lo = prev_end if prev_end is not None else -np.inf
            gap_hi = next_start if next_start is not None else np.inf
            gap_valleys = {
                "start": [(bt, d) for bt, d in valleys if gap_lo < bt <= s],
                "end": [(bt, d) for bt, d in valleys if e <= bt < gap_hi],
            }
            for side, t in (("start", s), ("end", e)):
                gv = gap_valleys[side]
                dq = min((abs(bt - t) for bt, _ in gv), default=None)
                dh = _nearest(hit_times, t)
                if dq is not None and dq > LQ_QUIET_WARN:
                    w = {"code": "boundary_off_quiet", "severity": "warn",
                         "side": side, "distance": round(float(dq), 3)}
                    # Outward snap: the deepest bottom within the radius, gap bounds already
                    # guarantee no overlap with neighbors and a non-shrinking interval.
                    cand = [(d, abs(bt - t), bt) for bt, d in gv
                            if abs(bt - t) <= LQ_SNAP_RADIUS]
                    if cand:
                        _, dist, bt = min(cand, key=lambda x: (-x[0], x[1]))
                        w["snap_t"] = round(float(bt), 3)
                        w["snap_distance"] = round(float(dist), 3)
                        w["snap_kind"] = "quiet"
                    if dh is not None:
                        w["nearest_hit"] = round(dh, 3)
                    add(item, w)
                elif dh is not None and dh > LQ_HIT_FAR and (dq is None or dq > LQ_QUIET_WARN):
                    add(item, {"code": "boundary_no_hit", "severity": "info", "side": side,
                               "distance": round(dh, 3)})

        if any(w["severity"] == "warn" for w in item["warnings"]):
            item["severity"] = "warn"
        elif not item["warnings"]:
            item["severity"] = "ok"
        items.append(item)

    return {"count": n, "severity_counts": sev_counts, "items": items}


# ------------------------------------------------------------------ Suggestions (defaults)

#: Objective suggestions computable from the annotations that do not depend on parameter search, used to explain "why these parameters are more suitable".
def suggest(gt: list[tuple[float, float]]) -> dict[str, Any]:
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
    # Data-driven parameter candidates (same statistics dynamic_grid uses); the optimizer searches
    # the union with its static grid, this is only for display / explanation.
    out["grid"] = {name: list(vals) for name, vals in dynamic_grid(gt)}
    return out
