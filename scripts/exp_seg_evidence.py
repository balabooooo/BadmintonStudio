"""Experiment: evidence-weighted hit density for segmentation.

Idea: the pose gate is binary and weakly discriminative on multi-court footage (~57% purity,
~44% recall). Instead of a hard keep/drop, weight every hit by its pose-swing evidence when
building the *hit density* curve, and multiply that density into the fused activity curve before
valley segmentation — so neighbouring-court hits (evidence ~0) stop filling the pauses between
rallies, while a few missed gates do not collapse in-rally density.

This script evaluates offline against the manual annotations of clip2 / clip5 (annotated first
section of each), reusing cached signals only (no AI re-run).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import annotation as AN  # noqa: E402
from bms.analysis import pipeline as P  # noqa: E402
from bms.analysis import pose as POSE  # noqa: E402
from bms.analysis import rally as RA  # noqa: E402
from bms.analysis import rally_vision as RV  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402

CLIPS = [
    ("clip2", ROOT / "data/projects/p_c83f8b920ee7.m_7b0cb5fd7121.analysis.json",
     ROOT / "data/annotations/0919_clip2_m_7b0cb5fd7121_960x540.anno.json"),
    ("clip5", ROOT / "data/projects/p_c83f8b920ee7.m_af2f4518f3e2.analysis.json",
     ROOT / "data/annotations/0919_clip5_m_af2f4518f3e2_960x540.anno.json"),
]


def ev_density(raw, ev, fps, n, window=2.0, power=1.0):
    """Hit density curve weighted by pose evidence (soft cross-court suppression)."""
    out = np.zeros(n, dtype=np.float32)
    if raw is None or raw.times.size == 0:
        return out
    w = np.power(np.clip(ev, 0, 1), power)
    keep = w > 0.02
    if not keep.any():
        return out
    idx = np.clip(np.round(raw.times[keep] * fps).astype(np.int64), 0, n - 1)
    np.add.at(out, idx, w[keep])
    out = RA.smooth(out, max(1, int(round(window * fps))))
    return RA.robust_norm(out)


def predict_with_evidence(ctx, params, hits_gated, ev_dens, weight_ev: float):
    """Mirror AN._predict but with activity *= (1-w + w*ev_density) before segmentation."""
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds),
        max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll), post_roll=float(params.post_roll),
        min_quiet=float(params.seg_min_quiet),
        prominence_ratio=float(params.seg_prominence),
        min_rest=float(params.seg_min_rest),
        min_core=float(params.seg_min_core),
    )
    act = ctx.fused.activity
    if ev_dens is not None and weight_ev > 0:
        a = RA.robust_norm(act)
        act = RA.robust_norm(a * ((1.0 - weight_ev) + weight_ev * ev_dens))
    fused = RA.FusedSignal(fps=ctx.fused.fps, duration=ctx.fused.duration, activity=act,
                           threshold_hi=ctx.fused.threshold_hi, threshold_lo=ctx.fused.threshold_lo,
                           audio_reliability=ctx.fused.audio_reliability)
    intervals, trace = P._segment_rallies(
        fused, params, ctx.duration,
        player_motion=ctx.player_motion, player_coverage=ctx.player_coverage,
        hits=hits_gated, opt_override=opt)
    method = str(trace.get("method") or "")
    if hits_gated is not None and getattr(hits_gated, "times", np.zeros(0)).size:
        intervals = RA.refine_with_hits(
            intervals, hits_gated, pre_roll=params.pre_roll, post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds, trim_start=method != "player_motion")
    intervals = RA.dedupe_overlaps(intervals, hits=hits_gated, fps=ctx.fps, activity=act)
    intervals = [iv for iv in intervals if iv.end - iv.start >= params.min_rally_seconds]
    intervals.sort(key=lambda v: v.start)
    intervals = P._join_abutting(intervals)
    return [(iv.start, iv.end) for iv in intervals if iv.end > ctx.lo and iv.start < ctx.hi]


def main() -> int:
    per_clip = {}
    for name, ajson, njson in CLIPS:
        res = AnalysisResult.model_validate(json.loads(ajson.read_text(encoding="utf-8")))
        anno = json.loads(njson.read_text(encoding="utf-8"))
        gt = sorted([(float(r["start"]), float(r["end"])) for r in anno["rallies"]])
        lo, hi = 0.0, max(b for _, b in gt) + 12.0
        ctx = AN._build_context(res, lo, hi)
        raw = ctx.hits_raw
        pose = ctx.pose
        ev = POSE.hit_swing_evidence(pose, raw.times, strengths=raw.strength,
                                     window=0.35, one_to_one=True)
        per_clip[name] = dict(res=res, gt=gt, lo=lo, hi=hi, ctx=ctx, raw=raw, ev=ev)

    # Grid: evidence density weight x power x seg params (coarse)
    best = None
    rows = []
    for w_ev, power, prom, core in itertools.product(
            [0.0, 0.4, 0.6, 0.8], [1.0, 2.0], [0.08, 0.10, 0.15], [1.2, 2.5]):
        tot_tp = tot_fp = tot_fn = 0
        detail = {}
        for name, d in per_clip.items():
            ctx, res, gt, lo, hi = d["ctx"], d["res"], d["gt"], d["lo"], d["hi"]
            base = res.params
            p = base.model_copy(update={
                "seg_prominence": prom, "seg_min_core": core, "seg_min_rest": 0.6,
                "seg_min_quiet": 0.6, "min_rally_seconds": 2.0})
            hits = AN._gated_hits(ctx, p)
            n = ctx.fused.activity.size
            dens = ev_density(d["raw"], d["ev"], ctx.fps, n, power=power) if w_ev > 0 else None
            preds = predict_with_evidence(ctx, p, hits, dens, w_ev)
            m = AN.metrics(preds, gt, 0.5)
            detail[name] = m
            tot_tp += m["tp"]; tot_fp += m["fp"]; tot_fn += m["fn"]
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / max(tot_tp + tot_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((f1, prec, rec, w_ev, power, prom, core,
                     detail["clip2"]["f1"], detail["clip5"]["f1"]))
    rows.sort(reverse=True)
    print("combined  P      R      F1   | w_ev pow  prom core | clip2  clip5")
    for r in rows[:15]:
        print("  %.3f  %.3f  %.3f | %.1f  %.1f  %.2f  %.1f | %.3f  %.3f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    print("...")
    for r in rows[-5:]:
        print("  %.3f  %.3f  %.3f | %.1f  %.1f  %.2f  %.1f | %.3f  %.3f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
