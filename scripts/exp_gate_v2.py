"""Experiment: composite hit attribution gate + corroborated hit-gap splitting.

Gate score per hit = max(w_pose * pose_ev, w_str * local_strength_pct)
  - local_strength_pct: percentile of the hit's strength among hits within +/-15 s
    (AGC-robust "our court is nearer the mic" cue; best single feature measured: AUC 0.79 vs pose 0.54)

Splitting: `split_by_hit_gaps` cuts at a large hit gap only when the gap centre also shows a
player-motion pause (pm < base + k*span). Prevents chopping long rallies when the gate misses
our hits (gate recall ~44%).
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
from bms.analysis.audio_hits import HitDetection  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402

CLIPS = [
    ("clip2", ROOT / "data/projects/p_c83f8b920ee7.m_7b0cb5fd7121.analysis.json",
     ROOT / "data/annotations/0919_clip2_m_7b0cb5fd7121_960x540.anno.json"),
    ("clip5", ROOT / "data/projects/p_c83f8b920ee7.m_af2f4518f3e2.analysis.json",
     ROOT / "data/annotations/0919_clip5_m_af2f4518f3e2_960x540.anno.json"),
]


def local_strength_pct(times: np.ndarray, strength: np.ndarray, half_win: float = 15.0) -> np.ndarray:
    n = times.size
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    order = np.argsort(times)
    ts, ss = times[order], strength[order]
    for i in range(n):
        lo = np.searchsorted(ts, ts[i] - half_win)
        hi = np.searchsorted(ts, ts[i] + half_win, side="right")
        nb = ss[lo:hi]
        out[order[i]] = float(np.mean(nb <= ss[i])) if nb.size else 0.5
    return out


def composite_gate(raw, pose, w_pose, w_str, thr):
    ev = POSE.hit_swing_evidence(pose, raw.times, strengths=raw.strength, window=0.35, one_to_one=True)
    sp = local_strength_pct(raw.times, raw.strength)
    score = np.maximum(w_pose * ev, w_str * sp)
    mask = score >= thr
    return mask, score


def split_by_hit_gaps_pm(intervals, hits, pm, fps, limit, min_side_hits=2, min_side_seconds=2.0,
                         pause_k=0.30):
    """split_by_hit_gaps + require a player-motion pause at the gap centre."""
    if hits is None or hits.times.size == 0 or not intervals:
        return intervals
    lim = float(limit)
    base = float(np.percentile(pm, 20))
    span = max(float(np.percentile(pm, 95)) - base, 1e-9)
    pause_thr = base + pause_k * span
    out: list = []

    def _pause(mid):
        i = int(mid * fps)
        a, b = max(0, i - int(0.6 * fps)), min(pm.size, i + int(0.6 * fps))
        return b > a and float(np.min(pm[a:b])) < pause_thr

    def _recurse(iv, depth=0):
        if depth > 6 or iv.end - iv.start < 2.0 * min_side_seconds:
            out.append(iv); return
        idx = AN._hit_idx_in_window(hits, iv.start, iv.end) if hasattr(AN, "_hit_idx_in_window") else RA._hit_idx_in_window(hits, iv.start, iv.end)
        if idx.size < 2 * min_side_hits:
            out.append(iv); return
        ts = hits.times[idx]
        gaps = np.diff(ts)
        k = int(np.argmax(gaps))
        if gaps[k] <= lim:
            out.append(iv); return
        left, right = ts[: k + 1], ts[k + 1:]
        if (left.size < min_side_hits or right.size < min_side_hits
                or (left[-1] - left[0]) < min_side_seconds
                or (right[-1] - right[0]) < min_side_seconds):
            out.append(iv); return
        cut = float(ts[k] + gaps[k] / 2.0)
        if not _pause(cut):      # no pause corroboration -> do not split
            out.append(iv); return
        a = RA.RallyInterval(start=iv.start, end=cut, confidence=iv.confidence)
        b = RA.RallyInterval(start=cut, end=iv.end, confidence=iv.confidence)
        _recurse(a, depth + 1); _recurse(b, depth + 1)

    for iv in intervals:
        _recurse(iv)
    return sorted(out, key=lambda v: v.start)


def predict(ctx, params, hits, pm, use_pm_guard):
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds), max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll), post_roll=float(params.post_roll),
        min_quiet=float(params.seg_min_quiet), prominence_ratio=float(params.seg_prominence),
        min_rest=float(params.seg_min_rest), min_core=float(params.seg_min_core))
    intervals, trace = P._segment_rallies(
        ctx.fused, params, ctx.duration,
        player_motion=ctx.player_motion, player_coverage=ctx.player_coverage,
        hits=hits, opt_override=opt)
    method = str(trace.get("method") or "")
    if hits is not None and getattr(hits, "times", np.zeros(0)).size:
        if use_pm_guard and pm is not None:
            lim = RA.hit_gap_limit(hits.times)
            intervals = split_by_hit_gaps_pm(intervals, hits, pm, ctx.fps, lim)
            intervals = RA.refine_with_hits(intervals, hits, pre_roll=params.pre_roll,
                                            post_roll=params.post_roll,
                                            tail_seconds=params.hit_tail_seconds,
                                            trim_start=method != "player_motion", split=False)
        else:
            intervals = RA.refine_with_hits(
                intervals, hits, pre_roll=params.pre_roll, post_roll=params.post_roll,
                tail_seconds=params.hit_tail_seconds, trim_start=method != "player_motion")
    intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=ctx.fps, activity=ctx.fused.activity)
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
        pm = ctx.player_motion
        per_clip[name] = dict(res=res, gt=gt, lo=lo, hi=hi, ctx=ctx, pm=pm)

    rows = []
    for w_pose, w_str, gthr, guard, prom, core in itertools.product(
            [0.4, 1.0], [0.6, 1.0], [0.30, 0.45, 0.60], [False, True], [0.08, 0.10], [1.8, 2.5]):
        tot = [0, 0, 0]
        det = {}
        for name, d in per_clip.items():
            ctx, res, gt = d["ctx"], d["res"], d["gt"]
            raw = ctx.hits_raw
            mask, score = composite_gate(raw, ctx.pose, w_pose, w_str, gthr)
            kr = float(mask.mean())
            if kr < 0.05 or kr > 0.97:
                hg = raw
            else:
                hg = POSE.filter_hits(raw, mask)
            p = res.params.model_copy(update={
                "seg_prominence": prom, "seg_min_core": core, "seg_min_rest": 0.6,
                "seg_min_quiet": 0.6, "min_rally_seconds": 2.0})
            preds = predict(ctx, p, hg, d["pm"], guard)
            m = AN.metrics(preds, gt, 0.5)
            tot[0] += m["tp"]; tot[1] += m["fp"]; tot[2] += m["fn"]
            det[name] = m["f1"]
        P_ = tot[0] / max(tot[0] + tot[1], 1)
        R_ = tot[0] / max(tot[0] + tot[2], 1)
        F = 2 * P_ * R_ / max(P_ + R_, 1e-9)
        rows.append((F, P_, R_, w_pose, w_str, gthr, guard, prom, core, det["clip2"], det["clip5"]))
    rows.sort(reverse=True)
    print("combF1  P      R    | wp  ws  gthr guard prom core | clip2 clip5")
    for r in rows[:20]:
        print("%.3f  %.3f  %.3f | %.1f %.1f  %.2f  %d   %.2f  %.1f | %.3f %.3f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
