"""Experiment v3: composite gate + two-tier hit anchoring + short-gap merge.

On top of v2 (composite pose/strength gate):
1. Two-tier refine: end anchors to last STRONG hit (score >= thr_hi); tightened when the next
   strong hit is far. Weak hits no longer extend rallies (fixes 6-9 s end overruns).
2. Short-gap merge: two adjacent segments separated by < merge_gap seconds are rejoined when a
   strong hit occurs within 1 s on BOTH sides of the gap (rally continues across a mid-rally
   lull the activity valley detector mistook for a pause).
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

sys.path.insert(0, str(ROOT / "scripts"))
from exp_gate_v2 import composite_gate, CLIPS  # noqa: E402


def refine_two_tier(intervals, hits, score, thr_hi, pre_roll, post_roll, tail_seconds,
                    trim_start, lim):
    """refine_with_hits variant: anchor on strong hits only."""
    if hits is None or hits.times.size == 0:
        return intervals
    t = hits.times
    strong_t = t[score >= thr_hi]
    anchor_t = strong_t if strong_t.size >= 2 else t
    out = []
    for iv in intervals:
        idx = np.nonzero((anchor_t >= iv.start - 1.0) & (anchor_t <= iv.end + 1.2))[0]
        if idx.size:
            first = float(anchor_t[idx[0]])
            inside = idx[anchor_t[idx] <= iv.end]
            last = float(anchor_t[inside[-1]]) if inside.size else first
            anchor = last + tail_seconds
            nxt = int(np.searchsorted(anchor_t, last + 1e-6, side="right"))
            gap_after = float(anchor_t[nxt] - last) if nxt < anchor_t.size else float("inf")
            if gap_after > lim:
                iv.end = min(iv.end, anchor)
            else:
                iv.end = max(iv.end, anchor)
            iv.end = max(iv.end, last + 0.05)
            if trim_start:
                iv.start = max(0.0, first - pre_roll)
            else:
                iv.start = min(iv.start, max(0.0, first - pre_roll))
        else:
            iv.start = max(0.0, iv.start - pre_roll)
            iv.end = iv.end + post_roll
        out.append(iv)
    return [iv for iv in out if iv.end > iv.start]


def merge_short_gaps(intervals, strong_t, merge_gap=4.0, side_win=1.0):
    if not strong_t.size:
        return intervals
    out = [intervals[0]]
    for cur in intervals[1:]:
        prev = out[-1]
        gap = cur.start - prev.end
        if 0 < gap <= merge_gap:
            left_ok = np.any((strong_t >= prev.end - side_win) & (strong_t <= prev.end + 0.3))
            right_ok = np.any((strong_t >= cur.start - 0.3) & (strong_t <= cur.start + side_win))
            if left_ok and right_ok:
                prev.end = cur.end
                continue
        out.append(cur)
    return out


def predict_v3(ctx, params, hits, score, thr_hi, merge_gap, split_lim_mult=1.0):
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
        # First pass with the standard refiner (gap splitting + anchoring on all gated hits)
        intervals = RA.refine_with_hits(
            intervals, hits, pre_roll=params.pre_roll, post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds, trim_start=method != "player_motion")
        strong_t = hits.times[score >= thr_hi]
        if strong_t.size >= 2:
            if merge_gap > 0:
                intervals = merge_short_gaps(intervals, strong_t, merge_gap=merge_gap)
            lim = RA.hit_gap_limit(strong_t) * split_lim_mult
            # Second pass: tighten ends against STRONG hits only (no re-splitting, no re-anchor)
            for iv in intervals:
                inside = strong_t[(strong_t >= iv.start) & (strong_t <= iv.end)]
                if inside.size == 0:
                    continue
                last = float(inside[-1])
                nxt = int(np.searchsorted(strong_t, last + 1e-6, side="right"))
                gap_after = float(strong_t[nxt] - last) if nxt < strong_t.size else float("inf")
                if gap_after > lim:
                    iv.end = max(min(iv.end, last + params.hit_tail_seconds), iv.start + 1.0)
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
        per_clip[name] = dict(res=res, gt=gt, lo=lo, hi=hi, ctx=ctx)

    rows = []
    for w_str, gthr, thr_hi, mgap, prom, core in itertools.product(
            [0.6], [0.40, 0.45, 0.50], [0.60, 0.70], [0.0, 3.0, 4.0], [0.08, 0.10], [1.8, 2.5]):
        tot = [0, 0, 0]
        det = {}
        for name, d in per_clip.items():
            ctx, res, gt = d["ctx"], d["res"], d["gt"]
            raw = ctx.hits_raw
            mask, score = composite_gate(raw, ctx.pose, 1.0, w_str, gthr)
            hg = POSE.filter_hits(raw, mask)
            sc = score[mask]
            p = res.params.model_copy(update={
                "seg_prominence": prom, "seg_min_core": core, "seg_min_rest": 0.6,
                "seg_min_quiet": 0.6, "min_rally_seconds": 2.0})
            preds = predict_v3(ctx, p, hg, sc, thr_hi, mgap)
            m = AN.metrics(preds, gt, 0.5)
            tot[0] += m["tp"]; tot[1] += m["fp"]; tot[2] += m["fn"]
            det[name] = m["f1"]
        P_ = tot[0] / max(tot[0] + tot[1], 1)
        R_ = tot[0] / max(tot[0] + tot[2], 1)
        F = 2 * P_ * R_ / max(P_ + R_, 1e-9)
        rows.append((F, P_, R_, w_str, gthr, thr_hi, mgap, prom, core, det["clip2"], det["clip5"]))
    rows.sort(reverse=True)
    print("combF1  P      R    | ws  gthr thr_hi mgap prom core | clip2 clip5")
    for r in rows[:20]:
        print("%.3f  %.3f  %.3f | %.1f  %.2f  %.2f  %.1f  %.2f  %.1f | %.3f %.3f" % tuple(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
