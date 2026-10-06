"""Evaluate rally segmentation using manual annotations (offline, no need to rerun AI).

This is what HANDOVER section 8.6 calls for: without a set of manual annotations
there is no basis for tuning segmentation parameters.

Annotation format (``data/annotations/*.anno.json``)::

    {
      "duration": 1800.4, "fps": 30.0,
      "focus": [0, 300],                 # time window covered by the annotation (seconds)
      "rallies": [{"start": 5.6, "end": 10.4}, ...]
    }

Analysis results use ``data/projects/*.analysis.json`` (containing signals/rallies/params).
The script will:

1. Restrict both the annotations and the analysis results to the ``focus`` time window
   (if the annotation only covers part of a clip, comparing against rallies from the
   whole clip underestimates precision);
2. Compute Precision / Recall / F1 with greedy IoU matching (default IoU >= 0.5);
3. Print the time position of every missed (FN) / false (FP) detection for easier triage;
4. With ``--sweep``, scan ``SegmentOptions`` (quiet-span scale) online and report the
   best combination.

Usage::

    .venv\\Scripts\\python.exe scripts\\eval_segmentation.py
    .venv\\Scripts\\python.exe scripts\\eval_segmentation.py --json <analysis json> --sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import pipeline as P          # noqa: E402
from bms.analysis import rally as RA            # noqa: E402
from bms.analysis import rally_vision as RV     # noqa: E402
from bms.core.models import AnalysisResult      # noqa: E402

ANNO_DIR = ROOT / "data" / "annotations"
DEFAULT_JSON = (ROOT / "data" / "projects"
                / "p_e274f1ae3e1e.m_aa6b31f616ee.analysis.json")


def load_annotation() -> tuple[list[tuple[float, float]], float, float]:
    files = sorted(ANNO_DIR.glob("*.anno.json"))
    if not files:
        raise SystemExit(f"没找到标注文件：{ANNO_DIR}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    gt = [(float(r["start"]), float(r["end"])) for r in data.get("rallies", [])
          if float(r["end"]) > float(r["start"])]
    focus = data.get("focus") or [0.0, float(data.get("duration", 0.0))]
    lo, hi = float(focus[0]), float(focus[1])
    gt = [(a, b) for a, b in gt if b > lo and a < hi]
    print(f"标注：{files[0].name}   窗口 [{lo:.0f}, {hi:.0f}]s   {len(gt)} 个回合")
    return gt, lo, hi


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gt, thr: float):
    cand = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    pi: set[int] = set()
    gi: set[int] = set()
    pairs = []
    for v, i, j in cand:
        if i in pi or j in gi:
            continue
        pi.add(i)
        gi.add(j)
        pairs.append((i, j, v))
    return pairs


def report(name: str, preds: list[tuple[float, float]], gt) -> dict:
    out = {"name": name, "n": len(preds)}
    for thr in (0.5, 0.3):
        pairs = match(preds, gt, thr)
        tp = len(pairs)
        fp = len(preds) - tp
        fn = len(gt) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if thr == 0.5:
            out.update(tp=tp, fp=fp, fn=fn, p=prec, r=rec, f1=f1)
        print(f"  {name:26s} n={len(preds):3d}  IoU>={thr:.1f}: "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    return out


def _as_intervals(res: AnalysisResult, lo: float, hi: float) -> list[tuple[float, float]]:
    return [(r.start, r.end) for r in res.rallies if r.end > lo and r.start < hi]


def _post_process(intervals, hits, fused, params, method: str):
    """Use the canonical finishing pipeline shared with run_analysis / resegment."""
    return P._finish_intervals(
        intervals, hits=hits, params=params, fused=fused,
        duration=fused.duration, method=method)


def build_fused(res: AnalysisResult) -> RA.FusedSignal:
    sig = res.signals or {}
    act = np.asarray(sig.get("activity_full") or [], dtype=np.float32)
    if act.size == 0:
        raise SystemExit("分析结果里没有 activity_full，无法离线重切分")
    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or (act.size / fps)
    med = float(np.median(act))
    return RA.FusedSignal(
        fps=fps, duration=duration, activity=act,
        threshold_hi=max(float(np.percentile(act, 78)), med * 1.25),
        threshold_lo=max(float(np.percentile(act, 55)) * 0.92, med * 1.05),
        audio_reliability=float(res.stats.get("audio_reliability", 1.0) or 1.0),
    )


def player_arrays(res: AnalysisResult, fused: RA.FusedSignal):
    sig = res.signals or {}
    pfps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    pm = sig.get("player_motion_full") or []
    cov = sig.get("player_coverage_full") or []
    player_motion = np.asarray(pm, dtype=np.float32) if pm and pfps > 0 else None
    player_coverage = (np.asarray(cov, dtype=np.float32)
                       if cov and len(cov) == len(pm) else None)
    return player_motion, player_coverage, pfps


def sweep(res: AnalysisResult, gt, hits, fused: RA.FusedSignal) -> None:
    player_motion, player_coverage, pfps = player_arrays(res, fused)
    if player_motion is None:
        print("\n没有存球员运动曲线（player_motion_full），无法扫切分尺度。")
        return
    print("\n=== 扫描 SegmentOptions（击球密度证据路径）===")
    print("  prom  core rest quiet min |    P     R    F1   n")
    best = None
    for prom, core, rest, quiet, minrally in itertools.product(
            [0.10, 0.15, 0.22, 0.30], [0.8, 1.2, 1.8, 2.5],
            [0.6, 0.8, 1.2], [0.6, 1.0], [2.0, 2.5, 3.0]):
        params = res.params.model_copy(update={"min_rally_seconds": minrally})
        opt = RV.SegmentOptions(min_rally=minrally, max_rally=params.max_rally_seconds,
                                pre_roll=params.pre_roll, post_roll=params.post_roll,
                                min_core=core, min_rest=rest,
                                min_quiet=quiet, prominence_ratio=prom)
        intervals, trace = P._segment_rallies(
            fused, params, fused.duration, player_motion=player_motion,
            player_coverage=player_coverage, hits=hits, opt_override=opt)
        intervals = _post_process(intervals, hits, fused, params, trace.get("method", ""))
        preds = [(iv.start, iv.end) for iv in intervals]
        pairs = match(preds, gt, 0.5)
        tp = len(pairs)
        fp = len(preds) - tp
        fn = len(gt) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if best is None or f1 > best[0]:
            best = (f1, prec, rec, len(preds), prom, core, rest, quiet, minrally)
            print(f"  {prom:.2f} {core:4.1f} {rest:4.1f} {quiet:4.1f} {minrally:3.1f} | "
                  f"{prec:.3f} {rec:.3f} {f1:.3f} {len(preds):3d}  *")
    if best:
        print(f"\n  最优: F1={best[0]:.3f} P={best[1]:.3f} R={best[2]:.3f}  "
              f"prominence={best[4]} min_core={best[5]} min_rest={best[6]} "
              f"min_quiet={best[7]} min_rally={best[8]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--sweep", action="store_true", help="扫描静默段尺度参数")
    args = ap.parse_args()

    gt, lo, hi = load_annotation()
    raw = json.loads(args.json.read_text(encoding="utf-8"))
    res = AnalysisResult.model_validate(raw)
    duration = float((res.signals.get("duration") or [0.0])[0])
    print(f"分析：{args.json.name}   时长 {duration:.1f}s   "
          f"audio_reliability={res.stats.get('audio_reliability')}")

    print("\n=== 对比 ===")
    report("存下来的回合（旧）", _as_intervals(res, lo, hi), gt)
    new_res = P.resegment(res.model_copy(deep=True), res.params, "balanced")
    report("resegment（新代码）", _as_intervals(new_res, lo, hi), gt)

    hits = P._rebuild_hits(res.signals)
    fused = build_fused(res)
    if args.sweep and hits is not None:
        sweep(res, gt, hits, fused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
