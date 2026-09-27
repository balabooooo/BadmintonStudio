"""Offline segmentation tuning / evaluation on clip2 + clip5 against manual annotations.

Reuses the exact resegment pipeline (`annotation._build_context` / `_predict`) so numbers match
what the app would produce. Runs:
  1. baseline metrics (current project params)
  2. staged grid search (segmentation params, then pose-gate threshold)
  3. per-error (FN/FP) attribution dump for diagnosis

Usage:
    .venv\\Scripts\\python.exe scripts\\tune_clip25.py [--save out.json]
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

from bms.analysis import annotation as AN  # noqa: E402
from bms.analysis import pipeline as P  # noqa: E402
from bms.analysis import rally as RA  # noqa: E402
from bms.analysis import rally_vision as RV  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402

CLIPS = [
    ("clip2", ROOT / "data/projects/p_c83f8b920ee7.m_7b0cb5fd7121.analysis.json",
     ROOT / "data/annotations/0919_clip2_m_7b0cb5fd7121_960x540.anno.json"),
    ("clip5", ROOT / "data/projects/p_c83f8b920ee7.m_af2f4518f3e2.analysis.json",
     ROOT / "data/annotations/0919_clip5_m_af2f4518f3e2_960x540.anno.json"),
]


def load_clip(ajson: Path, njson: Path):
    res = AnalysisResult.model_validate(json.loads(ajson.read_text(encoding="utf-8")))
    anno = json.loads(njson.read_text(encoding="utf-8"))
    gt = [(float(r["start"]), float(r["end"])) for r in anno.get("rallies", [])
          if float(r["end"]) > float(r["start"])]
    gt.sort()
    return res, anno, gt


def eval_window(gt: list[tuple[float, float]], margin: float = 12.0) -> tuple[float, float]:
    return 0.0, max(b for _, b in gt) + margin


def metrics_window(preds, gt, lo, hi, thr=0.5):
    pr = [(a, b) for a, b in preds if b > lo and a < hi]
    gg = [(a, b) for a, b in gt if b > lo and a < hi]
    m = AN.metrics(pr, gg, thr)
    pairs = AN.match_pairs(pr, gg, thr)
    return m, pr, gg, pairs


def error_report(name, pr, gg, pairs, ctx: AN._Context, hits):
    tp_g = {j for _, j, _ in pairs}
    tp_p = {i for i, _, _ in pairs}
    print(f"  --- {name} errors ---")
    for j, g in enumerate(gg):
        if j in tp_g:
            continue
        a, b = g
        pm = ctx.player_motion
        cov = ctx.player_coverage
        fps = ctx.fps
        i0, i1 = int(a * fps), min(int(b * fps), len(pm) - 1)
        pm_seg = float(np.mean(pm[i0:i1])) if pm is not None and i1 > i0 else -1
        cov_seg = float(np.mean(cov[i0:i1] > 0.5)) if cov is not None and i1 > i0 else -1
        nh = int(np.count_nonzero((hits.times >= a) & (hits.times <= b))) if hits is not None else -1
        print(f"  FN gt[{a:7.1f},{b:7.1f}] dur={b-a:5.1f} hits_in={nh:3d} pm={pm_seg:.3f} cov={cov_seg:.2f}")
    for i, p in enumerate(pr):
        if i in tp_p:
            continue
        a, b = p
        nh = int(np.count_nonzero((hits.times >= a) & (hits.times <= b))) if hits is not None else -1
        print(f"  FP pr[{a:7.1f},{b:7.1f}] dur={b-a:5.1f} hits_in={nh:3d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {}
    for name, ajson, njson in CLIPS:
        res, anno, gt = load_clip(ajson, njson)
        lo, hi = eval_window(gt)
        ctx = AN._build_context(res, lo, hi)
        base = res.params
        hits0 = AN._gated_hits(ctx, base)

        # Baseline: current params through the exact resegment path
        preds0 = AN._predict(ctx, base)
        m0, pr0, gg, pairs0 = metrics_window(preds0, gt, lo, hi)
        print(f"\n===== {name}  window [{lo:.0f},{hi:.0f}]  GT={len(gg)} =====")
        print(f"baseline: P={m0['precision']:.3f} R={m0['recall']:.3f} F1={m0['f1']:.3f} "
              f"(n={m0['n']} TP={m0['tp']} FP={m0['fp']} FN={m0['fn']})")
        error_report(name, pr0, gg, pairs0, ctx, hits0)
        report[name] = {"baseline": m0, "window": [lo, hi], "gt": len(gg)}

        # Stage A: segmentation grid
        best = None
        results = []
        for prom, core, rest, quiet, minr in itertools.product(
                [0.08, 0.10, 0.15, 0.22, 0.30], [0.8, 1.2, 1.8, 2.5],
                [0.6, 0.8, 1.2], [0.6, 0.8, 1.0], [2.0, 2.5, 3.0]):
            p = base.model_copy(update={
                "seg_prominence": prom, "seg_min_core": core, "seg_min_rest": rest,
                "seg_min_quiet": quiet, "min_rally_seconds": minr})
            try:
                preds = AN._predict(ctx, p, hits=hits0)
            except Exception:
                continue
            m, _, _, _ = metrics_window(preds, gt, lo, hi)
            results.append((m["f1"], m["recall"], m["precision"], m["n"],
                            dict(prom=prom, core=core, rest=rest, quiet=quiet, minr=minr)))
        results.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        print("stageA top5:")
        for r in results[:5]:
            print(f"  F1={r[0]:.3f} R={r[1]:.3f} P={r[2]:.3f} n={r[3]}  {r[4]}")
        best = results[0]
        report[name]["stageA_best"] = {"f1": best[0], "params": best[4]}

        # Stage B: pose gate threshold on top of stage A best
        pa = base.model_copy(update={
            "seg_prominence": best[4]["prom"], "seg_min_core": best[4]["core"],
            "seg_min_rest": best[4]["rest"], "seg_min_quiet": best[4]["quiet"],
            "min_rally_seconds": best[4]["minr"]})
        res_b = []
        for thr_g in [0.10, 0.12, 0.16, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60]:
            p = pa.model_copy(update={"pose_gate_threshold": thr_g})
            try:
                hg = AN._gated_hits(ctx, p)
                preds = AN._predict(ctx, p, hits=hg)
            except Exception:
                continue
            m, _, _, _ = metrics_window(preds, gt, lo, hi)
            res_b.append((m["f1"], m["recall"], m["precision"], m["n"], thr_g,
                          int(hg.times.size) if hg is not None else 0))
        res_b.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        print("stageB top5 (gate thr):")
        for r in res_b[:5]:
            print(f"  thr={r[4]:.2f} F1={r[0]:.3f} R={r[1]:.3f} P={r[2]:.3f} n={r[3]} kept_hits={r[5]}")
        report[name]["stageB_best"] = {"f1": res_b[0][0], "thr": res_b[0][4]}

    if args.save:
        args.save.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
