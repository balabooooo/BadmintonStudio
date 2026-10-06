"""LOCO calibration for the optional boundary-refinement template (P3).

For every annotated clip (clip1-clip3) this rebuilds the P3 boundary evidence from
the stored analysis signals (no AI rerun), then:

* positives  = labeled rally start/end times;
* negatives  = in-rally quiet-valley bottoms (the "wrong snap" failure set);
* a leave-one-clip-out (LOCO) rotation fits the statistical template (quantile
  normalization ranges + directional weights + positive-score threshold) on two
  clips and evaluates it on the third;
* the held-out clip is also segmented end-to-end with refinement ON vs OFF using
  the fitted template, so we can see whether the tie-breaker actually improves
  IoU F1 / +/-1.0s boundary-band F1 instead of just separating features.

The decision rule (agreed plan): ship the refinement ON by default only if the
LOCO macro metrics improve and no held-out clip regresses beyond the IoU floor
(0.005). Otherwise it stays an opt-in flag and this artifact documents why.

Usage::

    .venv\\Scripts\\python.exe scripts\\calibrate_boundary.py
    .venv\\Scripts\\python.exe scripts\\calibrate_boundary.py --max-move 2.0 --margin 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from bms.analysis import annotation as AN  # noqa: E402
from bms.analysis import boundary as BD  # noqa: E402
from bms.analysis import pose as POSE  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402
import eval_rallies as ER  # noqa: E402

EVAL_DIR = ROOT / "data" / "cache" / "eval"


# ------------------------------------------------------------------ clip loading


def _load_clip(pair: dict[str, Path]) -> dict:
    """Build everything one clip contributes to the LOCO rotation."""
    doc = json.loads(pair["annotation"].read_text(encoding="utf-8"))
    raw = json.loads(pair["analysis"].read_text(encoding="utf-8"))
    res = AnalysisResult.model_validate(raw)
    sig = res.signals or {}
    duration = float((sig.get("duration") or [0.0])[0]) or float(doc.get("duration", 0.0))
    gt = [(float(r["start"]), float(r["end"]))
          for r in AN.normalize_rallies(doc.get("rallies") or [])]
    lo, hi = ER._clip_window(doc, gt, duration)
    gt_w = ER._in_window(gt, lo, hi)

    ctx = AN._build_context(res, lo, hi)
    hits = AN._gated_hits(ctx, res.params)
    hit_times = np.asarray(hits.times if hits is not None else [], dtype=np.float32)

    ev = BD.evidence_from_signals(sig, hit_times)
    # Stored analyses written before P3 lack pose_overhead_full; upgrade the evidence
    # with the full overhead curve straight from the v2 pose npz when it resolves.
    overhead_src = "signals"
    if ev is not None and not sig.get("pose_overhead_full"):
        tag = str((res.stats or {}).get("pose_cache", "") or "")
        cache_dir = POSE.default_cache_dir()
        if tag and cache_dir is not None:
            ps = POSE._load_cache(cache_dir / f"{tag}.npz",
                                  n=len(sig.get("pose_swing_full") or []))
            if ps is not None:
                ev = BD.build_evidence(
                    fps=ev.fps, duration=ev.duration, motion=ev.motion,
                    coverage=ev.coverage, swing=ps.swing, swing_fps=ps.fps,
                    overhead=ps.overhead, pose_coverage=ps.coverage,
                    swing_quiet=ps.quiet, hit_times=hit_times)
                overhead_src = "pose_npz"

    pos = BD.positive_samples(gt_w, window=(lo, hi))
    neg = (BD.negative_samples(ev, gt_w, res.params.seg_min_quiet,
                               res.params.seg_prominence)
           if ev is not None else {"start": [], "end": []})
    rows: dict[str, dict[str, list]] = {}
    for side in ("start", "end"):
        rows[side] = {
            "pos": BD.feature_rows(ev, side, pos[side]) if ev else [],
            "neg": BD.feature_rows(ev, side, neg[side]) if ev else [],
        }

    return {
        "clip": pair["clip"],
        "res": res,
        "ctx": ctx,
        "gt": gt_w,
        "window": [lo, hi],
        "hits": hits,
        "ev": ev,
        "pos": pos,
        "neg": neg,
        "rows": rows,
        "overhead_src": overhead_src,
    }


# ------------------------------------------------------------------ end-to-end effect


def _segment_with(clips_clip: dict, tpl: BD.Template, *, refine: bool,
                  max_move: float, margin: float) -> list[tuple[float, float]]:
    ctx = clips_clip["ctx"]
    params = clips_clip["res"].params
    on = params.model_copy(update={
        "use_boundary_refine": refine,
        "boundary_max_move": max_move,
        "boundary_score_margin": margin,
    })
    # Rebuild evidence with the LOCO template (packed neutral curves would otherwise
    # mask the fitted weights); hit times are the same gated sequence for on/off.
    hit_times = clips_clip["hits"].times if clips_clip["hits"] is not None else None
    ev = BD.evidence_from_signals(ctx.sig, hit_times, template=tpl)
    ctx.boundary_ev = ev
    return AN._predict(ctx, on, hits=clips_clip["hits"])


def _metrics(preds, gt_w) -> dict:
    m = ER.evaluate(preds, gt_w)
    return {
        "n_pred": m["n_pred"],
        "iou_0.5": m["iou_0.5"]["f1"],
        "iou_0.3": m["iou_0.3"]["f1"],
        "band_f1_1.0": AN.boundary_band_f1(preds, gt_w, band=1.0),
        "score": AN.combo_score(m["iou_0.5"]["f1"],
                                AN.boundary_band_f1(preds, gt_w, band=1.0)),
        "start_median_err": m["boundary"]["start"]["median"],
        "end_median_err": m["boundary"]["end"]["median"],
    }


def _moved_count(off, on) -> int:
    n = min(len(off), len(on))
    moved = 0
    for i in range(n):
        if abs(off[i][0] - on[i][0]) > 1e-6 or abs(off[i][1] - on[i][1]) > 1e-6:
            moved += 1
    return moved + abs(len(off) - len(on))


# ------------------------------------------------------------------ LOCO rotation


def run_loco(clips: list[dict], max_move: float, margin: float) -> dict:
    by = {c["clip"]: c for c in clips}
    out: dict[str, Any] = {}
    for held in clips:
        name = held["clip"]
        train = [c for c in clips if c is not held]
        fits = {}
        merged_diag = {}
        for side in ("start", "end"):
            rp = [r for c in train for r in c["rows"][side]["pos"]]
            rn = [r for c in train for r in c["rows"][side]["neg"]]
            fits[side] = BD.fit_template(rp, rn, side, name=f"loco-{name}")
            merged_diag[side] = {
                "n_train_pos": len(rp), "n_train_neg": len(rn),
                "fit": fits[side].diagnostics,
            }
        tpl = BD.merge_templates(fits["start"].template, fits["end"].template,
                                 name=f"loco-{name}")

        held_diag = {}
        for side in ("start", "end"):
            held_diag[side] = {
                "n_pos": len(held["rows"][side]["pos"]),
                "n_neg": len(held["rows"][side]["neg"]),
                "pos": BD.acceptance(tpl, side, held["rows"][side]["pos"]),
                "neg": BD.acceptance(tpl, side, held["rows"][side]["neg"]),
            }

        off = _segment_with(held, tpl, refine=False, max_move=max_move, margin=margin)
        on = _segment_with(held, tpl, refine=True, max_move=max_move, margin=margin)
        gt_w = held["gt"]
        m_off = _metrics(off, gt_w)
        m_on = _metrics(on, gt_w)
        out[name] = {
            "train": [c["clip"] for c in train],
            "template": {
                "name": tpl.name,
                "weights": tpl.weights,
                "ranges": {k: [round(v[0], 4), round(v[1], 4)]
                           for k, v in tpl.ranges.items()},
                "thr_start": round(tpl.thr_start, 4),
                "thr_end": round(tpl.thr_end, 4),
            },
            "train_diagnostics": merged_diag,
            "held_diagnostics": held_diag,
            "off": m_off,
            "on": m_on,
            "delta": {k: round(m_on[k] - m_off[k], 4)
                      for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")
                      if isinstance(m_off[k], (int, float))},
            "moved": _moved_count(off, on),
        }
    # Macro summary
    macro = {}
    for variant in ("off", "on"):
        macro[variant] = {
            k: round(float(np.mean([out[c["clip"]][variant][k] for c in clips])), 4)
            for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")
        }
    macro["delta"] = {k: round(macro["on"][k] - macro["off"][k], 4)
                      for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")}
    macro["moved_total"] = sum(out[c["clip"]]["moved"] for c in clips)
    worst = {c["clip"]: out[c["clip"]]["delta"]["iou_0.5"] for c in clips}
    macro["worst_iou_delta"] = min(worst.values())
    macro["ship_default_on"] = bool(
        macro["delta"]["score"] > 0 and macro["worst_iou_delta"] >= -AN.OBJECTIVE_IOU_FLOOR)
    return {"loco": out, "macro": macro}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=list(ER.DEFAULT_CLIPS))
    ap.add_argument("--max-move", type=float, default=BD.DEFAULT_MAX_MOVE)
    ap.add_argument("--margin", type=float, default=BD.DEFAULT_SCORE_MARGIN)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pairs = ER.discover_pairs(include_all=False, only=tuple(args.clips))
    clips = []
    for p in pairs:
        c = _load_clip(p)
        clips.append(c)
        ev = c["ev"]
        print(f"[{c['clip']}] evidence={'yes' if ev else 'NO'} "
              f"available={ev.available_for_refine() if ev else False} "
              f"pose_cov={ev.pose_coverage if ev else 0:.3f} "
              f"overhead={c['overhead_src']} "
              f"pos={len(c['pos']['start'])}/{len(c['pos']['end'])} "
              f"neg={len(c['neg']['start'])}/{len(c['neg']['end'])}")
    if len(clips) < 2:
        print("need at least 2 paired clips for LOCO", file=sys.stderr)
        return 1

    result = run_loco(clips, args.max_move, args.margin)

    print("\n=== held-out feature acceptance (recall / false-alarm) ===")
    for name, d in result["loco"].items():
        for side in ("start", "end"):
            h = d["held_diagnostics"][side]
            print(f"{name:>12} {side:5s} pos_recall={h['pos']['accepted']:.2f} "
                  f"neg_falsealarm={h['neg']['accepted']:.2f} "
                  f"(n+={h['n_pos']} n-={h['n_neg']})")
    print("\n=== end-to-end LOCO effect (OFF -> ON) ===")
    print(f"{'clip':>12} {'IoU.5':>13} {'IoU.3':>13} {'band1.0':>13} {'score':>13} moved")
    for name, d in result["loco"].items():
        o, n = d["off"], d["on"]
        print(f"{name:>12} {o['iou_0.5']:.3f}->{n['iou_0.5']:.3f}  "
              f"{o['iou_0.3']:.3f}->{n['iou_0.3']:.3f}  "
              f"{o['band_f1_1.0']:.3f}->{n['band_f1_1.0']:.3f}  "
              f"{o['score']:.3f}->{n['score']:.3f}  {d['moved']}")
    m = result["macro"]
    print(f"{'MACRO':>12} {m['off']['iou_0.5']:.3f}->{m['on']['iou_0.5']:.3f}  "
          f"{m['off']['iou_0.3']:.3f}->{m['on']['iou_0.3']:.3f}  "
          f"{m['off']['band_f1_1.0']:.3f}->{m['on']['band_f1_1.0']:.3f}  "
          f"{m['off']['score']:.3f}->{m['on']['score']:.3f}  {m['moved_total']}")
    print(f"worst held-out IoU delta: {m['worst_iou_delta']:+.4f} | "
          f"ship default ON: {m['ship_default_on']}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = EVAL_DIR / f"boundary_calib_{ts}{tag}.json"
    artifact = {
        "generated_at": ts,
        "script": "scripts/calibrate_boundary.py",
        "boundary_version": BD.BOUNDARY_EVIDENCE_VERSION,
        "knobs": {"max_move": args.max_move, "margin": args.margin,
                  "iou_floor": AN.OBJECTIVE_IOU_FLOOR},
        "clips": [{"clip": c["clip"], "window": c["window"],
                   "evidence_available": bool(c["ev"] and c["ev"].available_for_refine()),
                   "pose_coverage": round(c["ev"].pose_coverage, 3) if c["ev"] else 0.0,
                   "overhead_source": c["overhead_src"],
                   "n_pos": {"start": len(c["pos"]["start"]), "end": len(c["pos"]["end"])},
                   "n_neg": {"start": len(c["neg"]["start"]), "end": len(c["neg"]["end"])}}
                  for c in clips],
        **result,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\nartifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
