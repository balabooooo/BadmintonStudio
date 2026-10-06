"""LOCO calibration for the fusion-weight optimizer stage (P4-2).

Stage E of ``annotation.optimize`` tunes the five fusion base weights
(``fuse_weight_players/motion/audio/shuttle/roi``) by coordinate descent on the
stored full-rate component curves (``component_*_full``). This script runs a
leave-one-clip-out (LOCO) rotation on clip1-clip3:

* train clips : run the full staged optimizer and take the five calibrated
  weights from each clip's ``best.params``;
* weight table: per-field median across the train clips;
* held clip   : score its own params with default weights vs its own params
  plus the LOCO table (segmentation scales / padding stay clip-local; only
  the weight table transfers);
* in-sample   : the held clip's own optimized weights are also scored, to
  expose the in-sample -> LOCO generalization gap.

Decision rule (agreed plan, same as P3): change the shipped default weights
only if the LOCO macro combo score improves and no held-out clip regresses
beyond the IoU floor (0.005). Otherwise defaults stay and this artifact
documents why.

Requires P4-1+ analyses (``component_*_full`` present in the project JSON);
older analyses must be re-run once to backfill those curves.

Usage::

    .venv\\Scripts\\python.exe scripts\\calibrate_weights.py
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
from bms.core.models import AnalysisResult  # noqa: E402
import eval_rallies as ER  # noqa: E402

EVAL_DIR = ROOT / "data" / "cache" / "eval"

WEIGHT_FIELDS = tuple(AN.WEIGHT_FIELDS.values())


# ------------------------------------------------------------------ clip loading


def _load_clip(pair: dict[str, Path]) -> dict:
    doc = json.loads(pair["annotation"].read_text(encoding="utf-8"))
    raw = json.loads(pair["analysis"].read_text(encoding="utf-8"))
    res = AnalysisResult.model_validate(raw)
    sig = res.signals or {}
    duration = float((sig.get("duration") or [0.0])[0]) or float(doc.get("duration", 0.0))
    gt = [(float(r["start"]), float(r["end"]))
          for r in AN.normalize_rallies(doc.get("rallies") or [])]
    lo, hi = ER._clip_window(doc, gt, duration)
    gt_w = ER._in_window(gt, lo, hi)
    has_components = AN._components_available(sig)
    return {
        "clip": pair["clip"], "res": res, "gt": gt_w,
        "window": [lo, hi], "has_components": has_components,
    }


# ------------------------------------------------------------------ evaluation


def _predict_with(clip: dict, params) -> list[tuple[float, float]]:
    lo, hi = clip["window"]
    ctx = AN._build_context(clip["res"], lo, hi, params)
    return AN._predict(ctx, params)


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


def _weights_of(params) -> dict[str, float]:
    return {f: round(float(getattr(params, f)), 4) for f in WEIGHT_FIELDS}


# ------------------------------------------------------------------ LOCO rotation


def _optimize_clip(clip: dict) -> dict:
    """Full staged optimize on one clip; returns best params + in-sample metrics."""
    lo, hi = clip["window"]
    out = AN.optimize(clip["res"], clip["gt"], focus=(lo, hi))
    best_params = clip["res"].params.model_copy(update=out["best"]["params"])
    return {
        "params": best_params,
        "weights": _weights_of(best_params),
        "baseline": {k: out["baseline"][k] for k in ("f1", "boundary_band_f1", "score")},
        "in_sample": {k: out["best"][k] for k in ("f1", "boundary_band_f1", "score")},
        "weights_stage": out["stages"].get("weights"),
        "tried": out["tried"],
    }


def run_loco(clips: list[dict]) -> dict[str, Any]:
    # In-sample optimization is clip-independent of the fold, so do it once per clip.
    insample: dict[str, dict] = {}
    for c in clips:
        insample[c["clip"]] = _optimize_clip(c)

    out: dict[str, Any] = {}
    for held in clips:
        name = held["clip"]
        train = [c for c in clips if c is not held]
        # Per-field median of the training clips' calibrated weights.
        table = {
            f: round(float(np.median([insample[c["clip"]]["weights"][f] for c in train])), 4)
            for f in WEIGHT_FIELDS
        }
        own = held["res"].params
        preds_base = _predict_with(held, own)
        preds_loco = _predict_with(held, own.model_copy(update=table))
        preds_is = _predict_with(held, insample[name]["params"])
        gt_w = held["gt"]
        m_base, m_loco, m_is = _metrics(preds_base, gt_w), _metrics(preds_loco, gt_w), \
            _metrics(preds_is, gt_w)
        out[name] = {
            "train": [c["clip"] for c in train],
            "loco_table": table,
            "in_sample_weights": insample[name]["weights"],
            "weights_accepted_moves": (insample[name]["weights_stage"] or {}).get(
                "accepted_moves", 0),
            "base": m_base,
            "loco": m_loco,
            "in_sample": m_is,
            "delta": {k: round(m_loco[k] - m_base[k], 4)
                      for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")},
            "in_sample_delta": {k: round(m_is[k] - m_base[k], 4)
                                for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")},
        }

    macro: dict[str, Any] = {}
    for variant in ("base", "loco", "in_sample"):
        macro[variant] = {
            k: round(float(np.mean([out[c["clip"]][variant][k] for c in clips])), 4)
            for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")
        }
    macro["delta"] = {k: round(macro["loco"][k] - macro["base"][k], 4)
                      for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")}
    macro["in_sample_delta"] = {
        k: round(macro["in_sample"][k] - macro["base"][k], 4)
        for k in ("iou_0.5", "iou_0.3", "band_f1_1.0", "score")}
    worst = min(out[c["clip"]]["delta"]["iou_0.5"] for c in clips)
    macro["worst_iou_delta"] = worst
    macro["ship_new_defaults"] = bool(
        macro["delta"]["score"] > 0 and worst >= -AN.OBJECTIVE_IOU_FLOOR)
    # Candidate default table = median of the three clips' own calibrated weights;
    # only meaningful when the LOCO verdict passes.
    macro["pooled_table"] = {
        f: round(float(np.median([insample[c["clip"]]["weights"][f] for c in clips])), 4)
        for f in WEIGHT_FIELDS
    }
    return {"loco": out, "macro": macro,
            "insample_summary": {c["clip"]: {
                "baseline": insample[c["clip"]]["baseline"],
                "in_sample": insample[c["clip"]]["in_sample"],
                "accepted_moves": (insample[c["clip"]]["weights_stage"] or {}).get(
                    "accepted_moves", 0),
            } for c in clips}}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=list(ER.DEFAULT_CLIPS))
    ap.add_argument("--analysis-dir", type=Path, default=None,
                    help="directory of backfilled *.analysis.json files "
                         "(default: data/projects)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pairs = ER.discover_pairs(include_all=False, only=tuple(args.clips),
                              analysis_dir=args.analysis_dir)
    clips = [_load_clip(p) for p in pairs]
    missing = [c["clip"] for c in clips if not c["has_components"]]
    for c in clips:
        print(f"[{c['clip']}] components_full={'yes' if c['has_components'] else 'NO'} "
              f"window={c['window'][0]:.1f}..{c['window'][1]:.1f} gt={len(c['gt'])}")
    if missing:
        print(f"\nERROR: {missing} lack component_*_full (pre-P4-1 analyses). "
              "Re-run the full analysis once to backfill the component curves, "
              "then rerun this script.", file=sys.stderr)
        return 2
    if len(clips) < 2:
        print("need at least 2 paired clips for LOCO", file=sys.stderr)
        return 1

    result = run_loco(clips)

    print("\n=== in-sample optimizer (per clip, no holdout) ===")
    for name, d in result["insample_summary"].items():
        b, s = d["baseline"], d["in_sample"]
        print(f"{name:>12} score {b['score']:.3f}->{s['score']:.3f} "
              f"iou.5 {b['f1']:.3f}->{s['f1']:.3f} weight_moves={d['accepted_moves']}")
    print("\n=== LOCO held-out effect (default weights -> LOCO table) ===")
    print(f"{'clip':>12} {'IoU.5':>13} {'IoU.3':>13} {'band1.0':>13} {'score':>13}")
    for name, d in result["loco"].items():
        b, n = d["base"], d["loco"]
        print(f"{name:>12} {b['iou_0.5']:.3f}->{n['iou_0.5']:.3f}  "
              f"{b['iou_0.3']:.3f}->{n['iou_0.3']:.3f}  "
              f"{b['band_f1_1.0']:.3f}->{n['band_f1_1.0']:.3f}  "
              f"{b['score']:.3f}->{n['score']:.3f}")
        print(f"{'':>12} loco_table={d['loco_table']}")
    m = result["macro"]
    print(f"{'MACRO LOCO':>12} {m['base']['iou_0.5']:.3f}->{m['loco']['iou_0.5']:.3f}  "
          f"{m['base']['iou_0.3']:.3f}->{m['loco']['iou_0.3']:.3f}  "
          f"{m['base']['band_f1_1.0']:.3f}->{m['loco']['band_f1_1.0']:.3f}  "
          f"{m['base']['score']:.3f}->{m['loco']['score']:.3f}")
    print(f"MACRO in-sample (overfit reference): score {m['base']['score']:.3f}->"
          f"{m['in_sample']['score']:.3f}")
    print(f"worst held-out IoU delta: {m['worst_iou_delta']:+.4f} | "
          f"ship new defaults: {m['ship_new_defaults']}")
    print(f"pooled candidate table: {m['pooled_table']}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = EVAL_DIR / f"weights_calib_{ts}{tag}.json"
    artifact = {
        "generated_at": ts,
        "script": "scripts/calibrate_weights.py",
        "factors": list(AN.WEIGHT_FACTORS),
        "knobs": {"min_gain": AN.WEIGHT_MIN_GAIN,
                  "iou_floor": AN.OBJECTIVE_IOU_FLOOR,
                  "max_passes": AN.WEIGHT_MAX_PASSES},
        "clips": [{"clip": c["clip"], "window": c["window"], "n_gt": len(c["gt"])}
                  for c in clips],
        "analysis_dir": str(args.analysis_dir) if args.analysis_dir else "data/projects",
        **result,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\nartifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
