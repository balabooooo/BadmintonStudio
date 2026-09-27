"""A/B scoring evaluation: compare old (balanced) vs new (highlight_pro) presets on annotated rallies.

Uses the optimized segmentation to generate rallies, then scores them with two presets and dumps
a side-by-side comparison for user review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import annotation as AN  # noqa: E402
from bms.analysis import rally as RA  # noqa: E402
from bms.analysis import scoring as SC  # noqa: E402
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
    gt = sorted([(float(r["start"]), float(r["end"])) for r in anno["rallies"]])
    return res, anno, gt


def score_with_preset(intervals, feats, preset_name: str):
    weights = SC.PRESETS.get(preset_name, SC.PRESETS["balanced"])
    quality = [{"sharpness": 0.5, "shake": 0.5, "subject_size": 0.3} for _ in intervals]
    return SC.score_rallies(feats, weights, quality)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--preset-a", default="balanced")
    ap.add_argument("--preset-b", default="highlight_pro")
    args = ap.parse_args()

    report: dict = {}
    for name, ajson, njson in CLIPS:
        res, anno, gt = load_clip(ajson, njson)
        lo, hi = 0.0, max(b for _, b in gt) + 12.0
        ctx = AN._build_context(res, lo, hi)

        # Use optimized segmentation params
        opt = res.params.model_copy(update={
            "seg_prominence": 0.10, "seg_min_core": 2.5,
            "seg_min_rest": 0.6, "seg_min_quiet": 0.6,
            "min_rally_seconds": 2.0,
            "pose_gate_threshold": 0.45,
        })
        preds = AN._predict(ctx, opt)
        m = AN.metrics(preds, gt, 0.5)

        # Build intervals with features
        hits = AN._gated_hits(ctx, opt)
        sig = res.signals
        fused = ctx.fused
        motion_dict = {
            "motion": np.asarray(sig.get("motion", []), dtype=np.float32),
            "fps": float(sig.get("motion_fps", [12.0])[0]),
        }
        players_dict = {
            "active_count": np.asarray(sig.get("active_count", []), dtype=np.float32),
            "active_speed": np.asarray(sig.get("active_speed", []), dtype=np.float32),
            "fps": float(sig.get("player_fps", [12.0])[0]),
        }
        intervals = [RA.RallyInterval(start=a, end=b) for a, b in preds]
        for iv in intervals:
            idx = RA._hit_idx_in_window(hits, iv.start, iv.end) if hits is not None else []
            iv.hit_indices = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        RA.attach_features(intervals, fused, hits=hits, motion=motion_dict, players=players_dict)

        feats = [iv.features for iv in intervals]
        scores_a = score_with_preset(intervals, feats, args.preset_a)
        scores_b = score_with_preset(intervals, feats, args.preset_b)

        # Build comparison rows (sorted by B score descending)
        rows = []
        for i, iv in enumerate(intervals):
            rows.append({
                "start": round(iv.start, 1),
                "end": round(iv.end, 1),
                "duration": round(iv.features.get("duration", 0), 1),
                "shots": int(iv.features.get("shot_count", 0)),
                "smash_proxy": int(iv.features.get("smash_proxy", 0)),
                "confrontation_streak": int(iv.features.get("confrontation_streak", 1)),
                "tempo": round(iv.features.get("tempo", 0), 2),
                "finish_tempo": round(iv.features.get("finish_tempo", 0), 2),
                "preset_a": round(scores_a[i]["total"], 1),
                "preset_b": round(scores_b[i]["total"], 1),
                "tags_a": scores_a[i].get("tags", []),
                "tags_b": scores_b[i].get("tags", []),
            })
        rows.sort(key=lambda r: -r["preset_b"])

        print(f"\n===== {name}  GT={len(gt)}  F1={m['f1']:.3f} =====")
        print(f"{'start':>6s} {'end':>6s} {'dur':>4s} {'shots':>5s} {'smash':>5s} {'confr':>5s} "
              f"{'tempo':>5s} {'ft':>5s} {args.preset_a:>10s} {args.preset_b:>12s} tags")
        for r in rows:
            print(f"{r['start']:6.1f} {r['end']:6.1f} {r['duration']:4.1f} {r['shots']:5d} "
                  f"{r['smash_proxy']:5d} {r['confrontation_streak']:5d} "
                  f"{r['tempo']:5.2f} {r['finish_tempo']:5.2f} "
                  f"{r['preset_a']:10.1f} {r['preset_b']:12.1f} {','.join(r['tags_b'])}")

        report[name] = {
            "metrics": m,
            "rallies": rows,
        }

    if args.save:
        args.save.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
