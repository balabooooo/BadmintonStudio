"""Multi-clip rally segmentation evaluation bench (offline, no AI rerun).

Why a new script when ``eval_segmentation.py`` exists
-----------------------------------------------------
The old script takes a single ``--json`` + the first annotation file it finds,
and reports only IoU P/R/F1. Tuning against three annotated clips needs:

* automatic annotation/analysis pairing by media id;
* one scoring window per clip (saved ``focus`` intersected with the annotation
  bounding box — un-annotated regions are *unscored*, exactly like
  :func:`bms.analysis.annotation.optimize`);
* boundary-localization metrics (start/end error quantiles + tolerance bands),
  count deviation and an over-split / under-split / spurious / missed breakdown;
* macro/micro aggregation across clips and a timestamped JSON artifact, so a
  later run can be diffed against the frozen baseline.

Pairing
-------
Annotation files are named ``<clip>_m_<mediahash>_<res>.anno.json`` and
analysis files ``<project>.m_<mediahash>.analysis.json``; the ``m_<12 hex>``
token is the stable key. Missing pairs are skipped with a warning (e.g. a clip
whose analysis JSON was never written).

Usage::

    .venv\\Scripts\\python.exe scripts\\eval_rallies.py
    .venv\\Scripts\\python.exe scripts\\eval_rallies.py --all --tag p0-baseline
    .venv\\Scripts\\python.exe scripts\\eval_rallies.py --mode stored
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import pipeline as P  # noqa: E402
from bms.analysis import annotation as AN  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402

ANNO_DIR = ROOT / "data" / "annotations"
PROJECT_DIR = ROOT / "data" / "projects"
EVAL_DIR = ROOT / "data" / "cache" / "eval"

_MEDIA_RE = re.compile(r"(m_[0-9a-f]{12})")
_IOU_THRESHOLDS = (0.3, 0.5)
_BANDS = (0.5, 1.0, 1.5)
#: Default validation set (user decision: clip1-3 only; LOCO).
DEFAULT_CLIPS = ("clip1", "clip2", "clip3")


# ------------------------------------------------------------------ pairing


def _media_token(name: str) -> str | None:
    m = _MEDIA_RE.search(name)
    return m.group(1) if m else None


def _clip_label(anno_name: str) -> str:
    """``0919_clip1_m_..._960x540`` -> ``0919_clip1`` (everything before the media token)."""
    stem = Path(anno_name).stem
    tok = _media_token(stem)
    if tok:
        stem = stem.split(tok)[0].rstrip("_")
    return stem


def discover_pairs(include_all: bool, only: set[str] | None,
                   analysis_dir: Path | None = None) -> list[dict[str, Path]]:
    annos = {p: _media_token(p.name) for p in sorted(ANNO_DIR.glob("*.anno.json"))}
    annos = {p: tok for p, tok in annos.items() if tok}
    analyses = {}
    for p in sorted((analysis_dir or PROJECT_DIR).glob("*.analysis.json")):
        tok = _media_token(p.name)
        if tok:
            analyses.setdefault(tok, p)
    pairs: list[dict[str, Path]] = []
    for anno_path, tok in annos.items():
        label = _clip_label(anno_path.name)
        short = label.split("_")[-1]
        if only is not None:
            wanted = label in only or short in only or f"clip{short[-1:]}" in only
        elif include_all:
            wanted = True
        else:
            wanted = any(c in label for c in DEFAULT_CLIPS)
        if not wanted:
            continue
        analysis_path = analyses.get(tok)
        if analysis_path is None:
            print(f"[skip] {label}: no analysis JSON for {tok}", file=sys.stderr)
            continue
        pairs.append({"clip": label, "token": tok,
                      "annotation": anno_path, "analysis": analysis_path})
    pairs.sort(key=lambda d: d["clip"])
    return pairs


# ------------------------------------------------------------------ metrics


def _clip_window(doc: dict, gt: list[tuple[float, float]], duration: float) -> tuple[float, float]:
    """Evaluation window = saved focus intersected with the annotation bounding box."""
    bbox_lo = min(a for a, _ in gt)
    bbox_hi = max(b for _, b in gt)
    focus = doc.get("focus")
    if focus and len(focus) == 2:
        lo = max(bbox_lo, float(focus[0]))
        hi = min(bbox_hi, float(focus[1]))
        if hi <= lo:  # degenerate focus: fall back to the bbox
            lo, hi = bbox_lo, bbox_hi
    else:
        lo, hi = bbox_lo, bbox_hi
    return max(0.0, lo), min(duration if duration > 0 else hi, hi)


def _in_window(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(a, b) for a, b in intervals if b > lo and a < hi]


def _classify_errors(preds: list[tuple[float, float]],
                     gt: list[tuple[float, float]],
                     pairs: list[tuple[int, int, float]]) -> dict[str, int]:
    """Label unmatched predictions / ground truth with an error type.

    * over_split: one GT rally is covered by >=2 overlapping predictions;
    * under_split: one prediction overlaps >=2 GT rallies;
    * shifted_fp / shifted_fn: one-to-one overlap exists but IoU@0.5 was not met;
    * spurious / missed: no temporal evidence on the other side.
    """
    used_p = {i for i, _, _ in pairs}
    used_g = {j for _, j, _ in pairs}

    def overlaps(idx: int, against: list[tuple[float, float]]) -> int:
        a, b = preds[idx]
        return sum(1 for c, d in against if AN.iou((a, b), (c, d)) >= 0.3)

    counts = {"over_split": 0, "under_split": 0, "spurious": 0, "missed": 0,
              "shifted_fp": 0, "shifted_fn": 0}
    for j, g in enumerate(gt):
        if j in used_g:
            continue
        n = sum(1 for p in preds if AN.iou(p, g) >= 0.3)
        if n >= 2:
            counts["over_split"] += 1
        elif n == 1:
            counts["shifted_fn"] += 1
        else:
            counts["missed"] += 1
    for i, p in enumerate(preds):
        if i in used_p:
            continue
        n = sum(1 for g in gt if AN.iou(p, g) >= 0.3)
        if n >= 2:
            counts["under_split"] += 1
        elif n == 1:
            counts["shifted_fp"] += 1
        else:
            counts["spurious"] += 1
    return counts


def evaluate(preds: list[tuple[float, float]], gt: list[tuple[float, float]]) -> dict:
    """Full metric bundle for one clip / one prediction variant."""
    out: dict[str, object] = {"n_pred": len(preds), "n_gt": len(gt),
                              "count_diff": len(preds) - len(gt)}
    for thr in _IOU_THRESHOLDS:
        out[f"iou_{thr}"] = AN.metrics(preds, gt, thr)
    # Boundary localization: shared helper pairs loosely (IoU>=0.3) so nearly-correct
    # pairs contribute boundary statistics even when they fail the strict IoU test,
    # and adds per-band boundary-point F1 (unmatched intervals count as FP/FN).
    bm = AN.boundary_metrics(preds, gt, match_thr=0.3, bands=_BANDS)
    boundary = {}
    for name in ("start", "end"):
        side = dict(bm[name])
        side.pop("errors", None)  # transient; pooled by the caller via the underscore keys
        boundary[name] = side
    out["boundary"] = boundary
    # Raw signed errors kept only in memory long enough for cross-clip pooling; stripped before
    # the artifact is written.
    out["_start_errors"] = bm["start"]["errors"]
    out["_end_errors"] = bm["end"]["errors"]
    strict_pairs = AN.match_pairs(preds, gt, 0.5)
    out["errors"] = _classify_errors(preds, gt, strict_pairs)
    return out


# ------------------------------------------------------------------ aggregation


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def aggregate(per_clip: list[dict], variants: list[str]) -> dict:
    agg: dict[str, object] = {}
    for variant in variants:
        macro: dict[str, object] = {}
        micro: dict[str, object] = {}
        all_start: list[float] = []
        all_end: list[float] = []
        err_sum = {"over_split": 0, "under_split": 0, "spurious": 0,
                   "missed": 0, "shifted_fp": 0, "shifted_fn": 0}
        for thr in _IOU_THRESHOLDS:
            precs, recs, f1s = [], [], []
            tp_s = fp_s = fn_s = 0
            for clip in per_clip:
                m = clip[variant]
                mm = m[f"iou_{thr}"]
                precs.append(mm["precision"])
                recs.append(mm["recall"])
                f1s.append(mm["f1"])
                tp_s += mm["tp"]
                fp_s += mm["fp"]
                fn_s += mm["fn"]
            macro[f"iou_{thr}"] = {
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "f1": round(float(np.mean(f1s)), 4),
            }
            p, r, f1 = _prf(tp_s, fp_s, fn_s)
            micro[f"iou_{thr}"] = {"tp": tp_s, "fp": fp_s, "fn": fn_s,
                                   "precision": p, "recall": r, "f1": f1}
        band_pool = {name: {b: [0, 0, 0] for b in _BANDS}
                     for name in ("start", "end")}
        for clip in per_clip:
            # Pool from the stored per-pair error lists (quantiles cannot be reconstructed from
            # quantiles) and the per-clip band TP/FP/FN counts.
            all_start.extend(clip[variant].get("_start_errors", []))
            all_end.extend(clip[variant].get("_end_errors", []))
            for name in ("start", "end"):
                for b in _BANDS:
                    bf = clip[variant]["boundary"][name]["bands_f1"][str(b)]
                    pp = band_pool[name][b]
                    pp[0] += bf["tp"]; pp[1] += bf["fp"]; pp[2] += bf["fn"]
            for k in err_sum:
                err_sum[k] += clip[variant]["errors"][k]
        pooled_b = {}
        for name, errs in (("start", all_start), ("end", all_end)):
            q = AN.error_quantiles(errs)
            abs_errs = [abs(e) for e in errs]
            q["bands"] = {str(band): (round(sum(1 for e in abs_errs if e <= band)
                                            / len(abs_errs), 3) if abs_errs else None)
                          for band in _BANDS}
            q["matched"] = len(errs)
            q["bands_f1"] = {}
            for b in _BANDS:
                tp, fp, fn = band_pool[name][b]
                f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
                q["bands_f1"][str(b)] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4)}
            pooled_b[name] = q
        agg[variant] = {"macro": macro, "micro": micro,
                        "boundary_pooled": pooled_b, "errors_sum": err_sum}
    return agg


# ------------------------------------------------------------------ per-clip run


def _params_snapshot(params) -> dict:
    keys = ("segment_mode", "seg_prominence", "seg_min_core", "seg_min_rest",
            "seg_min_quiet", "min_rally_seconds", "pre_roll", "post_roll",
            "hit_tail_seconds", "use_audio", "use_pose", "use_shuttle",
            "pose_gate_threshold", "hit_sensitivity")
    return {k: getattr(params, k) for k in keys if hasattr(params, k)}


def evaluate_pair(pair: dict[str, Path], modes: list[str]) -> dict:
    doc = json.loads(pair["annotation"].read_text(encoding="utf-8"))
    raw = json.loads(pair["analysis"].read_text(encoding="utf-8"))
    res = AnalysisResult.model_validate(raw)
    sig = res.signals or {}
    duration = float((sig.get("duration") or [0.0])[0]) or float(doc.get("duration", 0.0))

    gt = AN.normalize_rallies(doc.get("rallies") or [])
    gt = [(float(r["start"]), float(r["end"])) for r in gt]
    lo, hi = _clip_window(doc, gt, duration)
    gt_w = _in_window(gt, lo, hi)

    clip_out: dict[str, object] = {
        "clip": pair["clip"],
        "token": pair["token"],
        "annotation": pair["annotation"].name,
        "analysis": pair["analysis"].name,
        "duration": round(duration, 2),
        "window": [round(lo, 2), round(hi, 2)],
        "window_ratio": round((hi - lo) / duration, 3) if duration > 0 else None,
        "gt_count": len(gt_w),
        "params": _params_snapshot(res.params),
        "seg_method": (res.stats or {}).get("segmentation", {}).get("method"),
    }

    variants: dict[str, list[tuple[float, float]]] = {}
    if "stored" in modes:
        variants["stored"] = _in_window([(r.start, r.end) for r in res.rallies], lo, hi)
    if "resegment" in modes:
        new_res = P.resegment(copy.deepcopy(res), res.params, "balanced")
        variants["resegment"] = _in_window(
            [(r.start, r.end) for r in new_res.rallies], lo, hi)

    for name, preds in variants.items():
        # evaluate() attaches raw signed _start_errors/_end_errors for pooling; the artifact
        # writer strips those transient keys before writing JSON.
        clip_out[name] = evaluate(preds, gt_w)
    return clip_out


# ------------------------------------------------------------------ reporting


def _fmt_q(q: dict) -> str:
    if q.get("median") is None:
        return "  -  /  -  "
    return f"{q['median']:5.2f}/{q['p90']:5.2f}"


def print_report(clips: list[dict], modes: list[str], aggregate_doc: dict) -> None:
    for clip in clips:
        print(f"\n=== {clip['clip']}  window [{clip['window'][0]:.1f}, "
              f"{clip['window'][1]:.1f}]s  GT={clip['gt_count']}  "
              f"method={clip.get('seg_method')} ===")
        for name in modes:
            m = clip[name]
            m5, m3 = m["iou_0.5"], m["iou_0.3"]
            bs, be = m["boundary"]["start"], m["boundary"]["end"]
            print(f"  {name:9s} n={m['n_pred']:3d} dN={m['count_diff']:+d} | "
                  f"F1@.5={m5['f1']:.3f} (P={m5['precision']:.3f} R={m5['recall']:.3f} "
                  f"TP={m5['tp']} FP={m5['fp']} FN={m5['fn']}) | F1@.3={m3['f1']:.3f}")
            print(f"            start err med/p90={_fmt_q(bs)}s  "
                  f"end err med/p90={_fmt_q(be)}s  "
                  f"errors={m['errors']}")
    print("\n=== aggregate ===")
    for name in modes:
        a = aggregate_doc[name]
        for thr in _IOU_THRESHOLDS:
            ma, mi = a["macro"][f"iou_{thr}"], a["micro"][f"iou_{thr}"]
            print(f"  {name:9s} IoU>={thr:.1f}: macro F1={ma['f1']:.3f} "
                  f"(P={ma['precision']:.3f} R={ma['recall']:.3f}) | "
                  f"micro F1={mi['f1']:.3f} (TP={mi['tp']} FP={mi['fp']} FN={mi['fn']})")
        bs, be = a["boundary_pooled"]["start"], a["boundary_pooled"]["end"]
        print(f"  {name:9s} pooled start med/p90={_fmt_q(bs)}s  "
              f"end med/p90={_fmt_q(be)}s  errors={a['errors_sum']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include every paired clip (default: clip1-3 only)")
    ap.add_argument("--clip", action="append", default=None,
                    help="clip label to include (repeatable); e.g. --clip clip2")
    ap.add_argument("--mode", choices=("stored", "resegment", "both"), default="both")
    ap.add_argument("--tag", default="", help="optional tag appended to the artifact filename")
    ap.add_argument("--note", default="", help="free-text note stored in the artifact")
    args = ap.parse_args()

    only = set(args.clip) if args.clip else None
    modes = ["stored", "resegment"] if args.mode == "both" else [args.mode]
    pairs = discover_pairs(args.all, only)
    if not pairs:
        print("No paired annotation/analysis clips found.", file=sys.stderr)
        return 1

    clips = []
    for pair in pairs:
        try:
            clips.append(evaluate_pair(pair, modes))
        except Exception as e:  # noqa: BLE001
            print(f"[error] {pair['clip']}: {type(e).__name__}: {e}", file=sys.stderr)

    aggregate_doc = aggregate(clips, modes)
    print_report(clips, modes, aggregate_doc)

    # Strip the transient pooled error lists before writing the artifact.
    for clip in clips:
        for name in modes:
            clip[name].pop("_start_errors", None)
            clip[name].pop("_end_errors", None)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = EVAL_DIR / f"rally_eval_{ts}{suffix}.json"
    artifact = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "note": args.note,
        "window_rule": "saved focus intersected with annotation bounding box",
        "iou_thresholds": list(_IOU_THRESHOLDS),
        "boundary_bands": list(_BANDS),
        "clips": clips,
        "aggregate": aggregate_doc,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nArtifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
