"""One-time rebuild of per-frame player-box + keypoint-skeleton caches for annotated clips.

Why this script exists
----------------------
``PlayerSignal.frame_boxes`` was never persisted, and the pose npz cache stores only the derived
swing/ok/overhead curves (v1) — full 17-keypoint skeletons were dropped right after inference.
The annotation-page overlay (bbox + COCO skeleton) and the later boundary-template work both need
those artifacts without paying GPU cost on every app launch.

For each paired clip (default: clip1-3, the validation set) this script:

1. resolves the proxy video under ``data/cache/proxies`` via the ``m_<12hex>`` token;
2. computes the players cache tag from the **same inputs as the stored analysis**
   (``sample_fps`` capped at 12, effective ROI / polygon, viewpoint, size filter) and reuses
   ``data/cache/boxes/<tag>.npz`` when present — otherwise reruns ``analyze_players``;
3. resolves the pose npz for those boxes: reuses it when it already holds v2 keypoints,
   otherwise removes the stale v1 file (same key, no skeletons) and runs ``analyze_pose``;
4. patches the analysis JSON: ``stats.boxes_cache`` / ``stats.pose_cache`` references (references
   live with the other traces — ``signals`` is strictly numeric curves) plus pose curves in
   ``signals`` (same keys / rounding as ``pipeline._pack_signals``), atomically, keeping a
   one-time ``.visual_bak`` backup.

Idempotent: a second run skips both AI stages when the caches are fresh. Segmentation output is
not supposed to change — the three validation clips store zero hits, so adding pose curves cannot
affect hit gating; always re-run ``scripts/eval_rallies.py`` afterwards to confirm a zero diff.

Usage::

    .venv\\Scripts\\python.exe scripts\\rebuild_visual_cache.py
    .venv\\Scripts\\python.exe scripts\\rebuild_visual_cache.py --clip clip2
    .venv\\Scripts\\python.exe scripts\\rebuild_visual_cache.py --all --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.analysis import pipeline as P  # noqa: E402
from bms.analysis import players as PL  # noqa: E402
from bms.analysis import pose as POSE  # noqa: E402

ANNO_DIR = ROOT / "data" / "annotations"
PROJECT_DIR = ROOT / "data" / "projects"
PROXY_DIR = ROOT / "data" / "cache" / "proxies"

_MEDIA_RE = re.compile(r"(m_[0-9a-f]{12})")
DEFAULT_CLIPS = ("clip1", "clip2", "clip3")
BACKUP_SUFFIX = ".visual_bak"


# ------------------------------------------------------------------ pairing


def _media_token(name: str) -> str | None:
    m = _MEDIA_RE.search(name)
    return m.group(1) if m else None


def _clip_label(anno_name: str) -> str:
    stem = Path(anno_name).stem
    tok = _media_token(stem)
    if tok:
        stem = stem.split(tok)[0].rstrip("_")
    return stem


def discover_pairs(include_all: bool, only: set[str] | None) -> list[dict[str, Path]]:
    annos = {p: tok for p in sorted(ANNO_DIR.glob("*.anno.json"))
             if (tok := _media_token(p.name))}
    analyses = {}
    for p in sorted(PROJECT_DIR.glob("*.analysis.json")):
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
        proxies = sorted(PROXY_DIR.glob(f"*{tok}*.mp4"))
        if not proxies:
            print(f"[skip] {label}: no proxy video for {tok}", file=sys.stderr)
            continue
        pairs.append({"clip": label, "token": tok, "proxy": proxies[0],
                      "analysis": analysis_path})
    pairs.sort(key=lambda d: d["clip"])
    return pairs


# ------------------------------------------------------------------ per-clip rebuild


def _analysis_inputs(doc: dict) -> dict:
    """Reconstruct the analyze_players arguments from a stored analysis document."""
    params = doc.get("params") or {}
    stats = doc.get("stats") or {}
    calib = doc.get("calibration") if isinstance(doc.get("calibration"), dict) else None
    sample_fps = min(float(params.get("sample_fps") or 12.0), 12.0)
    raw_roi = stats.get("effective_roi")
    roi = tuple(float(v) for v in raw_roi) if raw_roi and len(raw_roi) == 4 else None
    poly = stats.get("effective_poly") or None
    size_filter = stats.get("player_size_filter") or None
    viewpoint = str((calib or {}).get("viewpoint") or "unknown")
    return {"sample_fps": sample_fps, "roi": roi, "roi_poly": poly,
            "size_filter": size_filter, "viewpoint": viewpoint}


def _progress_printer(label: str):
    last = [-1.0]

    def _report(p: float, stage: str) -> None:
        pct = int(round(max(0.0, min(1.0, float(p))) * 100))
        if pct >= last[0] + 5 or pct >= 100:
            last[0] = pct
            print(f"    {label}: {stage} {pct}%", flush=True)

    return _report


def _rebuild_clip(pair: dict[str, Path], device: str, make_backup: bool) -> dict:
    proxy = pair["proxy"]
    analysis_path = pair["analysis"]
    doc = json.loads(analysis_path.read_text(encoding="utf-8"))
    sig = doc.setdefault("signals", {})
    inputs = _analysis_inputs(doc)

    # ---- 1/2 player boxes: fresh npz cache or rerun detection ------------------------------
    tag = PL.boxes_cache_tag(str(proxy), **inputs)
    boxes_doc = PL.load_boxes_cache(tag)
    if boxes_doc is not None:
        frame_boxes = boxes_doc["frame_boxes"]
        fps = float(boxes_doc["fps"])
        btag = str(boxes_doc.get("tag") or tag)
        print(f"  boxes: cache hit ({btag}), {sum(len(f) for f in frame_boxes)} observations")
    else:
        print(f"  boxes: running player detection on {proxy.name} (device={device})")
        psig = PL.analyze_players(
            str(proxy),
            device=device,
            on_progress=_progress_printer("players"),
            **inputs,
        )
        frame_boxes = list(getattr(psig, "frame_boxes", []) or [])
        fps = float(getattr(psig, "fps", inputs["sample_fps"]) or inputs["sample_fps"])
        btag = str(getattr(psig, "cache_tag", "") or "")
        if not btag:
            raise RuntimeError("player box cache was not written (check data/cache/boxes)")
        print(f"  boxes: cached as {btag}, {sum(len(f) for f in frame_boxes)} observations")

    # ---- 3 pose: reuse v2 keypoints cache, otherwise rerun on the same boxes ----------------
    pose_dir = POSE.default_cache_dir()
    pose_path = POSE.pose_cache_path(pose_dir, str(proxy), fps, frame_boxes)
    pose_sig = None
    if POSE.load_keypoints(pose_path) is not None:
        pose_sig = POSE._load_cache(pose_path, len(frame_boxes))
        print(f"  pose: keypoint cache hit ({pose_path.stem})")
    else:
        if pose_path.exists():
            # Same cache key but an older-layout file (e.g. v1 curves only, or v2 with the buggy
            # crop->frame mapping): it must be replaced, otherwise analyze_pose treats the curve
            # cache as a hit and the corrected skeleton never gets written.
            print(f"  pose: replacing stale cache ({pose_path.stem}) with v{POSE.POSE_CACHE_VERSION}")
            pose_path.unlink()
        print(f"  pose: running pose estimation (device={device})")
        pose_sig = POSE.analyze_pose(
            str(proxy), frame_boxes, fps,
            device=device, cache_dir=pose_dir,
            on_progress=_progress_printer("pose"),
        )
        if pose_sig is None:
            print("  pose: no usable keypoints on this clip; skipping pose patch")
        else:
            pose_sig.trace["cache_tag"] = pose_path.stem
            print(f"  pose: cached as {pose_path.stem}, coverage={pose_sig.coverage:.3f}")

    # ---- 4 patch analysis JSON (mirror pipeline._pack_signals + run_analysis stats) ---------
    stats = doc.setdefault("stats", {})
    stats["boxes_cache"] = btag
    # Migrate pre-stats-layout tags written by an earlier revision of this script.
    sig.pop("boxes_cache", None)
    sig.pop("pose_cache", None)
    if pose_sig is not None:
        sig["pose_swing"] = P._downsample(pose_sig.swing)
        sig["pose_ok"] = P._downsample(pose_sig.ok)
        sig["pose_overhead"] = P._downsample(pose_sig.overhead)
        sig["pose_fps"] = [round(float(pose_sig.fps), 3)]
        sig["pose_coverage"] = [round(float(pose_sig.coverage), 4)]
        sig["pose_swing_full"] = [round(float(v), 4) for v in pose_sig.swing]
        sig["pose_quiet"] = [round(float(pose_sig.quiet), 5)]
        stats["pose_cache"] = pose_path.stem

    if make_backup:
        bak = analysis_path.with_suffix(analysis_path.suffix + BACKUP_SUFFIX)
        if not bak.exists():
            shutil.copy2(analysis_path, bak)
            print(f"  backup: {bak.name}")
    tmp = analysis_path.with_suffix(analysis_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, analysis_path)

    return {
        "clip": pair["clip"],
        "boxes_cache": btag,
        "boxes_observations": sum(len(f) for f in frame_boxes),
        "frames": len(frame_boxes),
        "pose_cache": pose_path.stem if pose_sig is not None else None,
        "pose_coverage": round(float(pose_sig.coverage), 4) if pose_sig is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include every paired clip (default: clip1-3 only)")
    ap.add_argument("--clip", action="append", default=None,
                    help="clip label to include (repeatable); e.g. --clip clip2")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                    help="compute device for the AI stages (auto-falls back to cpu)")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not keep the one-time .visual_bak copy of each analysis JSON")
    args = ap.parse_args()

    only = set(args.clip) if args.clip else None
    pairs = discover_pairs(args.all, only)
    if not pairs:
        print("No paired annotation/analysis/proxy clips found.", file=sys.stderr)
        return 1

    results = []
    started = time.time()
    for pair in pairs:
        print(f"\n=== {pair['clip']} ({pair['token']}) ===")
        try:
            results.append(_rebuild_clip(pair, args.device, make_backup=not args.no_backup))
        except Exception as e:  # noqa: BLE0000 - keep going on the other clips
            print(f"[error] {pair['clip']}: {type(e).__name__}: {e}", file=sys.stderr)

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['clip']}: frames={r['frames']} boxes_rows={r['boxes_observations']} "
              f"pose={r['pose_cache']} coverage={r['pose_coverage']}")
    print(f"elapsed {time.time() - started:.1f}s")
    print("\nNext: run scripts/eval_rallies.py and confirm metrics are unchanged vs the P0 baseline.")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
