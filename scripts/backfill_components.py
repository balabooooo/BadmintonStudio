"""Backfill ``component_*_full`` curves into analysis JSON copies (P4-2 support).

The P4-1 fusion-weight optimizer needs the five full-rate component curves
(``component_{players,motion,audio_hits,shuttle,roi}_full``). Analyses written
before P4-1 do not store them, and the curves cannot be faithfully rebuilt
offline (optical-flow court motion and derived player tracks were never
stored), so a full re-analysis is required.

This script re-runs the analysis for the annotated eval clips using each
project's **stored parameters**, but writes the new AnalysisResult JSONs to a
separate directory (default ``data/cache/eval/backfill``) instead of touching
the projects or the frozen baseline analyses. Point ``calibrate_weights.py``
at that directory with ``--analysis-dir``.

Proxy/audio artifacts under ``data/cache`` are reused when present; pose
npz caches are reused by tag as usual.

Usage::

    .venv\\Scripts\\python.exe scripts\\backfill_components.py
    .venv\\Scripts\\python.exe scripts\\backfill_components.py --clips 0919_clip1
    .venv\\Scripts\\python.exe scripts\\backfill_components.py --clips 0919_clip2 --use-shuttle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from bms.analysis.pipeline import run_analysis  # noqa: E402
from bms.config import ensure_dirs  # noqa: E402
from bms.core import store  # noqa: E402
from bms.core.models import AnalysisResult  # noqa: E402
from bms.logging_setup import setup_logging  # noqa: E402
import eval_rallies as ER  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "cache" / "eval" / "backfill"
DEFAULT_OUT_SHUTTLE = ROOT / "data" / "cache" / "eval" / "backfill_shuttle"

REQUIRED_FULL = tuple(f"component_{k}_full" for k in
                      ("players", "motion", "audio_hits", "shuttle", "roi"))


def _ids_from(analysis_path: Path) -> tuple[str, str]:
    # "<pid>.<mid>.analysis.json"
    stem = analysis_path.name[: -len(".analysis.json")]
    pid, mid = stem.split(".", 1)
    return pid, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=list(ER.DEFAULT_CLIPS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--weights-key", default="balanced")
    ap.add_argument("--use-shuttle", action="store_true",
                    help="enable shuttle trajectory tracking for the re-run (slow, GPU-heavy)")
    ap.add_argument("--shuttle-budget", type=float, default=0.0,
                    help="shuttle time budget in seconds; 0 = analyze the whole clip (coverage=1.0)")
    args = ap.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT_SHUTTLE if args.use_shuttle else DEFAULT_OUT

    ensure_dirs()
    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = ER.discover_pairs(include_all=False, only=tuple(args.clips))
    if not pairs:
        print("no paired annotation/analysis clips found", file=sys.stderr)
        return 1

    failures = 0
    for pair in pairs:
        clip = pair["clip"]
        analysis_path = pair["analysis"]
        pid, mid = _ids_from(analysis_path)
        proj = store.load_project(pid)
        if proj is None:
            print(f"[{clip}] project {pid} not found", file=sys.stderr)
            failures += 1
            continue
        media = next((m for m in proj.media if m.id == mid), None)
        if media is None:
            print(f"[{clip}] media {mid} not found in project", file=sys.stderr)
            failures += 1
            continue
        old = AnalysisResult.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        dest = args.out / analysis_path.name
        if dest.exists():
            print(f"[{clip}] {dest.name} already exists, skipping")
            continue

        print(f"\n=== [{clip}] {media.name} ({media.duration:.0f}s) -> {dest} ===",
              flush=True)
        last = [""]

        def on(p: float, stage: str, msg: str = "", _c=clip) -> None:
            line = f"  [{_c} {p * 100:5.1f}%] {stage:10s} {msg}"
            if line != last[0]:
                print(line, flush=True)
                last[0] = line

        t0 = time.time()
        run_params = old.params
        if args.use_shuttle:
            run_params = run_params.model_copy(update={
                "use_shuttle": True,
                "shuttle_budget_seconds": float(args.shuttle_budget),
            })
        res = run_analysis(media, run_params, on_progress=on,
                           weights_key=args.weights_key)
        dt = time.time() - t0
        if res.status != "done":
            print(f"[{clip}] analysis status={res.status} error={res.error}",
                  file=sys.stderr)
            failures += 1
            continue
        sig = res.signals or {}
        missing = [k for k in REQUIRED_FULL if not sig.get(k)]
        if missing:
            print(f"[{clip}] missing backfilled keys: {missing}", file=sys.stderr)
            failures += 1
            continue
        dest.write_text(res.model_dump_json(indent=1), encoding="utf-8")
        old_n = len(old.rallies)
        sh_note = ""
        if args.use_shuttle:
            trc = res.stats.get("shuttle_trace") or {}
            sh_note = f", shuttle_trace={trc}"
        print(f"[{clip}] done in {dt:.0f}s, rallies {old_n} -> {len(res.rallies)}, "
              f"wrote {dest.name}{sh_note}")

    print(f"\nbackfill finished, failures={failures}, out={args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
