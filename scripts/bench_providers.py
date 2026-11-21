"""Benchmark every available ONNX Runtime execution provider and cache the numbers.

P3 of ``docs/mobile-and-device-ai-plan.md``: on a machine with several EPs the
analysis should use the fastest one, not the one that happens to be first in a
hard-coded list. This script measures each provider that the installed ORT
actually exposes, writes the results to ``<data>/cache/inference_bench.json``
(keyed by a machine/driver/model fingerprint), and prints the resulting order.

On a CPU-only ORT build only the CPU row exists -- that is a valid result: it is
the baseline every accelerated provider must beat.

Usage:
    python scripts/bench_providers.py --golden data/golden/golden_clip4_xxx
    python scripts/bench_providers.py --golden data/golden/... --pose --batch 8 --iters 10
    python scripts/bench_providers.py --golden data/golden/... --providers CUDAExecutionProvider,DmlExecutionProvider
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _frames(golden: Path, count: int) -> list:
    import cv2
    import numpy as np

    meta = json.loads((golden / "meta.json").read_text(encoding="utf-8"))
    out = []
    for i in range(min(count, int(meta["frame_count"]))):
        path = golden / "frames" / f"f_{i:04d}.jpg"
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def _measure(call, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[min(len(times) - 1, int(0.9 * (len(times) - 1)))]
    return p50, p90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="directory from scripts/make_golden.py")
    ap.add_argument("--detect-model", default="yolo26n.dynamic.onnx")
    ap.add_argument("--pose-model", default="yolo26n-pose.static.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--pose-imgsz", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pose", action="store_true", help="also benchmark the pose model")
    ap.add_argument("--providers", default="", help="subset of providers; default = all available")
    ap.add_argument("--no-write", action="store_true", help="print only, do not touch the cache")
    args = ap.parse_args()

    from bms.inference import benchmark as BENCH
    from bms.inference import ort_backend as ORT
    from bms.inference import registry

    golden = Path(args.golden)
    if not (golden / "meta.json").exists():
        raise SystemExit(f"not a golden sample: {golden}")
    frames = _frames(golden, args.frames)[:args.batch]
    if not frames:
        raise SystemExit("no frames decoded")

    available = registry.available_providers()
    wanted = [p.strip() for p in args.providers.split(",") if p.strip()] or available
    providers = [p for p in wanted if p in available]
    if not providers:
        raise SystemExit(f"none of {wanted} are available; runtime exposes {available}")
    print(f"machine: {BENCH.fingerprint()}")
    print(f"providers: {providers}  batch: {len(frames)}")

    rows = []
    for provider in providers:
        chain = [provider] if provider == registry.CPU_PROVIDER else [provider, registry.CPU_PROVIDER]
        det = ORT.OrtDetector(model=args.detect_model, imgsz=args.imgsz, providers=chain)
        p50, p90 = _measure(lambda: det.predict(frames), args.warmup, args.iters)
        per_frame = round(p50 / len(frames), 2)
        print(f"  detect {provider:32s} p50 {p50:8.2f} ms  p90 {p90:8.2f} ms  {per_frame:6.2f} ms/frame")
        rows.append(("detect", provider, p50, p90))
        if not args.no_write:
            BENCH.record(args.detect_model, provider, p50, p90, args.imgsz, len(frames))

        if args.pose:
            pose = ORT.OrtPoseEstimator(model=args.pose_model, imgsz=args.pose_imgsz, providers=chain)
            patches = frames
            p50p, p90p = _measure(lambda: pose.predict(patches), args.warmup, args.iters)
            print(f"  pose   {provider:32s} p50 {p50p:8.2f} ms  p90 {p90p:8.2f} ms"
                  f"  {round(p50p / len(patches), 2):6.2f} ms/patch")
            rows.append(("pose", provider, p50p, p90p))
            if not args.no_write:
                BENCH.record(args.pose_model, provider, p50p, p90p, args.pose_imgsz, len(patches))

    print("\nprovider order (players):", BENCH.preferred_order(
        registry.TASK_PLAYERS, args.detect_model))
    print("provider order (pose):   ", BENCH.preferred_order(
        registry.TASK_POSE, args.pose_model))
    if not args.no_write:
        print("cache:", BENCH.cache_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
