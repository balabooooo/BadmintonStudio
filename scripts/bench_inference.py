"""Benchmark the detection / pose model on this machine (P0 baseline).

These numbers are the reference that later ONNX / execution-provider work (P2/P3)
must beat without changing results: same video, same frames, same ``imgsz`` and
``batch``. Latency is reported per batch (p50/p90) and per frame, because the
analysis loop calls the model once per batch of sampled frames.

Usage:
    python scripts/bench_inference.py "D:\\videos\\clip4.mp4" --device auto
    python scripts/bench_inference.py "D:\\videos\\clip4.mp4" --device cpu --batch 16
    python scripts/bench_inference.py --model models\\yolo26n-pose.pt --imgsz 256 --batch 32

When ``--video`` is omitted the script synthesises flat frames, which still
measures the model but not realistic box counts; prefer a real clip.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.inference.capabilities import probe  # noqa: E402
from bms.inference.scheduler import resolve_torch_device  # noqa: E402


def _sample_frames(video: str, count: int) -> tuple[list[np.ndarray], str]:
    """Read ``count`` distinct frames spread over the clip (as the analysis loop sees them)."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, total // max(1, count)) if total else 15
    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frame decoded from: {video}")
    return frames, f"{len(frames)} frames, stride {stride}"


def _synthetic_frames(count: int, width: int = 960, height: int = 720) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(count)]


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no numpy dependency so the numbers stay comparable)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def _measure(call, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        call()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return {
        "p50_ms": round(_percentile(times, 50) * 1000, 2),
        "p90_ms": round(_percentile(times, 90) * 1000, 2),
        "mean_ms": round(statistics.fmean(times) * 1000, 2),
        "iters": iters,
    }


def _bench_torch(
    model_path: str, frames: list[np.ndarray], batch: int, imgsz: int,
    device: str, warmup: int, iters: int,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    dev = resolve_torch_device(device)
    batches = [frames[i:i + batch] for i in range(0, len(frames), batch)]

    def call() -> None:
        for chunk in batches:
            model.predict(chunk, conf=0.10, imgsz=imgsz, device=dev, verbose=False)

    out = _measure(call, warmup, iters)
    out.update({
        "backend": "torch",
        "device": dev,
        "batches": len(batches),
        "per_frame_ms": round(out["p50_ms"] / max(1, len(frames)), 2),
    })
    return out


def _machine_info() -> dict:
    rep = probe()
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "capabilities": rep.as_dict(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001 - diagnostics only
        info["torch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - optional runtime
        info["onnxruntime"] = None
        info["ort_providers"] = []
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="", help="source clip (prefer a real match clip)")
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo26n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16, help="distinct frames fed to the model")
    ap.add_argument("--device", default="cuda", help="cuda | cpu | auto (auto == cuda)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "bench_inference.json"))
    args = ap.parse_args()

    if args.video:
        frames, source = _sample_frames(args.video, args.frames)
    else:
        frames = _synthetic_frames(args.frames)
        source = "synthetic"
    print(f"frames: {source}")

    device = "cuda" if args.device == "auto" else args.device
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": _machine_info(),
        "video": args.video or "(synthetic)",
        "source": source,
        "model": Path(args.model).name,
        "model_size": Path(args.model).stat().st_size if Path(args.model).exists() else 0,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "results": {
            "torch": _bench_torch(
                args.model, frames, args.batch, args.imgsz, device, args.warmup, args.iters,
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
