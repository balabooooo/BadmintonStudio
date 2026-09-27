"""Reconnaissance script: extract keyframes from a test video and output a contact sheet,
making it easier to confirm the camera angle and court by hand / multimodally.

Usage:
    python scripts/probe_video.py "<video path>" [--from 0] [--span 60] [--n 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.config import ensure_dirs, FRAMES_DIR  # noqa: E402
from bms.core import media as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--span", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ensure_dirs()
    info = M.probe_media(args.video)
    print("=== 媒体信息 ===")
    for k, v in info.model_dump().items():
        if k in ("path", "proxy_path", "audio_path", "poster"):
            continue
        print(f"  {k:16s} {v}")
    print(f"  size             {info.size / 1e9:.2f} GB")
    print(f"  aspect           {info.aspect:.3f}")
    print(f"  预估总帧数       {info.duration * info.fps:,.0f}")

    dur = info.duration
    start = max(0.0, min(args.start, max(0.0, dur - 1)))
    span = min(args.span, dur - start)
    times = [start + span * i / max(1, args.n - 1) for i in range(args.n)]

    out = Path(args.out) if args.out else (FRAMES_DIR / f"{Path(args.video).stem}_contact.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    p = M.extract_thumb_grid(info, times, out, thumb_h=270)
    print(f"\n联络图 -> {p}")
    for t in times:
        f = M.extract_frame(info, t, FRAMES_DIR / f"probe_{int(t):06d}.jpg", max_edge=1280)
        if f:
            print(f"  帧 t={t:8.2f}s -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
