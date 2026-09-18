"""快速生成一段低分辨率代理，用于算法实验。

用法:
    python scripts/make_test_proxy.py "<video>" --from 0 --to 300 --edge 480 --fps 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.config import PROXIES_DIR, ensure_dirs  # noqa: E402
from bms.core import ffmpeg as ff  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--to", dest="end", type=float, default=0.0)
    ap.add_argument("--edge", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ensure_dirs()
    span = (args.end - args.start) if args.end > args.start else 0
    tag = f"{int(args.start)}_{int(args.end)}_{args.edge}_{args.fps:g}"
    out = Path(args.out) if args.out else PROXIES_DIR / f"_test_{tag}.mp4"

    cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin"]
    if args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", args.video]
    if span > 0:
        cmd += ["-t", f"{span:.3f}"]
    cmd += [
        "-vf", f"scale={args.edge}:-2:flags=bilinear,fps={args.fps:g},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-an", "-loglevel", "error", str(out),
    ]
    t0 = time.time()
    r = ff.run(cmd)
    if not r.ok:
        print(r.stderr[-3000:])
        return 1
    print(f"{out}  ({time.time() - t0:.1f}s, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
