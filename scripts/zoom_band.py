"""把一段视频的中间横带裁出来放大拼图，方便肉眼核对球员位置与动作。

用法:
    python scripts/zoom_band.py <video> --times 0,10,20 --y0 0.33 --y1 0.66 --scale 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bms.core import ffmpeg as ff  # noqa: E402
from bms.core.media import probe_media  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--times", required=True)
    ap.add_argument("--y0", type=float, default=0.33)
    ap.add_argument("--y1", type=float, default=0.68)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--x1", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=2.4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    times = [float(x) for x in args.times.split(",")]
    out = Path(args.out) if args.out else ROOT / "data" / "cache" / "frames" / "zoom_band.jpg"
    tmp = out.parent / "_zoomtmp"
    tmp.mkdir(parents=True, exist_ok=True)

    tiles = []
    for i, t in enumerate(times):
        f = tmp / f"{i:03d}.png"
        r = ff.run([ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
                    "-ss", f"{t:.3f}", "-i", args.video, "-frames:v", "1", str(f)])
        if not r.ok or not f.is_file():
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        crop = img[int(h * args.y0):int(h * args.y1), int(w * args.x0):int(w * args.x1)]
        crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LANCZOS4)
        # 左上角标时间
        cv2.putText(crop, f"t={t:.0f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        sep = np.full((6, crop.shape[1], 3), 30, dtype=np.uint8)
        tiles.append(crop)
        tiles.append(sep)

    if not tiles:
        print("没有可用帧")
        return 1
    maxw = max(t.shape[1] for t in tiles)
    canvas = np.vstack([np.pad(t, ((0, 0), (0, maxw - t.shape[1]), (0, 0))) for t in tiles])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, canvas.shape)
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
