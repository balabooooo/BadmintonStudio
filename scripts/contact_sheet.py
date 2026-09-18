"""把若干时刻的画面拼成一张带时间戳的「接触印相」，用来肉眼核对回合边界。

为什么要它：判断「这一秒球落地了没有 / 还在不在打」只能看图，而一张一张读
太慢。把十几帧拼成一张网格，一次就能看完一段边界。

跑法::

    .venv\\Scripts\\python.exe scripts\\contact_sheet.py --times 46 48 49 50 --out x.jpg
    .venv\\Scripts\\python.exe scripts\\contact_sheet.py --range 44 72 2 --out y.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROXY = (ROOT / "data" / "cache" / "proxies"
         / "20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", type=float, nargs="*", default=[])
    ap.add_argument("--range", type=float, nargs=3, default=None,
                    metavar=("START", "END", "STEP"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--width", type=int, default=460)
    args = ap.parse_args()

    times = list(args.times)
    if args.range:
        a, b, s = args.range
        t = a
        while t <= b + 1e-6:
            times.append(round(t, 2))
            t += s
    if not times:
        print("没有给时刻")
        return 1

    cap = cv2.VideoCapture(str(PROXY))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for t in sorted(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src_fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        sc = args.width / w
        fr = cv2.resize(fr, (args.width, max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        # 时间戳画在左上角，带黑底描边，缩略图上也看得清
        label = f"{t:.1f}s"
        cv2.rectangle(fr, (0, 0), (110, 26), (0, 0, 0), -1)
        cv2.putText(fr, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()

    cols = max(1, args.cols)
    rows = int(np.ceil(len(tiles) / cols))
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 32, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / "data" / "cache" / "frames" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(tiles)} 帧 -> {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
