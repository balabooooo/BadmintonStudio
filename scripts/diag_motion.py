"""诊断：帧差 vs 时序中值背景建模，确认运动信号是否可用。

在固定机位下，``|frame - 时序中值背景|`` 能消掉全局闪烁与传感器噪声，
只保留真正在动的物体（球员 / 羽毛球）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bms.config import FRAMES_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="start", type=float, default=100.0)
    ap.add_argument("--to", dest="end", type=float, default=220.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * src_fps))

    grays: list[np.ndarray] = []
    times: list[float] = []
    i = int(args.start * src_fps)
    end = int(args.end * src_fps)
    while i < end:
        ok = cap.grab()
        if not ok:
            break
        if (i - int(args.start * src_fps)) % step == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                s = args.width / w
                fr = cv2.resize(fr, (args.width, max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA)
                grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
                times.append(i / src_fps)
        i += 1
    cap.release()

    n = len(grays)
    if n < 5:
        print("帧数不足")
        return 1
    print(f"读取 {n} 帧 @ {n / (times[-1] - times[0]):.2f} fps, 尺寸 {grays[0].shape}")

    # 1) 无处理帧差
    raw_diff = np.array([np.abs(grays[i] - grays[i - 1]).mean() for i in range(1, n)])
    # 2) 亮度归一后帧差
    def norm(g):
        m, s = g.mean(), g.std() + 1e-6
        return (g - m) / s
    ng = [norm(g) for g in grays]
    ndiff = np.array([np.abs(ng[i] - ng[i - 1]).mean() for i in range(1, n)])
    bright = np.array([g.mean() for g in grays])

    # 3) 时序中值背景（前后各 K 帧的滑动中值）
    K = 15
    idx = np.arange(n)
    bg = np.empty_like(np.stack(grays))
    stack = np.stack(grays)
    for i in range(n):
        a, b = max(0, i - K), min(n, i + K + 1)
        bg[i] = np.median(stack[a:b], axis=0)
    fg = np.abs(stack - bg)
    fg_mask = (fg > 12.0).astype(np.float32)
    changed = fg_mask.reshape(n, -1).mean(axis=1)
    fg_energy = fg.reshape(n, -1).mean(axis=1)

    # 输出背景图看一下
    cv2.imwrite(str(FRAMES_DIR / "bg_median.png"), np.clip(bg[n // 2], 0, 255).astype(np.uint8))
    for k in range(5):
        j = int(n * (k + 1) / 6)
        cv2.imwrite(str(FRAMES_DIR / f"fg_mask_{k}.png"), (fg_mask[j] * 255).astype(np.uint8))

    t = np.array(times)
    fig, axes = plt.subplots(5, 1, figsize=(22, 14), sharex=True)
    axes[0].plot(t, bright, lw=0.6); axes[0].set_ylabel("brightness")
    axes[1].plot(t[1:], raw_diff, lw=0.6, color="#a52"); axes[1].set_ylabel("raw frame diff")
    axes[2].plot(t[1:], ndiff, lw=0.6, color="#26a"); axes[2].set_ylabel("normalized diff")
    axes[3].plot(t, changed, lw=0.7, color="#2a8"); axes[3].set_ylabel("changed px ratio (>12)")
    axes[4].plot(t, fg_energy, lw=0.7, color="#82a"); axes[4].set_ylabel("fg energy")
    axes[4].set_xlabel("seconds")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FRAMES_DIR / "motion_methods.png"
    fig.savefig(out, dpi=90)
    print("图表 ->", out)

    for name, arr in (("brightness", bright), ("raw", raw_diff), ("norm", ndiff),
                      ("changed", changed), ("fgenergy", fg_energy)):
        print(f"  {name:10s} p5={np.percentile(arr,5):.4f} p50={np.percentile(arr,50):.4f} "
              f"p95={np.percentile(arr,95):.4f} max={arr.max():.4f} 变异系数={arr.std()/max(arr.mean(),1e-9):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
