"""验证脚本：视觉运动分析 + 活动热区。

用法:
    python scripts/test_motion.py "<proxy or video>" [--fps 15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bms.analysis import motion as MO  # noqa: E402
from bms.config import FRAMES_DIR, ensure_dirs  # noqa: E402
from bms.core.media import probe_media  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ensure_dirs()
    info = probe_media(args.video)
    t0 = time.time()
    sig = MO.analyze_motion(info, sample_fps=args.fps, work_width=args.width,
                            on=lambda p, m: None)
    print(f"分析耗时 {time.time() - t0:.1f}s  fps={sig.fps:.2f}  样本={sig.motion.size}")
    print(f"自动 ROI (归一化 x0,y0,x1,y1) = {tuple(round(v, 3) for v in sig.roi)}")
    for name in ("motion", "sharpness", "shake"):
        a = getattr(sig, name)
        print(f"  {name:10s} p5={np.percentile(a,5):.3f} p50={np.percentile(a,50):.3f} "
              f"p95={np.percentile(a,95):.3f} max={a.max():.3f}")

    t = np.arange(sig.motion.size) / sig.fps
    fig, axes = plt.subplots(4, 1, figsize=(22, 12))
    axes[0].plot(t, sig.motion, lw=0.7, color="#2a8")
    axes[0].set_ylabel("motion")
    axes[0].axhline(np.percentile(sig.motion, 70), color="r", ls="--", lw=0.8)
    axes[1].plot(t, sig.shake, lw=0.6, color="#a52")
    axes[1].set_ylabel("shake px")
    axes[2].plot(t, sig.sharpness, lw=0.6, color="#26a")
    axes[2].set_ylabel("sharpness")
    axes[3].plot(t, sig.cut, lw=0.6, color="#a2a")
    axes[3].set_ylabel("hist diff")
    axes[3].set_xlabel("seconds")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(args.out) if args.out else FRAMES_DIR / "motion_diag.png"
    fig.savefig(out, dpi=90)
    print("图表 ->", out)

    if sig.activity_map is not None:
        a = sig.activity_map / (sig.activity_map.max() + 1e-9)
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        im = ax2.imshow(a, cmap="inferno")
        ax2.set_title("activity map (time-accumulated frame diff)")
        fig2.colorbar(im, ax=ax2)
        fig2.tight_layout()
        out2 = FRAMES_DIR / "activity_map.png"
        fig2.savefig(out2, dpi=100)
        print("活动热区 ->", out2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
