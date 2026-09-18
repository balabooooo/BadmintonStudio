"""验证脚本：羽毛球候选检测与轨迹跟踪。

用法::

    python scripts/test_shuttle.py                        # 默认用 480p 测试代理跑 60 秒
    python scripts/test_shuttle.py --seconds 60 --tag r1 # 迭代时给诊断图加后缀
    python scripts/test_shuttle.py --sensitivity 0.3 --max-speed 0.6

产出：
* 控制台：轨迹列表、presence 非零率、max_candidate_speed 分位数
* ``data/cache/frames/shuttle_diag.png``（``--tag`` 时是 ``shuttle_diag_<tag>.png``）：
  子图1 presence 随时间；子图2 candidate_count 随时间；
  子图3 候选点最多的 4 帧 2x2，原帧上绿圈=所有候选点，黄粗线=属于有效轨迹的点
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bms.analysis import shuttle as SH  # noqa: E402
from bms.config import FRAMES_DIR, ensure_dirs  # noqa: E402

DEFAULT_VIDEO = ROOT / "data" / "cache" / "proxies" / "_test_0_300_480_15.mp4"


def _q(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


def _draw_panel(frame: np.ndarray, cands, tracked_pts, tracks_here, size=(960, 540)) -> np.ndarray:
    """在原帧上画候选点（绿圈）与有效轨迹点（黄色粗线）。"""
    vis = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
    sx = size[0] / frame.shape[1]
    sy = size[1] / frame.shape[0]

    # 有效轨迹：黄粗折线 + 黄实心点
    for pts in tracks_here:
        xy = np.array([[p[0] * size[0], p[1] * size[1]] for p in pts], np.int32)
        if len(xy) >= 2:
            cv2.polylines(vis, [xy], False, (0, 235, 255), 3, cv2.LINE_AA)
        for p in xy:
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 235, 255), -1, cv2.LINE_AA)

    # 所有候选点：绿圈
    for (x, y, _s) in cands:
        cv2.circle(vis, (int(round(x * size[0])), int(round(y * size[1]))), 9,
                   (0, 255, 0), 2, cv2.LINE_AA)
    del sx, sy
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--sample-fps", type=float, default=30.0)
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--min-points", type=int, default=SH.DEFAULT_MIN_POINTS)
    ap.add_argument("--max-gap", type=int, default=SH.DEFAULT_MAX_GAP)
    ap.add_argument("--min-speed", type=float, default=SH.DEFAULT_MIN_SPEED)
    ap.add_argument("--max-speed", type=float, default=SH.DEFAULT_MAX_SPEED)
    ap.add_argument("--max-resid", type=float, default=SH.DEFAULT_MAX_RESID)
    ap.add_argument("--min-span", type=float, default=SH.DEFAULT_MIN_SPAN)
    ap.add_argument("--min-accel", type=float, default=SH.DEFAULT_MIN_ACCEL)
    ap.add_argument("--window", type=int, default=SH.DEFAULT_WINDOW)
    ap.add_argument("--work-width", type=int, default=0)
    ap.add_argument("--court-gex", type=float, default=None,
                    help="场地先验阈值；None=按 sensitivity 自动。设 -999 可关闭")
    ap.add_argument("--roi", default="", help="归一化 x0,y0,x1,y1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    ensure_dirs()
    roi = None
    if args.roi:
        roi = tuple(float(v) for v in args.roi.split(","))  # type: ignore[assignment]

    kw = dict(
        sample_fps=args.sample_fps,
        roi=roi,
        max_seconds=args.seconds,
        sensitivity=args.sensitivity,
        work_width=args.work_width,
        window=args.window,
        min_points=args.min_points,
        max_gap=args.max_gap,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_resid=args.max_resid,
        min_span=args.min_span,
        min_accel=args.min_accel,
        court_gex=args.court_gex,
    )

    print(f"视频: {args.video}")
    print(f"参数: seconds={args.seconds} sensitivity={args.sensitivity} "
          f"min_points={args.min_points} max_gap={args.max_gap} "
          f"min_speed={args.min_speed} max_speed={args.max_speed} "
          f"max_resid={args.max_resid} min_span={args.min_span} "
          f"min_accel={args.min_accel} window={args.window} roi={roi} "
          f"court_gex={args.court_gex}")

    # ---- 规格入口 analyze_shuttle
    stages: list[tuple[str, float]] = []

    def on_prog(p: float, s: str) -> None:
        stages.append((s, p))

    t0 = time.time()
    sig = SH.analyze_shuttle(args.video, on_progress=on_prog, **kw)
    dt = time.time() - t0
    print(f"\n[analyze_shuttle] 耗时 {dt:.1f}s  采样 {sig.fps:.2f} fps  "
          f"时长 {sig.duration:.1f}s  帧数 {sig.presence.size}")

    # ---- 1) 轨迹列表
    print(f"\n=== 轨迹总数: {len(sig.tracks)} ===")
    print(f"{'#':>3} {'pts':>4} {'start':>7} {'end':>7} {'meanSpd':>8} "
          f"{'maxSpd':>8} {'span':>6} {'conf':>5}")
    for i, tr in enumerate(sig.tracks[: args.top]):
        print(f"{i:>3} {len(tr.points):>4} {tr.start:>7.2f} {tr.end:>7.2f} "
              f"{tr.mean_speed:>8.4f} {tr.max_speed:>8.4f} {tr.span:>6.3f} "
              f"{tr.confidence:>5.2f}")
    if len(sig.tracks) > args.top:
        print(f"    ... 其余 {len(sig.tracks) - args.top} 条省略")
    if sig.tracks:
        conf = np.array([t.confidence for t in sig.tracks])
        npts = np.array([len(t.points) for t in sig.tracks])
        print(f"轨迹置信度: p50={np.median(conf):.2f} max={conf.max():.2f} | "
              f"点数: p50={np.median(npts):.0f} max={npts.max()}")

    # ---- 2) presence / candidate_count / max_candidate_speed
    p = sig.presence
    nz = int((p > 0).sum())
    print(f"\n=== presence ===")
    print(f"非零率 {nz}/{p.size} = {100.0 * nz / max(1, p.size):.2f}%  "
          f"非零均值 {p[p > 0].mean() if nz else 0.0:.3f}  max {p.max():.3f}")
    if p.size:
        print(f"presence 总时长占比（>0.3）: "
              f"{100.0 * int((p > 0.3).sum()) / p.size:.2f}%")

    print(f"\n=== candidate_count ===")
    cc = sig.candidate_count
    print(f"p50={np.median(cc):.0f} p90={_q(cc, 90):.0f} p99={_q(cc, 99):.0f} "
          f"max={cc.max():.0f}  总候选={int(cc.sum())}  零候选帧="
          f"{int((cc == 0).sum())}/{cc.size}")

    print(f"\n=== max_candidate_speed (画面高/秒) ===")
    ms = sig.max_candidate_speed
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<5} = {_q(ms, q):.4f}")
    print(f"  max    = {float(ms.max()) if ms.size else 0.0:.4f}")

    # ---- 3) 诊断图
    _sig2, dbg = SH.analyze_shuttle_debug(args.video, **kw)
    cands_by_frame = dbg["candidates"]
    tracked = dbg["tracked"]
    step = int(dbg["step"])
    ww, wh = int(dbg["width"]), int(dbg["height"])
    print(f"\n[debug] 工作分辨率 {ww}x{wh}  step={step}  "
          f"有候选的帧 {len(cands_by_frame)}/{cc.size}")

    topframes = np.argsort(-cc)[:4]
    topframes = [int(f) for f in topframes if cc[f] > 0]
    print(f"候选最多的帧: {[(f, int(cc[f])) for f in topframes]}")

    # 这些帧所属的完整轨迹折线
    def tracks_of(f: int) -> list[list[tuple[float, float]]]:
        out = []
        for tr in _sig2.tracks:
            if any(pt.frame == f for pt in tr.points):
                out.append([(pt.x, pt.y) for pt in tr.points])
        return out

    cap = cv2.VideoCapture(str(args.video))
    panels = []
    for f in topframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f * step)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        panels.append(_draw_panel(fr, cands_by_frame.get(f, []), None, tracks_of(f)))
    cap.release()
    while len(panels) < 4:
        panels.append(np.zeros((540, 960, 3), np.uint8))

    t = np.arange(p.size) / max(sig.fps, 1e-6)
    fig = plt.figure(figsize=(17, 19), dpi=100)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.75, 0.75, 2.0, 2.0],
                          left=0.06, right=0.985, top=0.96, bottom=0.03,
                          hspace=0.28, wspace=0.05)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, p, lw=0.9, color="#c33")
    ax1.set_ylabel("presence")
    ax1.set_title(f"shuttle presence  (non-zero {100.0 * nz / max(1, p.size):.2f}%, "
                  f"{len(sig.tracks)} tracks)", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, max(t[-1] if t.size else 1, 1))

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, cc, lw=0.8, color="#268")
    ax2.set_ylabel("candidate count")
    ax2.set_xlabel("seconds")
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0, max(t[-1] if t.size else 1, 1))

    # 候选最多的 4 帧，2x2 拼图
    for k in range(4):
        ax = fig.add_subplot(gs[2 + k // 2, k % 2])
        ax.imshow(cv2.cvtColor(panels[k], cv2.COLOR_BGR2RGB))
        f = topframes[k] if k < len(topframes) else -1
        ax.set_title(f"frame {f}   n_cand={int(cc[f]) if f >= 0 else 0}   "
                     f"green circle = candidate,  yellow = valid track", fontsize=11)
        ax.axis("off")
    out = FRAMES_DIR / (f"shuttle_diag_{args.tag}.png" if args.tag else "shuttle_diag.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"诊断图 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
