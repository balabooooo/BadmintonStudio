"""验证脚本：人物检测 + 跟踪 + 比赛球员判定。

用法:
    python scripts/test_players.py [视频] [--seconds 120] [--fps 15]

产物:
    data/cache/frames/players_diag.png     活跃球员数量/速度随时间
    data/cache/frames/players_overlay.png  4 个时刻的框叠加（绿=比赛球员，红=其他人）
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

from bms.analysis import players as PL  # noqa: E402
from bms.config import FRAMES_DIR, PROXIES_DIR, ensure_dirs  # noqa: E402

DEFAULT_VIDEO = PROXIES_DIR / "_test_0_300_480_15.mp4"
MARK_TIMES = (10.0, 40.0, 70.0, 100.0)


def _find_box_at(track: PL.PlayerTrack, t: float, tol: float) -> tuple[float, float, float, float] | None:
    """找该轨迹在 t 秒附近的那一帧的框。"""
    best, best_dt = None, tol
    for i, tt in enumerate(track.times):
        d = abs(tt - t)
        if d <= best_dt:
            best, best_dt = i, d
    return track.boxes[best] if best is not None else None


def _duration(track: PL.PlayerTrack) -> float:
    return float(track.times[-1] - track.times[0]) if track.times else 0.0


def _x_range(track: PL.PlayerTrack) -> float:
    if not track.boxes:
        return 0.0
    cxs = [0.5 * (b[0] + b[2]) for b in track.boxes]
    return float(max(cxs) - min(cxs))


def _static_ratio(track: PL.PlayerTrack) -> float:
    sp = track.speeds[1:] if len(track.speeds) > 1 else []
    if not sp:
        return 1.0
    return float(sum(1 for s in sp if s < 0.05) / len(sp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ensure_dirs()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.exists():
        print(f"[错误] 视频不存在: {video}")
        return 1

    print(f"分析 {video.name}  max_seconds={args.seconds}  sample_fps={args.fps}  device={args.device}")
    t0 = time.time()
    last = {"p": -1.0}

    def on_progress(p: float, stage: str) -> None:
        if p - last["p"] >= 0.1:
            last["p"] = p
            print(f"  [{stage}] {p * 100:5.1f}%")

    sig = PL.analyze_players(
        str(video), sample_fps=args.fps, max_seconds=args.seconds,
        imgsz=args.imgsz, device=args.device, batch_hint=args.batch,
        on_progress=on_progress,
    )
    elapsed = time.time() - t0

    print("\n================ 汇总 ================")
    print(f"fps={sig.fps:.2f}  duration={sig.duration:.1f}s  轨迹数={len(sig.tracks)}  "
          f"耗时={elapsed:.1f}s  性能={sig.duration / max(elapsed, 1e-6):.2f} 处理秒/秒")

    print("\n---- 轨迹（按 active_score 降序，前 15 条） ----")
    print(f"{'id':>4} {'帧数':>6} {'meanA':>8} {'maxA':>8} {'meanV':>7} {'maxV':>7} "
          f"{'travel':>7} {'静态比':>6} {'x跨度':>6} {'时长':>6} {'score':>6}  状态")
    active_set = set(sig.active_player_ids)
    ranked = sorted(sig.tracks, key=lambda t: t.active_score, reverse=True)
    for tr in ranked[:15]:
        flag = "★ 比赛球员" if tr.track_id in active_set else ""
        print(f"{tr.track_id:>4} {tr.n:>6} {tr.mean_area:>8.4f} {tr.max_area:>8.4f} "
              f"{tr.mean_speed:>7.3f} {tr.max_speed:>7.3f} {tr.total_travel:>7.2f} "
              f"{_static_ratio(tr):>6.2f} {_x_range(tr):>6.3f} {_duration(tr):>6.1f} "
              f"{tr.active_score:>6.3f}  {flag}")

    print(f"\nactive_player_ids = {sig.active_player_ids}")
    for tr in ranked:
        if tr.track_id in active_set:
            print(f"   #{tr.track_id}: 帧数={tr.n} 覆盖={_duration(tr):.1f}s "
                  f"平均面积={tr.mean_area:.4f} 平均速度={tr.mean_speed:.3f} "
                  f"峰值速度={tr.max_speed:.3f} 低速帧占比={_static_ratio(tr):.2f} "
                  f"水平跨度={_x_range(tr):.3f} 得分={tr.active_score:.3f}")

    ac = sig.active_count
    sp = sig.active_speed
    print(f"\nactive_count: min={ac.min():.0f} median={np.median(ac):.0f} max={ac.max():.0f} "
          f"mean={ac.mean():.2f}")
    for q in (50, 75, 90, 95, 99):
        print(f"active_speed p{q}={np.percentile(sp, q):.3f}")
    print(f"active_speed max={sp.max():.3f}  max_speed p95={np.percentile(sig.max_speed, 95):.3f} "
          f"p99={np.percentile(sig.max_speed, 99):.3f} max={sig.max_speed.max():.3f}")
    print(f"crowd_speed mean={sig.crowd_speed.mean():.3f} p95={np.percentile(sig.crowd_speed, 95):.3f}")
    print(f"数组长度: active_count={ac.size} (期望 ceil(duration*fps)="
          f"{int(np.ceil(sig.duration * sig.fps))})  frame_boxes={len(sig.frame_boxes)}")

    # ---------------- 诊断图 ----------------
    t_axis = np.arange(ac.size) / sig.fps
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].plot(t_axis, ac, lw=0.9, color="#2a8")
    axes[0].set_ylabel("active_count")
    axes[0].set_title(f"active players per frame  (ids={sig.active_player_ids})")
    axes[1].plot(t_axis, sp, lw=0.9, color="#26a", label="active_speed")
    axes[1].plot(t_axis, sig.max_speed, lw=0.7, color="#a52", alpha=0.75, label="max_speed")
    axes[1].set_ylabel("speed (h/s)")
    axes[1].set_xlabel("seconds")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    diag = FRAMES_DIR / "players_diag.png"
    fig.savefig(diag, dpi=95)
    print(f"\n诊断图 -> {diag}")

    # ---------------- 叠加图 ----------------
    import cv2

    cap = cv2.VideoCapture(str(video))
    tol = 0.75 / sig.fps
    cells = []
    zooms = []
    for t in MARK_TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((270, 480, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        for tr in sig.tracks:
            box = _find_box_at(tr, t, tol)
            if box is None:
                continue
            x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
            if tr.track_id in active_set:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"#{tr.track_id}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        big = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        cv2.putText(big, f"t={t:.0f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cells.append(big)

        # 另外单独存一份 3x 放大（只保留球场横带），方便肉眼核对
        y0, y1 = int(0.26 * h), int(0.80 * h)
        band = frame[y0:y1]
        band = cv2.resize(band, (band.shape[1] * 3, band.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(band, f"t={t:.0f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        zoom_path = FRAMES_DIR / f"players_zoom_{int(t)}.png"
        cv2.imwrite(str(zoom_path), band)
        zooms.append(zoom_path)
    cap.release()

    top = np.hstack(cells[0:2])
    bottom = np.hstack(cells[2:4])
    overlay = np.vstack([top, bottom])
    out = FRAMES_DIR / "players_overlay.png"
    cv2.imwrite(str(out), overlay)
    print(f"叠加图 -> {out}  ({overlay.shape[1]}x{overlay.shape[0]})")
    print("放大图 -> " + ", ".join(p.name for p in zooms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
