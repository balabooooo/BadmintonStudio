"""姿态模块的端到端验证：球员框 -> 关键点 -> 挥拍信号 -> 击球归属。

跑法::

    .venv\\Scripts\\python.exe scripts\\test_pose.py --seconds 90

会打印：
* 姿态覆盖率、挥拍信号的分布；
* 每个音频击球在 ±0.35s 内的挥拍证据分，以及门控前后的保留比例；
* 按门控后的击球序列聚类，给出回合结构（和改动前的 0~69s 一整条对比）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import audio_hits as AH     # noqa: E402
from bms.analysis import players as PL        # noqa: E402
from bms.analysis import pose as POSE         # noqa: E402
from bms.analysis import rally as RA          # noqa: E402
from bms.core.models import AnalysisParams    # noqa: E402

PROXY = (ROOT / "data" / "cache" / "proxies"
         / "20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4")
WAV = (ROOT / "data" / "cache" / "audio"
       / "20260913_羽毛球_clip9_m_aa6b31f616ee_16000.wav")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--thr", type=float, default=0.22)
    args = ap.parse_args()

    p = AnalysisParams()
    t0 = time.time()
    sig = PL.analyze_players(str(PROXY), sample_fps=12.0, max_seconds=args.seconds,
                             viewpoint="side", roi_poly=p.court_poly)
    print(f"球员：{len(sig.tracks)} 条轨迹 / {len(sig.active_player_ids)} 个 active id / "
          f"{time.time() - t0:.1f}s")
    ac = np.asarray(sig.active_count, dtype=np.float32)
    print(f"      active_count 非零 {float(np.mean(ac > 0)):.2f}  "
          f"平均人数 {ac.mean():.2f}  最多 {ac.max():.0f}")

    t0 = time.time()
    pose = POSE.analyze_pose(str(PROXY), sig.frame_boxes, float(sig.fps), device="cuda")
    el = time.time() - t0
    if pose is None:
        print("姿态分析失败")
        return 1
    print(f"\n姿态：{el:.1f}s（{args.seconds:.0f}s 素材，折合 30 分钟约 "
          f"{el / args.seconds * 1800 / 60:.1f} 分钟）")
    print(f"      覆盖率 {pose.coverage:.3f}（旧实现没有这一路，等于 0）")
    sw = pose.swing
    print(f"      swing: p50={np.percentile(sw,50):.3f} p90={np.percentile(sw,90):.3f} "
          f"p99={np.percentile(sw,99):.3f} max={sw.max():.3f} 静息={pose.quiet:.3f}")
    print(f"      trace: {pose.trace}")

    hits = AH.detect_hits(WAV, sensitivity=p.hit_sensitivity)
    keep_t = hits.times <= args.seconds
    hits = POSE.filter_hits(hits, keep_t)
    ev = POSE.hit_swing_evidence(pose, hits.times)
    print(f"\n击球 {hits.times.size} 个（{args.seconds:.0f}s 窗口）")
    print(f"  证据分：p10={np.percentile(ev,10):.3f} p50={np.percentile(ev,50):.3f} "
          f"p90={np.percentile(ev,90):.3f}")
    for thr in (0.10, 0.15, 0.22, 0.30, 0.40, 0.50):
        print(f"  thr={thr:.2f} -> 保留 {float(np.mean(ev >= thr)) * 100:5.1f}%")

    mask, trace = POSE.gate_hits(hits, pose, threshold=args.thr)
    print(f"  门控 trace: {trace}")
    gated = POSE.filter_hits(hits, mask) if mask is not None else hits
    print(f"  门控后剩 {gated.times.size} 拍")

    # 门控前后各做一次「按击球间隔聚类」，直观对比回合结构
    dur = args.seconds
    iv = [RA.RallyInterval(start=0.0, end=dur)]
    print("\n=== 门控前：按击球间隔切分（MAX_INTRA_HIT_GAP=3.0s）===")
    show(RA.split_by_hit_gaps(iv, hits, limit=RA.MAX_INTRA_HIT_GAP))
    print("=== 门控后 ===")
    show(RA.split_by_hit_gaps(iv, gated, limit=RA.MAX_INTRA_HIT_GAP))
    return 0


def show(segs) -> None:
    if len(segs) <= 1:
        print(f"  只有 1 段（{segs[0].start:.1f}~{segs[0].end:.1f}s）—— 说明击球序列里"
              "没有超过门限的空档，切不开")
        return
    for s in sorted(segs, key=lambda v: v.start):
        print(f"  {s.start:7.2f} ~ {s.end:7.2f}s   时长 {s.end - s.start:6.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())
