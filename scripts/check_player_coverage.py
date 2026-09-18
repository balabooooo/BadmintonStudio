"""核对「逐时间窗挑选比赛球员」有没有把球员信号的覆盖补上。

旧行为（实测 30 分钟素材）：``frame_boxes`` 有 58% 的时间是空的 ——
前 300 秒和最后 600 秒完全没有球员信号，于是「球员运动切分」整段作废。
这个脚本跑一遍 :func:`bms.analysis.players.analyze_players`，按时间报告
``active_count`` 非零的比例以及球员框的数量，用来确认修复生效。

跑法::

    .venv\\Scripts\\python.exe scripts\\check_player_coverage.py [--seconds 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import players as PL  # noqa: E402
from bms.core.models import AnalysisParams  # noqa: E402

PROXY = (ROOT / "data" / "cache" / "proxies"
         / "20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = 全片")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    p = AnalysisParams()
    poly = p.court_poly
    t0 = time.time()

    def prog(v: float, stage: str) -> None:
        print(f"  [{v * 100:5.1f}%] {stage}", flush=True)

    sig = PL.analyze_players(
        str(PROXY),
        sample_fps=args.fps,
        max_seconds=args.seconds,
        viewpoint="side",
        roi_poly=poly,
        on_progress=prog,
    )
    el = time.time() - t0
    print(f"\n耗时 {el:.1f}s   轨迹 {len(sig.tracks)} 条   "
          f"active_ids {len(sig.active_player_ids)} 个")

    ac = np.asarray(sig.active_count, dtype=np.float32)
    n = ac.size
    dur = sig.duration
    print(f"active_count 非零比例 = {float(np.mean(ac > 0)):.3f}   "
          f"（旧实现实测 0.475）")
    print(f"缺失时间 = {float(np.mean(ac <= 0)) * dur:.0f}s / {dur:.0f}s")

    print("\n分段时间覆盖（每 300 秒）:")
    for a in range(0, int(dur), 300):
        i0, i1 = int(a * sig.fps), int(min(n, (a + 300) * sig.fps))
        if i1 <= i0:
            continue
        seg = ac[i0:i1]
        print(f"  {a:5d}-{a + 300:5d}s  有球员 {np.mean(seg > 0) * 100:5.1f}%   "
              f"平均人数 {seg.mean():.2f}   最多 {seg.max():.0f}")

    gaps = []
    bad = ac <= 0
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            if (j - i) / sig.fps >= 5.0:
                gaps.append((i / sig.fps, j / sig.fps))
            i = j
        else:
            i += 1
    print(f"\n连续缺失 >=5s 的时段共 {len(gaps)} 段，合计 "
          f"{sum(b - a for a, b in gaps):.0f}s")
    for a, b in gaps[:12]:
        print(f"    {a:7.1f} ~ {b:7.1f}s  ({b - a:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
