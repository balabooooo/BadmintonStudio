"""回合终点评估：拿一份**已经跑完的真实分析结果**，对比旧逻辑和新逻辑。

为什么可以这样比：分析结果 JSON 里存了 ``rallies``（旧代码的输出）、
``activity_full``、以及逐拍击球 ``hit_times/strength/confidence``。
于是「旧结果」直接读出来即可，「新结果」用 :func:`bms.analysis.pipeline.resegment`
在实际代码上重算 —— 不需要重跑 30 分钟的 AI 分析。

跑法::

    .venv\\Scripts\\python.exe scripts\\eval_rally_end.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import pipeline as P          # noqa: E402
from bms.analysis import rally as RA            # noqa: E402
from bms.core.models import AnalysisResult      # noqa: E402

DEFAULT_JSON = (ROOT / "data" / "projects"
                / "p_e274f1ae3e1e.m_aa6b31f616ee.analysis.json")


def describe(name: str, rallies: list, duration: float) -> dict:
    durs = np.asarray([r.end - r.start for r in rallies], dtype=np.float64)
    if rallies:
        gaps = np.asarray([max(0.0, rallies[i + 1].start - rallies[i].end)
                           for i in range(len(rallies) - 1)], dtype=np.float64)
    else:
        gaps = np.zeros(0)
    covered = float(durs.sum()) / max(duration, 1e-6)
    print(f"\n=== {name} ===")
    print(f"  回合数 {len(rallies)}   覆盖时长占比 {covered * 100:.1f}%")
    if durs.size:
        print(f"  时长: 平均 {durs.mean():.1f}s  中位 {np.median(durs):.1f}s  "
              f"最大 {durs.max():.1f}s  >30s 的有 {int((durs > 30).sum())} 个")
    if gaps.size:
        print(f"  回合间隔: 中位 {np.median(gaps):.1f}s  最小 {gaps.min():.2f}s  "
              f"<1s 的有 {int((gaps < 1.0).sum())} 个  <3s 的有 {int((gaps < 3.0).sum())} 个")
    return {"n": len(rallies), "covered": covered,
            "durs": durs, "gaps": gaps}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    res = AnalysisResult.model_validate(raw)
    duration = float(raw["signals"]["duration"][0])
    hit_times = np.asarray(raw["signals"]["hit_times"], dtype=np.float64)

    print(f"素材时长 {duration:.1f}s   音频击球 {hit_times.size} 个   "
          f"audio_reliability={res.stats.get('audio_reliability')}")

    old = sorted(res.rallies, key=lambda r: r.start)
    o = describe("旧逻辑（JSON 里存下来的真实输出）", old, duration)

    # 用「最后一拍 + hit_tail」以外的旧口径也要能对比，所以先看新口径
    new_res = P.resegment(res.model_copy(deep=True), res.params, "balanced")
    new = sorted(new_res.rallies, key=lambda r: r.start)
    n = describe("新逻辑（resegment，击球空档切分 + 终点锚定到最后一拍）", new, duration)

    print(f"\n=== 变化 ===")
    print(f"  回合数 {o['n']} -> {n['n']}")
    print(f"  覆盖时长占比 {o['covered'] * 100:.1f}% -> {n['covered'] * 100:.1f}%")
    if o["durs"].size and n["durs"].size:
        print(f"  最长回合 {o['durs'].max():.1f}s -> {n['durs'].max():.1f}s")
        print(f"  >30s 的回合 {int((o['durs'] > 30).sum())} -> {int((n['durs'] > 30).sum())}")

    # 逐条对照：新回合落在哪个旧回合里、切在哪里
    print(f"\n=== 新回合明细（前 30 条）===")
    print("    起  止      时长  拍数  来源旧回合(起-止)      切分方式")
    for r in new[:30]:
        f = r.features
        src = next((x for x in old if x.start <= (r.start + r.end) / 2 <= x.end), None)
        src_s = f"{src.start:6.1f}-{src.end:6.1f}" if src else "   (新)     "
        print(f" {r.start:6.1f} {r.end:6.1f} {r.duration:6.2f}  {f.shot_count:4d}   "
              f"{src_s}   {'锚定' if f.duration > 0 else ''}")

    # 击球空档的分布：用来核对 MAX_INTRA_HIT_GAP 这个门限合不合理
    hits = P._rebuild_hits(raw["signals"])
    if hits is not None and hits.times.size > 2:
        g = np.diff(np.sort(hits.times))
        g = g[g > 1e-6]
        print(f"\n=== 全部击球间隔分布（全片 {hits.times.size} 拍）===")
        for q in (50, 70, 80, 90, 95, 99):
            print(f"  p{q}: {np.percentile(g, q):5.2f}s", end="")
        print()
        print(f"  估计的段内拍间隔上限 hit_gap_limit = "
              f"{RA.hit_gap_limit(hits.times):.2f}s")
        for thr in (2.0, 3.0, 4.0, 5.0):
            print(f"  >{thr:.0f}s 的空档共 {int((g > thr).sum())} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
