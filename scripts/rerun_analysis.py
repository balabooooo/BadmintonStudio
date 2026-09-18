"""端到端重跑一次完整分析，用来验收切分改动。

复用工程里已经生成好的代理视频 / 音轨（``ensure_proxy`` / ``ensure_audio``
命中缓存就不会再转码），所以只花 AI 分析本身的时间。

跑法::

    .venv\\Scripts\\python.exe scripts\\rerun_analysis.py [--json <工程分析json>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import pipeline as P          # noqa: E402
from bms.core.models import AnalysisParams, MediaInfo  # noqa: E402

DEFAULT_PROJ = ROOT / "data" / "projects" / "p_e274f1ae3e1e.json"
DEFAULT_JSON = (ROOT / "data" / "projects"
                / "p_e274f1ae3e1e.m_aa6b31f616ee.analysis.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--reuse", action="store_true",
                    help="直接读上一次存下来的结果，不重跑分析")
    args = ap.parse_args()

    old = json.loads(args.json.read_text(encoding="utf-8"))
    proj = json.loads(DEFAULT_PROJ.read_text(encoding="utf-8"))
    media = MediaInfo.model_validate(proj["media"][0])
    params = AnalysisParams.model_validate(old["params"])
    out = ROOT / "data" / "cache" / "frames" / "analysis_new.json"

    if args.reuse and out.exists():
        print(f"复用 {out}")
        from bms.core.models import AnalysisResult

        res = AnalysisResult.model_validate_json(out.read_text(encoding="utf-8"))
        print(f"素材 {media.name}  {media.duration:.1f}s  （未重跑分析）")
        return report(res, old)

    print(f"素材 {media.name}  {media.duration:.1f}s  "
          f"proxy={Path(media.proxy_path or '').name}")
    print(f"params: pre_roll={params.pre_roll} post_roll={params.post_roll} "
          f"hit_tail={params.hit_tail_seconds} mode={params.segment_mode}")

    last = [0.0]

    def prog(p: float, stage: str, msg: str = "") -> None:
        # 进度回调会被高频调用，按 2% 打点，避免刷屏
        if p - last[0] >= 0.02 or p >= 1.0:
            last[0] = p
            print(f"  [{p * 100:5.1f}%] {stage} {msg}", flush=True)

    t0 = time.time()
    res = P.run_analysis(media, params, on_progress=prog)
    el = time.time() - t0
    print(f"\n分析完成，耗时 {el:.1f}s   状态 {res.status}")
    if res.status != "done":
        print(res.error)
        return 1
    # 先存盘再报表：报表脚本出错时不用重跑 5 分钟的 AI 分析
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.model_dump_json(indent=1), encoding="utf-8")
    print(f"结果已存 {out}")
    return report(res, old)


def _feat(r, key: str, default: float = 0.0) -> float:
    """同时吃得下 pydantic 的 RallyFeatures 和旧 JSON 里的普通 dict。"""
    f = getattr(r, "features", None)
    if f is None:
        return default
    if isinstance(f, dict):
        return float(f.get(key, default) or default)
    return float(getattr(f, key, default) or default)


def report(res, old: dict) -> int:
    def stats(rallies) -> dict:
        if not rallies:
            return {"n": 0}
        d = np.asarray([float(r.duration if hasattr(r, "duration")
                              else r["end"] - r["start"]) for r in rallies], dtype=np.float64)
        rs = sorted(rallies, key=lambda r: float(getattr(r, "start", r.get("start") if isinstance(r, dict) else 0)))
        st = [float(r.start if hasattr(r, "start") else r["start"]) for r in rs]
        en = [float(r.end if hasattr(r, "end") else r["end"]) for r in rs]
        g = np.asarray([max(0.0, st[i + 1] - en[i]) for i in range(len(rs) - 1)],
                       dtype=np.float64)
        return {"n": len(rallies), "mean": round(float(d.mean()), 2),
                "med": round(float(np.median(d)), 2), "max": round(float(d.max()), 2),
                "over30": int((d > 30).sum()),
                "min_gap": round(float(g.min()), 2) if g.size else 0.0,
                "gap_lt1": int((g < 1.0).sum()) if g.size else 0,
                "hit_anchored": int(sum(1 for r in rallies
                                        if _feat(r, "hit_anchored") > 0)),
                "zero_shot": int(sum(1 for r in rallies
                                     if _feat(r, "shot_count") == 0))}

    old_r = [{"start": r["start"], "end": r["end"], "duration": r["end"] - r["start"],
              "features": r.get("features") or {}} for r in old["rallies"]]
    o, n = stats(old_r), stats(res.rallies)
    print("\n=== 旧（存下来的结果）===")
    print("  " + json.dumps(o, ensure_ascii=False))
    print("=== 新（本次重跑）===")
    print("  " + json.dumps(n, ensure_ascii=False))
    print("\n=== 本次切分诊断 ===")
    for k in ("mode", "method", "count", "player_coverage", "activity_count",
              "quality", "merged_from"):
        if k in (res.stats.get("segmentation") or {}):
            print(f"  {k}: {res.stats['segmentation'][k]}")
    print(f"  audio_reliability: {res.stats.get('audio_reliability')}")
    print(f"  component_weights: {res.stats.get('component_weights')}")
    print(f"  player_trace: {json.dumps(res.stats.get('player_trace'), ensure_ascii=False)[:220]}")

    print("\n=== 新回合明细（前 40 条）===")
    print("     起      止    时长  拍数  锚定  tail_gap")
    for r in sorted(res.rallies, key=lambda x: x.start)[:40]:
        print(f" {r.start:7.2f} {r.end:7.2f} {r.duration:6.2f}  "
              f"{int(_feat(r, 'shot_count')):4d}   "
              f"{'Y' if _feat(r, 'hit_anchored') else '-'}    "
              f"{_feat(r, 'tail_gap'):6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
