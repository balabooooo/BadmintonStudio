"""命令行跑完整分析流水线，便于不开界面时调参。

用法:
    python scripts/run_analysis.py "<video>" [--no-players] [--no-shuttle] [--weights balanced]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.analysis.pipeline import run_analysis  # noqa: E402
from bms.config import ensure_dirs  # noqa: E402
from bms.core.media import probe_media  # noqa: E402
from bms.core.models import AnalysisParams  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default="balanced")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--min", dest="min_rally", type=float, default=3.0)
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--no-shuttle", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--viewpoint", default="auto",
                    choices=["auto", "rear", "side", "elevated", "overhead"])
    ap.add_argument("--segment-mode", default="auto",
                    choices=["auto", "activity", "hybrid"])
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--proxy", default="",
                    help="复用已有的代理视频（例如已缓存的 960x540 代理），"
                         "跳过「重新生成代理」这一步，方便快速迭代调参")
    ap.add_argument("--audio", default="", help="复用已有的 16k 单声道音轨 wav")
    args = ap.parse_args()

    ensure_dirs()
    info = probe_media(args.video)
    if args.proxy:
        info.proxy_path = args.proxy
        info.proxy_fps = info.fps
    if args.audio:
        info.audio_path = args.audio
    print(f"素材: {info.name}  {info.width}x{info.height}  {info.fps:.2f}fps  {info.duration:.1f}s")
    if info.proxy_path:
        print(f"复用代理: {info.proxy_path}")
    if info.audio_path:
        print(f"复用音轨: {info.audio_path}")

    params = AnalysisParams(
        hit_sensitivity=args.sensitivity,
        gap_seconds=args.gap,
        min_rally_seconds=args.min_rally,
        use_audio=not args.no_audio,
        use_players=not args.no_players,
        use_shuttle=not args.no_shuttle,
        sample_fps=args.fps,
        viewpoint=args.viewpoint,
        segment_mode=args.segment_mode,
        auto_calibrate=not args.no_calibrate,
    )

    last = [""]

    def on(p: float, stage: str, msg: str = "") -> None:
        line = f"  [{p * 100:5.1f}%] {stage:10s} {msg}"
        if line != last[0]:
            print(line, flush=True)
            last[0] = line

    t0 = time.time()
    res = run_analysis(info, params, on_progress=on, weights_key=args.weights)
    print(f"\n状态: {res.status}  耗时 {time.time() - t0:.1f}s")
    if res.error:
        print(res.error)
        return 1

    st = res.stats
    print(f"回合数: {st.get('count')}  有效时长: {st.get('active_duration', 0):.1f}s / {info.duration:.1f}s "
          f"({st.get('active_duration', 0) / max(info.duration, 1e-6) * 100:.1f}%)")
    print(f"总拍数: {st.get('total_shots')}  平均拍数: {st.get('avg_shots', 0):.1f}  最长: {st.get('max_shots')}")
    print(f"平均分: {st.get('avg_score', 0):.1f}  高分(>=70)回合数: {st.get('high_score_count')}")
    print(f"追踪: 击球 {st.get('hit_trace')}  球员 {st.get('player_trace')}  球 {st.get('shuttle_trace')}")
    w = res.signals.get("weights")
    print(f"融合权重 [players, motion, audio, shuttle, roi] = {w}")

    # ---- 切分与机位诊断（这是「切得准不准」最该看的两块信息）
    seg = (st or {}).get("segmentation") or {}
    print(f"\n切分: 方式={seg.get('method')}  模式={seg.get('mode')}  数量={seg.get('count')}"
          f"  球员检测覆盖={seg.get('player_coverage')}"
          f"  质量={json.dumps(seg.get('quality'), ensure_ascii=False)}"
          f"  合并来源={json.dumps(seg.get('merged_from'), ensure_ascii=False)}")
    cal = res.calibration or {}
    if cal:
        print(f"机位: {cal.get('viewpoint_label')}（置信度 {cal.get('confidence')}）"
              f"  场地色 {cal.get('court_color')}  占画面 {cal.get('court_area_ratio')}")
        for n in cal.get("notes", []):
            print(f"   · {n}")
    print(f"有效 ROI: {st.get('effective_roi')}")
    durs = [r.duration for r in res.rallies]
    if len(res.rallies) > 1:
        gaps = [max(0.0, res.rallies[i + 1].start - res.rallies[i].end)
                for i in range(len(res.rallies) - 1)]
        touching = sum(1 for g in gaps if g <= 0.1)
        print(f"时长 均值 {np.mean(durs):.1f}s 中位 {np.median(durs):.1f}s   "
              f"间隔 均值 {np.mean(gaps):.1f}s 最小 {min(gaps):.1f}s   "
              f"首尾相接 {touching} 个  无停顿(<=1s) {sum(1 for g in gaps if g <= 1.0)} 个")

    print("\n  序号    起(s)     止(s)   时长  拍数   总分  长度  强度  技术  精彩  标签")
    for r in res.rallies[:80]:
        print(f"  {r.index:4d}  {r.start:8.2f}  {r.end:8.2f}  {r.duration:5.1f}  {r.features.shot_count:4d}  "
              f"{r.scores.total:5.1f} {r.scores.length:5.1f} {r.scores.intensity:5.1f} "
              f"{r.scores.technique:5.1f} {r.scores.excitement:5.1f}  {','.join(r.tags)}")
    if len(res.rallies) > 80:
        print(f"  ... 其余 {len(res.rallies) - 80} 个省略")

    out = ROOT / "data" / "cache" / "last_analysis.json"
    out.write_text(res.model_dump_json(indent=1), encoding="utf-8")
    print(f"\n结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
