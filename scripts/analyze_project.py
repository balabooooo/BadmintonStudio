"""在已有工程上跑 AI 分析（可选自动剪辑/导出）。

用法:
    python scripts/analyze_project.py <project_id|工程名> [--weights highlight] [--cut] [--export 1080p]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"


def call(path: str, method: str = "GET", body: dict | None = None, timeout: float = 300) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait(jid: str, label: str) -> dict:
    t0 = time.time()
    last = ""
    while True:
        j = call(f"/api/jobs/{jid}")
        line = f"  [{j['progress'] * 100:5.1f}%] {j['stage']:9s} {j['message']}"
        if line != last:
            print(line, flush=True)
            last = line
        if j["status"] in ("done", "error", "cancelled"):
            print(f"  -> {label}: {j['status']} ({time.time() - t0:.0f}s)")
            if j["status"] == "error":
                print((j.get("error") or "")[-2500:])
            return j
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--weights", default="balanced")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--cut", action="store_true")
    ap.add_argument("--export", default="")
    ap.add_argument("--keep-ratio", type=float, default=0.0)
    args = ap.parse_args()

    pid = args.project
    if not pid.startswith("p_"):
        for p in call("/api/projects"):
            if p["name"] == args.project:
                pid = p["id"]
                break
        else:
            print("找不到工程:", args.project)
            return 1

    proj = call(f"/api/projects/{pid}")
    if not proj["media"]:
        print("工程里没有素材")
        return 1
    mid = proj["media"][0]["id"]
    print(f"工程 {proj['name']}  素材 {proj['media'][0]['name']}  {proj['media'][0]['duration']:.0f}s")

    print("[分析]")
    j = call(f"/api/projects/{pid}/analyze", "POST", {
        "media_id": mid, "weights": args.weights,
        "params": {"sample_fps": 12, "min_rally_seconds": 3.0, "gap_seconds": 3.0},
    })
    r = wait(j["job_id"], "analyze")
    if r["status"] != "done":
        return 1

    ana = call(f"/api/projects/{pid}/analysis/{mid}")
    st = ana["stats"]
    print(f"\n回合 {st['count']}  有效 {st['active_duration']:.0f}s / {st['duration']:.0f}s "
          f"({st['active_duration'] / max(st['duration'], 1) * 100:.1f}%)  平均分 {st['avg_score']:.1f}")
    print(f"音频可信度 {st.get('audio_reliability')}  融合权重 {st.get('component_weights')}")
    print(f"模块: 击球={st.get('hit_trace')}")
    print(f"      球员={st.get('player_trace')}")
    print(f"      羽毛球={st.get('shuttle_trace')}")

    rallies = ana["rallies"]
    if args.keep_ratio > 0:
        rallies = sorted(rallies, key=lambda r: -r["scores"]["total"])[: max(1, int(len(rallies) * args.keep_ratio))]
    sel = [r for r in rallies if r["scores"]["total"] >= args.min_score]

    print(f"\n  序号    起(s)     止(s)   时长  拍数   总分  标签")
    for r in sel[:60]:
        print(f"  {r['index']:4d}  {r['start']:8.1f}  {r['end']:8.1f}  {r['duration']:5.1f}  "
              f"{r['features']['shot_count']:4d}  {r['scores']['total']:5.1f}  {','.join(r['tags'])}")
    if len(sel) > 60:
        print(f"  ... 其余 {len(sel) - 60} 个省略")

    if args.cut:
        print("\n[自动剪辑]")
        body: dict = {"media_id": mid, "mode": "replace", "rally_ids": [r["id"] for r in sel]}
        tl = call(f"/api/projects/{pid}/timeline/auto-cut", "POST", body)
        print(f"  片段 {tl['clip_count']} 个  总时长 {tl['timeline']['duration']:.1f}s")

    if args.export:
        print("\n[导出]")
        presets = call("/api/export/presets")
        preset = next((p for p in presets if p["id"] == args.export), None)
        if preset is None:
            print("  未知预设:", args.export, [p["id"] for p in presets])
            return 1
        j = call(f"/api/projects/{pid}/export", "POST",
                 {"preset": preset, "name": f"clip9_{preset['id']}"})
        r = wait(j["job_id"], "export")
        if r["status"] == "done":
            print(f"  输出 {r['result']['path']}  {r['result']['size'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
