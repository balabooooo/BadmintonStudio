"""End-to-end API smoke test: create project -> import -> analyze -> auto-cut -> export.

Usage:
    python scripts/e2e_test.py "<video>" [--base http://127.0.0.1:8000] [--no-export]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path: str, method: str = "GET", body: dict | None = None, timeout: float = 120) -> dict:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode('utf-8', 'replace')[:600]}") from None


def wait_job(jid: str, label: str, timeout: float = 3600) -> dict:
    t0 = time.time()
    last = ""
    while True:
        j = call(f"/api/jobs/{jid}")
        line = f"  [{j['progress'] * 100:5.1f}%] {j['stage']:10s} {j['message']}"
        if line != last:
            print(line, flush=True)
            last = line
        if j["status"] in ("done", "error", "cancelled"):
            print(f"  -> {label}: {j['status']}  ({time.time() - t0:.1f}s)")
            if j["status"] == "error":
                print(j.get("error") or j.get("message"))
            return j
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{label} 超时")
        time.sleep(0.7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default="balanced")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    print("[1] 健康检查")
    print("   ", call("/api/health"))

    print("[2] 新建工程")
    proj = call("/api/projects", "POST", {"name": "E2E 测试工程"})
    pid = proj["id"]
    print("    project:", pid)

    print("[3] 导入素材")
    res = call(f"/api/projects/{pid}/media", "POST", {"paths": [args.video]})
    if res["failed"]:
        print("    导入失败:", res["failed"])
        return 1
    mid = res["added"][0]["id"]
    m = res["added"][0]
    print(f"    {m['name']}  {m['width']}x{m['height']}  {m['duration']:.1f}s  {m['fps']:.2f}fps  audio={m['has_audio']}")

    print("[4] 准备派生资源（代理视频 / 音轨）")
    j = call(f"/api/projects/{pid}/media/{mid}/prepare", "POST", {})
    wait_job(j["job_id"], "prepare")

    print("[5] AI 分析")
    j = call(
        f"/api/projects/{pid}/analyze",
        "POST",
        {"media_id": mid, "weights": args.weights, "params": {"sample_fps": 15}},
    )
    jr = wait_job(j["job_id"], "analyze")
    if jr["status"] != "done":
        return 1

    ana = call(f"/api/projects/{pid}/analysis/{mid}")
    st = ana["stats"]
    print(f"    回合 {st['count']}  有效时长 {st['active_duration']:.1f}s  总拍数 {st['total_shots']}  平均分 {st['avg_score']:.1f}")
    print(f"    模块: 击球={st.get('hit_trace')} 球员={st.get('player_trace')} 羽毛球={st.get('shuttle_trace')}")
    for r in ana["rallies"][:12]:
        print(f"      #{r['index']:3d} {r['start']:7.2f}-{r['end']:7.2f} {r['duration']:6.2f}s "
              f"{r['features']['shot_count']:3d}拍 {r['scores']['total']:5.1f}分  {','.join(r['tags'])}")

    print("[6] 按评分筛选并自动剪辑")
    body: dict = {"media_id": mid, "mode": "replace"}
    if args.min_score > 0:
        body["filter"] = {"min_score": args.min_score}
    tl = call(f"/api/projects/{pid}/timeline/auto-cut", "POST", body)
    print(f"    片段数 {tl['clip_count']}  时间线时长 {tl['timeline']['duration']:.1f}s")
    if tl["clip_count"] == 0:
        print("    !! 没有生成任何片段")
        return 1

    if args.no_export:
        print("[7] 跳过导出")
        return 0

    print("[7] 导出 1080p（NVENC）")
    presets = call("/api/export/presets")
    preset = next(p for p in presets if p["id"] == "yt1080p")
    j = call(f"/api/projects/{pid}/export", "POST", {"preset": preset, "name": f"e2e_{int(time.time())}"})
    jr = wait_job(j["job_id"], "export")
    if jr["status"] != "done":
        return 1
    result = jr["result"]
    print(f"    输出 {result['path']}  {result['size'] / 1e6:.1f} MB")

    print("[8] 导出记录")
    for f in call("/api/exports")[:5]:
        print(f"    {f['name']}  {f['size'] / 1e6:.1f} MB")
    print("\n全部通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
