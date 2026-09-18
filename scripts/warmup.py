"""为一段长视频建工程并只做「准备派生资源」（生成代理视频/音轨/封面）。

长视频（4K 30 分钟级）这一步可能跑十几分钟，所以单独拆出来，
之后反复调参做分析时就能复用缓存。

用法:
    python scripts/warmup.py "<video>" [--name 工程名] [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"


def call(path: str, method: str = "GET", body: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", default="")
    ap.add_argument("--project", default="", help="已有工程 id，直接复用")
    args = ap.parse_args()

    name = args.name or Path(args.video).stem
    if args.project:
        pid = args.project
    else:
        # 复用同名工程，避免重复建
        for p in call("/api/projects"):
            if p["name"] == name:
                pid = p["id"]
                break
        else:
            pid = call("/api/projects", "POST", {"name": name})["id"]
    print("project:", pid)

    proj = call(f"/api/projects/{pid}")
    mid = next((m["id"] for m in proj["media"] if m["path"] == args.video), None)
    if mid is None:
        res = call(f"/api/projects/{pid}/media", "POST", {"paths": [args.video]})
        if res["failed"]:
            print("导入失败:", res["failed"])
            return 1
        mid = res["added"][0]["id"]
        m = res["added"][0]
        print(f"media: {m['name']}  {m['width']}x{m['height']}  {m['duration']:.1f}s  {m['fps']:.2f}fps")
    else:
        print("media: 复用已有条目", mid)

    j = call(f"/api/projects/{pid}/media/{mid}/prepare", "POST", {})
    jid = j["job_id"]
    t0 = time.time()
    last = ""
    while True:
        st = call(f"/api/jobs/{jid}")
        line = f"[{st['progress'] * 100:5.1f}%] {st['stage']:8s} {st['message']}"
        if line != last:
            print(line, flush=True)
            last = line
        if st["status"] in ("done", "error", "cancelled"):
            print(f"-> {st['status']}  ({time.time() - t0:.1f}s)")
            if st["status"] == "error":
                print(st.get("error"))
                return 1
            break
        time.sleep(1.5)

    proj = call(f"/api/projects/{pid}")
    m = next(x for x in proj["media"] if x["id"] == mid)
    print("\n就绪:")
    print("  proxy :", m["proxy_path"])
    print("  audio :", m["audio_path"])
    print("  poster:", m["poster"])
    print(f"\n工程 id = {pid}   素材 id = {mid}")
    Path("data/last_project.txt").write_text(f"{pid}\n{mid}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
