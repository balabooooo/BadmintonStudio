"""导出指定预设并等待完成（用于验证竖屏自动跟随裁切等路径）。

用法:
    python scripts/test_export.py <project_id> <preset_id> [--name out]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("preset")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    presets = call("/api/export/presets")
    preset = next((p for p in presets if p["id"] == args.preset), None)
    if preset is None:
        print("未知预设:", args.preset, [p["id"] for p in presets])
        return 1
    print("preset:", preset["name"], preset["width"], "x", preset["height"],
          "auto_reframe =", preset["auto_reframe"])

    name = args.name or f"test_{args.preset}_{int(time.time())}"
    j = call(f"/api/projects/{args.project}/export", "POST", {"preset": preset, "name": name})
    print("job:", j["job_id"])

    t0 = time.time()
    last = ""
    while True:
        s = call(f"/api/jobs/{j['job_id']}")
        line = "{:5.1f}% {}".format(s["progress"] * 100, s["message"])
        if line != last:
            print(line, flush=True)
            last = line
        if s["status"] in ("done", "error", "cancelled"):
            print("->", s["status"], "{:.0f}s".format(time.time() - t0))
            if s["status"] == "error":
                print((s.get("error") or "")[-2000:])
                return 1
            r = s["result"]
            print("输出:", r["path"], round(r["size"] / 1e6, 1), "MB")
            return 0
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
