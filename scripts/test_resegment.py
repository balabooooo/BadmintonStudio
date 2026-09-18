"""快速重切分验证：调切分粒度后即时重建回合，并检查相邻回合是否还有重叠。

用法:
    python scripts/test_resegment.py <project_id> [--sensitivity 0.5]
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
    ap.add_argument("--sensitivity", type=float, default=None)
    args = ap.parse_args()

    proj = call(f"/api/projects/{args.project}")
    if not proj["media"]:
        print("工程没有素材")
        return 1
    mid = proj["media"][0]["id"]

    for sens in ([args.sensitivity] if args.sensitivity is not None else [0.15, 0.5, 0.85]):
        params = {"split_sensitivity": sens}
        t0 = time.time()
        res = call(f"/api/projects/{args.project}/resegment", "POST",
                   {"media_id": mid, "params": params})
        el = time.time() - t0
        rs = res["rallies"]
        ov = sum(1 for a, b in zip(rs, rs[1:]) if b["start"] < a["end"] - 0.01)
        durs = [r["duration"] for r in rs]
        avg = sum(durs) / max(1, len(durs))
        print(f"sensitivity={sens:<5} {el:5.2f}s  回合数={len(rs):3d}  平均时长={avg:5.1f}s  "
              f"最长={max(durs) if durs else 0:5.1f}s  重叠对={ov}  "
              f"占比={sum(durs) / max(1, res['stats']['duration']) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
