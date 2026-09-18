"""Inspect how tool results are stored in the session transcript."""

from __future__ import annotations

import json
from pathlib import Path

SESSION = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\session.jsonl")
REPORT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\shape_report.txt")

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


shown = 0
hit_kinds: dict[str, int] = {}
for i, raw in enumerate(SESSION.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
    if "pipeline.py" not in raw:
        continue
    try:
        rec = json.loads(raw)
    except Exception:
        continue
    kind = str(rec.get("type"))
    hit_kinds[kind] = hit_kinds.get(kind, 0) + 1
    if shown >= 3:
        continue
    shown += 1
    say(f"--- line {i}  type={kind}  top-level keys={list(rec.keys())}")
    data = rec.get("data")
    if isinstance(data, dict):
        say(f"    data keys={list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 200:
                say(f"      {k}: <str {len(v)} chars> head={v[:160]!r}")
            elif isinstance(v, list):
                say(f"      {k}: <list {len(v)}>")
                if v and isinstance(v[0], dict):
                    say(f"        [0] keys={list(v[0].keys())}")
            else:
                say(f"      {k}: {v!r}"[:200])
    elif isinstance(data, str):
        say(f"    data: <str {len(data)}> head={data[:160]!r}")
    else:
        say(f"    data: {type(data).__name__}")

say("")
say("含 pipeline.py 的记录按 type 统计:")
for k, v in sorted(hit_kinds.items(), key=lambda kv: -kv[1]):
    say(f"   {k}: {v}")

REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
