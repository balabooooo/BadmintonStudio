"""Walk tool/result records and dump their nested content shape."""

from __future__ import annotations

import json
from pathlib import Path

SESSION = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\session.jsonl")
REPORT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\shape2_report.txt")

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


shown = 0
for i, raw in enumerate(SESSION.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
    if "Showing lines" not in raw:
        continue
    rec = json.loads(raw)
    if rec.get("type") != "tool/result":
        continue
    shown += 1
    if shown > 2:
        continue
    msg = rec["data"]["message"]
    say(f"--- line {i}  message keys={list(msg.keys())}")
    cont = msg["content"]
    say(f"    content is list len={len(cont)}; [0] keys={list(cont[0].keys())}")
    inner = cont[0].get("content")
    say(f"    inner type={type(inner).__name__}")
    if isinstance(inner, list):
        say(f"      inner len={len(inner)}; [0] keys={list(inner[0].keys()) if inner else []}")
        for j, it in enumerate(inner[:3]):
            say(f"      inner[{j}] = {json.dumps(it, ensure_ascii=False)[:400]}")
    else:
        say(f"      inner = {json.dumps(inner, ensure_ascii=False)[:400]}")
say("")
say(f"含 'Showing lines' 的 tool/result 记录数 = {shown}")
REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
