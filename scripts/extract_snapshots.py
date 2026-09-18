"""Extract every pipeline.py read-snapshot from the session transcript.

Tool result text has the shape::

    <path>D:\\...\\pipeline.py</path>
    <type>file</type>
    <content>
    300: text of line 300
    301: ...
    </content>
    (Showing lines 300-499 of 1217)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SESSION = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\session.jsonl")
OUT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\snap")
OUT.mkdir(parents=True, exist_ok=True)

path_re = re.compile(r"<path>(?P<p>[^<]+)</path>")
body_re = re.compile(r"<content>\n(?P<body>.*?)\n?</content>", re.S)
tail_re = re.compile(r"\(Showing lines (?P<a>\d+)-(?P<b>\d+) of (?P<n>\d+)")
line_re = re.compile(r"^(?P<num>\d+): ?(?P<text>.*)$")

index: list[str] = []
snaps: list[tuple[Path, int, int, int, dict[int, str]]] = []

for raw in SESSION.read_text(encoding="utf-8", errors="replace").splitlines():
    if "pipeline.py" not in raw:
        continue
    try:
        rec = json.loads(raw)
    except Exception:
        continue
    if rec.get("type") != "tool/result":
        continue
    try:
        for item in rec["data"]["message"]["content"]:
            for sub in item.get("content", []):
                text = sub.get("text") or ""
                if "pipeline.py" not in text:
                    continue
                pm = path_re.search(text)
                if not pm or not pm.group("p").endswith("pipeline.py"):
                    continue
                bm = body_re.search(text)
                if not bm:
                    continue
                texts: dict[int, str] = {}
                for ln in bm.group("body").split("\n"):
                    lm = line_re.match(ln.rstrip("\r"))
                    if lm:
                        texts[int(lm.group("num"))] = lm.group("text")
                if not texts:
                    continue
                tm = tail_re.search(text)
                a, b = min(texts), max(texts)
                n_total = int(tm.group("n")) if tm else 0
                snaps.append((SESSION, a, b, n_total, texts))
    except Exception:
        continue

snaps.sort(key=lambda s: (s[1], s[3]))
for i, (_p, a, b, n_total, texts) in enumerate(snaps):
    body = [texts.get(k, "") for k in range(a, b + 1)]
    name = f"snap_{i:02d}_L{a:04d}-{b:04d}_of{n_total}.txt"
    (OUT / name).write_text("\n".join(body), encoding="utf-8")
    index.append(f"{name:46s} {a:5d}-{b:<5d} of {n_total}")

(OUT.parent / "snapshot_index.txt").write_text("\n".join(index), encoding="utf-8")
print(f"{len(snaps)} snapshots")
