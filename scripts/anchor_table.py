"""Anchor table: .pyc (pre-corruption) line numbers vs damaged-file line numbers.

Emits a text report pairing each top-level definition with its line number in the
compiled original and in the damaged source, so the offset per region is explicit.
"""

from __future__ import annotations

import marshal
import re
from pathlib import Path

PYC = Path(r"D:\Projects\BadmintonStudio\backend\bms\analysis\__pycache__"
           r"\pipeline.cpython-314.pyc")
DAMAGED = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
REPORT = DAMAGED.parent / "anchor_table.txt"

raw = PYC.read_bytes()
code = marshal.loads(raw[16:])

pyc_defs: list[tuple[int, str]] = []
for k in code.co_consts:
    if hasattr(k, "co_firstlineno") and k.co_name not in ("__annotate__",):
        pyc_defs.append((k.co_firstlineno, k.co_name))
pyc_defs.sort()

text = DAMAGED.read_text(encoding="utf-8").lstrip("\ufeff")
lines = text.splitlines()
dam_defs: list[tuple[int, str]] = []
for i, ln in enumerate(lines, 1):
    m = re.match(r"^(?:def|class)\s+(\w+)", ln)
    if m:
        dam_defs.append((i, m.group(1)))
    m2 = re.match(r"^@dataclass", ln)
    if m2:
        pass

by_name: dict[str, list[int]] = {}
for n, name in dam_defs:
    by_name.setdefault(name, []).append(n)

out: list[str] = []
out.append(f"{'函数':38s} {'原(.pyc)':>9s} {'损坏文件':>9s} {'偏移':>6s}")
out.append("-" * 70)
offsets: list[int] = []
for ln, name in pyc_defs:
    cand = by_name.get(name)
    if not cand:
        out.append(f"{name:38s} {ln:9d} {'—':>9s} {'—':>6s}")
        continue
    d = cand[0]
    off = d - ln
    offsets.append(off)
    out.append(f"{name:38s} {ln:9d} {d:9d} {off:6d}")

out.append("")
out.append(f"原文件最大行号（.pyc）= 1477，损坏文件行数 = {len(lines)}")
out.append(f"被删除的换行数 ≈ 1477 - {len(lines)} = {1477 - len(lines)}")
out.append(f"偏移范围 {min(offsets)} .. {max(offsets)}（负值 = 该处之前已丢行）")

REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
