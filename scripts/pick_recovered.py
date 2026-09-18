"""Pick the recovered candidate that is valid UTF-8 Python and looks like pipeline.py."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(r"D:\Projects\BadmintonStudio\backend\bms\analysis\pipeline.py")
DIR = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\recover")

want = ["def run_analysis", "def _segment_rallies", "def _merge_by_availability",
        "def resegment", "def _pack_signals"]

best = None
for p in sorted(DIR.glob("recovered_cp*.py")):
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"{p.name:24s} 不是合法 UTF-8: {e}")
        continue
    hits = sum(1 for w in want if w in text)
    try:
        ast.parse(text)
        ok = "AST OK"
    except SyntaxError as e:
        ok = f"SyntaxError line {e.lineno}"
    n_bad = text.count("\ufffd")
    print(f"{p.name:24s} 行数 {text.count(chr(10)) + 1:5d}  关键函数 {hits}/{len(want)}  "
          f"替换符 {n_bad}  {ok}")
    if ok == "AST OK" and hits == len(want):
        best = text

if best is None:
    print("\n没有找到完全恢复的候选")
    sys.exit(1)

# 备份现状，然后写回
if SRC.exists():
    (SRC.parent / "pipeline.py.corrupt").write_bytes(SRC.read_bytes())
SRC.write_text(best, encoding="utf-8")
print(f"\n已恢复并写回 {SRC}")
print("前 3 行：")
for line in best.splitlines()[:3]:
    print("   ", line[:100])
