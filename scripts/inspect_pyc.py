"""Compare the .pyc's line table against the corrupted source to see what was lost."""

from __future__ import annotations

import dis
import importlib.util
import marshal
import sys
from pathlib import Path

PYC = Path(r"D:\Projects\BadmintonStudio\backend\bms\analysis\__pycache__\pipeline.cpython-314.pyc")
BROKEN = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
REPORT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pyc_report.txt")

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


raw = PYC.read_bytes()
say(f"pyc header: {raw[:16]!r}")
code = marshal.loads(raw[16:])
say(f"module code name={code.co_name!r} firstlineno={code.co_firstlineno}")


def walk(c, depth=0):
    yield c
    for k in c.co_consts:
        if hasattr(k, "co_lines"):
            yield from walk(k, depth + 1)


maxline = 0
funcs: list[tuple[int, int, str]] = []
for c in walk(code):
    lines = {ln for _s, _e, ln in c.co_lines() if ln is not None}
    if lines:
        lo, hi = min(lines), max(lines)
        maxline = max(maxline, hi)
        if c.co_name != "<module>":
            funcs.append((lo, hi, c.co_name))

say(f"pyc 里出现的最大行号 = {maxline}")
say(f"pyc 里的顶层函数数量 = {len(funcs)}")

text = BROKEN.read_text(encoding="utf-8")
say(f"损坏文件的最后一个行号 = {text.count(chr(10)) + 1}")

say("")
say("pyc 记录的函数定义起始行（前 30 个，按行号排序）:")
for lo, hi, name in sorted(funcs)[:30]:
    say(f"   {lo:5d} ~ {hi:5d}   {name}")

# 收集所有字符串常量，统计有多少中文
strings: list[str] = []


def collect(c):
    for k in c.co_consts:
        if isinstance(k, str) and k:
            strings.append(k)
        elif hasattr(k, "co_consts"):
            collect(k)


collect(code)
cjk = sum(1 for s in strings for ch in s if 0x4E00 <= ord(ch) <= 0x9FFF)
say("")
say(f"pyc 里字符串常量 {len(strings)} 个，其中汉字 {cjk} 个")

# 关键：pyc 里还有没有 hit_times=(hits.times ...) 的痕迹？
say("")
say("pyc 里包含 'hits.times if hits is not None' 的常量（说明坏文件丢的是哪一版代码）:")
for s in strings:
    if "hits.times" in s:
        say(f"   {s[:120]!r}")

REPORT.write_text("\n".join(out), encoding="utf-8")
print(f"written {REPORT}")
