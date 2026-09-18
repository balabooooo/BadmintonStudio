"""Derive the mojibake decode table, matching whitespace-insensitively.

The corruption deleted newlines, so a literal's ASCII skeleton no longer appears
verbatim.  We strip ALL whitespace from both sides and search in the collapsed
text, keeping an index map back to the original positions.
"""

from __future__ import annotations

import json
import marshal
from pathlib import Path

PYC = Path(r"D:\Projects\BadmintonStudio\backend\bms\analysis\__pycache__"
           r"\pipeline.cpython-314.pyc")
CORRUPT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
OUTDIR = CORRUPT.parent
REPORT = OUTDIR / "decode_report2.txt"
MAPFILE = OUTDIR / "decode_map.json"

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


def is_ws(c: str) -> bool:
    return c in " \t\r\n"


def skel_nospace(s: str) -> tuple[str, list[int]]:
    """返回 (去掉空白的骨架, 每个骨架字符在原串里的位置)。"""
    buf: list[str] = []
    idx: list[int] = []
    for i, c in enumerate(s):
        if is_ws(c):
            continue
        buf.append("\x00" if ord(c) > 127 else c)
        idx.append(i)
    return "".join(buf), idx


raw = PYC.read_bytes()
code = marshal.loads(raw[16:])
strings: list[str] = []


def walk(c):
    for k in c.co_consts:
        if isinstance(k, str) and len(k) > 8:
            strings.append(k)
        elif hasattr(k, "co_consts"):
            walk(k)


walk(code)
say(f".pyc 字符串常量（>8 字符）{len(strings)} 个")

corrupt = CORRUPT.read_text(encoding="utf-8").lstrip("\ufeff")
cs, cidx = skel_nospace(corrupt)

mapping: dict[str, bytes] = {}
conflict: dict[str, set] = {}
lit_hits = 0
seg_pairs = 0

for lit in strings:
    if not any(ord(c) > 127 for c in lit):
        continue
    ls, lidx = skel_nospace(lit)
    if len(ls) < 8:
        continue
    pos = cs.find(ls)
    if pos < 0:
        continue
    lit_hits += 1
    # 逐字符走一遍：ASCII 骨架字符之间出现的非 ASCII 段要成对比较
    c_start = cidx[pos]
    c_end = cidx[pos + len(ls) - 1]
    # 在原串与损坏串上同时扫描非 ASCII 段
    def segments(text: str, lo: int, hi: int):
        res = []
        i = lo
        while i <= hi:
            if ord(text[i]) > 127:
                j = i
                while j <= hi and ord(text[j]) > 127:
                    j += 1
                res.append(text[i:j])
                i = j
            else:
                i += 1
        return res

    a = segments(lit, lidx[0], lidx[len(ls) - 1])
    b = segments(corrupt, c_start, c_end)
    if len(a) != len(b):
        continue
    for orig, moi in zip(a, b):
        nb = len(orig.encode("utf-8"))
        if len(moi) * 2 != nb:
            continue
        seg_pairs += 1
        rb = orig.encode("utf-8")
        for k, ch in enumerate(moi):
            bb = rb[2 * k:2 * k + 2]
            if ch in mapping and mapping[ch] != bb:
                conflict.setdefault(ch, {mapping[ch]}).add(bb)
            else:
                mapping.setdefault(ch, bb)

say(f"匹配上的字面量 {lit_hits} 个；成对非 ASCII 段 {seg_pairs} 个")
say(f"推出映射 {len(mapping)} 条；冲突字符 {len(conflict)} 条")

need: dict[str, int] = {}
for ch in corrupt:
    if ord(ch) > 127:
        need[ch] = need.get(ch, 0) + 1
total = sum(need.values())
good = {k: v for k, v in mapping.items() if k not in conflict}
covered = sum(v for k, v in need.items() if k in good)
say(f"损坏文件非 ASCII 字符 {total} 个；可用映射覆盖 {covered} "
    f"({covered / max(1, total) * 100:.2f}%)")
missing = sorted(((k, v) for k, v in need.items() if k not in good), key=lambda kv: -kv[1])
say(f"未覆盖字符 {len(missing)} 种，出现 {sum(v for _k, v in missing)} 次")
say("出现最多的 25 个:")
for k, v in missing[:25]:
    say(f"   U+{ord(k):04X} x{v}")

MAPFILE.write_text(json.dumps({k: v.hex() for k, v in good.items()}, ensure_ascii=False),
                   encoding="utf-8")
REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
