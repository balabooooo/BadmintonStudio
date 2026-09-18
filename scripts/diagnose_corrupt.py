"""Diagnose the corrupted pipeline.py without relying on console output.

Writes a plain-text report next to the file so it can be read back verbatim.
"""

from __future__ import annotations

import collections
from pathlib import Path

BROKEN = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
REPORT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\recover_report.txt")

lines: list[str] = []


def say(s: str) -> None:
    lines.append(s)


raw = BROKEN.read_bytes()
say(f"bytes={len(raw)}  bom={raw[:3]!r}")
text = raw.decode("utf-8")
say(f"decoded utf-8 ok, chars={len(text)}")
say(f"U+FFFD count={text.count(chr(0xFFFD))}")
say(f"newlines={text.count(chr(10))}")

non_ascii = [c for c in text if ord(c) > 127]
say(f"non-ascii chars={len(non_ascii)} distinct={len(set(non_ascii))}")
blocks = collections.Counter()
for c in set(non_ascii):
    o = ord(c)
    blocks[o >> 8] += 1
say("codepoint high-byte histogram (hex block -> distinct chars):")
for k in sorted(blocks):
    say(f"   U+{k:02X}xx : {blocks[k]}")

say("")
say("first 6 lines with repr (non-ascii shown as \\uXXXX, ascii kept):")
for i, ln in enumerate(text.splitlines()[:6], 1):
    safe = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in ln)
    say(f"  {i:3d}| {safe[:150]}")

say("")
say("scan for the literal 'hit_times' occurrences and their context:")
for i, ln in enumerate(text.splitlines(), 1):
    if "hit_times" in ln:
        safe = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in ln)
        say(f"  {i:5d}| {safe[:160]}")

say("")
say("tail 5 lines:")
for i, ln in enumerate(text.splitlines()[-5:], len(text.splitlines()) - 4):
    safe = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in ln)
    say(f"  {i:5d}| {safe[:150]}")

REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"report written to {REPORT}")
