"""Integrity check across the whole backend after the encoding accident."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(r"D:\Projects\BadmintonStudio")
REPORT = ROOT / "data" / "cache" / "frames" / "integrity.txt"

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


# 判定「是不是莫吉贝克」：一段文本里 CJK 字符比例高但整段读起来不是中文
def mojibake_score(text: str) -> float:
    # 典型的 GBK 误解码会出现大量生僻字与 PUA；用「生僻/罕见块」占比近似
    rare = 0
    total = 0
    for ch in text:
        o = ord(ch)
        if o < 128:
            continue
        total += 1
        # 常见中文常用字区间窄；莫吉贝克会大量落在 U+9000 以上与 PUA
        if o >= 0x9000 or 0xE000 <= o <= 0xF8FF or 0x2000 <= o <= 0x2FFF:
            rare += 1
    return rare / total if total else 0.0


files = sorted((ROOT / "backend").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")) \
    + sorted((ROOT / "tests").rglob("*.py"))

bad: list[str] = []
say(f"{'文件':60s} {'行数':>6s} {'语法':>6s} {'莫吉贝克':>8s}")
for p in files:
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        say(f"{str(p.relative_to(ROOT)):60s} {'':>6s} {'非UTF8':>6s}  {e}")
        bad.append(str(p))
        continue
    n = text.count("\n") + 1
    try:
        ast.parse(text)
        syn = "OK"
    except SyntaxError as e:
        syn = f"E{e.lineno}"
        bad.append(str(p))
    ms = mojibake_score(text)
    flag = "  <== 可疑" if ms > 0.25 else ""
    say(f"{str(p.relative_to(ROOT)):60s} {n:6d} {syn:>6s} {ms:8.3f}{flag}")

say("")
say("有问题的文件:")
for b in bad:
    say(f"   {b}")

REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
