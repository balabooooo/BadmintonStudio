"""Decompress the current DSH session transcript and pull out pipeline.py content."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SESS = Path(r"C:\Users\13127\.dsh\sessions\--D-Projects-Dsh--"
            r"\session-11db2bbf-6127-404b-9036-69ca390b4de3\session.jsonl.zstd")
OUT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames")
REPORT = OUT / "transcript_report.txt"

lines: list[str] = []


def say(s: str) -> None:
    lines.append(s)


def decompress(raw: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
        return zstd.decompress(raw)
    except Exception as e1:
        say(f"compression.zstd 不可用: {e1}")
    try:
        import zstandard  # type: ignore
        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30)
    except Exception as e2:
        say(f"zstandard 不可用: {e2}")
    raise RuntimeError("没有可用的 zstd 解码器")


raw = SESS.read_bytes()
say(f"session file {SESS.stat().st_size} bytes")
try:
    data = decompress(raw)
except Exception as e:
    say(f"解压失败: {e}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)
OUT.joinpath("session.jsonl").write_bytes(data)
say(f"解压后 {len(data)} bytes, 行数 {data.count(10) + 1}")

# 找出所有包含 pipeline.py 行号标记的记录，拼出各处 read 的结果
text = data.decode("utf-8", errors="replace")
chunks = re.findall(r"Showing lines \d+-\d+ of \d+", text)
say(f"含 'Showing lines a-b of N' 的片段 {len(chunks)} 处：")
for c in sorted(set(chunks)):
    say(f"   {c}")

REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"written {REPORT}")
