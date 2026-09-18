"""Derive the mojibake char -> original-bytes mapping from the transcript snapshots.

The corrupted file is ``original_utf8_bytes.decode(SOMECODEPAGE)`` written back as
UTF-8.  For an aligned run of original Chinese (N chars -> 3N utf-8 bytes) the
mojibake run should have 3N/2 chars, i.e. len(utf8) == 2 * len(mojibake).  Each
mojibake char then corresponds to exactly two of those bytes -- that is the
mapping we want.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SNAP = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\snap")
CORRUPT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
REPORT = CORRUPT.parent / "mapping_report.txt"
MAPFILE = CORRUPT.parent / "mojibake_map.json"

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


corrupt = CORRUPT.read_text(encoding="utf-8").lstrip("\ufeff")
corrupt_skeleton = "".join("\x00" if ord(c) > 127 else c for c in corrupt)

# 把所有非 ASCII 连续段抽出来
def runs(text: str) -> list[tuple[int, int, str]]:
    res = []
    i = 0
    n = len(text)
    while i < n:
        if ord(text[i]) > 127:
            j = i
            while j < n and ord(text[j]) > 127:
                j += 1
            res.append((i, j, text[i:j]))
            i = j
        else:
            i += 1
    return res


# 把损坏文件里的非 ASCII 段建索引：ASCII 前缀 -> 段内容
by_prefix: dict[str, list[str]] = {}
for a, b, seg in runs(corrupt):
    key = corrupt_skeleton[max(0, a - 24):a]
    by_prefix.setdefault(key, []).append(seg)

mapping: dict[str, bytes] = {}
conflicts = 0
pairs_used = 0
for p in sorted(SNAP.glob("snap_*.txt")):
    text = p.read_text(encoding="utf-8")
    for a, b, seg in runs(text):
        key = "".join("\x00" if ord(c) > 127 else c for c in text[max(0, a - 24):a])
        cands = by_prefix.get(key)
        if not cands:
            continue
        # 只接受「UTF-8 字节数正好是损坏段两倍」的候选
        nb = len(seg.encode("utf-8"))
        for c in cands:
            if len(c) * 2 != nb:
                continue
            pairs_used += 1
            raw = seg.encode("utf-8")
            for k, ch in enumerate(c):
                b2 = raw[2 * k:2 * k + 2]
                old = mapping.get(ch)
                if old is None:
                    mapping[ch] = b2
                elif old != b2:
                    conflicts += 1
            break

say(f"对齐并采纳的非 ASCII 段 {pairs_used} 个")
say(f"推出映射 {len(mapping)} 条，冲突 {conflicts} 条")

# 损坏文件里出现的全部非 ASCII 字符，看覆盖率
need = Counter(c for c in corrupt if ord(c) > 127)
covered = sum(v for k, v in need.items() if k in mapping)
say(f"损坏文件非 ASCII 字符 {sum(need.values())} 个，其中已有映射 {covered} "
    f"({covered / max(1, sum(need.values())) * 100:.2f}%)")
missing = [(k, v) for k, v in need.items() if k not in mapping]
say(f"缺失 {len(missing)} 种，累计出现 {sum(v for _k, v in missing)} 次")
say("缺失最多的 40 种（码点:次数）:")
for k, v in missing[:40]:
    say(f"   U+{ord(k):04X} : {v}")

MAPFILE.write_text(json.dumps({k: v.hex() for k, v in mapping.items()}, ensure_ascii=False),
                   encoding="utf-8")
REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
