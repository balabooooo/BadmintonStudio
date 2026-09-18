"""Put a known snapshot line next to its corrupted counterpart, verbatim."""

from __future__ import annotations

from pathlib import Path

CORRUPT = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\pipeline_broken.py")
SNAP = Path(r"D:\Projects\BadmintonStudio\data\cache\frames\snap")
REPORT = CORRUPT.parent / "sidebyside.txt"


def skel(s: str) -> str:
    return "".join("\x00" if ord(c) > 127 else c for c in s)


corrupt_lines = CORRUPT.read_text(encoding="utf-8").lstrip("\ufeff").splitlines()
corrupt_skel = [skel(x) for x in corrupt_lines]

out: list[str] = []


def say(s: str) -> None:
    out.append(s)


# 从快照里挑几行「中文密集」的行，在损坏文件里按 ASCII 骨架找对应
picked = 0
for p in sorted(SNAP.glob("snap_*.txt")):
    for ln in p.read_text(encoding="utf-8").splitlines():
        cn = sum(1 for c in ln if ord(c) > 127)
        if cn < 12:
            continue
        s = skel(ln)
        if len(s.strip()) < 20:
            continue
        for i, cs in enumerate(corrupt_skel):
            if cs.strip() == s.strip():
                say(f"### {p.name}")
                say(f"  原始 UTF-8 字节 : {ln.encode('utf-8').hex()}")
                say(f"  原始文本       : {ln!r}")
                say(f"  损坏文本       : {corrupt_lines[i]!r}")
                say(f"  损坏字节       : {corrupt_lines[i].encode('utf-8').hex()}")
                say(f"  原始非ASCII数  : {cn}   损坏非ASCII数: "
                    f"{sum(1 for c in corrupt_lines[i] if ord(c) > 127)}")
                say("")
                picked += 1
                break
        if picked >= 6:
            break
    if picked >= 6:
        break

say(f"共展示 {picked} 组")
REPORT.write_text("\n".join(out), encoding="utf-8")
print("written")
