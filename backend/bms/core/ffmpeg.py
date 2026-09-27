"""FFmpeg / FFprobe location and command execution wrapper.

Priority:
1. Environment variables ``BMS_FFMPEG`` / ``BMS_FFPROBE``
2. In-project ``tools/ffmpeg/bin``
3. The static build bundled with ``imageio-ffmpeg``
4. System PATH
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..config import TOOLS_DIR
from ..i18n import tr

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_ffmpeg_cache: str | None = None
_ffprobe_cache: str | None = None


def _candidate_dirs() -> list[Path]:
    return [TOOLS_DIR / "ffmpeg" / "bin", TOOLS_DIR / "bin", TOOLS_DIR]


def find_ffmpeg() -> str:
    global _ffmpeg_cache
    if _ffmpeg_cache:
        return _ffmpeg_cache

    env = os.environ.get("BMS_FFMPEG")
    if env and Path(env).is_file():
        _ffmpeg_cache = env
        return env

    for d in _candidate_dirs():
        for name in ("ffmpeg.exe", "ffmpeg"):
            p = d / name
            if p.is_file():
                _ffmpeg_cache = str(p)
                return _ffmpeg_cache

    try:  # static build bundled with imageio-ffmpeg
        import imageio_ffmpeg

        _ffmpeg_cache = imageio_ffmpeg.get_ffmpeg_exe()
        return _ffmpeg_cache
    except Exception:  # pragma: no cover - fallback when the dependency is missing
        pass

    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_cache = found
        return _ffmpeg_cache

    raise FileNotFoundError(tr("ffmpeg.not_found"))


def find_ffprobe() -> str | None:
    global _ffprobe_cache
    if _ffprobe_cache:
        return _ffprobe_cache
    env = os.environ.get("BMS_FFPROBE")
    if env and Path(env).is_file():
        _ffprobe_cache = env
        return env
    for d in _candidate_dirs():
        for name in ("ffprobe.exe", "ffprobe"):
            p = d / name
            if p.is_file():
                _ffprobe_cache = str(p)
                return _ffprobe_cache
    ff = Path(find_ffmpeg())
    sibling = ff.with_name("ffprobe.exe" if ff.suffix == ".exe" else "ffprobe")
    if sibling.is_file():
        _ffprobe_cache = str(sibling)
        return _ffprobe_cache
    found = shutil.which("ffprobe")
    _ffprobe_cache = found
    return found


def caps() -> dict[str, bool]:
    """Detect which capabilities the current ffmpeg build supports."""
    out = run([find_ffmpeg(), "-hide_banner", "-encoders"], capture=True).stdout
    result = {
        "nvenc_h264": "h264_nvenc" in out,
        "nvenc_hevc": "hevc_nvenc" in out,
        "qsv": "h264_qsv" in out,
        "libx264": "libx264" in out,
        "libx265": "libx265" in out,
    }
    dec = run([find_ffmpeg(), "-hide_banner", "-hwaccels"], capture=True).stdout
    result["cuda_decode"] = "cuda" in dec
    result["d3d11va"] = "d3d11va" in dec
    return result


# ------------------------------------------------------------------ Process execution


@dataclass
class ProcResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    ok: bool = field(default=True)

    def __post_init__(self) -> None:
        self.ok = self.returncode == 0


def _popen(cmd: Sequence[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(
        [str(c) for c in cmd],
        stdout=kw.pop("stdout", subprocess.PIPE),
        stderr=kw.pop("stderr", subprocess.PIPE),
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        **kw,
    )


def run(cmd: Sequence[str], capture: bool = True, check: bool = False) -> ProcResult:
    p = _popen(
        cmd,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    out, err = p.communicate()
    res = ProcResult(p.returncode, out or "", err or "")
    if check and not res.ok:
        raise RuntimeError(tr("ffmpeg.command_failed", code=res.returncode,
                              cmd=" ".join(map(str, cmd)), stderr=res.stderr[-4000:]))
    return res


_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_OUTTIME_RE = re.compile(r"out_time_us=(\d+)")


def run_with_progress(
    cmd: Sequence[str],
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    log_tail: int = 8000,
) -> ProcResult:
    """Run ffmpeg and parse the ``-progress`` output to report 0~1 progress."""
    p = _popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail: list[str] = []
    try:
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\r\n")
            if line:
                tail.append(line)
                if len(tail) > 400:
                    del tail[:200]
            m = _OUTTIME_RE.search(line)
            if m and total_seconds > 0 and on_progress:
                on_progress(min(1.0, (int(m.group(1)) / 1_000_000) / total_seconds))
            elif on_progress and total_seconds > 0:
                m2 = _TIME_RE.search(line)
                if m2:
                    secs = int(m2.group(1)) * 3600 + int(m2.group(2)) * 60 + float(m2.group(3))
                    on_progress(min(1.0, secs / total_seconds))
            if cancel and cancel():
                p.kill()
                break
    finally:
        p.wait()
    text = "\n".join(tail)
    return ProcResult(p.returncode, text[-log_tail:], "")


def probe_json(path: str | Path) -> dict:
    """Read media information with ffprobe."""
    exe = find_ffprobe()
    if not exe:
        raise FileNotFoundError(tr("ffmpeg.ffprobe_not_found"))
    res = run(
        [
            exe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if not res.ok:
        raise RuntimeError(tr("ffmpeg.probe_failed", stderr=res.stderr[-2000:]))
    return json.loads(res.stdout or "{}")


def ffconcat_escape(path: str) -> str:
    return str(path).replace("\\", "/").replace("'", "'\\''")
