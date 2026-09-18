"""FFmpeg / FFprobe 定位与命令执行封装。

优先级：
1. 环境变量 ``BMS_FFMPEG`` / ``BMS_FFPROBE``
2. 工程内 ``tools/ffmpeg/bin``
3. ``imageio-ffmpeg`` 自带的静态构建
4. 系统 PATH
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

    try:  # imageio-ffmpeg 自带静态构建
        import imageio_ffmpeg

        _ffmpeg_cache = imageio_ffmpeg.get_ffmpeg_exe()
        return _ffmpeg_cache
    except Exception:  # pragma: no cover - 依赖缺失时回退
        pass

    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_cache = found
        return _ffmpeg_cache

    raise FileNotFoundError(
        "找不到 ffmpeg。请将 ffmpeg.exe 放入 tools/ffmpeg/bin，"
        "或设置环境变量 BMS_FFMPEG。"
    )


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
    """探测当前 ffmpeg 构建支持哪些能力。"""
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


# ------------------------------------------------------------------ 进程执行


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
        raise RuntimeError(f"命令失败 ({res.returncode}): {' '.join(map(str, cmd))}\n{res.stderr[-4000:]}")
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
    """执行 ffmpeg 并解析 ``-progress`` 输出回报 0~1 进度。"""
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
    """用 ffprobe 读取媒体信息。"""
    exe = find_ffprobe()
    if not exe:
        raise FileNotFoundError("找不到 ffprobe")
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
        raise RuntimeError(f"ffprobe 失败: {res.stderr[-2000:]}")
    return json.loads(res.stdout or "{}")


def ffconcat_escape(path: str) -> str:
    return str(path).replace("\\", "/").replace("'", "'\\''")
