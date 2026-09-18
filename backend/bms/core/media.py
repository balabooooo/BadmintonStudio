"""媒体探测与派生资源生成（代理视频 / 音频 / 缩略图 / 雪碧图）。"""

from __future__ import annotations

import json
import math
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Callable

from ..config import AUDIO_DIR, PROXIES_DIR, THUMBS_DIR, PROXY_FPS, PROXY_MAX_EDGE, AUDIO_SR
from . import ffmpeg as ff
from .models import MediaInfo

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str = "") -> None:
    pass


# ------------------------------------------------------------------ 探测


def _probe_pyav(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        v = next((s for s in c.streams if s.type == "video"), None)
        a = next((s for s in c.streams if s.type == "audio"), None)
        duration = float(c.duration / av.time_base) if c.duration else 0.0
        info: dict = {
            "duration": duration,
            "width": int(v.codec_context.width) if v else 0,
            "height": int(v.codec_context.height) if v else 0,
            "fps": float(v.average_rate) if v and v.average_rate else 30.0,
            "vcodec": v.codec_context.name if v else "",
            "acodec": a.codec_context.name if a else None,
            "has_audio": a is not None,
            "nb_frames": int(v.frames) if v and v.frames else 0,
        }
        # 旋转元数据
        rot = 0
        if v is not None:
            for k in ("rotate", "rotation"):
                if k in v.metadata:
                    try:
                        rot = int(float(v.metadata[k]))
                    except Exception:
                        pass
        if not rot:
            try:
                rot = int(v.side_data.get("displaymatrix") or 0) if v else 0
            except Exception:
                rot = 0
        if rot in (90, 270):
            info["width"], info["height"] = info["height"], info["width"]
            if v is not None and v.average_rate:
                pass
        info["rotation"] = rot % 360
        if info["duration"] <= 0 and v is not None and v.duration and v.time_base:
            info["duration"] = float(v.duration * v.time_base)
        if info["duration"] <= 0 and info["nb_frames"] and info["fps"]:
            info["duration"] = info["nb_frames"] / info["fps"]
        return info


def _probe_ffprobe(path: Path) -> dict:
    raw = ff.probe_json(path)
    fmt = raw.get("format", {})
    v = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = 30.0
    if v:
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = v.get(key)
            if val and val not in ("0/0", "N/A"):
                try:
                    fps = float(Fraction(val))
                    break
                except Exception:
                    pass
    rot = 0
    if v:
        rot = int(v.get("tags", {}).get("rotate", 0) or 0)
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
    w = int(v.get("width", 0)) if v else 0
    h = int(v.get("height", 0)) if v else 0
    if rot % 180 == 90:
        w, h = h, w
    return {
        "duration": float(fmt.get("duration", 0) or 0),
        "width": w,
        "height": h,
        "fps": fps,
        "vcodec": (v or {}).get("codec_name", ""),
        "acodec": (a or {}).get("codec_name") if a else None,
        "has_audio": a is not None,
        "nb_frames": int((v or {}).get("nb_frames", 0) or 0),
        "rotation": rot % 360,
    }


def stable_media_id(path: str | Path) -> str:
    """用文件路径（含大小）算一个稳定 id。

    随机的 UUID 会让「同一个文件重新探测」变成一个新素材，进而重复生成
    代理视频、让已有分析结果失效。用路径哈希可以保证幂等。
    """
    import hashlib

    p = Path(path)
    key = str(p.resolve()).lower()
    try:
        key += f"|{p.stat().st_size}"
    except OSError:
        pass
    return "m_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def probe_media(path: str | Path) -> MediaInfo:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"媒体文件不存在: {p}")
    try:
        info = _probe_pyav(p)
    except Exception:
        info = _probe_ffprobe(p)
    if info.get("duration", 0) <= 0 or info.get("width", 0) <= 0:
        try:
            fb = _probe_ffprobe(p)
            for k, v in fb.items():
                if not info.get(k):
                    info[k] = v
        except Exception:
            pass
    return MediaInfo(
        id=stable_media_id(p),
        path=str(p),
        name=p.stem,
        size=p.stat().st_size,
        duration=float(info.get("duration", 0.0)),
        fps=float(info.get("fps", 30.0) or 30.0),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        rotation=int(info.get("rotation", 0)),
        vcodec=str(info.get("vcodec", "")),
        acodec=info.get("acodec"),
        has_audio=bool(info.get("has_audio")),
    )


# ------------------------------------------------------------------ 派生资源


def proxy_target(media: MediaInfo) -> tuple[int, int]:
    w, h = media.width or 1920, media.height or 1080
    if max(w, h) <= PROXY_MAX_EDGE:
        tw, th = w, h
    else:
        s = PROXY_MAX_EDGE / max(w, h)
        tw, th = int(round(w * s)), int(round(h * s))
    tw -= tw % 2
    th -= th % 2
    return max(2, tw), max(2, th)


def _scale_filter(w: int, h: int, fps: float) -> str:
    return f"scale={w}:{h}:flags=bilinear,fps={fps:g},setsar=1"


# ------------------------------------------------------------------ 硬件加速


_HW_PROBE: dict[str, bool] = {}


def hardware_available() -> bool:
    """检测「CUDA 解码 + scale_cuda + NVENC」这条链路是否真的能用。

    只信 ffmpeg 的 `-hwaccels` 声明不够——很多构建声明了 cuda 却没有可用的
    CUDA 设备，所以这里真跑一次 2 帧的短转码来判定，结果缓存起来。
    """
    import os

    if os.environ.get("BMS_FORCE_SOFTWARE"):
        return False
    if "ok" in _HW_PROBE:
        return _HW_PROBE["ok"]
    try:
        caps = ff.caps()
        if not (caps.get("cuda_decode") and caps.get("nvenc_h264")):
            _HW_PROBE["ok"] = False
            return False
        out = PROXIES_DIR / "_hwprobe.mp4"
        cmd = [
            ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=0.3",
            "-vf", "format=nv12,hwupload_cuda,scale_cuda=320:-2:format=yuv420p",
            "-c:v", "h264_nvenc", "-preset", "p1", "-f", "mp4", str(out),
        ]
        ok = ff.run(cmd).ok and out.is_file() and out.stat().st_size > 512
        out.unlink(missing_ok=True)
        _HW_PROBE["ok"] = bool(ok)
    except Exception:
        _HW_PROBE["ok"] = False
    return _HW_PROBE["ok"]


def hwaccel_input_args(media: MediaInfo) -> list[str]:
    """高分辨率素材才值得开硬件解码；小素材走软件路径反而更省事。"""
    if not hardware_available():
        return []
    if max(media.width or 0, media.height or 0) < 1440:
        return []
    return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]


def hw_scale_filter(w: int, h: int, fps: float) -> str:
    return f"scale_cuda={w}:{h}:format=yuv420p,fps={fps:g},setsar=1"


def nvenc_args(quality: int = 26, bitrate: str = "4M") -> list[str]:
    """NVENC 参数。

    注意：实测这个 ffmpeg 构建的 ``h264_nvenc`` **只在帧来自 CUDA 显存时可用**
    （输入走 ``-hwaccel_output_format cuda`` 或滤镜链末尾有 ``hwupload_cuda``），
    否则会报 "No capable devices found"。所以这里始终配合 GPU 滤镜链使用。
    """
    return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
            "-cq", str(quality), "-b:v", bitrate, "-maxrate", _mul(bitrate, 2),
            "-bufsize", _mul(bitrate, 4), "-spatial-aq", "1"]


def _mul(br: str, k: float) -> str:
    import re as _re

    m = _re.match(r"^(\d+(?:\.\d+)?)([kKmM]?)$", br.strip())
    if not m:
        return br
    return f"{float(m.group(1)) * k:.0f}{m.group(2) or ''}"


def _probe_duration(path: Path) -> float:
    """快速读取文件时长，用来校验派生资源是否完整。"""
    try:
        import av

        with av.open(str(path)) as c:
            if c.duration:
                return float(c.duration / av.time_base)
            v = next((s for s in c.streams if s.type == "video"), None)
            if v is not None and v.duration and v.time_base:
                return float(v.duration * v.time_base)
    except Exception:
        pass
    return 0.0


def _is_valid(path: Path, expected: float, tol: float = 0.06, min_bytes: int = 4096) -> bool:
    """判断已有派生文件能否直接复用。

    必须同时满足：文件够大、能被解出、时长和源差不多。
    任务被取消时 ffmpeg 会留下一个没有 moov 的残file，只判断「文件存在」
    会导致后续全部复用这个坏文件。
    """
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    if expected <= 0:
        return True
    d = _probe_duration(path)
    if d <= 0:
        return False
    return abs(d - expected) <= max(2.0, expected * tol)


def ensure_proxy(media: MediaInfo, on: Progress = _noop, cancel=None) -> MediaInfo:
    """生成低分辨率代理视频（AI 分析与网页预览都用它）。

    优先走「CUDA 解码 → scale_cuda → NVENC」这条全 GPU 链路：实测 4K HEVC
    能跑到 8 倍实时以上，比纯 CPU 快一个数量级。失败自动退回软件编码。

    写入始终走 ``.part`` 临时文件，成功校验后才改名，避免中断留下坏文件。
    """
    src = Path(media.path)
    tw, th = proxy_target(media)
    out = PROXIES_DIR / f"{src.stem}_{media.id}_{tw}x{th}.mp4"
    media.proxy_width, media.proxy_height = tw, th
    media.proxy_fps = min(PROXY_FPS, media.fps or PROXY_FPS)

    if _is_valid(out, media.duration, min_bytes=65536):
        on(0.6, "复用已有代理视频")
        media.proxy_path = str(out)
        return media
    out.unlink(missing_ok=True)

    part = out.with_suffix(".part.mp4")
    hw_in = hwaccel_input_args(media)
    attempts: list[tuple[str, list[str]]] = []

    if hw_in:
        attempts.append(("硬件加速", [
            ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", *hw_in, "-i", str(src),
            "-vf", hw_scale_filter(tw, th, media.proxy_fps),
            *nvenc_args(26),
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart", "-progress", "pipe:1", "-loglevel", "error", str(part),
        ]))

    attempts.append(("软件编码", [
        ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-i", str(src),
        "-vf", _scale_filter(tw, th, media.proxy_fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "30",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart", "-progress", "pipe:1", "-loglevel", "error", str(part),
    ]))

    last_err = ""
    for label, cmd in attempts:
        variants = [cmd]
        # 无声源回退：把音频参数换掉
        if "-c:a" in cmd:
            v2 = [c for c in cmd if c not in ("-c:a", "aac", "-b:a", "128k", "-ac", "2")]
            v2.insert(v2.index("-c:v"), "-an")
            variants.append(v2)
        for cmd_v in variants:
            part.unlink(missing_ok=True)
            res = ff.run_with_progress(
                cmd_v, media.duration, lambda p, m=label: on(0.6 * p, f"生成代理视频（{m}）"), cancel
            )
            if res.ok and _is_valid(part, media.duration, min_bytes=65536):
                part.replace(out)
                media.proxy_path = str(out)
                return media
            last_err = (res.stdout or "")[-2000:]
    part.unlink(missing_ok=True)
    raise RuntimeError(f"代理视频生成失败:\n{last_err}")


def ensure_audio(media: MediaInfo, on: Progress = _noop, cancel=None) -> MediaInfo:
    """提取单声道 16k PCM，用于击球/回合音频分析（同样原子写入 + 校验）。"""
    if not media.has_audio:
        media.audio_path = None
        return media
    src = Path(media.path)
    out = AUDIO_DIR / f"{src.stem}_{media.id}_{AUDIO_SR}.wav"
    if _is_valid(out, media.duration, min_bytes=8192):
        media.audio_path = str(out)
        return media
    out.unlink(missing_ok=True)
    part = out.with_suffix(".part.wav")
    cmd = [
        ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
        "-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        "-c:a", "pcm_s16le", "-progress", "pipe:1", "-loglevel", "error", str(part),
    ]
    res = ff.run_with_progress(cmd, media.duration, lambda p: on(0.15 + 0.15 * p, "提取音轨"), cancel)
    if not res.ok or not _is_valid(part, media.duration, min_bytes=8192):
        part.unlink(missing_ok=True)
        media.audio_path = None
        return media
    part.replace(out)
    media.audio_path = str(out)
    return media


def ensure_poster(media: MediaInfo, at: float | None = None, on: Progress = _noop,
                  force: bool = False) -> MediaInfo:
    """生成封面。

    默认取片长 25% 处；但如果分析已经知道哪些时间点有内容，调用方应该传入
    ``at``（例如最高分回合的中间时刻），否则很容易抓到「有人正走过镜头」的黑历史。
    ``at`` 会参与文件名，所以换了更好的一帧会直接覆盖旧封面。
    """
    src = Path(media.proxy_path or media.path)
    t = at if at is not None else min(max(media.duration * 0.25, 1.0), 600.0)
    tag = f"{int(t)}" if at is not None else "default"
    out = THUMBS_DIR / f"{media.id}_poster_{tag}.jpg"
    if force or not out.is_file():
        cmd = [
            ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
            "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
            "-vf", "scale=640:-2",
            "-q:v", "3", str(out),
        ]
        if not ff.run(cmd).ok or not out.is_file():
            out = THUMBS_DIR / f"{media.id}_poster.jpg"
            ff.run([
                ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
                "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                "-vf", "scale=640:-2", "-q:v", "3", str(out),
            ])
    if out.is_file():
        media.poster = str(out)
    return media


def make_sprite(
    media: MediaInfo,
    count: int = 100,
    cols: int = 10,
    tile_w: int = 160,
    on: Progress = _noop,
) -> dict | None:
    """生成时间线用的缩略图雪碧图。"""
    src = Path(media.proxy_path or media.path)
    if media.duration <= 0:
        return None
    count = max(10, min(count, 400))
    rows = math.ceil(count / cols)
    tw = tile_w - (tile_w % 2)
    th = int(round(tw / max(0.1, media.aspect)))
    th -= th % 2
    interval = media.duration / count
    out = THUMBS_DIR / f"{media.id}_sprite.jpg"
    cmd = [
        ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-i", str(src),
        "-vf", f"fps=1/{interval:.6f},scale={tw}:{th},tile={cols}x{rows}",
        "-frames:v", "1", "-q:v", "5", str(out),
    ]
    res = ff.run(cmd)
    if not res.ok or not out.is_file():
        return None
    return {
        "path": str(out),
        "count": count,
        "cols": cols,
        "rows": rows,
        "tile_w": tw,
        "tile_h": th,
        "interval": interval,
    }


# ------------------------------------------------------------------ 抽帧


def extract_frame(media: MediaInfo, t: float, out_png: Path, max_edge: int = 0) -> Path | None:
    src = Path(media.proxy_path or media.path)
    vf = f"scale={max_edge}:-2" if max_edge else None
    cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-frames:v", "1", "-q:v", "2", str(out_png)]
    res = ff.run(cmd)
    return out_png if res.ok and out_png.is_file() else None


def extract_thumb_grid(
    media: MediaInfo,
    times: list[float],
    out_path: Path,
    thumb_h: int = 108,
    gap: int = 4,
) -> Path | None:
    """把若干时刻的帧纵向拼成一张联络图。

    用 ffmpeg 逐帧取样而非 OpenCV：在超长 HEVC（4K/20GB 级）上
    ``cv2.VideoCapture.set(POS_MSEC)`` 会 seek 到关键帧后返回黑帧。
    """
    import cv2
    import numpy as np

    src = Path(media.proxy_path or media.path)
    frames: list[np.ndarray] = []
    tmp = out_path.parent / f".thumb_{out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(times):
        f = tmp / f"{i:04d}.jpg"
        cmd = [
            ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
            "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", f"scale=-2:{thumb_h}",
            "-q:v", "4", str(f),
        ]
        if ff.run(cmd).ok and f.is_file():
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
    _rmtree(tmp)
    if not frames:
        return None
    max_w = max(f.shape[1] for f in frames)
    rows = len(frames)
    canvas = np.zeros((rows * (thumb_h + gap) - gap, max_w, 3), dtype=np.uint8) + 24
    for i, f in enumerate(frames):
        y = i * (thumb_h + gap)
        canvas[y : y + f.shape[0], : f.shape[1]] = f
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _rmtree(p: Path) -> None:
    import shutil as _sh

    _sh.rmtree(p, ignore_errors=True)


def disk_usage(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def cleanup_cache(keep_ids: set[str]) -> None:
    for d in (PROXIES_DIR, AUDIO_DIR):
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and not any(k in f.name for k in keep_ids):
                try:
                    f.unlink()
                except OSError:
                    pass


__all__ = [
    "probe_media",
    "ensure_proxy",
    "ensure_audio",
    "ensure_poster",
    "make_sprite",
    "extract_frame",
    "extract_thumb_grid",
    "proxy_target",
    "disk_usage",
    "cleanup_cache",
    "shutil",
]
