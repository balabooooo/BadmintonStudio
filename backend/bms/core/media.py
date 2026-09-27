"""Media probing and derived asset generation (proxy video / audio / thumbnails / sprite sheet)."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Callable

from loguru import logger

from ..config import AUDIO_DIR, PROXIES_DIR, THUMBS_DIR, PROXY_FPS, PROXY_MAX_EDGE, AUDIO_SR
from ..i18n import tr
from . import ffmpeg as ff
from .models import MediaInfo

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str = "") -> None:
    pass


def _part_path(out: Path) -> Path:
    """Unique ``.part`` sibling of ``out`` (same directory, same final extension).

    A fixed ``.part`` name lets two concurrent callers (e.g. a prepare job and the analysis
    pipeline) feed their ffmpeg processes into the same file and corrupt each other's output.
    """
    return out.with_name(
        f"{out.stem}.{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:6]}.part{out.suffix}"
    )


# ------------------------------------------------------------------ Probing


_pyav_error: str | None = None
_pyav_checked = False


def _pyav_ok() -> bool:
    """Whether PyAV is available.

    Windows with "Smart App Control / AppLocker" enabled blocks PyAV's unsigned native
    extension (DLL load failed); probe only once and remember the failure so it is not
    triggered repeatedly for every file.
    """
    global _pyav_error, _pyav_checked
    if not _pyav_checked:
        _pyav_checked = True
        try:
            import av  # noqa: F401
        except Exception as e:  # noqa: BLE001
            _pyav_error = f"{type(e).__name__}: {e}"
    return _pyav_error is None


def probe_backend_status() -> dict:
    """For the "Settings / Environment" page: explains which chain is actually used to probe media."""
    pyav = _pyav_ok()
    ffprobe = ff.find_ffprobe()
    return {
        "active": "pyav" if pyav else ("ffprobe" if ffprobe else "ffmpeg"),
        "pyav": pyav,
        "pyav_error": _pyav_error,
        "ffprobe": ffprobe,
        "ffmpeg": ff.find_ffmpeg(),
    }


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
        # Rotation metadata
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


_FFMPEG_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
# Of the form `Stream #0:0[0x1](und): Video: hevc ...`; the order of the bracket/paren groups is not fixed.
_FFMPEG_VIDEO_RE = re.compile(r"Stream #\d+:\d+[^:]*:\s*Video:\s*([A-Za-z0-9_]+)")
_FFMPEG_AUDIO_RE = re.compile(r"Stream #\d+:\d+[^:]*:\s*Audio:\s*([A-Za-z0-9_]+)")
_FFMPEG_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FFMPEG_FPS_RE = re.compile(r"([\d.]+)\s*fps\b")
# Handles both `displaymatrix: rotation of -90.00 degrees` and `rotation : -90` outputs
_FFMPEG_ROT_RE = re.compile(r"(?:rotation of\s*|rotation\s*:\s*)(-?[\d.]+)")


def _probe_ffmpeg(path: Path) -> dict:
    """Fallback when ffprobe is unavailable: parse ``ffmpeg -i`` stderr metadata.

    On Windows ``imageio-ffmpeg`` ships only ffmpeg, not ffprobe; PyAV may in turn be
    blocked by app-control policy, so a probe path that depends only on ffmpeg must be kept.
    """
    res = ff.run([ff.find_ffmpeg(), "-hide_banner", "-i", str(path)])
    text = res.stderr or ""
    duration = 0.0
    m = _FFMPEG_DUR_RE.search(text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    vid = _FFMPEG_VIDEO_RE.search(text)
    aud = _FFMPEG_AUDIO_RE.search(text)
    width = height = 0
    fps = 0.0
    if vid:
        # Parse only the video line, to avoid mistaking the audio sample rate/bitrate for the resolution.
        line = text[vid.start():].splitlines()[0]
        size = _FFMPEG_SIZE_RE.search(line)
        if size:
            width, height = int(size.group(1)), int(size.group(2))
        fr = _FFMPEG_FPS_RE.search(line)
        if fr:
            fps = float(fr.group(1))
    rot = 0
    rm = _FFMPEG_ROT_RE.search(text)
    if rm:
        rot = int(float(rm.group(1))) % 360
    if rot % 180 == 90:
        width, height = height, width
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps or 30.0,
        "vcodec": vid.group(1) if vid else "",
        "acodec": aud.group(1) if aud else None,
        "has_audio": aud is not None,
        "nb_frames": 0,
        "rotation": rot,
    }


def stable_media_id(path: str | Path) -> str:
    """Compute a stable id from the file path (including size).

    A random UUID makes "probing the same file again" become a new media item, which then
    regenerates the proxy video and invalidates existing analysis results. Hashing the path
    guarantees idempotence.
    """
    import hashlib

    p = Path(path)
    key = str(p.resolve()).lower()
    try:
        key += f"|{p.stat().st_size}"
    except OSError:
        pass
    return "m_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _probe_media_raw(p: Path) -> dict:
    """Try PyAV / ffprobe / ffmpeg in order; use whichever first returns a complete duration + resolution."""
    info: dict = {}
    errors: list[str] = []
    probes = [_probe_ffprobe, _probe_ffmpeg]
    if _pyav_ok():
        probes.insert(0, _probe_pyav)
    for probe in probes:
        try:
            got = probe(p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{probe.__name__}: {type(e).__name__}: {e}")
            continue
        for k, v in got.items():
            if not info.get(k):
                info[k] = v
        if info.get("duration", 0) > 0 and info.get("width", 0) > 0:
            return info
    if not info:
        raise RuntimeError(tr("media.probe_failed", detail="；".join(errors)))
    return info


def probe_media(path: str | Path) -> MediaInfo:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(tr("media.file_not_found", path=p))
    info = _probe_media_raw(p)
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


# ------------------------------------------------------------------ Derived assets


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


def proxy_source(media: MediaInfo) -> Path:
    """Resolve "prefer the proxy but fall back to the source" for every read path.

    ``media.proxy_path`` may point at a file that no longer exists (it was removed by the cache
    cleanup). The old ``media.proxy_path or media.path`` only fell back when the field was *empty*,
    so a stale non-empty path was handed to cv2/ffmpeg and failed. Fall back whenever the file is
    gone, not only when the field is unset.
    """
    if media.proxy_path:
        p = Path(media.proxy_path)
        if p.is_file():
            return p
    return Path(media.path)


def proxy_stem(media: MediaInfo) -> str:
    """Deterministic name of the proxy file (the same formula :func:`ensure_proxy` writes).

    Callers that key artifacts on the proxy name (rally annotations) need it even when
    ``proxy_path`` has been cleared from the project, so it must not depend on the file existing.
    """
    tw, th = proxy_target(media)
    return f"{Path(media.path).stem}_{media.id}_{tw}x{th}"


def _scale_filter(w: int, h: int, fps: float) -> str:
    return f"scale={w}:{h}:flags=bilinear,fps={fps:g},setsar=1"


# ------------------------------------------------------------------ Hardware acceleration


_HW_PROBE: dict[str, bool] = {}


def hardware_available() -> bool:
    """Check whether the "CUDA decode + scale_cuda + NVENC" chain really works.

    Trusting ffmpeg's `-hwaccels` declaration alone is not enough -- many builds declare cuda
    but have no usable CUDA device, so this actually runs a 2-frame transcode to decide and
    caches the result.
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
    """Only high-resolution media is worth hardware decoding; small media is actually easier through the software path."""
    if not hardware_available():
        return []
    if max(media.width or 0, media.height or 0) < 1440:
        return []
    return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]


def hw_scale_filter(w: int, h: int, fps: float) -> str:
    return f"scale_cuda={w}:{h}:format=yuv420p,fps={fps:g},setsar=1"


def nvenc_args(quality: int = 26, bitrate: str = "4M") -> list[str]:
    """NVENC parameters.

    Note: in this ffmpeg build ``h264_nvenc`` **only works when frames come from CUDA
    memory** (input uses ``-hwaccel_output_format cuda`` or the filter chain ends with
    ``hwupload_cuda``), otherwise it reports "No capable devices found". So it is always
    used together with a GPU filter chain here.
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
    """Quickly read the file duration, used to verify that a derived asset is complete."""
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
    try:  # fall back to ffmpeg when PyAV is blocked / ffprobe is unavailable
        return float(_probe_ffmpeg(path).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def _is_valid(path: Path, expected: float, tol: float = 0.06, min_bytes: int = 4096) -> bool:
    """Decide whether an existing derived file can be reused directly.

    Must satisfy all of: the file is large enough, it can be decoded, and its duration is
    close to the source. When a job is cancelled ffmpeg leaves behind a broken file without
    a moov atom; checking only "the file exists" would cause that bad file to be reused
    everywhere afterwards.
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
    """Generate a low-resolution proxy video (used by both AI analysis and web preview).

    Prefer the fully GPU path "CUDA decode -> scale_cuda -> NVENC": measured 4K HEVC can run
    at over 8x real time, an order of magnitude faster than pure CPU. Fall back to software
    encoding automatically on failure.

    Writes always go through a ``.part`` temporary file and are renamed only after
    verification, to avoid leaving a broken file on interruption.
    """
    src = Path(media.path)
    tw, th = proxy_target(media)
    out = PROXIES_DIR / f"{proxy_stem(media)}.mp4"
    media.proxy_width, media.proxy_height = tw, th
    media.proxy_fps = min(PROXY_FPS, media.fps or PROXY_FPS)

    if _is_valid(out, media.duration, min_bytes=65536):
        on(0.6, tr("media.reuse_proxy"))
        media.proxy_path = str(out)
        return media
    out.unlink(missing_ok=True)
    logger.info("proxy generate: {!r} -> {} ({}x{} @ {:.2f}fps)",
                src.name, out.name, tw, th, media.proxy_fps)

    part = _part_path(out)
    hw_in = hwaccel_input_args(media)
    attempts: list[tuple[str, list[str]]] = []

    if hw_in:
        attempts.append((tr("media.hw_accel"), [
            ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", *hw_in, "-i", str(src),
            "-vf", hw_scale_filter(tw, th, media.proxy_fps),
            *nvenc_args(26),
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart", "-progress", "pipe:1", "-loglevel", "error", str(part),
        ]))

    attempts.append((tr("media.sw_encode"), [
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
        # Silent-source fallback: swap out the audio arguments
        if "-c:a" in cmd:
            v2 = [c for c in cmd if c not in ("-c:a", "aac", "-b:a", "128k", "-ac", "2")]
            v2.insert(v2.index("-c:v"), "-an")
            variants.append(v2)
        for cmd_v in variants:
            part.unlink(missing_ok=True)
            res = ff.run_with_progress(
                cmd_v, media.duration, lambda p, m=label: on(0.6 * p, tr("media.proxy_generating", mode=m)), cancel
            )
            if res.ok and _is_valid(part, media.duration, min_bytes=65536):
                part.replace(out)
                media.proxy_path = str(out)
                logger.info("proxy ready: {} ({})", out.name, label)
                return media
            last_err = (res.stdout or "")[-2000:]
    part.unlink(missing_ok=True)
    logger.error("proxy failed for {!r}: {}", src.name, last_err[-500:])
    raise RuntimeError(tr("media.proxy_failed", err=last_err))


def ensure_audio(media: MediaInfo, on: Progress = _noop, cancel=None) -> MediaInfo:
    """Extract mono 16k PCM for hit/rally audio analysis (same atomic write + verification)."""
    if not media.has_audio:
        media.audio_path = None
        return media
    src = Path(media.path)
    out = AUDIO_DIR / f"{src.stem}_{media.id}_{AUDIO_SR}.wav"
    if _is_valid(out, media.duration, min_bytes=8192):
        media.audio_path = str(out)
        return media
    out.unlink(missing_ok=True)
    part = _part_path(out)
    cmd = [
        ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
        "-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        "-c:a", "pcm_s16le", "-progress", "pipe:1", "-loglevel", "error", str(part),
    ]
    res = ff.run_with_progress(cmd, media.duration, lambda p: on(0.15 + 0.15 * p, tr("media.extract_audio")), cancel)
    if not res.ok or not _is_valid(part, media.duration, min_bytes=8192):
        part.unlink(missing_ok=True)
        media.audio_path = None
        logger.warning("audio extraction failed for {!r} (continuing without audio)", src.name)
        return media
    part.replace(out)
    media.audio_path = str(out)
    logger.info("audio ready: {}", out.name)
    return media


def ensure_poster(media: MediaInfo, at: float | None = None, on: Progress = _noop,
                  force: bool = False) -> MediaInfo:
    """Generate a poster.

    Defaults to 25% of the clip length; but if the analysis already knows which time points
    have content, the caller should pass ``at`` (for example the midpoint of the highest-scoring
    rally), otherwise it easily captures an embarrassing "someone walking past the lens" frame.
    ``at`` is part of the filename, so switching to a better frame directly overwrites the old poster.
    """
    src = proxy_source(media)
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
    """Generate the thumbnail sprite sheet used by the timeline."""
    src = proxy_source(media)
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


# ------------------------------------------------------------------ Frame extraction


def extract_frame(media: MediaInfo, t: float, out_png: Path, max_edge: int = 0) -> Path | None:
    src = proxy_source(media)
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
    """Vertically stitch frames from several time points into a contact sheet.

    Use ffmpeg to sample frame by frame rather than OpenCV: on very long HEVC files
    (4K/20GB class) ``cv2.VideoCapture.set(POS_MSEC)`` seeks to a keyframe and returns a
    black frame.
    """
    import cv2
    import numpy as np

    src = proxy_source(media)
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
    "proxy_source",
    "proxy_stem",
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
