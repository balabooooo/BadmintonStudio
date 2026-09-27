"""Timeline export.

Strategy
--------
- **Single-pass filter graph**: with a single media source and a moderate number of clips (<= 80),
  use one ``trim`` + ``concat`` filter_complex to encode and output in one go: fastest and best quality.
- **Segmented fallback**: when crossing media or with too many clips, first render an intermediate
  file per segment with uniform parameters, then merge them with the concat demuxer, to avoid one
  overly long command blowing up.
- **Automatic vertical follow**: use the per-frame subject horizontal center ``subject_x`` obtained
  during analysis to generate a smooth crop-path expression, cropping the ultra-wide frame to 9:16
  while always following the players.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from ..core import ffmpeg as ff
from ..core.models import Clip, ExportPreset, MediaInfo, Project, Timeline, Track
from ..i18n import tr

Progress = Callable[[float, str], None]


def _noop(p: float, m: str = "") -> None:
    pass


# ------------------------------------------------------------------ Utilities


def pick_encoder(preset: ExportPreset, caps: dict[str, bool]) -> tuple[list[str], str]:
    """Pick encoder parameters.

    Returns ``(encoder args, filter chain tail)``. The second value adapts the pixel format to what
    that encoder can consume:

    - NVENC needs frames in CUDA memory (measured: this build's h264_nvenc reports
      "No capable devices found" when given system-memory frames), so the tail must be
      ``format=nv12,hwupload_cuda``;
    - software encoding just needs ``format=yuv420p``.
    """
    want = preset.encoder
    if want == "auto":
        want = "nvenc" if (caps.get("nvenc_h264") or caps.get("nvenc_hevc")) else "x264"
    if preset.vcodec == "hevc":
        if want == "nvenc" and caps.get("nvenc_hevc"):
            return (["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                     "-cq", "23", "-b:v", preset.video_bitrate,
                     "-maxrate", _x2(preset.video_bitrate), "-bufsize", _x2(_x2(preset.video_bitrate)),
                     "-spatial-aq", "1"], "format=nv12,hwupload_cuda")
        if want == "qsv" and caps.get("qsv"):
            return (["-c:v", "hevc_qsv", "-global_quality", "23", "-b:v", preset.video_bitrate],
                    "format=nv12")
        return (["-c:v", "libx265", "-preset", "medium", "-crf", str(preset.crf or 22),
                 "-b:v", preset.video_bitrate], "format=yuv420p")
    if want == "nvenc" and caps.get("nvenc_h264"):
        return (["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", "21", "-b:v", preset.video_bitrate,
                 "-maxrate", _x2(preset.video_bitrate), "-bufsize", _x2(_x2(preset.video_bitrate)),
                 "-spatial-aq", "1"], "format=nv12,hwupload_cuda")
    if want == "qsv" and caps.get("qsv"):
        return (["-c:v", "h264_qsv", "-global_quality", "22", "-b:v", preset.video_bitrate],
                "format=nv12")
    return (["-c:v", "libx264", "-preset", "medium", "-crf", str(preset.crf or 20),
             "-b:v", preset.video_bitrate], "format=yuv420p")


def encoder_plans(preset: ExportPreset, caps: dict[str, bool]) -> list[tuple[str, list[str], str]]:
    """Provide the encoding plans to try in order, falling back to software encoding when hardware fails."""
    venc, vtail = pick_encoder(preset, caps)
    codec = venc[1] if len(venc) > 1 else ""
    is_hw = ("nvenc" in codec) or ("qsv" in codec)
    plans: list[tuple[str, list[str], str]] = [
        (tr("render.hw_encode") if is_hw else tr("render.sw_encode"), venc, vtail)
    ]
    if is_hw:
        fallback_v = "libx265" if preset.vcodec == "hevc" else "libx264"
        plans.append((
            tr("render.sw_encode_fallback"),
            ["-c:v", fallback_v, "-preset", "medium", "-crf", str(preset.crf or 21),
             "-b:v", preset.video_bitrate],
            "format=yuv420p",
        ))
    return plans


def _x2(br: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)?)([kKmM]?)$", br.strip())
    if not m:
        return br
    v, unit = float(m.group(1)), m.group(2)
    return f"{v * 2:.0f}{unit or ''}"


def has_audio(media: MediaInfo) -> bool:
    return bool(media.has_audio)


# ------------------------------------------------------------------ Vertical follow path


def _subject_path(proj: Project, media_id: str, t0: float, t1: float) -> tuple[float, float, float] | None:
    """Return the subject's horizontal center and left/right span within the time range (normalized 0~1).

    Use the **left/right envelope** rather than "the mean of all player box centers": under a side
    camera setup the two players are at opposite ends of the frame, the mean falls exactly between
    them, and cropping by the mean aligns with neither. The envelope also makes the rule "both
    players are in frame" actually work (same for doubles).
    """
    res = proj.analyses.get(media_id)
    if res is None:
        return None
    sx = res.signals.get("subject_x") or []
    sl = res.signals.get("subject_left") or []
    sr = res.signals.get("subject_right") or []
    fps_l = res.signals.get("subject_fps") or [0.0]
    fps = float(fps_l[0]) if fps_l else 0.0
    span_total = float(res.signals.get("duration", [0.0])[0] or 0.0)
    if fps <= 0 or span_total <= 0:
        return None

    def _avg(arr: list[float]) -> float | None:
        if not arr:
            return None
        n = len(arr)
        a = max(0, int(t0 / max(span_total, 1e-6) * n))
        b = min(n, max(a + 1, int(t1 / max(span_total, 1e-6) * n)))
        seg = arr[a:b]
        return (sum(seg) / len(seg)) if seg else None

    left = _avg(sl)
    right = _avg(sr)
    if left is not None and right is not None and right > left:
        return ((left + right) / 2.0, left, right)
    cx = _avg(sx)
    if cx is None:
        return None
    w = _avg(res.signals.get("subject_w") or []) or 0.25
    return (cx, cx - w / 2.0, cx + w / 2.0)


def _crop_expr_for_clip(preset: ExportPreset, src_w: int, src_h: int,
                        subject: tuple[float, float, float] | None,
                        margin: float = 0.08) -> str | None:
    """Generate the crop filter string; returning None means no cropping is needed.

    **The crop box must exactly match the target aspect ratio.** The old implementation, when
    "doubles does not fit", widened the crop box horizontally beyond the target ratio and then
    ``_to_target_size`` scaled it directly to the target size -- that stretched the image
    horizontally and made people look fat. The correct approach is either to **scale the crop box
    proportionally** (leaving margin on all sides) or, when it truly does not fit, fall back to
    "pivot on the midpoint of the two players", rather than changing the shape of the crop box.
    """
    if not preset.auto_reframe:
        return None
    ar = preset.width / preset.height
    src_ar = src_w / max(1, src_h)
    if src_ar <= ar:
        return None  # the source is already narrow enough, no horizontal crop needed

    def clamp_box(cw: int, ch: int, cx_norm: float) -> str:
        cw = max(2, min(int(cw), src_w) // 2 * 2)
        ch = max(2, min(int(ch), src_h) // 2 * 2)
        x = int(round(cx_norm * src_w - cw / 2))
        y = (src_h - ch) // 2
        x = max(0, min(x, src_w - cw))
        y = max(0, min(y, src_h - ch))
        return f"crop={cw}:{ch}:{x}:{y}"

    # Crop box based on full height
    ch = src_h
    cw = min(src_w, int(round(ch * ar / 2)) * 2)
    if cw < 2:
        return None

    if subject is None:
        return clamp_box(cw, ch, 0.5)

    cx, left, right = subject
    need = (right - left) + margin
    if need > cw / max(src_w, 1):
        # A wider field of view is needed: scale up **proportionally** until wide enough or limited by the source frame
        k = min(need / (cw / max(src_w, 1)), src_h / max(ch, 1) * 1.0)
        k = max(1.0, k)
        nw = min(src_w, cw * k)
        nh = min(src_h, ch * k)
        # Preserve the aspect ratio
        if nw / max(nh, 1) > ar:
            nh = min(src_h, nw / ar)
        else:
            nw = min(src_w, nh * ar)
        cw, ch = int(nw), int(nh)
    return clamp_box(cw, ch, float(cx))


def _fit_chain(preset: ExportPreset, is_vertical_src: bool) -> str:
    """Fit a frame of any size into the target resolution."""
    if preset.auto_reframe and preset.height > preset.width:
        # Already cropped to the target aspect ratio, scale directly to the target size
        return f"scale={preset.width}:{preset.height}:flags=lanczos,setsar=1,fps={preset.fps:g}"
    return (f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={preset.fps:g}")


# ------------------------------------------------------------------ Single-pass export


def _single_pass(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
                 on: Progress, cancel) -> tuple[bool, str]:
    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    if track is None:
        return False, tr("render.empty_timeline")
    clips: list[Clip] = sorted(track.clips, key=lambda c: c.tl_start)
    if len(clips) > 80:
        return False, tr("render.too_many_clips")
    media_ids = {c.media_id for c in clips}
    if len(media_ids) != 1:
        return False, tr("render.cross_media")
    media = next((m for m in proj.media if m.id == next(iter(media_ids))), None)
    if media is None:
        return False, tr("render.media_not_found")

    src = Path(media.path)
    if not src.is_file():
        return False, tr("render.source_missing")
    src_w, src_h = media.width or 1920, media.height or 1080
    has_a = has_audio(media)
    total = sum(max(0.05, (c.src_out - c.src_in) / max(c.speed, 1e-3)) for c in clips)

    parts: list[str] = []
    concat_in: list[str] = []
    for i, c in enumerate(clips):
        a = max(0.0, float(c.src_in))
        b = max(a + 0.04, float(c.src_out))
        spd = max(0.1, min(8.0, float(c.speed)))
        chain: list[str] = [
            f"trim=start={a:.4f}:end={b:.4f}",
            "setpts=(PTS-STARTPTS)" + (f"/{spd:.4f}" if abs(spd - 1) > 1e-3 else ""),
        ]
        subj = _subject_path(proj, c.media_id, a, b) if preset.auto_reframe else None
        crop = _crop_expr_for_clip(preset, src_w, src_h, subj)
        if crop:
            chain.append(crop)
        chain.append(_to_target_size(preset))
        parts.append(f"[0:v]{','.join(chain)}[v{i}]")
        if has_a:
            ach: list[str] = [f"atrim=start={a:.4f}:end={b:.4f}", "asetpts=PTS-STARTPTS"]
            if abs(spd - 1) > 1e-3:
                ach.append(f"atempo={_atempo_chain(spd)}")
            if abs(float(c.volume) - 1.0) > 1e-3:
                ach.append(f"volume={float(c.volume):.3f}")
            ach.append("aresample=48000")
            parts.append(f"[0:a]{','.join(ach)}[a{i}]")
            concat_in.append(f"[v{i}][a{i}]")
        else:
            concat_in.append(f"[v{i}]")

    n = len(clips)
    a_flag = "1" if has_a else "0"
    graph = ";".join(parts) + ";" + "".join(concat_in) + f"concat=n={n}:v=1:a={a_flag}[vcat]"
    if has_a:
        graph += "[acat]"
    if not has_a:
        graph += f";anullsrc=channel_layout=stereo:sample_rate=48000,atrim=0:{total:.3f}[acat]"

    caps = ff.caps()
    last_err = ""
    for label, venc, vtail in encoder_plans(preset, caps):
        graph_full = graph + f";[vcat]{vtail}[vout]"
        cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-i", str(src),
               "-filter_complex", graph_full, "-map", "[vout]", "-map", "[acat]"]
        cmd += venc + ["-r", f"{preset.fps:g}", "-movflags", "+faststart",
                       "-c:a", "aac", "-b:a", preset.audio_bitrate, "-ac", "2",
                       "-progress", "pipe:1", "-loglevel", "error", str(out)]
        out.unlink(missing_ok=True)
        res = ff.run_with_progress(cmd, total, lambda p, lb=label: on(0.05 + 0.9 * p, tr("render.encoding", mode=lb)), cancel)
        if res.ok and out.is_file() and out.stat().st_size > 1024:
            return True, ""
        last_err = (res.stdout or "")[-2500:]
        on(0.05, tr("render.encoder_retry", mode=label))
    return False, last_err


def _to_target_size(preset: ExportPreset) -> str:
    """Clip-level scaling: scale directly when vertical and already cropped, otherwise scale proportionally (no outer pad)."""
    if preset.auto_reframe and preset.height > preset.width:
        return f"scale={preset.width}:{preset.height}:flags=lanczos,setsar=1"
    return (f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")


def _atempo_chain(speed: float) -> str:
    """atempo supports only 0.5~2.0 per instance; chain multiple for values beyond that."""
    s = float(speed)
    parts: list[str] = []
    while s > 2.0:
        parts.append("2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("0.5")
        s /= 0.5
    parts.append(f"{s:.4f}")
    return ",".join(parts)


# ------------------------------------------------------------------ Segmented fallback


def _segmented(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
               on: Progress, cancel) -> tuple[bool, str]:
    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    if track is None:
        return False, tr("render.empty_timeline")
    clips: list[Clip] = sorted(track.clips, key=lambda c: c.tl_start)
    media_by_id = {m.id: m for m in proj.media}
    tmp = Path(tempfile.mkdtemp(prefix="bms_seg_"))
    caps = ff.caps()
    plans = encoder_plans(preset, caps)
    segs: list[Path] = []
    try:
        total = sum(max(0.05, (c.src_out - c.src_in) / max(c.speed, 1e-3)) for c in clips)
        done = 0.0
        # Uniform encoding plan: `-c copy` merging requires all intermediate segments to have the
        # same encoding parameters; per-segment fallback would make one segment libx264 and another
        # nvenc, and the final merge would fail outright.
        chosen: tuple[str, list[str], str] | None = None
        for i, c in enumerate(clips):
            if cancel and cancel():
                return False, tr("job.cancelled")
            m = media_by_id.get(c.media_id)
            if m is None:
                continue
            src = Path(m.path)
            if not src.is_file():
                continue
            spd = max(0.1, min(8.0, float(c.speed)))
            clip_dur = max(0.05, c.src_out - c.src_in)
            subj = _subject_path(proj, c.media_id, c.src_in, c.src_out) if preset.auto_reframe else None
            crop = _crop_expr_for_clip(preset, m.width or 1920, m.height or 1080, subj)
            base_vf: list[str] = []
            if crop:
                base_vf.append(crop)
            base_vf.append(_to_target_size(preset))
            if abs(spd - 1) > 1e-3:
                base_vf.append(f"setpts=PTS/{spd:.4f}")
            base_vf.append(f"fps={preset.fps:g}")

            # Every segment must carry an audio track with consistent parameters: this used to be
            # `-an`, and the merged output had no sound at all (sparse highlights take exactly this
            # path). Fill in silence when the source has no audio track.
            has_a = has_audio(m)
            audio_in: list[str] = []
            if has_a:
                af: list[str] = []
                if abs(spd - 1) > 1e-3:
                    af.append(f"atempo={_atempo_chain(spd)}")
                if abs(float(c.volume) - 1.0) > 1e-3:
                    af.append(f"volume={float(c.volume):.3f}")
                af.append("aresample=48000")
                audio_args = ["-map", "0:v:0", "-map", "0:a:0?", "-af", ",".join(af)]
            else:
                audio_in = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
                audio_args = ["-map", "0:v:0", "-map", "1:a:0"]

            seg = tmp / f"{i:05d}.mp4"
            ok = False
            err = ""
            for label, venc, vtail in ([chosen] if chosen else plans):
                vf = ",".join(base_vf + [vtail])
                # -progress pipe:1 cannot be omitted: this command uses -loglevel error, and without
                # it there is no out_time_us output, on_progress is never called, and the progress bar
                # on the job panel stays stuck at 4% forever.
                cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin",
                       "-ss", f"{c.src_in:.4f}", "-i", str(src)] + audio_in + [
                       "-t", f"{clip_dur:.4f}",
                       "-vf", vf] + audio_args + venc + \
                      ["-r", f"{preset.fps:g}", "-c:a", "aac", "-b:a", preset.audio_bitrate,
                       "-ac", "2", "-ar", "48000",
                       "-loglevel", "error", "-progress", "pipe:1", "-nostats", str(seg)]
                seg.unlink(missing_ok=True)
                r = ff.run_with_progress(
                    cmd, clip_dur,
                    lambda p, lb=label: on(0.02 + 0.75 * (done + p * clip_dur) / total, tr("render.segment_rendering", mode=lb)),
                    cancel,
                )
                if r.ok and seg.is_file() and seg.stat().st_size > 1024:
                    ok = True
                    chosen = (label, venc, vtail)
                    break
                err = (r.stdout or "")[-1200:]
            if not ok:
                return False, tr("render.segment_failed", index=i, err=err)
            segs.append(seg)
            done += clip_dur

        if not segs:
            return False, tr("render.no_renderable_clips")
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{ff.ffconcat_escape(str(s))}'\n" for s in segs), encoding="utf-8")
        cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-f", "concat", "-safe", "0",
               "-i", str(lst), "-c", "copy", "-movflags", "+faststart",
               "-progress", "pipe:1", "-loglevel", "error", str(out)]
        r = ff.run_with_progress(cmd, 1.0, lambda p: on(0.8 + 0.18 * p, tr("render.merging")), cancel)
        if not r.ok or not out.is_file():
            return False, tr("render.merge_failed", err=(r.stdout or '')[-1500:])
        return True, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ Entry point


def _render_clips(proj: Project, clips: list[Clip], preset: ExportPreset, out: Path,
                  on_progress: Progress, cancel) -> Path:
    """Render a group of clips into one file: single-pass preferred, segmented fallback when sparse/failed."""
    span = 0.0
    content = 0.0
    if clips:
        span = max(c.src_out for c in clips) - min(c.src_in for c in clips)
        content = sum(max(0.05, (c.src_out - c.src_in) / max(c.speed, 1e-3)) for c in clips)
    sparse = bool(clips) and content > 1 and span > content * 1.5

    one = Timeline(tracks=[Track(name=tr("timeline.track_default", n=1), kind="video", clips=list(clips))], fps=preset.fps)
    ok, err = (False, tr("render.sparse_direct"))
    if not sparse:
        ok, err = _single_pass(proj, one, preset, out, on_progress, cancel)
    else:
        on_progress(0.03, tr("render.sparse_ratio", ratio=f"{span / max(content, 1e-6):.1f}"))

    if not ok:
        if cancel and cancel():
            raise RuntimeError(tr("job.cancelled"))
        on_progress(0.04, tr("render.switch_segmented", reason=str(err)[:60]))
        ok, err2 = _segmented(proj, one, preset, out, on_progress, cancel)
        if not ok:
            raise RuntimeError(tr("render.export_failed", err=err2))
    return out


def _export_separate(proj: Project, clips: list[Clip], preset: ExportPreset, out: Path,
                     on_progress: Progress, cancel) -> dict:
    """Export each clip to its own file (flattened in the same directory), named ``<name>_01.mp4``."""
    n = len(clips)
    paths: list[Path] = []
    for i, c in enumerate(clips):
        if cancel and cancel():
            raise RuntimeError(tr("job.cancelled"))
        seg_out = out.with_name(f"{out.stem}_{i + 1:02d}{out.suffix}")
        base = i / n

        def on(p: float, m: str, b: float = base, idx: int = i + 1) -> None:
            on_progress(min(0.99, b + p / n), tr("render.separate_progress", index=idx, total=n, message=m))

        _render_clips(proj, [c], preset, seg_out, on, cancel)
        paths.append(seg_out)
    on_progress(1.0, tr("render.done"))
    return {
        "paths": [str(p) for p in paths],
        "count": n,
        "preset": preset.model_dump(mode="json"),
        "dir": str(out.parent),
    }


def export_timeline(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
                    on_progress: Progress = _noop, cancel=None, separate: bool = False) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    on_progress(0.01, tr("render.preparing"))

    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    clips = sorted(track.clips, key=lambda c: c.tl_start) if track else []
    if not clips:
        raise RuntimeError(tr("render.export_failed", err=tr("render.empty_timeline")))

    if separate:
        return _export_separate(proj, clips, preset, out, on_progress, cancel)

    _render_clips(proj, clips, preset, out, on_progress, cancel)
    on_progress(1.0, tr("render.done"))
    size = out.stat().st_size if out.is_file() else 0
    return {
        "path": str(out),
        "size": size,
        "preset": preset.model_dump(mode="json"),
        "dir": str(out.parent),
    }
