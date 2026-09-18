"""时间线导出。

策略
----
- **单趟滤镜图**：同一素材、片段数适中（<= 80）时，用一条 ``trim`` +
  ``concat`` 的 filter_complex 一次编码输出，速度最快、画质最好。
- **分段回退**：跨素材或片段过多时，先渲染每段统一参数的中间文件，再用
  concat 分离器合并，避免一条超长命令爆掉。
- **竖屏自动跟随**：利用分析阶段得到的逐帧主体横向中心 ``subject_x``，
  生成平滑的裁剪路径表达式，把超广角画面裁成 9:16 并始终跟着球员。
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from ..core import ffmpeg as ff
from ..core.models import Clip, ExportPreset, MediaInfo, Project, Timeline

Progress = Callable[[float, str], None]


def _noop(p: float, m: str = "") -> None:
    pass


# ------------------------------------------------------------------ 工具


def pick_encoder(preset: ExportPreset, caps: dict[str, bool]) -> tuple[list[str], str]:
    """挑选编码器参数。

    返回 ``(编码器参数, 滤镜链尾部)``。第二个值用来把像素格式接到该编码器能吃的样子：

    - NVENC 需要帧在 CUDA 显存里（实测这个构建的 h264_nvenc 拿到系统内存帧会报
      "No capable devices found"），所以尾部要 ``format=nv12,hwupload_cuda``；
    - 软件编码只要 ``format=yuv420p``。
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
    """给出可依次尝试的编码方案，硬件失败时自动退回软件编码。"""
    venc, vtail = pick_encoder(preset, caps)
    codec = venc[1] if len(venc) > 1 else ""
    is_hw = ("nvenc" in codec) or ("qsv" in codec)
    plans: list[tuple[str, list[str], str]] = [
        ("硬件编码" if is_hw else "软件编码", venc, vtail)
    ]
    if is_hw:
        fallback_v = "libx265" if preset.vcodec == "hevc" else "libx264"
        plans.append((
            "软件编码（回退）",
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


# ------------------------------------------------------------------ 竖屏跟随路径


def _subject_path(proj: Project, media_id: str, t0: float, t1: float) -> tuple[float, float, float] | None:
    """返回该时间段内主体的横向中心、左右跨度（归一化 0~1）。

    用**左右包络**而不是「所有球员框中心的均值」：侧方机位下两名球员分别
    在画面两端，均值正好落在他们中间，按均值裁切等于谁都没对准。
    包络还能让「两个人都进画面」这条规则真正生效（双打同理）。
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
    """生成 crop 滤镜字符串；返回 None 表示不需要裁切。

    **裁切框必须严格等于目标宽高比。** 旧实现在「双打放不下」时会把裁切框
    横向加宽到超过目标比例，再由 ``_to_target_size`` 直接缩放成目标尺寸 ——
    那等于把画面横向拉伸了，人会被拉胖。正确做法是要么**等比放大裁切框**
    （左右、上下都还有余量），要么在确实装不下时退回「以两人中点为轴」，
    而不是改变裁切框的形状。
    """
    if not preset.auto_reframe:
        return None
    ar = preset.width / preset.height
    src_ar = src_w / max(1, src_h)
    if src_ar <= ar:
        return None  # 源本身够窄，无需横向裁

    def clamp_box(cw: int, ch: int, cx_norm: float) -> str:
        cw = max(2, min(int(cw), src_w) // 2 * 2)
        ch = max(2, min(int(ch), src_h) // 2 * 2)
        x = int(round(cx_norm * src_w - cw / 2))
        y = (src_h - ch) // 2
        x = max(0, min(x, src_w - cw))
        y = max(0, min(y, src_h - ch))
        return f"crop={cw}:{ch}:{x}:{y}"

    # 以满高度为基准的裁切框
    ch = src_h
    cw = min(src_w, int(round(ch * ar / 2)) * 2)
    if cw < 2:
        return None

    if subject is None:
        return clamp_box(cw, ch, 0.5)

    cx, left, right = subject
    need = (right - left) + margin
    if need > cw / max(src_w, 1):
        # 需要更宽的视野：**等比**放大，直到够宽或顶到源画面
        k = min(need / (cw / max(src_w, 1)), src_h / max(ch, 1) * 1.0)
        k = max(1.0, k)
        nw = min(src_w, cw * k)
        nh = min(src_h, ch * k)
        # 保持比例
        if nw / max(nh, 1) > ar:
            nh = min(src_h, nw / ar)
        else:
            nw = min(src_w, nh * ar)
        cw, ch = int(nw), int(nh)
    return clamp_box(cw, ch, float(cx))


def _fit_chain(preset: ExportPreset, is_vertical_src: bool) -> str:
    """把任意尺寸的画面塞进目标分辨率。"""
    if preset.auto_reframe and preset.height > preset.width:
        # 已经按目标宽高比裁过，直接缩放到目标尺寸
        return f"scale={preset.width}:{preset.height}:flags=lanczos,setsar=1,fps={preset.fps:g}"
    return (f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={preset.fps:g}")


# ------------------------------------------------------------------ 单趟导出


def _single_pass(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
                 on: Progress, cancel) -> tuple[bool, str]:
    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    if track is None:
        return False, "时间线为空"
    clips: list[Clip] = sorted(track.clips, key=lambda c: c.tl_start)
    if len(clips) > 80:
        return False, "片段过多"
    media_ids = {c.media_id for c in clips}
    if len(media_ids) != 1:
        return False, "跨素材"
    media = next((m for m in proj.media if m.id == next(iter(media_ids))), None)
    if media is None:
        return False, "找不到素材"

    src = Path(media.path)
    if not src.is_file():
        return False, "源文件不存在"
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
        res = ff.run_with_progress(cmd, total, lambda p, lb=label: on(0.05 + 0.9 * p, f"编码中（{lb}）"), cancel)
        if res.ok and out.is_file() and out.stat().st_size > 1024:
            return True, ""
        last_err = (res.stdout or "")[-2500:]
        on(0.05, f"{label}失败，尝试其他编码器")
    return False, last_err


def _to_target_size(preset: ExportPreset) -> str:
    """片段级缩放：竖屏且已裁切时直接缩放，否则等比缩放（外层不再 pad）。"""
    if preset.auto_reframe and preset.height > preset.width:
        return f"scale={preset.width}:{preset.height}:flags=lanczos,setsar=1"
    return (f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")


def _atempo_chain(speed: float) -> str:
    """atempo 单次只支持 0.5~2.0，超出的串接。"""
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


# ------------------------------------------------------------------ 分段回退


def _segmented(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
               on: Progress, cancel) -> tuple[bool, str]:
    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    if track is None:
        return False, "时间线为空"
    clips: list[Clip] = sorted(track.clips, key=lambda c: c.tl_start)
    media_by_id = {m.id: m for m in proj.media}
    tmp = Path(tempfile.mkdtemp(prefix="bms_seg_"))
    caps = ff.caps()
    plans = encoder_plans(preset, caps)
    segs: list[Path] = []
    try:
        total = sum(max(0.05, (c.src_out - c.src_in) / max(c.speed, 1e-3)) for c in clips)
        done = 0.0
        # 统一编码方案：`-c copy` 合并要求所有中间片段编码参数一致，
        # 逐段回退会让某一段变成 libx264、另一段是 nvenc，最后合并直接失败。
        chosen: tuple[str, list[str], str] | None = None
        for i, c in enumerate(clips):
            if cancel and cancel():
                return False, "已取消"
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

            # 每个片段都必须带音轨，且参数一致：以前这里是 `-an`，合并出来的成片
            # 完全没有声音（稀疏高光走的正是这条路）。源没有音轨时补一段静音。
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
                # -progress pipe:1 不能省：这段命令用的是 -loglevel error，
                # 没有它就不会有 out_time_us 输出，on_progress 永远不回调，
                # 任务面板上的进度条会一直停在 4% 不动。
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
                    lambda p, lb=label: on(0.02 + 0.75 * (done + p * clip_dur) / total, f"分段渲染（{lb}）"),
                    cancel,
                )
                if r.ok and seg.is_file() and seg.stat().st_size > 1024:
                    ok = True
                    chosen = (label, venc, vtail)
                    break
                err = (r.stdout or "")[-1200:]
            if not ok:
                return False, f"片段 {i} 渲染失败: {err}"
            segs.append(seg)
            done += clip_dur

        if not segs:
            return False, "没有可渲染的片段"
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{ff.ffconcat_escape(str(s))}'\n" for s in segs), encoding="utf-8")
        cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-f", "concat", "-safe", "0",
               "-i", str(lst), "-c", "copy", "-movflags", "+faststart",
               "-progress", "pipe:1", "-loglevel", "error", str(out)]
        r = ff.run_with_progress(cmd, 1.0, lambda p: on(0.8 + 0.18 * p, "合并片段"))
        if not r.ok or not out.is_file():
            return False, f"合并失败: {(r.stdout or '')[-1500:]}"
        return True, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 入口


def export_timeline(proj: Project, timeline: Timeline, preset: ExportPreset, out: Path,
                    on_progress: Progress = _noop, cancel=None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    on_progress(0.01, "准备导出")

    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    clips = sorted(track.clips, key=lambda c: c.tl_start) if track else []

    # 单趟滤镜图必须从一个输入顺序解码到最后一个片段（trim 不能跳读），
    # 所以「片段稀疏」时解码量会远超实际输出。片段之间的空隙越大，
    # 分段导出（每段各自 -ss 快速定位）越划算。
    span = 0.0
    content = 0.0
    if clips:
        span = max(c.src_out for c in clips) - min(c.src_in for c in clips)
        content = sum(max(0.05, (c.src_out - c.src_in) / max(c.speed, 1e-3)) for c in clips)
    sparse = bool(clips) and content > 1 and span > content * 1.5

    ok, err = (False, "片段稀疏，直接分段导出")
    if not sparse:
        ok, err = _single_pass(proj, timeline, preset, out, on_progress, cancel)
    else:
        on_progress(0.03, f"片段较稀疏（解码跨度是输出的 {span / max(content, 1e-6):.1f} 倍），走分段导出")

    if not ok:
        on_progress(0.04, f"改用分段导出（{str(err)[:60]}）")
        ok, err2 = _segmented(proj, timeline, preset, out, on_progress, cancel)
        if not ok:
            raise RuntimeError(f"导出失败：{err2}")
    on_progress(1.0, "完成")
    size = out.stat().st_size if out.is_file() else 0
    return {"path": str(out), "size": size, "preset": preset.model_dump(mode="json")}
