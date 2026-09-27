"""Transcribe a clip's audio with faster-whisper, for tuning voice-command detection.

The voice-command scorer (see ``bms/analysis/speech.py``) runs this same engine; this script
exposes it standalone so you can inspect what Whisper hears around a given timestamp, compare
model sizes, and check whether a near-homophone mis-recognition needs the fuzzy match.

Usage:
    python scripts/transcribe_speech.py "<video|wav>" [--around 638] [--phrases 好球 漂亮]
    python scripts/transcribe_speech.py data/cache/audio/xxx_16000.wav --around 638 --model medium

A video input reuses (or extracts) the cached 16k mono wav via ``media.ensure_audio``.
Model weights are downloaded to the HuggingFace cache, not into the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.config import ensure_dirs  # noqa: E402
from bms.core.ct2_cuda import ensure_cuda_dlls  # noqa: E402
from bms.core.media import ensure_audio, probe_media  # noqa: E402


def _resolve_audio(src: str) -> Path:
    """Return a 16k mono wav path: use a wav directly, otherwise reuse/extract the cached audio."""
    p = Path(src)
    if p.suffix.lower() == ".wav":
        return p
    info = probe_media(src)
    if not info.has_audio:
        raise SystemExit(f"素材没有音轨：{src}")
    info = ensure_audio(info, on=lambda pr, m: print(f"  [{pr * 100:5.1f}%] {m}", flush=True))
    if not info.audio_path:
        raise SystemExit(f"无法提取音轨：{src}")
    return Path(info.audio_path)


def _pick_device(device: str) -> tuple[str, str]:
    """Resolve (device, compute_type).

    ``auto`` prefers CUDA float16 when CTranslate2 reports a device, but CTranslate2 needs its
    own CUDA/cuBLAS runtime (torch's bundled CUDA does not satisfy it), so a failed load or a
    lazy ``cublas64_*.dll`` error falls back to CPU int8. Force with ``--device cuda|cpu``.
    """
    if device == "cpu":
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001
        pass
    return "cpu", "int8"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="视频或 16k 单声道 wav")
    ap.add_argument("--model", default="small", help="faster-whisper 模型名或本地目录（默认 small）")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--around", type=float, default=None, help="重点查看的时间点（秒）")
    ap.add_argument("--window", type=float, default=30.0, help="--around 前后查看的窗口（秒）")
    ap.add_argument("--phrases", nargs="*", default=[], help="要匹配的口令短语")
    ap.add_argument("--no-vad", action="store_true", help="关闭 VAD 过滤（短促喊声可能被 VAD 丢掉）")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=None, help="解码温度（默认由模型决定）")
    ap.add_argument("--no-condition", action="store_true",
                    help="不参考上一段文本（condition_on_previous_text=False，减少重复幻觉）")
    ap.add_argument("--start", type=float, default=None, help="只转写该时间之后的音频（秒）")
    ap.add_argument("--end", type=float, default=None, help="只转写该时间之前的音频（秒）")
    ap.add_argument("--normalize", action="store_true",
                    help="按 speech._normalize_level 放大安静音频（语音口令分析默认会做）")
    ap.add_argument("--out", default="", help="结果 JSON 输出路径（默认 data/cache/last_transcript.json）")
    args = ap.parse_args()

    ensure_dirs()
    cuda_dirs = ensure_cuda_dlls()
    if cuda_dirs:
        print(f"已注册 CUDA DLL 目录: {', '.join(cuda_dirs)}")
    wav = _resolve_audio(args.source)
    print(f"音轨: {wav}")

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    prompt = "".join(args.phrases) or None
    vad = not args.no_vad
    device, compute_type = _pick_device(args.device)

    audio = decode_audio(str(wav))
    sr = 16000
    offset = 0.0
    if args.start is not None:
        offset = max(0.0, args.start)
        audio = audio[int(offset * sr):]
    if args.end is not None:
        audio = audio[: int(max(0.0, args.end - offset) * sr)]
    gain = 1.0
    if args.normalize:
        from bms.analysis.speech import _normalize_level

        audio, gain = _normalize_level(audio)
    print(f"音频: {audio.size / sr:.1f}s（起点 {offset:.1f}s，归一化增益 {gain:.2f}）")

    print(f"加载模型 {args.model}（device={device}, compute_type={compute_type}）…")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001
        if device != "cuda":
            raise
        print(f"CUDA 加载失败（{type(e).__name__}: {e}），回退 CPU int8")
        device, compute_type = "cpu", "int8"
        model = WhisperModel(args.model, device=device, compute_type=compute_type)

    def run() -> tuple[list, object]:
        kwargs: dict = dict(
            language=args.language or None,
            vad_filter=vad,
            word_timestamps=True,
            initial_prompt=prompt,
            beam_size=args.beam_size,
            condition_on_previous_text=not args.no_condition,
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        it, info = model.transcribe(audio, **kwargs)
        return list(it), info

    print(f"转写中（language={args.language}, vad_filter={vad}, word_timestamps=True）…")
    t0 = time.time()
    try:
        segments_raw, info = run()
    except Exception as e:  # noqa: BLE001
        if device != "cuda":
            raise
        print(f"CUDA 运行失败（{type(e).__name__}: {e}），回退 CPU int8 重试")
        device, compute_type = "cpu", "int8"
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
        segments_raw, info = run()
    duration = float(getattr(info, "duration", 0.0) or 0.0) or audio.size / sr
    print(f"识别语言: {info.language}（{info.language_probability:.2f}）  时长: {duration:.1f}s")

    segments: list[dict] = []
    words: list[dict] = []
    last_pct = -1.0
    for seg in segments_raw:
        text = (seg.text or "").strip()
        segments.append({"start": round(seg.start + offset, 3), "end": round(seg.end + offset, 3), "text": text})
        for w in (seg.words or []):
            words.append({"start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
                          "word": w.word, "prob": round(w.probability, 3)})
        if duration > 0:
            pct = min(100.0, seg.end / duration * 100.0)
            if pct - last_pct >= 5.0:
                last_pct = pct
                print(f"  [{pct:5.1f}%] {seg.end:7.1f}s  {text}", flush=True)
    elapsed = time.time() - t0

    print(f"\n转写完成：{len(segments)} 段 / {len(words)} 词，耗时 {elapsed:.1f}s")
    print("\n全部片段：")
    for s in segments:
        print(f"  [{s['start']:7.1f} - {s['end']:7.1f}]  {s['text']}")

    if args.around is not None:
        lo, hi = args.around - args.window, args.around + args.window
        print(f"\n{args.around:.1f}s 附近 ±{args.window:.0f}s 的词级时间戳：")
        near = [w for w in words if lo <= w["start"] <= hi]
        if not near:
            print("  （该窗口内没有识别到词）")
        for w in near:
            print(f"  {w['start']:8.3f} - {w['end']:8.3f}  {w['word']}  (p={w['prob']})")

    hits: list[dict] = []
    for phrase in args.phrases:
        for s in segments:
            if phrase in s["text"]:
                hits.append({"t": s["start"], "phrase": phrase, "text": s["text"]})
    if args.phrases:
        print("\n口令命中：")
        if not hits:
            print(f"  未命中 {args.phrases}")
        for h in hits:
            print(f"  {h['t']:8.1f}s  {h['phrase']}  «{h['text']}»")

    out = Path(args.out) if args.out else ROOT / "data" / "cache" / "last_transcript.json"
    payload = {
        "model": args.model,
        "device": device,
        "language": info.language,
        "duration": duration,
        "phrases": args.phrases,
        "hits": hits,
        "segments": segments,
        "words": words,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
