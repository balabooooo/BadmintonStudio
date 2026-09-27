"""Voice command detection (optional): recognize 1~2 short user-configured phrases to add points to rallies.

Why faster-whisper + near-homophone matching
--------------------------------------------
The requirement only cares about fixed shouts like "好球" / "漂亮", not transcribing the whole
commentary. The previous Vosk grammar-constrained approach was fast but very brittle on a faint
far-field shout: the small Vosk model simply dropped the phrase. Whisper does open transcription
and is far more robust for short Chinese shouts, but it is **not** grammar-constrained, so it can
mis-hear the first syllable (a real "好球" often comes out as "到球" / "倒球").

Therefore the matching runs in two passes:

1. **exact** — the recognized characters equal the configured phrase;
2. **near-homophone** — same length, the **last character identical**, and the tone-less pinyin of
   every character within a small edit distance. That lets ``到球`` / ``倒球`` match ``好球``
   (``dao`` vs ``hao``, distance 1) while rejecting ``打球`` (``da``, distance 2) and ``要求``
   (last character differs).

Near-homophone matching needs :mod:`pypinyin`; if it is missing the module silently falls back to
exact matching (same degrade-gracefully convention as the player / pose / shuttle modules).

The input audio track directly uses the 16k mono PCM16 produced by
:func:`bms.core.media.ensure_audio`, so no extra decoding dependency is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import MODELS_DIR
from ..core.ct2_cuda import ensure_cuda_dlls
from ..i18n import tr
from .audio_hits import load_wav_mono

#: Maximum number of supported phrases
MAX_PHRASES = 2
#: Maximum number of characters per phrase
MAX_PHRASE_CHARS = 3
#: Supported faster-whisper model sizes (larger = more accurate, slower, bigger download)
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
#: Default model: medium is the smallest size that reliably hears a faint short shout.
DEFAULT_MODEL = "medium"
#: Allowed pinyin edit distance per character when near-homophone matching.
_MAX_PINYIN_DISTANCE = 1
#: Minimum probability of the token covering the phrase's **final** character. The final character
#: (e.g. the distinctive "球") is a reliable anchor; requiring it to be confident filters out the
#: Whisper "prompt echo" hallucination, which sprinkles low-confidence copies of the phrase into
#: silent stretches.
_MIN_ANCHOR_PROB = 0.4

#: Targeted recall pass (see ``refine_phrases``): length of each re-decoded window.
REFINE_WINDOW_S = 3.0
#: Step between consecutive re-decoded windows.
REFINE_HOP_S = 1.5
#: How far outside a rally region the sliding windows may reach.
REFINE_PAD_S = 1.0
#: Events closer than this (seconds) are treated as the same shout when merging / de-duplicating.
REFINE_DEDUP_S = 1.5

Progress = Callable[[float, str], None]


def _noop(p: float, m: str = "") -> None:
    pass


# ------------------------------------------------------------------ Phrase sanitizing


def sanitize_phrases(
    raw: object,
    max_phrases: int = MAX_PHRASES,
    max_chars: int = MAX_PHRASE_CHARS,
) -> list[str]:
    """Sanitize user-input phrases: strip whitespace, drop empty strings, dedupe, limit count, truncate each to ``max_chars``.

    The limits are also enforced on the **backend**, not only via the UI: project JSON, the
    command line, and data saved by old versions can all bypass the UI. The convention is to
    "drop / truncate the excess" rather than raise an error — a single mistyped word should not
    fail the entire analysis.
    """
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw, (list, tuple, set)):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        # Remove all whitespace (spaces in Chinese phrases are almost always a mistyped input)
        s = "".join(item.split())
        if not s:
            continue
        s = s[:max_chars]
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_phrases:
            break
    return out


# ------------------------------------------------------------------ Model locating


def _is_model_dir(p: Path) -> bool:
    """A faster-whisper model directory must contain the CTranslate2 weight and config."""
    try:
        return p.is_dir() and (p / "model.bin").exists() and (p / "config.json").exists()
    except OSError:
        return False


def resolve_model(model: str = DEFAULT_MODEL) -> str:
    """Resolve the whisper model to load.

    Priority: ``BMS_SPEECH_MODEL`` (a local directory or a size / HuggingFace repo id) -> a
    pre-downloaded ``models/faster-whisper-<size>`` (or ``whisper-<size>``) -> the size name,
    which faster-whisper resolves / downloads from HuggingFace on first use.
    Model files are not committed to the repo; pre-fetch them with
    ``scripts/fetch_speech_model.py``.
    """
    env = os.environ.get("BMS_SPEECH_MODEL")
    if env:
        p = Path(env).expanduser()
        if _is_model_dir(p):
            return str(p)
        return env
    size = str(model or DEFAULT_MODEL)
    base = Path(MODELS_DIR)
    for name in (f"faster-whisper-{size}", f"whisper-{size}"):
        p = base / name
        if _is_model_dir(p):
            return str(p)
    return size


# ------------------------------------------------------------------ Result structure


@dataclass
class SpeechDetection:
    """Detected phrase events."""

    #: Time at which each phrase occurred (seconds), sorted before returning
    times: np.ndarray
    #: Canonical phrases corresponding one-to-one with ``times``
    labels: list[str]
    #: Whether the module actually ran (False when a dependency / model / audio is missing)
    available: bool = False
    engine: str = "none"
    trace: dict[str, Any] = field(default_factory=dict)


def _empty(trace: dict[str, Any], engine: str = "none") -> SpeechDetection:
    return SpeechDetection(times=np.zeros(0), labels=[], available=False,
                           engine=engine, trace=trace)


# ------------------------------------------------------------------ Matching


def _pinyin(text: str) -> list[str] | None:
    """Tone-less pinyin per character, or ``None`` when pypinyin is unavailable."""
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        return [p for p in lazy_pinyin(text, style=Style.NORMAL, errors="ignore") if p]
    except Exception:  # noqa: BLE001
        return None


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _matches(candidate: str, phrase: str, fuzzy: bool) -> bool:
    """Whether a recognized ``candidate`` string counts as the configured ``phrase``."""
    if candidate == phrase:
        return True
    if not fuzzy:
        return False
    if len(phrase) < 2 or len(candidate) != len(phrase):
        return False
    # Anchor on the final character (e.g. the distinctive "球" in "好球") to avoid false positives.
    if candidate[-1] != phrase[-1]:
        return False
    pc = _pinyin(candidate)
    pp = _pinyin(phrase)
    if not pc or not pp or len(pc) != len(pp):
        return False
    return sum(_edit_distance(pc[i], pp[i]) for i in range(len(pp))) <= _MAX_PINYIN_DISTANCE


def _match_tokens_scored(tokens: list[tuple[str, float, float]], phrases: list[str],
                         fuzzy: bool, min_anchor_prob: float = 0.0,
                         ) -> list[tuple[float, str, float, float]]:
    """Find matching phrases within one segment's word list; return [(time, phrase, anchor_p, first_p)].

    ``tokens`` are ``(text, start, probability)`` from Whisper's word timestamps. Whisper may split
    a phrase across word tokens (or merge several into one), so characters are flattened into a
    stream and a window of the phrase length is slid over it. A window only counts when the token
    covering its **last** character is at least ``min_anchor_prob`` confident (see
    ``_MIN_ANCHOR_PROB``). Each phrase is reported at most once per segment, at its first
    qualifying occurrence.
    """
    hits: list[tuple[float, str, float, float]] = []
    if not tokens:
        return hits
    chars: list[tuple[str, float, float]] = []
    for text, start, prob in tokens:
        for ch in text:
            chars.append((ch, start, prob))
    stream = "".join(ch for ch, _, _ in chars)
    for phrase in phrases:
        lp = len(phrase)
        if lp == 0 or lp > len(stream):
            continue
        for i in range(len(stream) - lp + 1):
            if not _matches(stream[i:i + lp], phrase, fuzzy):
                continue
            if chars[i + lp - 1][2] < min_anchor_prob:
                continue
            hits.append((chars[i][1], phrase, chars[i + lp - 1][2], chars[i][2]))
            break
    return hits


def _match_tokens(tokens: list[tuple[str, float, float]], phrases: list[str],
                  fuzzy: bool, min_anchor_prob: float = 0.0) -> list[tuple[float, str]]:
    """Backward-compatible wrapper around :func:`_match_tokens_scored` (drops the probabilities)."""
    return [(t, phrase) for t, phrase, _, _ in
            _match_tokens_scored(tokens, phrases, fuzzy, min_anchor_prob)]


# ------------------------------------------------------------------ Audio level


#: Percentile of ``|x|`` used as the loudness reference when normalizing.
_NORM_REF_PCT = 95.0
#: Target reference level after normalization (fraction of full scale).
_NORM_TARGET = 0.25
#: Never amplify by more than this factor (avoids blowing up near-silent / noise-only audio).
_NORM_MAX_GAIN = 8.0


def _normalize_level(
    x: np.ndarray,
    target: float = _NORM_TARGET,
    max_gain: float = _NORM_MAX_GAIN,
) -> tuple[np.ndarray, float]:
    """Boost quiet audio toward ``target`` so a faint far-field shout still reaches the recognizer.

    Whisper is *mostly* gain-invariant, but very quiet footage (a distant spectator at a quiet
    moment) still loses the shout to the noise floor. Normalizing the level recovers recall on
    such clips. Only amplification is applied (never attenuation), and the gain is capped, so
    loud / already-clipped audio is left untouched.
    """
    if x.size == 0:
        return x, 1.0
    ref = float(np.percentile(np.abs(x), _NORM_REF_PCT))
    if ref < 1e-4:
        return x, 1.0
    gain = min(max_gain, max(1.0, target / ref))
    if gain <= 1.0001:
        return x, 1.0
    return np.clip(x * gain, -1.0, 1.0), gain


# ------------------------------------------------------------------ Detection


#: Cache of loaded models keyed by (model id, device, compute type); loading is expensive.
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _pick_device() -> tuple[str, str]:
    """Prefer CUDA float16 when CTranslate2 reports a GPU, else CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001
        pass
    return "cpu", "int8"


def _load_model(model_id: str, device: str, compute_type: str) -> Any:
    key = (model_id, device, compute_type)
    model = _MODEL_CACHE.get(key)
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_id, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = model
    return model


def detect_phrases(
    wav_path: str | Path,
    phrases: list[str],
    on_progress: Progress = _noop,
    cancel: Callable[[], bool] | None = None,
    model: str = DEFAULT_MODEL,
    fuzzy: bool = True,
) -> SpeechDetection:
    """Recognize phrases occurring in the audio track; return an event list with timestamps.

    Missing dependency / missing model / unreadable audio always **degrades**: ``available=False``
    + ``trace`` explains the reason, and no exception is raised (otherwise it would be swallowed
    by the pipeline's try/except and the UI would only see "no bonus").
    """
    clean = sanitize_phrases(phrases)
    if not clean:
        return _empty({"disabled": tr("speech.no_phrases")})

    # Read the audio first: it is cheap and lets a bad path / silent clip fail without loading a model.
    try:
        x, sr = load_wav_mono(wav_path)
    except Exception as e:  # noqa: BLE001
        return _empty({"error": tr("speech.read_audio_failed", detail=f"{type(e).__name__}: {e}"),
                       "phrases": clean})
    if x.size < sr // 4:
        return _empty({"error": tr("speech.audio_too_short"), "phrases": clean})

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as e:  # noqa: BLE001 - dependency is optional
        return _empty(
            {"error": tr("speech.whisper_missing", detail=f"{type(e).__name__}: {e}"),
             "hint": tr("speech.whisper_hint"), "phrases": clean},
            engine="missing",
        )

    ensure_cuda_dlls()
    model_id = resolve_model(model)
    device, compute_type = _pick_device()
    try:
        engine_model = _load_model(model_id, device, compute_type)
    except Exception as e:  # noqa: BLE001
        if device == "cuda":
            try:
                device, compute_type = "cpu", "int8"
                engine_model = _load_model(model_id, device, compute_type)
            except Exception as e2:  # noqa: BLE001
                return _empty({"error": tr("speech.load_model_failed", detail=f"{type(e2).__name__}: {e2}"),
                               "model": model_id, "phrases": clean}, engine="whisper")
        else:
            return _empty({"error": tr("speech.load_model_failed", detail=f"{type(e).__name__}: {e}"),
                           "model": model_id, "phrases": clean}, engine="whisper")

    # Level-normalize quiet audio before recognition (see ``_normalize_level``).
    x, applied_gain = _normalize_level(x)

    events: list[tuple[float, str]] = []
    try:
        segments, info = engine_model.transcribe(
            x,
            language="zh",
            vad_filter=False,
            word_timestamps=True,
            beam_size=5,
            # Seed the decoder with the configured phrases: a faint short shout is otherwise
            # frequently dropped or mangled by the open language model. ``condition_on_previous_text``
            # must stay off or the prompt leaks into later segments and produces a flood of false
            # positives (verified: on a test clip, cond=True yielded 30 hits vs 3 with cond=False).
            initial_prompt="".join(clean) or None,
            condition_on_previous_text=False,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0) or (x.size / max(sr, 1))
        for seg in segments:
            if cancel is not None and cancel():
                return _empty({"cancelled": True, "model": model_id,
                               "phrases": clean, "count": len(events)}, engine="whisper")
            words = [(str(w.word).strip(), float(w.start), float(getattr(w, "probability", 1.0) or 1.0))
                     for w in (seg.words or []) if str(w.word).strip()]
            if not words:
                text = str(seg.text or "").strip()
                if text:
                    # No word timestamps: fall back to the whole segment, with a probability derived
                    # from no_speech_prob (a silent stretch is where the prompt echo appears).
                    words = [(text, float(seg.start),
                              max(0.0, 1.0 - float(getattr(seg, "no_speech_prob", 0.0) or 0.0)))]
            events.extend(_match_tokens(words, clean, fuzzy, min_anchor_prob=_MIN_ANCHOR_PROB))
            on_progress(min(1.0, float(seg.end) / duration), tr("speech.recognizing"))
    except Exception as e:  # noqa: BLE001
        return _empty({"error": tr("speech.recognize_failed", detail=f"{type(e).__name__}: {e}"),
                       "device": device, "model": model_id, "phrases": clean}, engine="whisper")

    events.sort(key=lambda e: e[0])
    return SpeechDetection(
        times=np.asarray([t for t, _ in events], dtype=np.float64),
        labels=[p for _, p in events],
        available=True,
        engine="whisper",
        trace={"engine": "whisper", "model": model_id, "device": device, "phrases": clean,
               "fuzzy": bool(fuzzy), "count": len(events), "gain": round(applied_gain, 2)},
    )


# ------------------------------------------------------------------ Targeted recall pass


def _candidate_windows(
    regions: list[tuple[float, float]],
    window_s: float = REFINE_WINDOW_S,
    hop_s: float = REFINE_HOP_S,
    pad_s: float = REFINE_PAD_S,
) -> list[tuple[float, float]]:
    """Slide short windows over each ``(start, end)`` region; return sorted, de-duplicated windows.

    A region is padded by ``pad_s`` on both sides because a shout often sits right at (or just
    outside) a rally boundary. Windows start at the padded region start and step by ``hop_s``; one
    final window is appended so the tail is always covered. Windows are clamped to ``>= 0``.
    """
    win = max(0.1, float(window_s))
    hop = max(0.1, float(hop_s))
    pad = max(0.0, float(pad_s))
    out: list[tuple[float, float]] = []
    for a, b in regions or []:
        a = max(0.0, float(a) - pad)
        b = max(a + win, float(b) + pad)
        first = len(out)
        t = a
        while t + win <= b + 1e-9:
            out.append((round(t, 3), round(t + win, 3)))
            t += hop
        if len(out) == first or out[-1][1] < b - 1e-9:
            tail = max(a, b - win)
            out.append((round(tail, 3), round(tail + win, 3)))
    seen: set[tuple[float, float]] = set()
    uniq: list[tuple[float, float]] = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _normalize_base_events(
    events: list[dict[str, Any] | tuple[float, str]] | None,
) -> list[tuple[float, str]]:
    """Coerce the two accepted base-event shapes into ``[(t, phrase)]``.

    The main full-file pass is serialized as ``{"t": float, "phrase": str}`` dicts by
    :func:`bms.analysis.pipeline._speech_events` (that shape is what ``stats`` stores and what
    resegmentation reads back), while the merge helpers historically took ``(t, phrase)`` tuples.
    Accepting both at this boundary avoids an accidental ``float('t')`` when a caller hands over
    the canonical dict form. Malformed entries are skipped, mirroring :func:`_merge_events`.
    """
    out: list[tuple[float, str]] = []
    for e in events or []:
        try:
            if isinstance(e, dict):
                out.append((float(e.get("t", 0.0)), str(e.get("phrase", ""))))
            else:
                out.append((float(e[0]), str(e[1])))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return out


def _merge_events(
    base: list[tuple[float, str]] | None,
    refined: list[tuple[float, str, float, float]] | None,
    dedup_s: float = REFINE_DEDUP_S,
) -> list[tuple[float, str]]:
    """Merge full-file ``base`` events with short-window ``refined`` events, de-duplicating by time.

    ``base`` entries are ``(t, phrase)`` (already accepted by the full-file pass, treated as trusted
    with anchor probability 1.0); ``refined`` entries are ``(t, phrase, anchor_p, first_p)``. Events
    closer than ``dedup_s`` are one shout; the entry with the **highest anchor probability** wins
    (ties keep the earliest time). Returns sorted ``[(t, phrase)]``.
    """
    items: list[tuple[float, str, float]] = []
    for e in base or []:
        try:
            items.append((float(e[0]), str(e[1]), 1.0))
        except (TypeError, ValueError, IndexError):
            continue
    for e in refined or []:
        try:
            items.append((float(e[0]), str(e[1]), float(e[2])))
        except (TypeError, ValueError, IndexError):
            continue
    items.sort(key=lambda x: (x[0], -x[2]))
    merged: list[tuple[float, str, float]] = []
    for t, ph, p in items:
        if merged and t - merged[-1][0] < dedup_s:
            if p > merged[-1][2]:
                merged[-1] = (t, ph, p)
            continue
        merged.append((t, ph, p))
    return [(t, ph) for t, ph, _ in merged]


def refine_phrases(
    wav_path: str | Path,
    phrases: list[str],
    regions: list[tuple[float, float]],
    base_events: list[dict[str, Any] | tuple[float, str]] | None = None,
    on_progress: Progress = _noop,
    cancel: Callable[[], bool] | None = None,
    model: str = DEFAULT_MODEL,
    fuzzy: bool = True,
    window_s: float = REFINE_WINDOW_S,
    hop_s: float = REFINE_HOP_S,
    pad_s: float = REFINE_PAD_S,
    min_anchor_prob: float = _MIN_ANCHOR_PROB,
) -> SpeechDetection:
    """Second-chance recall pass: re-decode tight windows inside ``regions`` and merge the hits.

    Why this exists: the full-file decode is unstable on an ambiguous 30 s chunk -- a clear short
    shout can collapse into unrelated text (a real ``好球`` is heard as ``8比6`` / ``打球``) because
    its first syllable depends on the surrounding context. A tightly cropped window is far more
    reliable (verified 5/5 on the reported clip). Instead of re-decoding everything, only short
    windows over the candidate regions (rally spans) are re-decoded, so the cost is bounded by the
    active rally duration.

    Unlike :func:`detect_phrases`, the windows are decoded **without** ``initial_prompt`` (and
    without ``hotwords``): on a short, ambiguous window the prompt biases the model into emitting a
    copy of the phrase over non-speech audio, which floods the clip with false positives (on the
    reported clip, prompt on -> 32 hits vs prompt off -> exactly the 1 missed real shout). The main
    pass already covers the easy/direct hits; this pass only needs to recover the ones it dropped.
    Missing dependency / model / audio degrades to ``available=False`` with the base events
    preserved by the caller.
    """
    clean = sanitize_phrases(phrases)
    base = _normalize_base_events(base_events)
    if not clean or not regions:
        return _empty({"refine": {"disabled": tr("speech.refine_disabled")}, "count": len(base)},
                      engine="whisper")

    try:
        x, sr = load_wav_mono(wav_path)
    except Exception as e:  # noqa: BLE001
        return _empty({"error": tr("speech.read_audio_failed", detail=f"{type(e).__name__}: {e}"),
                       "phrases": clean})
    if x.size < sr // 4:
        return _empty({"error": tr("speech.audio_too_short"), "phrases": clean})

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as e:  # noqa: BLE001 - dependency is optional
        return _empty({"error": tr("speech.whisper_missing", detail=f"{type(e).__name__}: {e}"),
                       "hint": tr("speech.whisper_hint"), "phrases": clean}, engine="missing")

    ensure_cuda_dlls()
    model_id = resolve_model(model)
    device, compute_type = _pick_device()
    try:
        engine_model = _load_model(model_id, device, compute_type)
    except Exception as e:  # noqa: BLE001
        if device != "cuda":
            return _empty({"error": tr("speech.load_model_failed", detail=f"{type(e).__name__}: {e}"),
                           "model": model_id, "phrases": clean}, engine="whisper")
        try:
            device, compute_type = "cpu", "int8"
            engine_model = _load_model(model_id, device, compute_type)
        except Exception as e2:  # noqa: BLE001
            return _empty({"error": tr("speech.load_model_failed", detail=f"{type(e2).__name__}: {e2}"),
                           "model": model_id, "phrases": clean}, engine="whisper")

    x, applied_gain = _normalize_level(x)
    windows = _candidate_windows(regions, window_s, hop_s, pad_s)
    refined: list[tuple[float, str, float, float]] = []
    n = len(windows)
    error: str | None = None
    try:
        for i, (ws, we) in enumerate(windows):
            if cancel is not None and cancel():
                break
            lo = max(0, int(round(ws * sr)))
            hi = min(x.size, int(round(we * sr)))
            if hi - lo < sr // 4:
                continue
            # No initial_prompt / hotwords here: on a short window the prompt echo floods the
            # result with false positives (see the function docstring).
            segments, _info = engine_model.transcribe(
                x[lo:hi],
                language="zh",
                vad_filter=False,
                word_timestamps=True,
                beam_size=5,
                initial_prompt=None,
                condition_on_previous_text=False,
            )
            for seg in segments:
                words = [(str(w.word).strip(), ws + float(w.start),
                          float(getattr(w, "probability", 1.0) or 1.0))
                         for w in (seg.words or []) if str(w.word).strip()]
                if not words:
                    text = str(seg.text or "").strip()
                    if text:
                        words = [(text, ws + float(seg.start),
                                  max(0.0, 1.0 - float(getattr(seg, "no_speech_prob", 0.0) or 0.0)))]
                refined.extend(_match_tokens_scored(words, clean, fuzzy, min_anchor_prob))
            on_progress(min(1.0, (i + 1) / max(n, 1)), tr("speech.refining"))
    except Exception as e:  # noqa: BLE001 - keep the partial result rather than fail the analysis
        error = f"{type(e).__name__}: {e}"

    merged = _merge_events(base, refined)
    trace: dict[str, Any] = {
        "engine": "whisper", "model": model_id, "device": device, "phrases": clean,
        "fuzzy": bool(fuzzy), "gain": round(applied_gain, 2), "count": len(merged),
        "refine": {"windows": n, "regions": len(regions), "base": len(base),
                   "added": max(0, len(merged) - len(base))},
    }
    if error:
        trace["refine"]["error"] = error
    return SpeechDetection(
        times=np.asarray([t for t, _ in merged], dtype=np.float64),
        labels=[p for _, p in merged],
        available=True,
        engine="whisper",
        trace=trace,
    )


__all__ = [
    "SpeechDetection",
    "MAX_PHRASES",
    "MAX_PHRASE_CHARS",
    "WHISPER_SIZES",
    "DEFAULT_MODEL",
    "REFINE_WINDOW_S",
    "REFINE_HOP_S",
    "REFINE_PAD_S",
    "REFINE_DEDUP_S",
    "sanitize_phrases",
    "resolve_model",
    "detect_phrases",
    "refine_phrases",
]
