"""Rally segmentation fusion engine.

Fuses four complementary signals — audio hits, frame motion, player activity, and shuttle presence —
into a per-frame confidence that "this is a rally", then cuts rally intervals with a hysteresis state machine.

Design principles
-----------------
1. **Each signal is robustly normalized first** (quantile stretching) to avoid different dimensions overpowering each other.
2. **Weights adapt to reliability**: if a signal has poor discriminative power (Gaussian distribution,
   no bimodality), it is automatically down-weighted; this keeps things working when "the audio track is
   polluted by AGC" or "no players are detected in the frame".
3. **Hysteresis + minimum duration + gap merging**: avoid chopping a single rally apart, and avoid
   footsteps / audience movement triggering false rallies.
4. **Boundary snapping to audio hits**: when available, align the rally start to the first hit (serve).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio_hits import HitDetection, cluster_rallies

EPS = 1e-9


# ------------------------------------------------------------------ Utilities


def robust_norm(a: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    """Quantile-stretch to 0~1; insensitive to outliers."""
    if a is None or a.size == 0:
        return np.zeros(0, dtype=np.float32)
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_q), np.percentile(a, hi_q)
    if hi - lo < EPS:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Moving average."""
    if a.size == 0 or win <= 1:
        return a
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(ap, k, mode="valid")[: a.size].astype(np.float32)


def resample(a: np.ndarray, src_fps: float, dst_fps: float, dst_len: int) -> np.ndarray:
    """Linearly resample a signal along the time axis to the target length."""
    if a is None or a.size == 0:
        return np.zeros(dst_len, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-9 and a.size == dst_len:
        return a.astype(np.float32)
    src_t = np.arange(a.size) / max(src_fps, EPS)
    dst_t = np.arange(dst_len) / max(dst_fps, EPS)
    return np.interp(dst_t, src_t, a.astype(np.float32)).astype(np.float32)


def spikes_to_signal(times: np.ndarray, values: np.ndarray, fps: float, length: int,
                     decay: float = 1.0) -> np.ndarray:
    """Turn discrete events (hits) into a per-frame signal: assign at the hit, then exponential decay."""
    out = np.zeros(length, dtype=np.float32)
    if times is None or times.size == 0:
        return out
    idx = np.clip(np.round(np.asarray(times) * fps).astype(np.int64), 0, max(0, length - 1))
    v = values if values is not None and len(values) == len(times) else np.ones(len(times), dtype=np.float32)
    np.maximum.at(out, idx, np.asarray(v, dtype=np.float32))
    if decay <= 0:
        return out
    # Exponential-decay impulse response: y[i] = max(out[i], decay * y[i-1])
    for i in range(1, length):
        p = out[i - 1] * decay
        if p > out[i]:
            out[i] = p
    return out


def hit_density_signal(
    hits: HitDetection | None,
    fps: float,
    length: int,
    window: float = 2.0,
    min_confidence: float = 0.15,
) -> np.ndarray:
    """Turn audio hits into a density curve (0~1) for "is there continuous hitting around here".

    Why it is needed: in a multi-court gym a single hit sound is untrustworthy (the neighboring court
    is also making noise), but "several shots occurred within a stretch of time" is still the most
    direct observation that our match is underway. Measured (on footage with `audio_reliability` of
    only 0.23): the average hit density in a 2-second window within rallies is 1.6x that between
    rallies, with a discriminative AUC≈0.78; whereas the AUC of frame motion / player speed is only
    0.5~0.6. The old fusion squeezed this signal down to weight 0.13 via `audio_reliability`, which
    amounts to discarding the most informative signal.

    A sliding-window count is used (rather than single-pulse decay) because the key evidence is
    "density" rather than "how loud some sound is": an isolated shot (picking up the shuttle, the
    neighboring court) will not form density; only continuous rallying will.
    """
    out = np.zeros(max(0, int(length)), dtype=np.float32)
    if hits is None or hits.times.size == 0 or length <= 0:
        return out
    keep = hits.confidence >= min_confidence
    times = hits.times[keep] if keep.any() else hits.times
    if times.size == 0:
        return out
    idx = np.clip(np.round(times * fps).astype(np.int64), 0, out.size - 1)
    np.add.at(out, idx, 1.0)
    out = smooth(out, max(1, int(round(window * fps))))
    return robust_norm(out)


def discriminative_power(a: np.ndarray) -> float:
    """Estimate whether a signal "has a bimodal structure", as an adaptive weight.

    Measured by (p90 - p50) / (p90 - p10): the more the values concentrate in the upper part, the more
    it looks like a sparse-event signal where "a few moments are clearly high", and the more it deserves
    a high weight.
    """
    if a is None or a.size < 32:
        return 0.35
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    if p90 - p10 < EPS:
        return 0.05
    sep = float((p90 - p50) / (p90 - p10))
    # sep≈0.5 means uniform/symmetric (no information); the closer to 0.8~0.95, the more it looks like an impulse signal
    return float(np.clip((sep - 0.42) / 0.45, 0.05, 1.0))


# ------------------------------------------------------------------ Fusion


@dataclass
class FusedSignal:
    fps: float
    duration: float
    activity: np.ndarray                 # fused activity 0~1
    components: dict[str, np.ndarray] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    threshold_hi: float = 0.5
    threshold_lo: float = 0.3
    #: Reliability of the audio hit signal 0~1 (near 0 with multiple courts / AGC pollution)
    audio_reliability: float = 1.0


@dataclass
class RallyInterval:
    start: float
    end: float
    confidence: float = 0.5
    #: Indices of audio hits falling within the interval
    hit_indices: list[int] = field(default_factory=list)
    serve_time: float | None = None
    serve_side: str = "unknown"
    receive_time: float | None = None
    receive_side: str = "unknown"
    features: dict[str, float] = field(default_factory=dict)


def hit_reliability(hits: HitDetection | None, duration: float, fps: float,
                    n: int) -> tuple[float, float]:
    """Assess how trustworthy the audio hit signal actually is.

    A multi-court gym + camera AGC make "hit sounds" appear uniformly across the whole timeline; such
    a signal contains almost no rally start/end information, and forcibly weighting it would glue a
    single rally into several minutes.

    Returns ``(reliability 0~1, gate strength quantile)``:

    - ``coverage``: fraction of time covered by "a hit within 2 seconds". A real match is usually
      0.25~0.55; if > 0.8 it means the sound is almost always there, basically noise or other courts.
    - ``rate``: average touches per second. The realistic upper bound for a single badminton court is about 2.5/s.
    - ``burst``: fraction of hits with adjacent separation < 0.35s, which is high in real rallies.
    """
    if hits is None or hits.times.size < 8 or duration <= 0:
        return 0.0, 0.0
    t = hits.times
    rate = t.size / max(duration, 1e-6)
    # Coverage: split the timeline into 2-second bins and count the fraction of bins containing hits
    nb = max(1, int(np.ceil(duration / 2.0)))
    idx = np.clip((t / 2.0).astype(np.int64), 0, nb - 1)
    coverage = float(np.unique(idx).size / nb)
    gaps = np.diff(t)
    burst = float(np.mean(gaps < 0.35)) if gaps.size else 0.0

    rate_ok = float(np.clip((3.2 - rate) / 2.2, 0.05, 1.0))       # >3.2/s is judged as noise
    cover_ok = float(np.clip((0.86 - coverage) / 0.36, 0.05, 1.0))  # >0.86 is judged as noise
    burst_ok = float(np.clip(burst / 0.35, 0.2, 1.0))
    rel = float(np.clip(rate_ok * cover_ok * (0.55 + 0.45 * burst_ok), 0.0, 1.0))

    # Strength threshold: keep only the relatively strong portion of hits
    gate = float(np.percentile(hits.strength, 35)) if hits.strength.size else 0.0
    return rel, gate


def fuse(
    fps: float,
    duration: float,
    hits: HitDetection | None = None,
    motion: dict[str, np.ndarray] | None = None,
    players: dict[str, np.ndarray] | None = None,
    shuttle: dict[str, np.ndarray] | None = None,
    roi_activity: np.ndarray | None = None,
) -> FusedSignal:
    """Fuse the various baseline signals into a single activity curve."""
    n = max(1, int(round(duration * fps)))
    comp: dict[str, np.ndarray] = {}

    # --- Audio: hit density + strength (reliability gating and strength filtering first)
    audio_rel = 0.0
    if hits is not None and hits.times.size:
        audio_rel, gate = hit_reliability(hits, duration, fps, n)
        keep = (hits.strength >= gate) & (hits.confidence >= 0.15)
        times = hits.times[keep]
        strength = hits.strength[keep]
        if times.size:
            density = spikes_to_signal(times, strength, fps, n, decay=0.93)
            cnt = np.zeros(n, dtype=np.float32)
            idx = np.clip(np.round(times * fps).astype(np.int64), 0, n - 1)
            np.add.at(cnt, idx, 1.0)
            dens = smooth(cnt, max(1, int(fps * 2.0)))
            comp["audio_hits"] = 0.6 * robust_norm(dens) + 0.4 * robust_norm(density)
        else:
            comp["audio_hits"] = np.zeros(n, dtype=np.float32)
    else:
        comp["audio_hits"] = np.zeros(n, dtype=np.float32)

    # --- Frame motion
    if motion:
        mfps = float(motion.get("fps", fps))
        court = motion.get("court_motion")
        if court is None or not np.any(court):
            court = motion.get("motion", np.zeros(0))
        m = resample(court, mfps, fps, n)
        comp["motion"] = robust_norm(smooth(m, max(1, int(fps * 0.8))))
    else:
        comp["motion"] = np.zeros(n, dtype=np.float32)

    # --- Player activity
    if players:
        pfps = float(players.get("fps", fps))
        cnt = resample(players.get("active_count", np.zeros(0)), pfps, fps, n)
        spd = resample(players.get("active_speed", np.zeros(0)), pfps, fps, n)
        mspd = resample(players.get("max_speed", np.zeros(0)), pfps, fps, n)
        on_court = np.clip(cnt / 2.0, 0.0, 1.0)
        # Player speed is pulse-like: run a step, stop, run another step. So "how much of the last few
        # seconds was spent moving fast" distinguishes "rallying" from "walking" better than the instantaneous average speed.
        thr = float(np.percentile(spd, 72)) if spd.size else 0.0
        burst = smooth((spd > thr).astype(np.float32), max(1, int(fps * 3.0)))
        comp["players"] = (
            0.34 * robust_norm(smooth(spd, max(1, int(fps * 1.2))), 10, 92)
            + 0.34 * robust_norm(burst)
            + 0.18 * robust_norm(smooth(mspd, max(1, int(fps * 0.8))), 10, 92)
            + 0.14 * robust_norm(smooth(on_court * spd, max(1, int(fps * 2.0))), 10, 92)
        ).astype(np.float32)
    else:
        comp["players"] = np.zeros(n, dtype=np.float32)

    # --- Shuttle presence
    if shuttle:
        sfps = float(shuttle.get("fps", fps))
        pres = resample(shuttle.get("presence", np.zeros(0)), sfps, fps, n)
        spd = resample(shuttle.get("max_candidate_speed", np.zeros(0)), sfps, fps, n)
        comp["shuttle"] = (0.6 * robust_norm(smooth(pres, max(1, int(fps * 0.5)))) +
                           0.4 * robust_norm(smooth(spd, max(1, int(fps * 0.5))))).astype(np.float32)
    else:
        comp["shuttle"] = np.zeros(n, dtype=np.float32)

    # --- External ROI activity
    if roi_activity is not None and roi_activity.size:
        rfps = float(motion.get("fps", fps)) if motion else fps
        comp["roi"] = robust_norm(resample(roi_activity, rfps, fps, n))
    else:
        comp["roi"] = np.zeros(n, dtype=np.float32)

    # --- Adaptive weights (audio is additionally multiplied by its reliability)
    base = {"players": 1.35, "motion": 1.0, "audio_hits": 0.95, "shuttle": 0.9, "roi": 0.8}
    weights: dict[str, float] = {}
    for k, v in comp.items():
        if not np.any(v):
            weights[k] = 0.0
            continue
        w = base.get(k, 0.5) * (0.35 + 0.65 * discriminative_power(v))
        if k == "audio_hits":
            w *= audio_rel
        weights[k] = w
    total = sum(weights.values())
    if total < EPS:
        # When all signals are unavailable, degrade to "the whole clip is a candidate", to be fixed by later audio / manual edits
        weights = {"audio_hits": 1.0}
        total = 1.0
        comp["audio_hits"] = np.ones(n, dtype=np.float32) * 0.5

    activity = np.zeros(n, dtype=np.float32)
    for k, v in comp.items():
        activity += (weights[k] / total) * v
    activity = smooth(activity, max(1, int(fps * 1.2)))

    # --- Adaptive thresholds (dual-threshold hysteresis)
    p_hi = float(np.percentile(activity, 78))
    p_lo = float(np.percentile(activity, 55))
    med = float(np.median(activity))
    hi = max(p_hi, med * 1.25)
    lo = max(p_lo * 0.92, med * 1.05)
    return FusedSignal(fps=fps, duration=duration, activity=activity,
                       components=comp, weights=weights, threshold_hi=hi, threshold_lo=lo,
                       audio_reliability=audio_rel)


def _fps_of(sig: dict | None, key: str, default: float) -> float:
    if not sig:
        return default
    try:
        return float(sig.get("fps", default))
    except Exception:
        return default


# ------------------------------------------------------------------ State machine


def segment(
    sig: FusedSignal,
    gap_seconds: float = 3.0,
    min_seconds: float = 3.0,
    max_seconds: float = 120.0,
    min_on_seconds: float = 0.7,
    min_off_seconds: float = 1.6,
    pad_start: float = 0.0,
    pad_end: float = 0.0,
    target_seconds: float = 28.0,
    split_sensitivity: float = 0.5,
) -> list[RallyInterval]:
    """A hysteresis state machine + secondary splitting by "typical rally duration".

    Why secondary splitting is needed: in training / multi-shuttle practice players pause only a few
    seconds between rallies, so the activity curve does not collapse below the low threshold and the
    hysteresis state machine glues several consecutive rallies into one "rally" lasting tens or even
    hundreds of seconds. So over-long intervals are recursively split at the **lowest activity point**
    until each segment does not exceed a reasonable multiple of ``target_seconds``.

    ``split_sensitivity`` 0~1 controls how aggressive it is: higher means a shorter target duration and a higher exit threshold.
    """
    a = sig.activity
    fps = sig.fps
    n = a.size
    if n == 0:
        return []

    s = float(np.clip(split_sensitivity, 0.0, 1.0))
    # Target duration: conservative 48s -> aggressive 16s
    target = float(np.clip(target_seconds, 8.0, 180.0)) * (1.35 - 0.7 * s)
    target = float(np.clip(target, 8.0, 180.0))

    hi = sig.threshold_hi * (1.0 - 0.12 * s)
    lo = sig.threshold_lo * (1.0 + 0.28 * s)
    on_need = max(1, int(round(min_on_seconds * fps)))
    # The more aggressive, the earlier it decides "this segment has ended"
    off_need = max(1, int(round(min_off_seconds * fps * (1.35 - 0.7 * s))))

    intervals: list[tuple[int, int]] = []
    state = 0
    on_run = off_run = 0
    start_i = 0
    for i in range(n):
        if state == 0:
            on_run = on_run + 1 if a[i] >= hi else 0
            if on_run >= on_need:
                state = 1
                start_i = i - on_run + 1
                off_run = 0
        else:
            off_run = off_run + 1 if a[i] < lo else 0
            if off_run >= off_need:
                intervals.append((start_i, i - off_run + 1))
                state = 0
                on_run = 0
    if state == 1:
        intervals.append((start_i, n - 1))

    if not intervals:
        return []

    # Merge intervals that are very close (also affected by how aggressive it is)
    merged: list[list[int]] = [list(intervals[0])]
    gap_frames = int(round(max(0.6, gap_seconds * (1.3 - 0.75 * s)) * fps))
    for s0, e0 in intervals[1:]:
        if s0 - merged[-1][1] <= gap_frames:
            merged[-1][1] = e0
        else:
            merged.append([s0, e0])

    out: list[RallyInterval] = []
    min_f = int(round(min_seconds * fps))
    max_f = int(round(max_seconds * fps)) if max_seconds > 0 else 10**9
    target_f = int(round(target * fps))
    for s0, e0 in merged:
        if e0 - s0 < min_f:
            continue
        for ps, pe in _split_by_valley(sig, s0, e0, target_f, max(min_f, int(target_f * 0.35))):
            if pe - ps < min_f:
                continue
            conf = float(np.clip(np.mean(a[ps : pe + 1]) / max(sig.threshold_hi, EPS), 0, 1.5)) / 1.5
            out.append(RallyInterval(
                start=max(0.0, ps / fps - pad_start),
                end=min(sig.duration, (pe + 1) / fps + pad_end),
                confidence=float(np.clip(conf, 0.05, 1.0)),
            ))
    return out


def _split_by_valley(sig: FusedSignal, s: int, e: int, target_f: int, min_f: int) -> list[tuple[int, int]]:
    """Recursively split an overly long interval at the "local activity minimum".

    The cut point is chosen as the minimum rather than an equal division, so it lands on the "picking
    up the shuttle / wiping sweat" gap as much as possible; at the same time a little margin is left
    around the cut point, to avoid chopping off a shot's follow-through.
    """
    if e - s <= target_f * 1.35 or target_f <= 0:
        return [(s, e)]
    a = sig.activity
    rel = int(np.argmin(a[s + min_f : e - min_f])) + s + min_f if (e - min_f) - (s + min_f) > 2 else (s + e) // 2
    if rel <= s + min_f or rel >= e - min_f:
        rel = (s + e) // 2
    left = _split_by_valley(sig, s, rel, target_f, min_f)
    right = _split_by_valley(sig, rel, e, target_f, min_f)
    return left + right


# ------------------------------------------------------------------ Boundary snapping and serve detection


def thin_shots(times: np.ndarray, strength: np.ndarray, confidence: np.ndarray,
               min_gap: float = 0.28) -> np.ndarray:
    """Pick a physically plausible sequence out of the candidate hits.

    In badminton, two hits by the same side cannot be less than about 0.3 seconds apart (amateurs are
    slower), so "4 shots per second" within a rally is almost certainly detection noise. Here the hits
    are greedily picked from highest to lowest strength, ensuring the separation between any two selected
    hits is >= ``min_gap``, then sorted by time.
    """
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    score = strength * (0.5 + 0.5 * np.clip(confidence, 0, 1))
    order = np.argsort(-score)
    chosen: list[int] = []
    for i in order:
        t = times[i]
        if all(abs(t - times[j]) >= min_gap for j in chosen):
            chosen.append(int(i))
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.array(sorted(chosen), dtype=np.int64)


#: Conservative lower bound for the **maximum** separation (seconds) between two hits within a rally.
#:
#: This is based on physics rather than tuning: the shot interval within a rally is determined by the
#: shuttle's flight time — measured on amateur footage, the maximum within a segment is 2.4~3.6 seconds,
#: with the vast majority falling in 0.4~1.5 seconds; whereas between two rallies there is necessarily
#: picking up the shuttle / changing serve / walking back to the receiving position, and the time with
#: **nobody hitting** is generally >= 4 seconds. So "a large gap in the hit sequence" is almost exactly a
#: rally boundary — this is far more reliable than "whether the players stopped", because players keep
#: walking while picking up the shuttle and the activity never collapses.
MAX_INTRA_HIT_GAP = 3.0

#: Seconds to look on each side when taking the hit window. The boundaries given by segmentation are
#: themselves coarse (they may fall after the serve, or before the last shot), so looking a little
#: outward is needed to catch the serve / follow-through shot. But it **must not be too large**, otherwise
#: it will swallow hits from the neighboring rally (see :func:`refine_with_hits`).
_HIT_TOL_BEFORE = 1.0
_HIT_TOL_AFTER = 1.2


def hit_gap_limit(
    times: np.ndarray,
    floor: float = MAX_INTRA_HIT_GAP,
    mult: float = 1.8,
    quantile: float = 70.0,
    cap: float = 4.5,
) -> float:
    """Estimate the "maximum shot separation allowed within a rally".

    Take a **typical** shot interval (p70) and multiply by a margin, rather than taking a high quantile:
    the higher the quantile, the more it is itself raised by "the large gaps of seconds to tens of seconds
    between rallies" — measured after gating (only our own hits remain), p80 is already at 2.42 seconds,
    and multiplying by 1.8 gives a threshold of 4.36 seconds, so real pauses of 3.7 seconds and 3.1 seconds
    all fail to be split, and a rally of over 20 seconds just stays there.

    ``floor`` is the physical lower bound (the shot interval within a rally is determined by the shuttle's
    flight time), and ``cap`` prevents the threshold from being relaxed into uselessness when the sequence
    itself is very sparse.
    """
    if times is None or len(times) < 3:
        return float(floor)
    g = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    g = g[g > 1e-6]
    if g.size < 2:
        return float(floor)
    return float(min(cap, max(floor, mult * float(np.percentile(g, quantile)))))


def _hit_idx_in_window(hits: HitDetection, t0: float, t1: float,
                       min_confidence: float = 0.18) -> np.ndarray:
    """Indices of hits "like our match" within ``[t0, t1]``.

    Two filters: a confidence threshold + the physical plausibility screening of :func:`thin_shots`
    (two hits by the same side cannot be less than about 0.3 seconds apart; "4 shots per second" must be noise).
    """
    t = hits.times
    idx = np.nonzero((t >= t0) & (t <= t1))[0]
    if idx.size == 0:
        return idx
    idx = idx[hits.confidence[idx] >= min_confidence]
    if idx.size == 0:
        return idx
    sub = thin_shots(t[idx], hits.strength[idx], hits.confidence[idx])
    return idx[sub]


def split_by_hit_gaps(
    intervals: list[RallyInterval],
    hits: HitDetection | None,
    limit: float | None = None,
    min_side_hits: int = 2,
    min_side_seconds: float = 2.0,
) -> list[RallyInterval]:
    """Split intervals "containing an overly large shot separation" at the middle of the gap.

    This is the correct solution to "one rally containing the next rally". The old implementation relied
    on "player quiet segments" and "activity valleys", both of which fail in a multi-court gym: players
    walk while picking up the shuttle (activity does not collapse), and full-frame motion is continuously
    lit up by the neighboring court (the valley disappears). So several rallies get glued into an interval
    of tens of seconds. But **whether the shuttle is being hit** is the direct observation of this: once the
    hit sequence contains a gap far exceeding the normal shot interval, those two stretches must be two rallies.

    ``min_side_hits`` / ``min_side_seconds`` are guardrails against false splits: both sides must really have
    enough shots and each be able to support a rally, otherwise it is better to leave it unsplit.
    """
    if hits is None or hits.times.size == 0 or not intervals:
        return intervals
    lim = float(limit) if limit is not None else hit_gap_limit(hits.times)
    out: list[RallyInterval] = []

    def _recurse(iv: RallyInterval, depth: int = 0) -> None:
        if depth > 6 or iv.end - iv.start < 2.0 * min_side_seconds:
            out.append(iv)
            return
        idx = _hit_idx_in_window(hits, iv.start, iv.end)
        if idx.size < 2 * min_side_hits:
            out.append(iv)
            return
        ts = hits.times[idx]
        gaps = np.diff(ts)
        k = int(np.argmax(gaps))
        if gaps[k] <= lim:
            out.append(iv)
            return
        left, right = ts[: k + 1], ts[k + 1:]
        # Guardrail: both sides must have enough material, otherwise this is not "two rallies" but "a few missed hits"
        if (left.size < min_side_hits or right.size < min_side_hits
                or (left[-1] - left[0]) < min_side_seconds
                or (right[-1] - right[0]) < min_side_seconds):
            out.append(iv)
            return
        cut = float(ts[k] + gaps[k] / 2.0)
        a = RallyInterval(start=iv.start, end=cut, confidence=iv.confidence)
        b = RallyInterval(start=cut, end=iv.end, confidence=iv.confidence)
        for nb in (a, b):
            nb.serve_side = iv.serve_side
            nb.receive_side = iv.receive_side
        _recurse(a, depth + 1)
        _recurse(b, depth + 1)

    for iv in intervals:
        _recurse(iv)
    return sorted(out, key=lambda v: v.start)


def refine_with_hits(
    intervals: list[RallyInterval],
    hits: HitDetection,
    pre_roll: float = 1.0,
    post_roll: float = 1.6,
    tail_seconds: float = 0.9,
    min_hits: int = 1,
    split: bool = True,
    trim_start: bool = True,
    gap_limit: float | None = None,
) -> list[RallyInterval]:
    """Correct rally boundaries with audio hits, and **allow tightening the end point**.

    The old implementation wrote ``iv.end = max(iv.end, last + tail)`` here — the end could only be pushed
    later, never pulled earlier. So no matter how long the interval given by segmentation was (measured:
    a 69-second "rally" with 72 shots), the hit information **could not** cut off the stretch after the
    shuttle landed. This is exactly the direct cause of "a rally containing a long time after the shuttle lands".

    Current rules:

    1. First split intervals by large gaps in the hit sequence (:func:`split_by_hit_gaps`), eliminating
       "one rally containing the next rally";
    2. The end is **anchored to the last shot**: ``iv.end = last + tail_seconds``.
       But tightening is only dared when there is evidence — the criterion is "the next hit globally is more
       than ``gap_limit`` away from ``last``": if even the neighboring court's sound is absent, this stretch
       really has nobody playing. If there is a hit immediately after, the shuttle is still in flight, so the
       original later end is kept.
    3. The start is still aligned to ``pre_roll`` before the first hit (serve);
       ``trim_start=False`` is used for player-motion segmentation (its start is "the player starts moving",
       which already includes the serve preparation and should not be delayed).
    """
    if hits is None or hits.times.size == 0:
        for iv in intervals:
            iv.start = max(0.0, iv.start - pre_roll)
            iv.end = iv.end + post_roll
        return intervals

    t = hits.times
    lim = float(gap_limit) if gap_limit is not None else hit_gap_limit(t)
    if split:
        intervals = split_by_hit_gaps(intervals, hits, limit=lim)

    # When taking the hit window, **do not cross into neighbors**: one hit can belong to only one rally.
    # Without this clamping, both sides of the cut would count the same shot (the left as "last shot", the
    # right as "first hit"), so the left interval keeps 0.9s forward and the right keeps 1.2s backward, and
    # the two segments overlap by 2.1 seconds — downstream dedupe_overlaps cuts in the middle, the cut is
    # again end-to-end, and finally _join_abutting merges it back, making the split pointless.
    ordered = sorted(intervals, key=lambda v: v.start)
    bounds: list[tuple[float, float]] = []
    for k, iv in enumerate(ordered):
        prev_end = ordered[k - 1].end if k > 0 else -np.inf
        next_start = ordered[k + 1].start if k + 1 < len(ordered) else np.inf
        lo = max(iv.start - _HIT_TOL_BEFORE, prev_end)
        hi = min(iv.end + _HIT_TOL_AFTER, next_start)
        if hi < lo:
            lo = hi = (iv.start + iv.end) / 2.0
        bounds.append((lo, hi))

    out: list[RallyInterval] = []
    for iv, (lo, hi) in zip(ordered, bounds):
        idx = _hit_idx_in_window(hits, lo, hi)
        iv.hit_indices = idx.tolist()
        if idx.size >= min_hits:
            first = float(t[idx[0]])
            # The "last shot" is only searched inside the interval: the extra 1.6 seconds on the right of
            # the window is the remediation margin for "the active interval ending earlier than the last shot",
            # and must not be used as the end anchor.
            inside = idx[t[idx] <= iv.end]
            last = float(t[inside[-1]]) if inside.size else first
            anchor = last + max(0.0, tail_seconds)

            # How far is the next hit globally from the last shot? If it exceeds gap_limit, this stretch really has ended.
            nxt = int(np.searchsorted(t, last + 1e-6, side="right"))
            gap_after = float(t[nxt] - last) if nxt < t.size else float("inf")
            if gap_after > lim:
                iv.end = min(iv.end, anchor)          # tighten
            else:
                iv.end = max(iv.end, anchor)          # there are more shots after, do not cut it off
            iv.end = max(iv.end, last + 0.05)
            if trim_start:
                iv.start = max(0.0, first - pre_roll)
            else:
                iv.start = min(iv.start, max(0.0, first - pre_roll))
            iv.features["hit_anchored"] = 1.0
            iv.features["tail_gap"] = round(min(gap_after, 999.0), 2)
        else:
            iv.start = max(0.0, iv.start - pre_roll)
            iv.end = iv.end + post_roll
        if idx.size >= 1:
            iv.serve_time = float(t[idx[0]])
        if idx.size >= 2:
            iv.receive_time = float(t[idx[1]])
        out.append(iv)
    return [iv for iv in out if iv.end > iv.start]


def attach_features(
    intervals: list[RallyInterval],
    sig: FusedSignal,
    hits: HitDetection | None = None,
    motion: dict[str, np.ndarray] | None = None,
    players: dict[str, np.ndarray] | None = None,
    shuttle: dict[str, np.ndarray] | None = None,
) -> list[RallyInterval]:
    """Compute objective features for each rally, for subsequent scoring."""
    fps = sig.fps
    for iv in intervals:
        a, b = int(iv.start * fps), int(iv.end * fps)
        a, b = max(0, a), min(sig.activity.size, max(b, a + 1))
        seg = sig.activity[a:b]
        # Start from the existing features rather than creating a new dict: the hit_anchored / tail_gap
        # written in during boundary anchoring are diagnostic information that the UI relies on to explain
        # "why the end is here". The old code swapped in a brand-new dict here, throwing them all away.
        f: dict[str, float] = dict(iv.features)
        f.update({
            "duration": float(iv.end - iv.start),
            "activity_mean": float(seg.mean()) if seg.size else 0.0,
            "activity_peak": float(seg.max()) if seg.size else 0.0,
            "confidence": float(iv.confidence),
        })
        if hits is not None and iv.hit_indices:
            hs = hits.strength[iv.hit_indices]
            hc = hits.confidence[iv.hit_indices]
            ts = hits.times[iv.hit_indices]
            f["shot_count"] = float(len(iv.hit_indices))
            f["hit_strength_mean"] = float(hs.mean())
            f["hit_strength_p90"] = float(np.percentile(hs, 90))
            f["hit_conf_mean"] = float(hc.mean())
            if len(ts) > 1:
                d = np.diff(ts)
                f["tempo"] = float(1.0 / max(np.median(d), 1e-3))
                f["rally_span"] = float(ts[-1] - ts[0])
                # Late-rally tempo (the denser, the more intense)
                tail = d[-max(2, len(d) // 3):]
                f["finish_tempo"] = float(1.0 / max(np.median(tail), 1e-3))
                # Tempo variance: high variance = unpredictable/unstable exchange
                f["tempo_variance"] = float(np.var(d)) if d.size else 0.0
                # Confrontation streak: longest run of gaps <= 0.85s (rapid fire exchange)
                if d.size:
                    streak = max_run = 1
                    for gap in d:
                        if gap <= 0.85:
                            streak += 1
                            max_run = max(max_run, streak)
                        else:
                            streak = 1
                    f["confrontation_streak"] = float(max_run)
                else:
                    f["confrontation_streak"] = 1.0
            # Smash proxy: hits whose strength is in the top 15% of the rally (simple proxy for smash)
            if hs.size:
                thr = float(np.percentile(hs, 85))
                f["smash_proxy"] = float(np.count_nonzero(hs >= thr))
            else:
                f["smash_proxy"] = 0.0
        if players:
            pfps = float(players.get("fps", fps))
            f["player_speed_mean"] = _seg_mean(players.get("active_speed"), pfps, iv.start, iv.end)
            f["player_speed_max"] = _seg_max(players.get("max_speed"), pfps, iv.start, iv.end)
            f["active_count_mean"] = _seg_mean(players.get("active_count"), pfps, iv.start, iv.end)
        if motion:
            mfps = float(motion.get("fps", fps))
            f["motion_mean"] = _seg_mean(motion.get("court_motion", motion.get("motion")), mfps, iv.start, iv.end)
            f["motion_peak"] = _seg_max(motion.get("motion"), mfps, iv.start, iv.end)
        if shuttle:
            sfps = float(shuttle.get("fps", fps))
            f["shuttle_presence"] = _seg_mean(shuttle.get("presence"), sfps, iv.start, iv.end)
            f["shuttle_speed_p90"] = _seg_pct(shuttle.get("max_candidate_speed"), sfps, iv.start, iv.end, 90)
            f["shuttle_speed_max"] = _seg_max(shuttle.get("max_candidate_speed"), sfps, iv.start, iv.end)
        iv.features = f
    return intervals


def _slice(arr, fps, t0, t1):
    if arr is None or len(arr) == 0:
        return None
    a = max(0, int(t0 * fps))
    b = min(len(arr), max(a + 1, int(t1 * fps)))
    if b <= a:
        return None
    return np.asarray(arr[a:b], dtype=np.float32)


def _seg_mean(arr, fps, t0, t1) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(s.mean()) if s is not None and s.size else 0.0


def _seg_max(arr, fps, t0, t1) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(s.max()) if s is not None and s.size else 0.0


def _seg_pct(arr, fps, t0, t1, q) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(np.percentile(s, q)) if s is not None and s.size else 0.0


def dedupe_overlaps(intervals: list[RallyInterval], hits: HitDetection | None = None,
                    fps: float = 15.0, activity: np.ndarray | None = None) -> list[RallyInterval]:
    """Eliminate overlap between adjacent rallies.

    During boundary snapping each rally keeps ``pre_roll`` forward and ``tail`` backward, so two nearby
    rallies cover each other by a few seconds. Using them directly to build the timeline would show the
    same footage twice. Here the overlap region is cut at "the largest hit separation" (or the lowest
    activity point), which is more natural than simply taking the midpoint.
    """
    if len(intervals) < 2:
        return intervals
    intervals = sorted(intervals, key=lambda v: v.start)
    out = [intervals[0]]
    for cur in intervals[1:]:
        prev = out[-1]
        if cur.start >= prev.end - 0.05:
            out.append(cur)
            continue
        a, b = cur.start, min(prev.end, cur.end)
        cut = _best_cut(a, b, hits, fps, activity)
        prev.end = min(prev.end, cut)
        cur.start = max(cur.start, cut)
        if cur.end - cur.start >= 0.5:
            out.append(cur)
    return [iv for iv in out if iv.end - iv.start >= 0.5]


def _best_cut(a: float, b: float, hits: HitDetection | None, fps: float,
              activity: np.ndarray | None) -> float:
    """Pick the moment within [a, b] that most resembles "between two rallies"."""
    if b - a < 0.25:
        return (a + b) / 2
    if hits is not None and hits.times.size:
        ht = hits.times
        i0, i1 = int(np.searchsorted(ht, a)), int(np.searchsorted(ht, b))
        seg = ht[i0:i1]
        if seg.size >= 2:
            gaps = np.diff(seg)
            k = int(np.argmax(gaps))
            return float(seg[k] + gaps[k] / 2.0)
        if seg.size == 1:
            # Only one hit falls in the overlap region; cut a little before it
            return float(max(a, min(seg[0] - 0.15, b)))
    if activity is not None and activity.size and fps > 0:
        i0, i1 = max(0, int(a * fps)), min(activity.size, int(b * fps))
        if i1 - i0 >= 2:
            return float((np.argmin(activity[i0:i1]) + i0) / fps)
    return (a + b) / 2


# ------------------------------------------------------------------ Audio-only fallback


def fallback_from_audio(hits: HitDetection, params: Any) -> list[RallyInterval]:
    """When there is no visual signal at all, fall back to pure audio clustering (old-school but usable)."""
    clusters = cluster_rallies(
        hits,
        gap_seconds=getattr(params, "gap_seconds", 3.2),
        min_seconds=getattr(params, "min_rally_seconds", 2.0),
        max_seconds=getattr(params, "max_rally_seconds", 120.0),
    )
    out = []
    for c in clusters:
        out.append(RallyInterval(start=c.start, end=c.end,
                                 confidence=float(np.clip(len(c.hits) / 12.0, 0.15, 1.0)),
                                 hit_indices=list(c.hits)))
    return out
