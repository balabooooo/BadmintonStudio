"""Audio-independent shot estimation for rally scoring.

Why it exists
-------------
Audio hit detection is the traditional source of "how many shots this rally had",
but it is unreliable in a multi-court gym: every neighboring court's contact sound
looks like one of our hits, and heavy background noise hides soft touches. Turning
hit-sound detection off used to leave ``shot_count = 0``, which collapsed the
length / technique / excitement / highlight dimensions of the score even when the
footage itself was perfectly good (measured on the 2026-10-11 clips: every rally
scored ~20 because no shot feature existed).

This module builds a **visual shot-event sequence** with exactly the same shape as
:class:`bms.analysis.audio_hits.HitDetection`, so both sources feed *one* scoring
formula in :mod:`bms.analysis.scoring` and switching the audio switch cannot make
the two systems drift apart.

Sources, in priority order
--------------------------
1. **Pose swing peaks** (``pose.swing_peaks``): the wrist speed relative to the
   shoulder line is a direct physical proxy for "a racket swing happened". With
   full pose coverage this is essentially one peak per shot; measured on clip4 it
   produced 259 events against 259 audio hits accepted by the pose gate, and the
   resulting score distribution differed by only ~1 point on average.
2. **Duration prior**: used when pose is disabled, unavailable, or has too little
   coverage. A fixed exchange tempo places shots evenly inside each rally so scoring
   still produces a complete, comparable value. The trace marks this explicitly
   instead of pretending it is a measurement.

Every rally with no events from the chosen source is filled by the duration prior
(``filled_intervals`` in the trace), because a detection gap inside one rally must
not silently score that rally as "zero shots" while its neighbours are normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import audio_hits as AH

#: Minimum pose coverage before swing peaks are trusted as the shot source.
POSE_MIN_COVERAGE = 0.35
#: Fallback exchange tempo (shots per second) used only to fill detection gaps.
#: 1.6 measured from the pose gate's accepted-hit rate on the reference clips
#: (259 accepted hits over 135.7 s of active rally time ≈ 1.9/s; 1.6 stays slightly
#: conservative so fast doubles are not saturated).
PRIOR_SHOT_RATE = 1.6
#: Upper bound for the number of prior shots generated inside one rally.
MAX_PRIOR_SHOTS = 60


@dataclass
class VisualShots:
    """Visual shot events plus the diagnostics explaining where they came from."""

    hits: AH.HitDetection
    trace: dict[str, Any] = field(default_factory=dict)


def _empty_hits() -> AH.HitDetection:
    return AH.HitDetection(
        times=np.zeros(0, dtype=np.float64),
        strength=np.zeros(0, dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        envelope=np.zeros(0, dtype=np.float32),
        env_fps=1.0,
    )


def _detection(times: np.ndarray, strength: np.ndarray, confidence: np.ndarray) -> AH.HitDetection:
    order = np.argsort(times)
    return AH.HitDetection(
        times=np.asarray(times[order], dtype=np.float64),
        strength=np.asarray(strength[order], dtype=np.float32),
        confidence=np.asarray(confidence[order], dtype=np.float32),
        envelope=np.zeros(0, dtype=np.float32),
        env_fps=1.0,
    )


def from_pose(pose: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shot events from ``PoseSignal.swing``; empty when pose is unreliable."""
    if pose is None or getattr(pose, "n", 0) < 8:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if float(getattr(pose, "coverage", 0.0) or 0.0) < POSE_MIN_COVERAGE:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    from . import pose as POSE

    idx, prom = POSE.swing_peaks(pose)
    if idx.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    fps = float(getattr(pose, "fps", 0.0) or 0.0)
    if fps <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    times = idx.astype(np.float64) / fps
    strength = np.clip(np.asarray(prom, dtype=np.float32), 0.0, 1.0)
    return times, strength, strength.copy()


def _span(iv: Any) -> tuple[float, float] | None:
    start = getattr(iv, "start", None)
    end = getattr(iv, "end", None)
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if end <= start:
        return None
    return start, end


def _prior_events(intervals: Sequence[Any], rate: float,
                  strength: float, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Place ``rate`` shots per second evenly inside each interval, returning events and count."""
    times: list[float] = []
    strengths: list[float] = []
    confs: list[float] = []
    filled = 0
    for iv in intervals:
        span = _span(iv)
        if span is None:
            continue
        start, end = span
        n = int(round((end - start) * max(rate, 0.0)))
        n = min(MAX_PRIOR_SHOTS, max(1, n))
        if n <= 0:
            continue
        step = (end - start) / (n + 1)
        for k in range(1, n + 1):
            times.append(start + step * k)
            strengths.append(strength)
            confs.append(confidence)
        filled += 1
    return (np.asarray(times, dtype=np.float64),
            np.asarray(strengths, dtype=np.float32),
            np.asarray(confs, dtype=np.float32), filled)


def estimate(
    *,
    pose: Any = None,
    intervals: Sequence[Any] | None = None,
    duration: float = 0.0,
    prior_rate: float = PRIOR_SHOT_RATE,
) -> VisualShots:
    """Build the shot-event sequence used by scoring when audio hits are unavailable.

    ``intervals`` is optional; when given, every rally without a pose event is filled
    by the duration prior so no rally silently scores as "zero shots".
    """
    trace: dict[str, Any] = {}
    source = "duration_prior"
    times = np.zeros(0)
    strength = np.zeros(0)
    confidence = np.zeros(0)

    pose_times, pose_strength, pose_conf = from_pose(pose)
    if pose_times.size:
        source = "pose"
        times, strength, confidence = pose_times, pose_strength, pose_conf
        trace["pose_coverage"] = round(float(getattr(pose, "coverage", 0.0) or 0.0), 3)
        trace["pose_events"] = int(pose_times.size)

    filled = 0
    if intervals:
        ivs = list(intervals)
        # Which rallies already contain at least one event from the primary source?
        missing: list[Any] = []
        if times.size:
            for iv in ivs:
                span = _span(iv)
                if span is None:
                    continue
                a, b = span
                if not bool(np.any((times >= a) & (times <= b))):
                    missing.append(iv)
        else:
            missing = [(iv) for iv in ivs if _span(iv) is not None]
        if missing:
            if times.size:
                fill_strength = float(np.median(strength))
                fill_conf = float(np.median(confidence))
            else:
                fill_strength = 0.5
                fill_conf = 0.5
            f_t, f_s, f_c, filled = _prior_events(missing, prior_rate, fill_strength, fill_conf)
            times = np.concatenate([times, f_t])
            strength = np.concatenate([strength, f_s])
            confidence = np.concatenate([confidence, f_c])
        if filled and source == "duration_prior":
            trace["filled_intervals"] = filled
    elif not times.size:
        # No intervals and no visual event: return an empty sequence. The caller
        # (scoring) still runs; it simply has no shot evidence for this stretch.
        pass

    if duration > 0:
        keep = (times >= 0.0) & (times <= float(duration) + 1e-6)
        times, strength, confidence = times[keep], strength[keep], confidence[keep]

    trace.update({
        "source": source,
        "events": int(times.size),
        "filled_intervals": int(filled),
        "prior_rate": round(float(prior_rate), 3),
    })
    return VisualShots(hits=_detection(times, strength, confidence), trace=trace)
