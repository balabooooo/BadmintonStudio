"""Player-motion segmentation engine: actually finds "where this rally starts and where it ends".

Why it is needed
----------------
The old segmentation was built entirely on **full-frame fused activity**: audio hits, full-frame
motion, player speed, and shuttle presence were weighted-summed, then smoothed with a 1.2-second
moving average, then split into intervals by a "dual-threshold hysteresis state machine", and
finally over-long intervals were recursively split by a typical duration.

This approach failed on the measured footage for three reasons, all fundamental:

1. **Full-frame motion contains no rally information.** The gym simultaneously has audience
   movement, the neighboring court's shuttle, and lighting changes, which keep full-frame motion
   energy high at **any** moment. So the activity curve simply cannot collapse between rallies, and
   the hysteresis state machine never considers "this segment has ended".
   Measured: 30 minutes of footage yielded only 8 intervals, one of which was 236 seconds and another 138 seconds.
2. **The 1.2-second smoothing flattened out the "pauses".** The few seconds of silence between
   rallies get smoothed away by the 1.2-second average and added to the persistent background
   motion, so the valleys disappear.
3. **So "segmentation" became "equal division".** Over-long intervals fall into ``_split_by_valley``,
   and that function only recurses when ``e - s > target * 1.35``; because a 30-second interval is
   indeed slightly longer than the 28-second target, it finds the lowest point inside each interval
   and cuts again — producing a bunch of nearly equal-length fragments (measured: 32 rallies
   averaging 20.1 seconds, median 19.8 seconds, with many fragments **butting end-to-end**: the
   previous one's end exactly equals the next one's start). End-to-end boundaries cannot exist in a
   real match, because after a point there is necessarily a pause for picking up the shuttle / changing serve.

How this module works
---------------------
The semantics of a rally are: **serve -> rally -> dead ball -> pick up / prepare -> next serve**.
The players are the only reliable observation target that runs through the whole process:

* during a rally, both match players keep moving **continuously and fast**;
* after the dead ball they stop (catching their breath, watching the shuttle, picking it up), which is the **only**
  long pause that necessarily appears between every rally.

So the signal that should really be used is not "how noisy the full frame is" but "**how long these two players have not moved**".
Only the players' motion trajectories are needed; there is no need to know where the shuttle is. This module:

1. derives "how much this player moved in each frame" from the detection-box sequence (``box_motion``);
2. performs "valley + prominence" analysis on a **local time scale** (4 seconds by default),
   cutting the curve into alternating "active / quiet segments" — no manual threshold is needed,
   since prominence inherently requires the valley to be sufficiently lower than the peaks on either side;
3. defines rally boundaries using **quiet segments** (rather than active segments). Quiet segments are
   real physical pauses; active segments may be polluted by background motion.
4. boundary snapping: the start retreats to the real "standing still" moment within the quiet segment, and
   the end takes the last obvious movement;
5. finally validates once with **whether the shuttle is in flight** (when available): a rally's core
   interval should repeatedly show the shuttle; if the shuttle is never visible, this segment is actually
   picking up the shuttle / changing ends.

When there is no player signal at all (detection failure / non-match footage), it automatically falls
back to the old path of :func:`bms.analysis.rally.segment`, with unchanged behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------ Utilities


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Moving average (edge padding)."""
    if a.size == 0 or win <= 1:
        return a.astype(np.float32)
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(ap, k, mode="valid")[: a.size].astype(np.float32)


def _robust_norm(a: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros(0, dtype=np.float32)
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_q), np.percentile(a, hi_q)
    if hi - lo < EPS:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resample_to(a: np.ndarray, src_fps: float, dst_fps: float, n: int) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros(n, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-9 and a.size == n:
        return a.astype(np.float32)
    src_t = np.arange(a.size) / max(src_fps, EPS)
    dst_t = np.arange(n) / max(dst_fps, EPS)
    return np.interp(dst_t, src_t, a.astype(np.float32)).astype(np.float32)


def box_motion(boxes_per_frame: list, fps: float, aspect: float = 1.78,
               window: float = 1.0) -> np.ndarray:
    """Turn "per-frame player boxes" into a "how much the player moved per second" signal.

    Args:
        boxes_per_frame: one list per frame, whose elements are ``(track_id, x1, y1, x2, y2)``
            normalized coordinates; ``frame_boxes`` has exactly this structure.
        fps: frame rate of the sequence.
        aspect: frame aspect ratio, used to convert x displacement to the "frame height" scale.
        window: time window (seconds) over which "movement" is measured. The cumulative displacement
            within a sliding window is used instead of per-frame displacement because players
            "run a step then stop" — per-frame displacement frequently drops to zero, whereas whether
            there was any movement within a 1-second window is the criterion for "are they playing".

    Returns:
        An array whose length equals the frame count, in units of "cumulative displacement within the window / frame height".
    """
    n = len(boxes_per_frame)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ids: set[int] = set()
    for fr in boxes_per_frame:
        for item in fr or ():
            ids.add(int(item[0]))
    if not ids:
        return np.zeros(n, dtype=np.float32)
    col = {tid: k for k, tid in enumerate(sorted(ids))}
    pos = np.full((n, len(ids)), np.nan, dtype=np.float32)   # normalized x (already converted using frame height)
    ys = np.full((n, len(ids)), np.nan, dtype=np.float32)    # normalized y
    for i, fr in enumerate(boxes_per_frame):
        for item in fr or ():
            k = col[int(item[0])]
            pos[i, k] = 0.5 * (float(item[1]) + float(item[3])) * aspect
            ys[i, k] = 0.5 * (float(item[2]) + float(item[4]))

    # Differencing nan yields nan -> treat as "no displacement" (an occluded player should not count as moving)
    step = np.hypot(np.diff(pos, axis=0), np.diff(ys, axis=0))     # unit: frame height
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    # A single-frame displacement over 25% of frame height is basically an identity switch, so drop it
    step = np.clip(step, 0.0, 0.25)
    step = np.vstack([np.zeros((1, step.shape[1]), dtype=np.float32), step])

    win_f = max(1, int(round(window * fps)))
    # Cumulative displacement of each player within the window, then take "the most active player"
    kern = np.ones(win_f, dtype=np.float32)
    per_player = np.empty_like(step)
    for k in range(step.shape[1]):
        per_player[:, k] = np.convolve(step[:, k], kern, mode="same")
    return per_player.max(axis=1).astype(np.float32)


def detection_coverage(boxes_per_frame: list, fps: float) -> np.ndarray:
    """Per-frame "whether a match player was detected", then a morphological closing over time.

    A single-player frame still counts as valid: when only one of the two players is detected, that
    player's motion still indicates "the shuttle is in flight". But times where **no person is
    detected for a whole stretch** must be marked separately — during that time we know nothing about
    "whether the players moved", so it must not be treated as "the players did not move".
    This is exactly where the old implementation tended to lose rallies: a stretch of silence caused
    by detection failure would be treated as one long pause, swallowing the real rallies on either side of it.
    """
    n = len(boxes_per_frame)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ok = np.asarray([1.0 if fr else 0.0 for fr in boxes_per_frame], dtype=np.float32)
    # Closing operation: fill detection holes no longer than 1.5 seconds (a player being blocked for a few frames does not affect the decision)
    from scipy.ndimage import binary_closing

    gap = max(1, int(round(1.5 * fps)))
    closed = binary_closing(ok > 0.5, structure=np.ones(gap * 2 + 1, dtype=bool))
    return closed.astype(np.float32)


def blend_with_activity(player_motion: np.ndarray, coverage: np.ndarray,
                        activity: np.ndarray) -> np.ndarray:
    """Where the player signal is missing, fill it in with frame activity.

    The fusion is `covered * player + (1 - covered) * activity`:
    when players are present, fully trust the players (player motion is clean "in-rally evidence");
    when players are absent, fall back to frame activity (it is not clean, but far better than "always 0").
    Both are first normalized to 0~1, so blending introduces no dimensional issues.
    """
    n = player_motion.size
    if n == 0 or activity is None or activity.size != n:
        return player_motion
    cov = np.clip(coverage.astype(np.float32), 0.0, 1.0)
    return (cov * player_motion + (1.0 - cov) * activity).astype(np.float32)


def audio_visual_evidence(
    player_motion: np.ndarray,
    hit_density: np.ndarray | None,
    weight_hits: float = 0.7,
) -> np.ndarray:
    """Multiply "players are moving" and "hits are continuous around here" into a single rally-evidence curve.

    Player motion alone is not discriminative enough (players also walk when picking up the shuttle,
    and the neighboring court is also moving; measured AUC≈0.57); hit density alone is fooled by
    "the neighboring court happening to also rally a few shots". **Multiplying** the two requires both
    to hold at once, which exactly corresponds to "our match is rallying": between rallies, if either
    signal collapses, the evidence collapses, making the valleys clear.

    The reason for using ``(1-weight) + weight*hits`` rather than a direct product: with a direct
    product, any fluctuation in hit density would crush the whole evidence segment, and short rallies
    are especially easy to swallow; keeping a floor lets "the players really are moving fast" itself
    support a candidate, so a missed hit does not make the rally disappear.

    Both signals are robustly normalized first to avoid dimensional differences.
    """
    if player_motion is None or player_motion.size == 0:
        return hit_density if hit_density is not None else np.zeros(0, dtype=np.float32)
    p = _robust_norm(player_motion)
    if hit_density is None or hit_density.size != p.size or not np.any(hit_density):
        return p
    h = _robust_norm(hit_density)
    w = float(np.clip(weight_hits, 0.0, 1.0))
    return _robust_norm(p * ((1.0 - w) + w * h))


# ------------------------------------------------------------------ Quiet-segment detection


@dataclass
class QuietSpan:
    """A quiet interval where "the players basically did not move" (frame indices)."""

    start: int
    end: int
    depth: float          # depth of the valley relative to the peaks on either side (0~1)
    floor: float          # absolute level of the valley floor
    bottom: int = 0       # frame index of the actual minimum (snap target; span may be wider)


def find_quiet_spans(
    m: np.ndarray,
    fps: float,
    min_quiet: float = 1.0,
    max_quiet: float = 0.0,
    prominence_ratio: float = 0.30,
) -> list[QuietSpan]:
    """Find "quiet segments" on the player-motion curve.

    The approach is **valley + prominence**, not "below some threshold":

    1. ``scipy.signal.find_peaks`` finds local minima (find peaks on the negated curve);
    2. each minimum gets a prominence — how far the valley dropped relative to the higher "saddle"
       on either side. Prominence is normalized by some quantile of the whole clip's motion intensity,
       so "a pause in a quiet gym" and "a pause in a noisy gym" can both be caught with the same parameters;
    3. minima with insufficient prominence are dropped outright (that is not a pause, just normal fluctuation of motion intensity);
    4. if adjacent minima are closer than ``min_quiet``, only the deeper one is kept.

    Args:
        m: player-motion curve (the output of ``box_motion``).
        fps: frame rate.
        min_quiet: minimum separation between two candidate valleys for them to count as "two pauses".
        max_quiet: maximum duration of a single quiet segment (seconds); 0 = unlimited. An over-long
            quiet segment suggests there may be no match at all in this stretch (rest, changing ends),
            and is left to the caller to truncate by maximum silence.
        prominence_ratio: prominence threshold = this ratio x (p95 - p20). The default 0.30 means
            the valley must be at least 30% lower than the "typical active level".

    Returns:
        A time-ordered list of :class:`QuietSpan`.
    """
    if m is None or m.size < max(8, int(fps)):
        return []
    from scipy.signal import find_peaks, peak_prominences

    base = float(np.percentile(m, 20))
    top = float(np.percentile(m, 95))
    span = max(top - base, EPS)
    min_prom = prominence_ratio * span

    dist = max(1, int(round(min_quiet * fps)))
    idx, props = find_peaks(-m, distance=dist, prominence=min_prom)
    if idx.size == 0:
        return []
    prom = props.get("prominences")
    if prom is None:
        prom = peak_prominences(-m, idx)[0]

    quiet = m <= base + 0.35 * span
    out: list[QuietSpan] = []
    for k, i in enumerate(idx):
        # Expand the valley to both sides until it "no longer belongs to the quiet", giving the quiet segment's width
        a = i
        while a > 0 and quiet[a - 1]:
            a -= 1
        b = i
        while b + 1 < m.size and quiet[b + 1]:
            b += 1
        # A quiet segment must have at least a little width, otherwise it is just a single jitter of the curve
        if b - a < max(1, int(0.25 * fps)):
            a = max(0, i - int(0.15 * fps))
            b = min(m.size - 1, i + int(0.15 * fps))
        out.append(QuietSpan(start=a, end=b, bottom=int(i),
                             depth=float(np.clip(prom[k] / span, 0.0, 1.5)),
                             floor=float(m[i])))
    if max_quiet > 0:
        lim = int(round(max_quiet * fps))
        for q in out:
            if q.end - q.start > lim:
                mid = (q.start + q.end) // 2
                q.start, q.end = mid - lim // 2, mid + lim // 2
    return out


# ------------------------------------------------------------------ Main segmentation


@dataclass
class SegmentSignals:
    """All signals used for segmentation (all at the same frame rate)."""

    fps: float
    duration: float
    #: Player motion (same scale after 0~1 normalization)
    player_motion: np.ndarray
    #: Fused activity (product of the old path, used as backup evidence)
    activity: np.ndarray | None = None
    #: Whether the shuttle is in flight per frame (0~1), may be empty
    shuttle: np.ndarray | None = None
    #: Audio hit times (seconds), may be empty
    hit_times: np.ndarray | None = None
    #: Whether a player signal was actually obtained
    has_players: bool = False
    #: Per-frame "whether player detection is valid" (``detection_coverage``)
    coverage: np.ndarray | None = None
    #: Fraction of time player detection is valid
    coverage_ratio: float = 0.0


@dataclass
class SegmentOptions:
    min_rally: float = 2.0
    max_rally: float = 120.0
    #: Minimum separation between two quiet valleys for them to count as "two pauses". The old
    #: default of 1.0s plus lots of noise would cut a bunch of false boundaries inside a single
    #: rally, so this cannot be too small.
    min_quiet: float = 0.7
    #: Prominence threshold for quiet valleys (relative to p95-p20). The old default of 0.30 is too
    #: high: in multi-court footage players also walk while picking up the shuttle, so valleys are
    #: inherently shallow, and most inter-rally pauses are judged "not a valley", gluing adjacent rallies together.
    prominence_ratio: float = 0.18
    #: How long to keep before "standing still" within the quiet segment (receiving-ready motion)
    pre_roll: float = 1.0
    #: How long to keep after the rally ends (follow-through after the shuttle lands)
    post_roll: float = 1.6
    #: A quiet segment shorter than this does not count as "rally end" (avoid treating one long pause as a rally boundary)
    min_rest: float = 0.8
    #: The minimum allowed rally length; shorter candidates are dropped. The old default of 2.5s has a
    #: "knife-edge effect" on amateur footage: measured cases where a whole segment was rejected because
    #: its longest continuous movement stretch was 2.42s (0.08s short). In amateur rallies players have many
    #: moments of "standing and watching the shuttle", so continuous movement stretches are inherently short.
    min_core: float = 1.0


@dataclass
class RawSegment:
    start: float
    end: float
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def segment_by_player_motion(
    sig: SegmentSignals,
    opt: SegmentOptions | None = None,
) -> list[RawSegment]:
    """Segment rallies using "player motion + quiet segments".

    Interval definition: start at **the first obvious movement after a quiet segment ends**, and end
    at **the last obvious movement before the next quiet segment begins**. What this yields is the
    time window where "the shuttle is in flight", excluding the walking of picking up the shuttle,
    and never crossing the pause between two points.
    """
    opt = opt or SegmentOptions()
    m = sig.player_motion
    fps = sig.fps
    n = m.size
    if n == 0 or not sig.has_players:
        return []

    quiet = find_quiet_spans(m, fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio)
    if not quiet:
        return []

    base = float(np.percentile(m, 20))
    top = float(np.percentile(m, 95))
    span = max(top - base, EPS)
    # Threshold for "obvious movement": quiet floor + 25% of the dynamic range
    move_thr = base + 0.25 * span
    moving = m >= move_thr

    # Stretches where player detection is invalid for a long time: the "silence" there is fake, so it cannot be used as a boundary
    unknown = None
    if sig.coverage is not None and sig.coverage.size == n:
        unknown = sig.coverage < 0.5
    first_known = 0
    if unknown is not None:
        known_idx = np.nonzero(~unknown)[0]
        if known_idx.size == 0:
            return []
        first_known = int(known_idx[0])

    def _boundary_ok(i: int) -> bool:
        if unknown is None:
            return True
        a = max(0, i - int(round(0.75 * fps)))
        b = min(n, i + int(round(0.75 * fps)) + 1)
        return bool(np.any(~unknown[a:b]))

    def _add(segs_out: list[RawSegment], lo: int, hi: int, meta: dict) -> None:
        if hi <= lo or not _boundary_ok(lo) or not _boundary_ok(hi - 1):
            return
        core = _trim_to_motion(moving, lo, hi, m, fps, opt)
        if core is None:
            return
        s_i, e_i, score = core
        if not _boundary_ok(s_i) or not _boundary_ok(e_i):
            return
        start = max(0.0, s_i / fps - opt.pre_roll)
        end = min(sig.duration, (e_i + 1) / fps + opt.post_roll)
        if end - start < opt.min_rally:
            return
        if opt.max_rally > 0 and end - start > opt.max_rally:
            segs_out.extend(_split_long(m, fps, start, end, opt))
            return
        segs_out.append(RawSegment(start=start, end=end, score=score, meta=dict(meta)))

    # ---- Quiet segments -> rallies: between two adjacent quiet segments is a candidate rally
    segs: list[RawSegment] = []
    for k in range(len(quiet) - 1):
        q0, q1 = quiet[k], quiet[k + 1]
        if q1.start <= q0.end:
            continue
        rest = (q0.end - q0.start) / fps
        if rest < opt.min_rest:
            continue
        if q1.start - q0.end < int(opt.min_core * fps):
            continue
        _add(segs, q0.end, q1.start,
             {"rest_before": round(rest, 2), "quiet_depth": round(q1.depth, 2)})

    # ---- Head/tail: if the video starts mid-rally, the stretch before the first quiet segment is also a rally
    first = quiet[0]
    if first.start / fps > opt.min_core and first.start > first_known:
        _add(segs, first_known, first.start, {"head": True})
    last = quiet[-1]
    if sig.duration - last.end / fps > opt.min_core:
        _add(segs, last.end, n, {"tail": True})

    segs = _dedupe(segs)
    return sorted(segs, key=lambda s: s.start)


def _trim_to_motion(moving: np.ndarray, lo: int, hi: int, m: np.ndarray,
                    fps: float, opt: SegmentOptions) -> tuple[int, int, float] | None:
    """Tighten the candidate interval ``[lo, hi)`` to the core stretch where "there really is movement".

    What tightening does: quiet-segment boundaries are defined by "below the quiet floor", but players
    already stop counting as quiet before they have fully come to a standstill; conversely, the head
    and tail of a movement segment are often a slow start / inertial follow-through.

    The approach is to find "the largest connected clump" rather than "the first movement segment":
    in a long multi-shot rally players have brief moments of standing and watching the shuttle (for
    example when it flies overhead, both look up and freeze); if truncated by "the first segment", a
    rally would be cut into two halves.
    """
    if hi <= lo:
        return None
    idx = np.nonzero(moving[lo:hi])[0]
    if idx.size == 0:
        return None
    gap = max(1, int(round(0.35 * fps)))
    # Split into connected clumps where "separation > gap"
    splits = np.nonzero(np.diff(idx) > gap)[0]
    bounds = np.concatenate([[0], splits + 1, [idx.size]])
    best: tuple[int, int] | None = None
    best_len = -1
    for k in range(len(bounds) - 1):
        a, b = int(idx[bounds[k]]), int(idx[bounds[k + 1] - 1])
        if b - a > best_len:
            best_len, best = b - a, (a, b)
    if best is None:
        return None
    s_i, e_i = best
    seg = m[lo + s_i: lo + e_i + 1]
    base = float(np.percentile(m, 20))
    span = max(float(np.percentile(m, 95)) - base, EPS)
    score = float(np.clip((float(seg.mean()) - base) / span, 0.0, 1.0))
    # A too-low movement duty ratio means this stretch is mostly a little walking, not a rally
    duty = float(np.count_nonzero(moving[lo + s_i: lo + e_i + 1])) / max(1, e_i - s_i + 1)
    if duty < 0.35 or (e_i - s_i) / fps < opt.min_core:
        return None
    return lo + s_i, lo + e_i, score


def _split_long(m: np.ndarray, fps: float, start: float, end: float,
                opt: SegmentOptions) -> list[RawSegment]:
    """Split an over-long rally at its deepest internal quiet valley."""
    a, b = int(start * fps), int(end * fps)
    if b - a < int(2 * opt.min_core * fps):
        return [RawSegment(start=start, end=end)]
    inner = find_quiet_spans(m[a:b], fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio * 0.7)
    if not inner:
        return [RawSegment(start=start, end=end)]
    best = max(inner, key=lambda q: q.depth)
    cut = (best.start + best.end) // 2
    if cut < int(opt.min_core * fps) or (b - a) - cut < int(opt.min_core * fps):
        return [RawSegment(start=start, end=end)]
    left = _split_long(m, fps, start, (a + cut) / fps, opt)
    right = _split_long(m, fps, (a + cut) / fps, end, opt)
    return left + right


def _dedupe(segs: list[RawSegment]) -> list[RawSegment]:
    """Drop mutually overlapping candidates, keeping the more "substantial" one; and eliminate end-to-end abutment."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[RawSegment] = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]
        if cur.start >= prev.end - 0.05:
            out.append(cur)
            continue
        # Overlap: keep whichever is longer
        if (cur.end - cur.start) > (prev.end - prev.start):
            out[-1] = cur
    return out


# ------------------------------------------------------------------ Main entry


def segment_visual(
    fps: float,
    duration: float,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    activity: np.ndarray | None = None,
    shuttle_presence: np.ndarray | None = None,
    shuttle_fps: float = 0.0,
    hit_times: np.ndarray | None = None,
    hit_density: np.ndarray | None = None,
    opt: SegmentOptions | None = None,
) -> tuple[list[RawSegment], float]:
    """Pipeline-facing entry: segment using player motion, and return the valid coverage of player detection.

    There is an easily overlooked but crucial point in the approach: **times when player detection is
    invalid must not be treated as "the players did not move"**. In real footage the match players
    are often undetected for more than half the time (on a 960x540 proxy a person is only 80~120 pixels
    tall); if the player-motion curve is used directly, those holes get read as one long pause and the
    real rallies on either side are swallowed. Measured on 5 minutes of footage, the old and new
    approaches gave 5 rallies and 1 rally respectively.

    So here:
      1. first compute per-frame "whether detection is valid" (``detection_coverage``, with a 1.5-second closing operation);
      2. where detection is invalid, **fill in with frame activity** (``blend_with_activity``);
      3. boundaries are only allowed to fall where detection is valid (otherwise the "silence" there is untrustworthy).

    Returns:
        ``(rally list, player detection coverage 0~1)``. When coverage is very low the caller should
        switch to activity segmentation.
    """
    opt = opt or SegmentOptions()
    n = max(1, int(round(duration * fps)))
    boxes = player_boxes or []
    has_players = bool(boxes) and player_fps > 0 and len(boxes) >= int(fps * 2)
    if not has_players:
        return [], 0.0

    m = box_motion(boxes, player_fps, window=1.0)
    m = _resample_to(m, player_fps, fps, n)
    cov = _resample_to(detection_coverage(boxes, player_fps), player_fps, fps, n)
    coverage_ratio = float(np.mean(cov > 0.5))
    m = _robust_norm(_smooth(m, max(1, int(fps * 0.5))))
    act = _resample_to(activity, fps, fps, n) if activity is not None else None
    if act is not None and np.any(act):
        act = _robust_norm(_smooth(act, max(1, int(fps * 1.2))))
        m = blend_with_activity(m, cov, act)
    # Hit density is the most discriminative signal on this footage (see `audio_visual_evidence`).
    # Multiply it into player motion, so a rally requires both "players are moving" and "hits are continuous around here".
    if hit_density is not None and hit_density.size == n:
        m = audio_visual_evidence(m, hit_density)
    if not np.any(m):
        return [], coverage_ratio

    sig = SegmentSignals(
        fps=fps, duration=duration, player_motion=m,
        activity=act,
        shuttle=(_resample_to(shuttle_presence, shuttle_fps, fps, n)
                 if shuttle_presence is not None and shuttle_fps > 0 else None),
        hit_times=hit_times, has_players=True,
        coverage=cov, coverage_ratio=coverage_ratio,
    )
    segs = segment_by_player_motion(sig, opt)
    if segs:
        segs = verify_with_shuttle(segs, sig, opt)
    return segs, coverage_ratio


def segment_activity(
    activity: np.ndarray,
    fps: float,
    duration: float,
    opt: SegmentOptions | None = None,
    shuttle_presence: np.ndarray | None = None,
    shuttle_fps: float = 0.0,
) -> list[RawSegment]:
    """Segment using only the fused activity curve (the primary approach when the player signal is unavailable).

    The differences from the old path (``rally.segment``) are exactly why the old path segmented poorly:

    1. **No unconditional recursive equal division.** The old path cut at the lowest point inside an
       interval whenever it was even slightly longer than the target duration; this is precisely the
       source of "32 rallies averaging 20 seconds, with many fragments end-to-end". Here, cutting only
       happens when **the valley's prominence is sufficient**, and the cut point is the valley center
       (the real pause) rather than any arbitrary lowest point.
    2. **Adaptive prominence.** The prominence threshold is set as a ratio of (p95 - p20) rather than a
       hard-coded absolute value, so a quiet gym and a noisy gym use the same set of parameters.
    3. **A pause must be left between adjacent rallies.** The output will not be end-to-end.
    """
    opt = opt or SegmentOptions()
    n = activity.size
    if n < max(8, int(fps * 2)):
        return []
    base = float(np.percentile(activity, 20))
    top = float(np.percentile(activity, 95))
    span = max(top - base, EPS)
    quiet = find_quiet_spans(activity, fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio)

    # Active threshold: quiet floor + 30% of the dynamic range
    live_thr = base + 0.30 * span
    live = activity >= live_thr

    def trim(lo: int, hi: int) -> tuple[int, int] | None:
        if hi - lo < int(opt.min_core * fps):
            return None
        idx = np.nonzero(live[lo:hi])[0]
        if idx.size == 0:
            return None
        gap = max(1, int(round(0.8 * fps)))
        splits = np.nonzero(np.diff(idx) > gap)[0]
        bounds = np.concatenate([[0], splits + 1, [idx.size]])
        best: tuple[int, int] | None = None
        best_len = -1
        for k in range(len(bounds) - 1):
            a, b = int(idx[bounds[k]]), int(idx[bounds[k + 1] - 1])
            if b - a > best_len:
                best_len, best = b - a, (a, b)
        if best is None or (best[1] - best[0]) / fps < opt.min_core:
            return None
        return lo + best[0], lo + best[1]

    segs: list[RawSegment] = []
    span_bounds: list[tuple[int, int]] = []
    if quiet:
        if quiet[0].start > int(opt.min_core * fps):
            span_bounds.append((0, quiet[0].start))
        for k in range(len(quiet) - 1):
            span_bounds.append((quiet[k].end, quiet[k + 1].start))
        if n - quiet[-1].end > int(opt.min_core * fps):
            span_bounds.append((quiet[-1].end, n))
    else:
        span_bounds.append((0, n))

    for lo, hi in span_bounds:
        core = trim(lo, hi)
        if core is None:
            continue
        s_i, e_i = core
        start = max(0.0, s_i / fps - opt.pre_roll)
        end = min(duration, (e_i + 1) / fps + opt.post_roll)
        if end - start < opt.min_rally:
            continue
        core_act = activity[s_i:e_i + 1]
        score = float(np.clip((float(core_act.mean()) - base) / span, 0.0, 1.0))
        if opt.max_rally > 0 and end - start > opt.max_rally:
            segs.extend(_split_long(activity, fps, start, end, opt))
            continue
        segs.append(RawSegment(start=start, end=end, score=score))

    segs = _dedupe(segs)
    if shuttle_presence is not None and shuttle_fps > 0:
        sig = SegmentSignals(
            fps=fps, duration=duration, player_motion=np.zeros(0), has_players=False,
            shuttle=_resample_to(shuttle_presence, shuttle_fps, fps, n),
        )
        segs = verify_with_shuttle(segs, sig, opt)
    return sorted(segs, key=lambda s: s.start)


def verify_with_shuttle(segs: list[RawSegment], sig: SegmentSignals,
                        opt: SegmentOptions) -> list[RawSegment]:
    """Validate rallies with "whether the shuttle is in flight".

    Once the shuttle is visible, that shot is indeed being played; conversely, if the shuttle is
    almost never seen in a candidate rally's core interval, it is more likely picking up the shuttle /
    changing ends. It only acts when the shuttle signal is **reliable enough** (a certain presence rate
    across the clip), to avoid turning this module into a new source of inaccuracy.
    """
    sh = sig.shuttle
    if sh is None or sh.size == 0:
        return segs
    cover = float(np.mean(sh > 0.15))
    if cover < 0.05 or cover > 0.9:      # too little = cannot detect; too much = always noise
        return segs
    fps = sig.fps
    out: list[RawSegment] = []
    for s in segs:
        a = max(0, int(s.start * fps))
        b = min(sh.size, max(a + 1, int(s.end * fps)))
        core = sh[a:b]
        if core.size == 0:
            out.append(s)
            continue
        hit_ratio = float(np.mean(core > 0.15))
        s.meta["shuttle_ratio"] = round(hit_ratio, 3)
        if hit_ratio < 0.02:
            # No shuttle visible in the whole segment: not a rally, drop it
            continue
        # Tighten the boundaries using "the first/last frame where the shuttle exists", but keep the breathing room of pre/post roll
        nz = np.nonzero(core > 0.15)[0]
        if nz.size >= 2:
            s.start = max(s.start, (a + int(nz[0])) / fps - opt.pre_roll * 0.5)
            s.end = min(s.end, (a + int(nz[-1]) + 1) / fps + opt.post_roll * 0.5)
        if s.end - s.start >= opt.min_rally:
            out.append(s)
    return out


__all__ = [
    "QuietSpan",
    "RawSegment",
    "SegmentOptions",
    "SegmentSignals",
    "audio_visual_evidence",
    "blend_with_activity",
    "box_motion",
    "detection_coverage",
    "find_quiet_spans",
    "segment_activity",
    "segment_by_player_motion",
    "segment_visual",
    "verify_with_shuttle",
]
