"""Serve/end-of-rally boundary evidence model (P3).

The regular segmenter (:mod:`bms.analysis.rally_vision`) places rally boundaries at quiet
valleys plus fixed pre/post rolls. This module is an *optional* boundary refinement:
given the cached full-frame-rate signals (player motion, pose swing/overhead, hit times),
it builds per-frame "boundary evidence" curves, calibrated with a statistical template
from manual labels, and lets :func:`refine_boundaries` snap a boundary to a nearby quiet
valley **only when the evidence there is clearly stronger**.

Design constraints (agreed plan):

* **Tie-breaker only**: the switch (``AnalysisParams.use_boundary_refine``) defaults to
  ``False``; with the switch off, no pose, or insufficient pose coverage, the output must
  be byte-for-byte identical to the current finishing chain.
* **No AI reruns**: every feature comes from cached numpy arrays / stored curves. Resegment
  reads the evidence curves packed at analysis time (and can rebuild them from
  ``player_motion_full`` / ``pose_swing_full`` for older projects).
* **Guardrails**: minimum evidence channels, score threshold + gain margin, max movement,
  never cross a neighboring rally, never shrink a rally below ``MIN_RALLY_LEN``.
* Templates are plain statistics (quantile normalization ranges + directional weights +
  positive-score quantile thresholds), fit LOCO by ``scripts/calibrate_boundary.py``.

Feature channels are all oriented "larger = more boundary-like":

* ``motion_contrast`` — mean motion on the rally side minus the outside (pause) side;
* ``motion_slope``   — onset (start) / decay (end) slope of the smoothed motion curve;
* ``swing``          — strongest swing on the rally side above the pose quiet level;
* ``overhead``       — fraction of overhead-arm frames on the rally side;
* ``hit``            — proximity of the first/last gated hit (computed live, not packed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import rally_vision as RV

EPS = 1e-9

# ---------------------------------------------------------------- Constants

#: Version of the packed evidence curves / this module's signal contract.
BOUNDARY_EVIDENCE_VERSION = 1

#: Feature windows (seconds) around a start / end boundary: (outside, rally-side).
START_WIN = (2.0, 1.5)
END_WIN = (1.5, 1.5)
#: Slope estimation window (seconds) and how far around the boundary the max slope is sought.
SLOPE_WIN = 0.3
SLOPE_LOOK = 0.7
#: Swing/overhead rally-side window (seconds).
SWING_WIN = 1.5
#: First/last hit farther than this contributes zero hit evidence.
HIT_GAP_CAP = 4.0

#: Refinement is skipped entirely when pose coverage (fraction of valid frames) is below this.
REFINE_MIN_POSE_COV = 0.35
#: Local detection coverage required around a candidate snap point.
REFINE_MIN_LOCAL_COV = 0.5
#: Minimum legal rally length after a snap (mirrors the label-quality snap guard).
MIN_RALLY_LEN = 0.5
#: Default max snap distance (seconds); overridable via AnalysisParams.
DEFAULT_MAX_MOVE = 3.0
#: Minimum score gain over the current position for a snap to fire.
DEFAULT_SCORE_MARGIN = 0.12
#: At least this many channels must individually look boundary-like at the candidate.
MIN_CHANNELS = 2
#: A channel counts as "firing" above this normalized value.
CHANNEL_FIRE = 0.5

#: Motion-only channel names (packed into the evidence curves).
MOTION_CHANNELS = ("motion_contrast", "motion_slope", "swing", "overhead")
HIT_CHANNEL = "hit"
ALL_CHANNELS = MOTION_CHANNELS + (HIT_CHANNEL,)


@dataclass(frozen=True)
class Template:
    """Statistical scoring template (ranges + directional weights + thresholds).

    ``ranges[ch] = (lo, hi)`` robustly maps a raw feature to 0..1; ``weights[ch]`` are
    non-negative directional weights; ``thr_*`` are the per-side acceptance thresholds
    (positive-score quantiles from calibration).
    """

    ranges: dict[str, tuple[float, float]]
    weights: dict[str, float]
    thr_start: float
    thr_end: float
    name: str = "neutral"

    def motion_weight(self) -> float:
        return float(sum(max(0.0, self.weights.get(c, 0.0)) for c in MOTION_CHANNELS))

    def hit_weight(self) -> float:
        return float(max(0.0, self.weights.get(HIT_CHANNEL, 0.0)))


#: Placeholder until LOCO calibration freezes the real constants. Refinement ships OFF,
#: so this template never affects production output; the calibration script replaces it.
NEUTRAL_TEMPLATE = Template(
    ranges={
        "motion_contrast": (0.0, 0.4),
        "motion_slope": (0.0, 0.5),
        "swing": (0.1, 1.2),
        "overhead": (0.0, 1.0),
        "hit": (0.25, 1.0),
    },
    weights={c: 1.0 for c in ALL_CHANNELS},
    thr_start=0.55,
    thr_end=0.55,
    name="neutral",
)

# The frozen calibrated template is patched in here after LOCO (P3/P6). ``None`` means
# "use the neutral placeholder".
_TEMPLATE: Template | None = None


def get_template() -> Template:
    return _TEMPLATE or NEUTRAL_TEMPLATE


def set_template(tpl: Template | None) -> None:
    """Install the calibrated template (test/CLI hook; production keeps the frozen default)."""
    global _TEMPLATE
    _TEMPLATE = tpl


# ------------------------------------------------------------- array helpers


def _resample(arr: np.ndarray | None, src_fps: float, dst_fps: float, n: int) -> np.ndarray | None:
    if arr is None or arr.size == 0 or src_fps <= 0:
        return None
    return RV._resample_to(np.asarray(arr, dtype=np.float32), src_fps, dst_fps, n)


def _roll_mean(a: np.ndarray, w: int, forward: bool) -> np.ndarray:
    """Mean of the ``w`` samples after (forward) or before (backward) each index."""
    n = a.size
    w = max(1, min(w, n))
    csum = np.concatenate(([0.0], np.cumsum(a.astype(np.float64))))
    idx = np.arange(n)
    if forward:
        hi = np.minimum(idx + w, n)
        lo = idx
    else:
        hi = idx
        lo = np.maximum(idx - w, 0)
    denom = np.maximum(hi - lo, 1)
    return ((csum[hi] - csum[lo]) / denom).astype(np.float32)


def _directed_max_filter(a: np.ndarray, w: int, forward: bool) -> np.ndarray:
    """Max over the ``w`` samples after (forward) or before (backward) each index."""
    from scipy.ndimage import maximum_filter1d

    n = a.size
    w = max(1, min(w, n))
    origin = (w - 1) // 2 if forward else -((w - 1) // 2)
    return maximum_filter1d(a.astype(np.float32), size=w, origin=origin, mode="nearest")


# ------------------------------------------------------------- evidence curves


@dataclass
class BoundaryEvidence:
    """Per-frame boundary evidence aligned to the segmentation grid (``fps``)."""

    fps: float
    duration: float
    #: The motion curve used both for scoring and for finding quiet valleys (grid fps).
    motion: np.ndarray
    #: Raw per-channel feature arrays (grid fps), per side.
    channels: dict[str, dict[str, np.ndarray]]
    #: Packed motion-part total score curves (no hit channel), per side.
    curves: dict[str, np.ndarray]
    #: Detection coverage at grid fps (optional).
    coverage: np.ndarray | None
    #: Fraction of frames with valid pose.
    pose_coverage: float
    hit_times: np.ndarray
    template: Template
    swing_quiet: float = 0.0
    has_pose: bool = False

    @property
    def n(self) -> int:
        return int(self.motion.size)

    def frame(self, t: float) -> int:
        return int(round(float(t) * self.fps))

    def time(self, i: int) -> float:
        return i / self.fps

    def hit_feature(self, side: str, frame: int) -> float:
        """1..0 proximity of the first (start) / last (end) hit on the rally side."""
        if self.hit_times is None or self.hit_times.size == 0:
            return 0.0
        t = frame / self.fps
        if side == "start":
            later = self.hit_times[self.hit_times >= t - 0.05]
            if later.size == 0:
                return 0.0
            gap = float(later[0]) - t
        else:
            earlier = self.hit_times[self.hit_times <= t + 0.05]
            if earlier.size == 0:
                return 0.0
            gap = t - float(earlier[-1])
        return float(np.clip(1.0 - max(0.0, gap) / HIT_GAP_CAP, 0.0, 1.0))

    def channel_values(self, side: str, frame: int) -> dict[str, float]:
        """Raw channel values at one frame, including the live hit channel."""
        frame = int(np.clip(frame, 0, self.n - 1))
        out = {c: float(self.channels[side][c][frame]) for c in MOTION_CHANNELS}
        out[HIT_CHANNEL] = self.hit_feature(side, frame)
        return out

    def score(self, side: str, frame: int) -> float:
        """Total score (motion curve + live hit channel) at one frame."""
        frame = int(np.clip(frame, 0, self.n - 1))
        tpl = self.template
        wm, wh = tpl.motion_weight(), tpl.hit_weight()
        total_w = wm + wh
        if total_w <= EPS:
            return 0.0
        motion_part = float(self.curves[side][frame]) * wm
        hit_part = _normalize(tpl, HIT_CHANNEL, self.hit_feature(side, frame)) * wh
        return float(np.clip((motion_part + hit_part) / total_w, 0.0, 1.0))

    def local_coverage_ok(self, frame: int) -> bool:
        if self.coverage is None:
            return True
        w = max(1, int(round(0.5 * self.fps)))
        a = max(0, frame - w)
        b = min(self.n, frame + w + 1)
        return bool(np.mean(self.coverage[a:b] >= 0.5) >= REFINE_MIN_LOCAL_COV)

    def available_for_refine(self) -> bool:
        return self.has_pose and self.pose_coverage >= REFINE_MIN_POSE_COV


def _normalize(tpl: Template, ch: str, value: float) -> float:
    lo, hi = tpl.ranges.get(ch, (0.0, 1.0))
    if hi - lo <= EPS:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _side_channels(motion: np.ndarray, swing: np.ndarray | None,
                   overhead: np.ndarray | None, swing_quiet: float,
                   win: tuple[float, float], slope_dir: int,
                   fps: float) -> dict[str, np.ndarray]:
    """Raw per-frame features for one side. ``slope_dir`` +1 = onset (start), -1 = decay (end)."""
    n = motion.size
    w_out = max(1, int(round(win[0] * fps)))
    w_in = max(1, int(round(win[1] * fps)))
    # Motion contrast: rally-side mean minus outside mean.
    if slope_dir > 0:
        outside = _roll_mean(motion, w_out, forward=False)
        inside = _roll_mean(motion, w_in, forward=True)
    else:
        outside = _roll_mean(motion, w_out, forward=True)
        inside = _roll_mean(motion, w_in, forward=False)
    contrast = inside - outside

    # Max onset/decay slope (motion units per second) around each frame.
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    k = max(1, int(round(SLOPE_WIN * fps)))
    sm = np.convolve(motion, np.ones(k) / k, mode="same")
    d = np.empty(n, dtype=np.float32)
    d[:k] = 0.0
    d[k:] = (sm[k:] - sm[:-k]) / max(k / fps, EPS)
    look = max(1, int(round(SLOPE_LOOK * fps)))
    if slope_dir > 0:
        slope = maximum_filter1d(d, size=2 * look + 1, mode="nearest")
    else:
        slope = -minimum_filter1d(d, size=2 * look + 1, mode="nearest")

    out: dict[str, np.ndarray] = {
        "motion_contrast": contrast.astype(np.float32),
        "motion_slope": slope.astype(np.float32),
    }
    sw_win = max(1, int(round(SWING_WIN * fps)))
    if swing is not None:
        near = (_directed_max_filter(swing, sw_win, forward=True) if slope_dir > 0
                else _directed_max_filter(swing, sw_win, forward=False))
        out["swing"] = np.clip(near - swing_quiet, 0.0, None).astype(np.float32)
    else:
        out["swing"] = np.zeros(n, dtype=np.float32)
    if overhead is not None:
        out["overhead"] = (_roll_mean(overhead, sw_win, forward=True) if slope_dir > 0
                           else _roll_mean(overhead, sw_win, forward=False)).astype(np.float32)
    else:
        out["overhead"] = np.zeros(n, dtype=np.float32)
    return out


def _pack_curve(channels: dict[str, np.ndarray], tpl: Template) -> np.ndarray:
    """Weighted motion-part score (normalized by the active motion-channel weights)."""
    acc = np.zeros_like(motion_ref(channels), dtype=np.float64)
    wsum = 0.0
    for ch in MOTION_CHANNELS:
        w = max(0.0, tpl.weights.get(ch, 0.0))
        if w <= EPS:
            continue
        lo, hi = tpl.ranges.get(ch, (0.0, 1.0))
        norm = np.clip((channels[ch] - lo) / max(hi - lo, EPS), 0.0, 1.0)
        acc += w * norm
        wsum += w
    if wsum <= EPS:
        return np.zeros_like(acc, dtype=np.float32)
    return (acc / wsum).astype(np.float32)


def motion_ref(channels: dict[str, np.ndarray]) -> np.ndarray:
    return channels["motion_contrast"]


def build_evidence(*, fps: float, duration: float,
                   motion: np.ndarray | None,
                   coverage: np.ndarray | None = None,
                   swing: np.ndarray | None = None,
                   swing_fps: float = 0.0,
                   overhead: np.ndarray | None = None,
                   pose_coverage: float = 0.0,
                   swing_quiet: float = 0.0,
                   hit_times: Iterable[float] | None = None,
                   template: Template | None = None) -> BoundaryEvidence | None:
    """Build per-frame boundary evidence from cached full-rate arrays.

    All pose inputs are optional: without pose the evidence is still built (motion-only),
    but :meth:`BoundaryEvidence.available_for_refine` returns False, so refinement is a
    no-op and production output is unchanged.
    """
    if motion is None or np.asarray(motion).size < max(8, int(round(fps))):
        return None
    motion = np.asarray(motion, dtype=np.float32)
    n = motion.size
    fps = float(fps)
    duration = float(duration) if duration > 0 else n / fps
    tpl = template or get_template()

    swing_r = _resample(np.asarray(swing, dtype=np.float32) if swing is not None else None,
                        swing_fps or fps, fps, n)
    over_r = _resample(np.asarray(overhead, dtype=np.float32) if overhead is not None else None,
                       swing_fps or fps, fps, n)
    cov_r = _resample(np.asarray(coverage, dtype=np.float32) if coverage is not None else None,
                      fps, fps, n)
    has_pose = swing_r is not None and pose_coverage > 0.0

    ch_start = _side_channels(motion, swing_r, over_r, swing_quiet, START_WIN, +1, fps)
    ch_end = _side_channels(motion, swing_r, over_r, swing_quiet, END_WIN, -1, fps)
    channels = {"start": ch_start, "end": ch_end}
    curves = {
        "start": _pack_curve(ch_start, tpl),
        "end": _pack_curve(ch_end, tpl),
    }
    hits = (np.asarray(sorted(float(t) for t in hit_times), dtype=np.float32)
            if hit_times is not None else np.zeros(0, dtype=np.float32))
    return BoundaryEvidence(
        fps=fps, duration=duration, motion=motion,
        channels=channels, curves=curves, coverage=cov_r,
        pose_coverage=float(pose_coverage), hit_times=hits,
        template=tpl, swing_quiet=float(swing_quiet), has_pose=bool(has_pose),
    )


def evidence_from_signals(sig: dict[str, Any], hit_times: Iterable[float] | None,
                          template: Template | None = None) -> BoundaryEvidence | None:
    """Rebuild evidence from a stored analysis ``signals`` dict (resegment / optimizer).

    Prefers the packed motion-part score curves (identical to the full run); otherwise
    rebuilds channels from the stored full-rate arrays (motion + swing are always full;
    overhead only when a future analysis packed ``pose_overhead_full``).
    """
    full = sig.get("activity_full") or []
    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or (len(full) / fps if full else 0.0)
    if not full or duration <= 0:
        return None
    n = len(full)

    pfps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    pm_full = sig.get("player_motion_full") or []
    if pm_full and pfps > 0:
        motion = RV._resample_to(np.asarray(pm_full, dtype=np.float32), pfps, fps, n)
        cov_raw = sig.get("player_coverage_full") or []
        coverage = (np.asarray(cov_raw, dtype=np.float32)
                    if cov_raw and len(cov_raw) == len(pm_full) else None)
        coverage = _resample(coverage, pfps, fps, n)
    else:
        motion = np.asarray(full, dtype=np.float32)
        coverage = None

    pose_fps = float((sig.get("pose_fps") or [0.0])[0]) or 0.0
    swing = sig.get("pose_swing_full") or []
    overhead = sig.get("pose_overhead_full") or []
    pose_cov = float((sig.get("pose_coverage") or [0.0])[0]) or 0.0
    quiet = float((sig.get("pose_quiet") or [0.0])[0]) or 0.0
    tpl = template or get_template()

    packed_ok = (int((sig.get("boundary_version") or [0])[0]) == BOUNDARY_EVIDENCE_VERSION
                 and sig.get("boundary_start_full") and sig.get("boundary_end_full"))
    if packed_ok:
        ev = build_evidence(fps=fps, duration=duration, motion=motion, coverage=coverage,
                            swing=swing, swing_fps=pose_fps,
                            overhead=overhead or None, pose_coverage=pose_cov,
                            swing_quiet=quiet, hit_times=hit_times, template=tpl)
        if ev is not None:
            ev.curves["start"] = np.asarray(sig["boundary_start_full"], dtype=np.float32)
            ev.curves["end"] = np.asarray(sig["boundary_end_full"], dtype=np.float32)
        return ev

    return build_evidence(fps=fps, duration=duration, motion=motion, coverage=coverage,
                          swing=swing or None, swing_fps=pose_fps,
                          overhead=overhead or None, pose_coverage=pose_cov,
                          swing_quiet=quiet, hit_times=hit_times, template=tpl)


def pack_signals(ev: BoundaryEvidence) -> dict[str, list[float]]:
    """Signal-dict entries to persist with the analysis (P5 visualization + resegment)."""
    return {
        "boundary_version": [BOUNDARY_EVIDENCE_VERSION],
        "boundary_start_full": [round(float(v), 4) for v in ev.curves["start"]],
        "boundary_end_full": [round(float(v), 4) for v in ev.curves["end"]],
    }


def boundary_meta(ev: BoundaryEvidence | None) -> dict[str, Any]:
    """Stats-side metadata (kept across resegment via keep_keys)."""
    if ev is None:
        return {"available": False, "version": BOUNDARY_EVIDENCE_VERSION}
    tpl = ev.template
    return {
        "available": ev.available_for_refine(),
        "version": BOUNDARY_EVIDENCE_VERSION,
        "template": tpl.name,
        "pose_coverage": round(ev.pose_coverage, 3),
        "w_motion": round(tpl.motion_weight(), 3),
        "w_hit": round(tpl.hit_weight(), 3),
        "thr_start": round(tpl.thr_start, 3),
        "thr_end": round(tpl.thr_end, 3),
    }


# ------------------------------------------------------------- sampling


def positive_samples(gt: list[tuple[float, float]],
                     window: tuple[float, float] | None = None
                     ) -> dict[str, list[float]]:
    """Labeled boundary times (positives), split by side."""
    lo, hi = (window or (0.0, float("inf")))
    return {
        "start": [float(a) for a, b in gt if b > a and lo <= a <= hi],
        "end": [float(b) for a, b in gt if b > a and lo <= b <= hi],
    }


def negative_samples(ev: BoundaryEvidence, gt: list[tuple[float, float]],
                     seg_min_quiet: float, seg_prominence: float,
                     margin: float = 1.5) -> dict[str, list[float]]:
    """In-rally quiet-valley bottoms (negatives): the P3 "wrong snap" set.

    Only bottoms strictly inside a labeled rally (``margin`` seconds from either label
    edge) are used — external gaps ARE boundaries and must not contaminate the negatives.
    Start/end share the same negative pool (a pause inside a rally is neither a serve nor
    an end boundary).
    """
    spans = RV.find_quiet_spans(ev.motion, ev.fps,
                                min_quiet=float(seg_min_quiet),
                                prominence_ratio=float(seg_prominence))
    bottoms = [q.bottom / ev.fps for q in spans]
    neg: list[float] = []
    for t in bottoms:
        for a, b in gt:
            if a + margin <= t <= b - margin:
                neg.append(float(t))
                break
    return {"start": neg, "end": list(neg)}


def feature_rows(ev: BoundaryEvidence, side: str, times: list[float]) -> list[dict[str, float]]:
    return [ev.channel_values(side, ev.frame(t)) for t in times]


# ------------------------------------------------------------- calibration


@dataclass
class FittedTemplate:
    template: Template
    diagnostics: dict[str, Any] = field(default_factory=dict)


def fit_template(rows_pos: list[dict[str, float]],
                 rows_neg: list[dict[str, float]],
                 side: str, *, name: str = "fitted",
                 pos_quantile: float = 0.15) -> FittedTemplate:
    """Fit robust ranges (pooled q15/q85) + directional weights + positive threshold.

    Weights come from the standardized median difference (pos vs neg); channels that do
    not separate in the expected direction get zero weight. The acceptance threshold is
    the ``pos_quantile`` quantile of the *positive* scores (high recall on real
    boundaries by construction).
    """
    ranges: dict[str, tuple[float, float]] = {}
    weights: dict[str, float] = {}
    diagnostics: dict[str, Any] = {"channels": {}}
    if not rows_pos or not rows_neg:
        # Without both classes every fitted weight/threshold is degenerate (all-zero
        # weights or a threshold of 0); refuse rather than emit an "accept anything"
        # template.
        base = NEUTRAL_TEMPLATE
        tpl = Template(ranges=dict(base.ranges), weights=dict(base.weights),
                       thr_start=base.thr_start, thr_end=base.thr_end,
                       name=f"{name}-insufficient")
        diagnostics["degenerate"] = {"n_pos": len(rows_pos), "n_neg": len(rows_neg)}
        return FittedTemplate(template=tpl, diagnostics=diagnostics)
    pos_mat = {c: np.asarray([r.get(c, 0.0) for r in rows_pos], dtype=np.float64)
               for c in ALL_CHANNELS}
    neg_mat = {c: np.asarray([r.get(c, 0.0) for r in rows_neg], dtype=np.float64)
               for c in ALL_CHANNELS}
    for ch in ALL_CHANNELS:
        p, ng = pos_mat[ch], neg_mat[ch]
        pool = np.concatenate([p, ng])
        lo = float(np.quantile(pool, 0.15))
        hi = float(np.quantile(pool, 0.85))
        if hi - lo <= EPS:
            hi = lo + 1.0
        ranges[ch] = (round(lo, 5), round(hi, 5))
        scale = max(hi - lo, EPS)
        d = (float(np.median(p)) - float(np.median(ng))) / scale if p.size and ng.size else 0.0
        w = float(np.clip(4.0 * d, 0.0, 2.0))
        weights[ch] = round(w, 4)
        diagnostics["channels"][ch] = {
            "median_pos": round(float(np.median(p)), 4) if p.size else None,
            "median_neg": round(float(np.median(ng)), 4) if ng.size else None,
            "separation": round(d, 3), "weight": round(w, 3),
        }
    thr = 0.55
    tpl = Template(ranges=ranges, weights=weights,
                   thr_start=0.55, thr_end=0.55, name=name)
    if rows_pos:
        sc = _scores_with(tpl, side, rows_pos)
        # Small epsilon so boundary cases at the quantile are accepted (rounding the
        # threshold upward to 4 dp rejected all positives on near-perfect separation).
        thr = float(np.quantile(sc, pos_quantile)) - 1e-6
    if side == "start":
        tpl = Template(ranges=ranges, weights=weights,
                       thr_start=round(thr, 6), thr_end=NEUTRAL_TEMPLATE.thr_end, name=name)
    else:
        tpl = Template(ranges=ranges, weights=weights,
                       thr_start=NEUTRAL_TEMPLATE.thr_start, thr_end=round(thr, 6), name=name)
    diagnostics["n_pos"] = len(rows_pos)
    diagnostics["n_neg"] = len(rows_neg)
    diagnostics[f"thr_{side}"] = round(thr, 6)
    return FittedTemplate(template=tpl, diagnostics=diagnostics)


def merge_templates(start: Template, end: Template, name: str = "loco") -> Template:
    """Combine side-specific fits (ranges/weights averaged; thresholds kept per side)."""
    ranges: dict[str, tuple[float, float]] = {}
    weights: dict[str, float] = {}
    for ch in ALL_CHANNELS:
        a, b = start.ranges.get(ch, NEUTRAL_TEMPLATE.ranges[ch]), \
            end.ranges.get(ch, NEUTRAL_TEMPLATE.ranges[ch])
        ranges[ch] = (round((a[0] + b[0]) / 2, 5), round((a[1] + b[1]) / 2, 5))
        weights[ch] = round((start.weights.get(ch, 0.0) + end.weights.get(ch, 0.0)) / 2, 4)
    return Template(ranges=ranges, weights=weights,
                    thr_start=start.thr_start, thr_end=end.thr_end, name=name)


def _scores_with(tpl: Template, side: str, rows: list[dict[str, float]]) -> np.ndarray:
    wm = sum(max(0.0, tpl.weights.get(c, 0.0)) for c in MOTION_CHANNELS)
    wh = max(0.0, tpl.weights.get(HIT_CHANNEL, 0.0))
    total = wm + wh
    out = []
    for r in rows:
        ms = 0.0
        for ch in MOTION_CHANNELS:
            w = max(0.0, tpl.weights.get(ch, 0.0))
            lo, hi = tpl.ranges[ch]
            ms += w * np.clip((r.get(ch, 0.0) - lo) / max(hi - lo, EPS), 0.0, 1.0)
        lo, hi = tpl.ranges[HIT_CHANNEL]
        hn = wh * np.clip((r.get(HIT_CHANNEL, 0.0) - lo) / max(hi - lo, EPS), 0.0, 1.0)
        out.append((ms + hn) / total if total > EPS else 0.0)
    return np.asarray(out, dtype=np.float64)


def acceptance(tpl: Template, side: str, rows: list[dict[str, float]]) -> dict[str, float]:
    """Fraction of samples passing the threshold (recall on positives, false alarm on negs)."""
    if not rows:
        return {"accepted": 0.0, "n": 0}
    sc = _scores_with(tpl, side, rows)
    thr = tpl.thr_start if side == "start" else tpl.thr_end
    return {"accepted": round(float(np.mean(sc >= thr)), 4), "n": len(rows),
            "score_q15": round(float(np.quantile(sc, 0.15)), 4),
            "score_median": round(float(np.median(sc)), 4)}


# ------------------------------------------------------------- tie-breaker


@dataclass
class _Snap:
    index: int
    side: str
    frm: float
    to: float
    score_frm: float
    score_to: float
    depth: float


def refine_boundaries(intervals: list, ev: BoundaryEvidence | None,
                      params: Any, duration: float,
                      *, method: str = "") -> tuple[list, dict[str, Any]]:
    """Snap interval starts/ends to better-supported nearby quiet-valley bottoms.

    Pure tie-breaker: returns the input list untouched (same objects) when the switch is
    off, evidence is missing, pose coverage is insufficient, or no candidate clears the
    threshold + gain + guardrail filters. See module docstring for the guardrail list.
    """
    if not getattr(params, "use_boundary_refine", False):
        return intervals, {"status": "off"}
    if ev is None:
        return intervals, {"status": "no_evidence"}
    if not ev.available_for_refine():
        return intervals, {"status": "pose_coverage_low",
                           "pose_coverage": round(ev.pose_coverage, 3)}
    if not intervals:
        return intervals, {"status": "empty"}

    max_move = float(getattr(params, "boundary_max_move", DEFAULT_MAX_MOVE) or DEFAULT_MAX_MOVE)
    margin = float(getattr(params, "boundary_score_margin", DEFAULT_SCORE_MARGIN)
                   or DEFAULT_SCORE_MARGIN)
    spans = RV.find_quiet_spans(
        ev.motion, ev.fps,
        min_quiet=float(getattr(params, "seg_min_quiet", 0.6)),
        prominence_ratio=float(getattr(params, "seg_prominence", 0.10)))
    valleys = [(q.bottom, q.depth) for q in spans]
    tpl = ev.template

    snaps: list[_Snap] = []
    out = list(intervals)
    m = len(out)
    for i, iv in enumerate(out):
        for side in ("start", "end"):
            b = float(iv.start if side == "start" else iv.end)
            frm = int(np.clip(ev.frame(b), 0, ev.n - 1))
            thr = tpl.thr_start if side == "start" else tpl.thr_end

            # Legal snap window: inside [b - R, b + R], never past the neighbor / the
            # other edge, and never producing a sub-MIN_RALLY_LEN rally.
            lo_t = max(0.0, b - max_move)
            hi_t = min(duration, b + max_move)
            if side == "start":
                prev_end = float(out[i - 1].end) if i > 0 else 0.0
                lo_t = max(lo_t, prev_end)
                hi_t = min(hi_t, float(iv.end) - MIN_RALLY_LEN)
            else:
                next_start = float(out[i + 1].start) if i + 1 < m else duration
                hi_t = min(hi_t, next_start)
                lo_t = max(lo_t, float(iv.start) + MIN_RALLY_LEN)
            if hi_t <= lo_t:
                continue
            lo_f, hi_f = ev.frame(lo_t), ev.frame(hi_t)

            cand = [(vf, depth) for vf, depth in valleys
                    if lo_f <= vf <= hi_f and vf != frm]
            if not cand:
                continue
            score_frm = ev.score(side, frm)
            best: tuple[int, float, float] | None = None
            for vf, depth in cand:
                if not ev.local_coverage_ok(vf):
                    continue
                sc = ev.score(side, vf)
                if sc < thr or sc < score_frm + margin:
                    continue
                # Minimum independent channels must fire at the candidate.
                vals = ev.channel_values(side, vf)
                fires = 0
                for ch in ALL_CHANNELS:
                    w = tpl.weights.get(ch, 0.0)
                    if w > EPS and _normalize(tpl, ch, vals[ch]) >= CHANNEL_FIRE:
                        fires += 1
                if fires < MIN_CHANNELS:
                    continue
                if best is None or sc > best[1]:
                    best = (vf, sc, depth)
            if best is None:
                continue
            vf, sc, depth = best
            to = round(ev.time(vf), 3)
            if side == "start":
                iv.start = to
            else:
                iv.end = to
            snaps.append(_Snap(index=i, side=side, frm=round(b, 3), to=to,
                               score_frm=round(score_frm, 3), score_to=round(sc, 3),
                               depth=round(float(depth), 3)))

    out.sort(key=lambda v: v.start)
    trace: dict[str, Any] = {
        "status": "applied" if snaps else "no_candidate",
        "method": method,
        "moved": len(snaps),
        "thresholds": {"start": tpl.thr_start, "end": tpl.thr_end},
        "max_move": max_move,
        "margin": margin,
        "snaps": [s.__dict__ for s in snaps],
    }
    return out, trace
