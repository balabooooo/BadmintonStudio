"""Badminton shuttlecock candidate detection and trajectory tracking — pure CV, no neural network.

Design rationale
----------------
The camera is **completely static** (measured inter-frame global displacement median 0.04 px, max < 1 px
on this project's footage), so no optical flow / registration is done; instead **temporal statistics**
are used for background modeling. In the frame the shuttlecock is a **very small, bright white, fast-moving point**:
about 8~20 px on 4K original, only 1~2.5 px on a 480p proxy. So the core of this module is not "find
bright things" (a gym has lots of constantly bright white lines, lights, white walls, white shoes),
but "find a **just-brightened**, **isolated** small white dot, and require it to form **a physically
plausible parabolic trajectory**".

Four filters in series
----------------------
1. **Whiteness channel**: ``W = min(R, G, B)``. The shuttle is achromatic (all three channels high),
   while the court is saturated green (R and B very low). Measured green court ``W ≈ 33``, white
   line/shoe ``W ≈ 200~255``, so finding "bright white dots" on ``W`` is much cleaner than on
   grayscale (on grayscale the contrast between a white line and green court is only ~120, on
   ``W`` it is ~170, and the green court's texture is flattened overall).
2. **Temporal novelty**: take the 85th percentile of the pixel in a window of K frames before and
   after as a reference, ``NOV = W - p85``. Constantly bright objects (white lines, lights) have
   ``NOV ≈ 0``; a fast-passing shuttle has a large ``NOV``. The reference is first **globally
   brightness-aligned** to cancel camera auto-exposure drift (this footage measurably has a +22
   gray-level rise at the start; without alignment hundreds of false candidates explode in the
   first 15 frames).
3. **Court prior + isolation** (three complementary criteria for "does it look like an isolated shuttle"):
   * **Green court around it**: the candidate point itself is **white** (do NOT require it to be green!),
     but the ring of background around it must be green court. In a gym, players' skin, wooden walls,
     and floor reflections get tinted by the court's green light and carry a green excess, so looking
     only at the candidate's own color cannot separate them — measured, looking only at the candidate
     color gives 1144 candidates (1.3/frame), while looking at the surrounding background leaves only
     68 (0.08/frame), the single strongest denoising criterion in this module.
   * **Brightness isolation**: the fraction of neighbors with brightness comparable to the candidate
     must be low. White shoes and white clothes form large white regions, have poor isolation, and are dropped.
   * **Novelty isolation**: the fraction of neighbors with equally high novelty must be low. When a
     whole white object moves, novelty is everywhere, while an isolated small white dot has high
     novelty only at itself.
4. **Speed gating + parabola fitting**: the shuttle is the fastest object in the scene; candidate
   points are associated frame to frame by velocity extrapolation, then a quadratic curve is fitted
   to the trajectory and large residuals are dropped. Random noise can hardly satisfy the three
   requirements "position, velocity, curvature" simultaneously for 5 consecutive frames, which is
   the final and strongest filter.

Known limitations (important, please treat honestly)
----------------------------------------------------
* **Not 100% accurate tracking**. The goal is only to provide auxiliary signals (estimating hit
  moments, shuttle speed, rally intensity) for cross-validating audio hit detection; do not treat it as ground truth.
* **White court lines are a fundamental blind spot**: when the shuttle flies above a white line the
  ``W`` background itself is bright, ``NOV`` is close to 0, and a miss is inevitable. The module uses
  a "static white region mask" to exclude these regions directly (prefer a miss over a false positive).
* When the shuttle is against a player's body, racket, or white shoes, it cannot be distinguished from
  "a moving large white object", causing misses or false detections.
* **On a low-bitrate proxy this module is essentially ineffective — this has been confirmed by
  measurement, so please do not assume it always works.** The only test footage available to this
  project is a 480x270 / 15fps / **441 bytes/frame** proxy: the 8x downsampling "smears" any 1~2 px
  small white dot together with the green court, so that
  * the candidate points' whiteness ``W`` all bunch into the narrow band 151~174 (the shuttle
    completely overlaps with far-field white shoes, and not a single candidate has ``W >= 190``, see
    the iteration log in the module header comments);
  * the temporal noise standard deviation of a flat court is already 11 gray levels at p90 and 37 at
    p99, the same order as the target amplitude;
  * in those 60 seconds the audio detected **91 real hits** (the shuttle really is flying), but when
    the sensitivity was tuned to 0.8/1.0 the fraction of detected presence frames aligned to real hit
    moments **within 0.2 s was only 12%/31%, lower than the ~61% of random guessing** — i.e. what is
    detected is noise, not the shuttle.
  Conclusion: **this proxy is not enough to support visual shuttlecock detection**; higher
  resolution/bitrate footage is needed (the described 4K original: the shuttle is 8~20 px, which is
  exactly what this module is designed for).
* Applies only to a **static camera**. Handheld/panning footage must be globally registered first,
  otherwise this module is meaningless.
* The default speed gate upper limit 0.6 frame-heights/second is a conservative value per
  specification; a real smash's frame speed far exceeds it, and on 4K footage ``max_speed`` should be
  raised to 3~6 (with ``max_gap`` relaxed accordingly), otherwise smashes are dropped by the gate.

Iteration log (measured on the 480x270 test proxy, to explain where these thresholds come from)
------------------------------------------------------------------------------------------------
* Round 1: temporal median background + brightness threshold + small connected components. **Failed**:
  candidates almost all landed on white court lines, ceiling lights, and white walls (130~180 candidates per frame).
* Round 2: added "temporal 85th percentile novelty" and global brightness alignment. The hundreds of
  false candidates that exploded at the start due to auto-exposure rise were suppressed, candidates
  dropped to p50=0 / p90=2; but the top events all landed on **moving white shoes**.
* Round 3: added isolation criteria. First wrote it wrong — used "the candidate point itself should be
  greenish" as the court prior, but the shuttle is white with green excess ≈ 0, which excluded the true
  target first, giving 0 candidates; after changing to **"the ring of background around the candidate
  must be green court"**, candidates dropped from 1144 (1.3/frame) to 72 (0.08/frame).
* Round 4: added trajectory-level physical constraints (minimum span + fitted acceleration lower bound).
  The white shoes of a walking player can only fit a uniform straight line (acceleration ≈ 0) and were
  correctly rejected → 0 trajectories. This is precisely the honest conclusion for this footage: there
  is no shuttle among the remaining candidates.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

from ..i18n import tr

# ---------------------------------------------------------------- public data structures


@dataclass
class ShuttlePoint:
    frame: int
    time: float
    x: float          # normalized 0~1
    y: float          # normalized 0~1
    score: float      # 0~1 confidence


@dataclass
class ShuttleTrack:
    """A continuous shuttlecock flight trajectory."""

    points: list[ShuttlePoint] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    #: Mean/max pixel speed (normalized by frame height, unit: frame-heights/second)
    mean_speed: float = 0.0
    max_speed: float = 0.0
    #: Total displacement from track start to end (normalized)
    span: float = 0.0
    confidence: float = 0.0


@dataclass
class ShuttleSignal:
    fps: float
    duration: float
    tracks: list[ShuttleTrack] = field(default_factory=list)
    #: Confidence 0~1 that "a shuttlecock is present in the frame" for each frame
    presence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Number of candidate points detected per frame
    candidate_count: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Maximum speed among all candidate points per frame (normalized/second)
    max_candidate_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------- constants / defaults

#: Reference height for threshold calibration. Scale-related thresholds are scaled proportionally to the working resolution based on this height.
REF_HEIGHT = 480.0
#: Default temporal window radius K (K frames before and after participate in the statistics)
DEFAULT_WINDOW = 15
#: Minimum number of points required for a track
DEFAULT_MIN_POINTS = 5
#: Maximum number of frames allowed to be lost during association
DEFAULT_MAX_GAP = 2
#: Speed gate (frame-heights/second). The shuttle is the fastest object in the scene: below the
#: lower bound it is a static white dot, and above the upper bound it is physically impossible for a
#: shuttle. Increase this value for high-speed smash scenes.
DEFAULT_MIN_SPEED = 0.02
DEFAULT_MAX_SPEED = 0.60
#: Maximum RMS residual allowed for the parabola fit (frame heights)
DEFAULT_MAX_RESID = 0.035
#: Minimum track span (frame heights): a "trajectory" short enough to be almost motionless in place is not a shuttle
DEFAULT_MIN_SPAN = 0.04
#: Minimum fitted acceleration of a track (frame-heights/second²).
#: Physical basis: in flight a shuttle is subject to gravity (~9.8 m/s²) plus air resistance, so its
#: trajectory necessarily curves noticeably; while "the white shoes of a player walking on court"
#: only produce an approximately uniform straight line.
#: Converted for a typical camera angle (frame height covers about 8 m), 9.8 m/s² ≈ 1.2
#: frame-heights/second², and the fitted acceleration of a high clear / flat drive falls between
#: 0.3~1.5. The default 0.25 is a lower bound with margin left.
#: **Note**: this value depends on the camera angle and the real distance covered by the frame;
#: changing the camera angle requires calibration. Set it to 0 to disable this criterion.
DEFAULT_MIN_ACCEL = 0.25
#: Upper bound on the working resolution width. Native 4K per-pixel processing is too slow and
#: unnecessary; scaling to this width suffices (after 3840 -> 1280 the shuttle is still 2.5~6.5 px).
DEFAULT_WORK_WIDTH = 1280
#: Calibrated range of connected-component area (@480 row height, unit px²)
_AREA_MIN_REF = 1.0
_AREA_MAX_REF = 80.0


# ---------------------------------------------------------------- low-level utilities


def _white_map(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute "whiteness" W and "green excess" GEX from a BGR frame."""
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    w = np.minimum(np.minimum(r, g), b)
    gex = g - 0.5 * (r + b)
    return w, gex


def _odd(v: float, lo: int = 3) -> int:
    k = int(round(v))
    if k < lo:
        k = lo
    if k % 2 == 0:
        k += 1
    return k


def _pctl_axis0(stack: np.ndarray, q: float) -> np.ndarray:
    """Compute the per-pixel q-th percentile along the time axis for a (T, H, W) stack.

    Uses :func:`numpy.partition` (O(T)) rather than :func:`numpy.percentile` (which sorts): 3~4x
    faster at this module's window length (T≈31); no interpolation is done, and the error is far
    smaller than the threshold margin.
    """
    t = stack.shape[0]
    k = int(round((t - 1) * q / 100.0))
    k = max(0, min(t - 1, k))
    return np.partition(stack, k, axis=0)[k]


def _sensitivity_thresholds(sensitivity: float, scale: float, work_h: int,
                            court_gex: float | None = None) -> dict:
    """Convert sensitivity(0~1) and the resolution into a set of actual thresholds.

    Resolution adaptation: area is scaled by ``scale**2`` (area is a 2D quantity, so it scales as the
    square rather than linearly — linear scaling would make the area upper bound too small at high
    resolution and drop the shuttle), while the isolation window and opening kernel scale linearly
    with ``scale``.

    The defaults were calibrated by measurement on this project's 480x270 test proxy (see the
    iteration log in the module docstring).
    """
    s = float(np.clip(sensitivity, 0.0, 1.0))
    area_min = max(1, int(round(_AREA_MIN_REF * scale * scale)))
    area_max = max(area_min + 1, int(round(_AREA_MAX_REF * scale * scale)))
    return {
        # Absolute whiteness lower bound. The shuttle is white: measured white lines/shoes are
        # 200~255, while player skin, wooden walls, and floor reflections are all 100~160, so this
        # keeps the vast majority of clutter out.
        "w": 170.0 - 40.0 * s,
        # Novelty threshold (relative to the temporal 85th-percentile reference)
        "nov": 36.0 - 18.0 * s,
        # Local contrast: the candidate must be clearly brighter than the median of the surrounding ring
        "contrast": 58.0 - 26.0 * s,
        # Court prior: lower bound on the median green excess of the background **ring around** the candidate (note: not the candidate itself)
        "bg_gex": 55.0 - 20.0 * s if court_gex is None else float(court_gex),
        # Opening kernel size; at too low a resolution the target itself is only 1~2 px and the
        # opening would eat it too, so it degenerates to 0 at low resolution (no opening)
        "open": 3 if work_h >= 360 else 0,
        "area_min": area_min,
        "area_max": area_max,
        "iso_win": _odd(11.0 * scale, 5),
        # Brightness isolation: upper bound on the fraction of neighbors with brightness comparable to the candidate (inside white shoes/clothes it approaches 1)
        "iso_w": 0.20 + 0.30 * s,
        # novelty isolation: upper bound on the fraction of neighbors with equally high novelty
        "iso_n": 0.05 + 0.20 * s,
    }


# ---------------------------------------------------------------- candidate detection


def _detect_candidates(
    w: np.ndarray,
    w_ref: np.ndarray,
    gex: np.ndarray,
    static_white: np.ndarray | None,
    mask_roi: np.ndarray | None,
    th: dict,
) -> list[tuple[float, float, float, float]]:
    """Extract candidate points from a single frame, returning [(x, y, score, novelty), ...] (pixel coordinates).

    ``gex`` is **this frame's green-excess map**, used only to judge whether the area **around** the
    candidate is green court. Note that it must not be used to require the candidate itself to be
    green — the shuttle is white with green excess near 0, and an early version fell into exactly this
    reversed prior (see the iteration log in the module docstring).
    """
    import cv2

    nov = w - w_ref
    m = (nov > th["nov"]) & (w > th["w"])
    if static_white is not None:
        m &= ~static_white
    if mask_roi is not None:
        m &= mask_roi
    if not m.any():
        return []

    m8 = m.astype(np.uint8)
    osz = th["open"]
    if osz >= 3:
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz, osz)))
        if not m8.any():
            return []

    nl, _lab, st, cent = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if nl <= 1:
        return []

    excess = np.maximum(nov, 0.0)
    h, wd = w.shape
    S = th["iso_win"]
    half = S // 2
    out: list[tuple[float, float, float, float]] = []
    for j in range(1, nl):
        area = int(st[j, cv2.CC_STAT_AREA])
        if area < th["area_min"] or area > th["area_max"]:
            continue
        x0, y0 = int(st[j, cv2.CC_STAT_LEFT]), int(st[j, cv2.CC_STAT_TOP])
        x1 = x0 + int(st[j, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(st[j, cv2.CC_STAT_HEIGHT])

        # Sub-pixel centroid: gray-weighted by novelty within the connected component
        sub = excess[y0:y1, x0:x1]
        tot = float(sub.sum())
        if tot <= 1e-6:
            cx, cy = float(cent[j][0]), float(cent[j][1])
        else:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cx = float((sub * xs).sum() / tot)
            cy = float((sub * ys).sum() / tot)

        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < wd and 0 <= iy < h):
            continue
        nov_peak = float(nov[iy, ix])
        lvl = float(w[iy, ix])

        # Local window centered on the candidate
        ax0, ax1 = max(0, ix - half), min(wd, ix + half + 1)
        ay0, ay1 = max(0, iy - half), min(h, iy + half + 1)
        pw = w[ay0:ay1, ax0:ax1]
        pn = nov[ay0:ay1, ax0:ax1]
        pg = gex[ay0:ay1, ax0:ax1]
        if pw.size < 9:
            continue
        bg = float(np.median(pw))

        # ---- local contrast: the candidate must be clearly brighter than the median of the surrounding ring
        contrast = lvl - bg
        if contrast < th["contrast"]:
            continue

        # Remove the central 3x3 core so the candidate itself is not counted as a "neighbor"
        core = np.zeros_like(pw, dtype=bool)
        core[max(0, iy - ay0 - 1):min(pw.shape[0], iy - ay0 + 2),
             max(0, ix - ax0 - 1):min(pw.shape[1], ix - ax0 + 2)] = True
        n_out = float(max(1, int((~core).sum())))

        # ---- court prior: whether the surrounding ring of background is green court
        ring = pg[~core]
        if ring.size < 4 or float(np.median(ring)) < th["bg_gex"]:
            continue

        # ---- brightness isolation: fraction of neighbors "comparable in brightness to the candidate"
        # (threshold taken at 60% between bg~lvl, hence independent of absolute brightness, applicable to dark and bright backgrounds)
        iso_w = float(((pw > bg + 0.60 * contrast) & ~core).sum()) / n_out
        if iso_w > th["iso_w"]:
            continue

        # ---- novelty isolation: fraction of neighbors with equally high novelty
        # When white shoes/clothes move as a whole, novelty is everywhere, and this rejects them
        # entirely; around an isolated small white dot novelty ≈ 0, so the ratio is near 0.
        nbg = float(np.median(pn))
        iso_n = float(((pn > nbg + 0.5 * (nov_peak - nbg)) & ~core).sum()) / n_out
        if iso_n > th["iso_n"]:
            continue

        sn = min(1.0, nov_peak / max(1.0, th["nov"] * 3.0))
        sc = min(1.0, contrast / max(1.0, th["contrast"] * 2.5))
        sw_ = max(0.0, 1.0 - iso_w / max(1e-6, th["iso_w"]))
        sn_ = max(0.0, 1.0 - iso_n / max(1e-6, th["iso_n"]))
        score = float(np.clip(0.35 * sn + 0.30 * sc + 0.15 * sw_ + 0.20 * sn_, 0.0, 1.0))
        out.append((cx, cy, score, nov_peak))
    return out


# ---------------------------------------------------------------- trajectory association


def _associate(
    per_frame: dict[int, list[tuple[float, float, float, float]]],
    fps: float,
    height: int,
    min_points: int,
    max_gap: int,
    max_speed: float,
) -> list[list[tuple[int, float, float, float]]]:
    """String per-frame candidate points into trajectories by "velocity extrapolation + nearest neighbor".

    Returns ``[[(frame, x, y, score), ...], ...]``, each sorted by frame number ascending.
    """
    frames = sorted(per_frame)
    active: list[list[tuple[int, float, float, float]]] = []
    done: list[list[tuple[int, float, float, float]]] = []

    for f in frames:
        cands = per_frame[f]
        used = [False] * len(cands)
        # Process the "longest-stalled" trajectories first, so new trajectories do not steal an old one's target
        active.sort(key=lambda trk: trk[-1][0])
        still: list[list[tuple[int, float, float, float]]] = []
        for trk in active:
            gap = f - trk[-1][0]
            if gap > max_gap + 1:          # gap too long, terminate the trajectory
                if len(trk) >= min_points:
                    done.append(trk)
                continue

            # Extrapolate the predicted position using the velocity of the last two points
            if len(trk) >= 2:
                pf, px, py, _ = trk[-2]
                lf, lx, ly, _ = trk[-1]
                dt = max(1e-6, (lf - pf) / fps)
                step = gap / fps
                pred_x = lx + (lx - px) / dt * step
                pred_y = ly + (ly - py) / dt * step
            else:
                pred_x, pred_y = trk[-1][1], trk[-1][2]

            # The search radius is determined by the speed upper bound, plus a 2 px floor to keep sub-pixel jitter from stalling it
            radius = max(2.0, max_speed * height * (gap / fps))
            best, best_cost = -1, 1e18
            for k, (cx, cy, sc, _nv) in enumerate(cands):
                if used[k]:
                    continue
                d = float(np.hypot(cx - pred_x, cy - pred_y))
                if d > radius:
                    continue
                cost = d / radius - 0.25 * sc      # the closer the better and the higher the candidate confidence the better
                if cost < best_cost:
                    best, best_cost = k, cost
            if best >= 0:
                used[best] = True
                cx, cy, sc, _nv = cands[best]
                trk.append((f, cx, cy, sc))
            still.append(trk)
        active = still

        # Unclaimed candidate points start new trajectories
        for k, (cx, cy, sc, _nv) in enumerate(cands):
            if not used[k]:
                active.append([(f, cx, cy, sc)])

    for trk in active:
        if len(trk) >= min_points:
            done.append(trk)
    return done


def _fit_track(
    pts: list[tuple[int, float, float, float]],
    fps: float,
    height: int,
    width: int,
    min_speed: float,
    max_speed: float,
    max_resid: float,
    min_points: int,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
) -> tuple[ShuttleTrack | None, float]:
    """Quadratic (parabola) fit and scoring; returns ``(track, RMS residual / frame height)``.

    Physical basis: subject to gravity + air resistance, over a short period the shuttle's trajectory
    is approximately a parabola in the frame, so a quadratic fit of position against time suffices.
    Besides a small residual, it also requires that **the fitted acceleration "not be too small"**:
    uniform straight-line motion (a walking person, a slowly moving white object) also has a small
    residual, but its quadratic term is close to 0; a real shuttle necessarily carries the curvature
    caused by gravity. This separates "physically like a shuttle" from "mathematically fits well".
    """
    n = len(pts)
    if n < min_points:
        return None, 1e9
    t = np.array([p[0] for p in pts], dtype=np.float64) / fps
    x = np.array([p[1] for p in pts], dtype=np.float64)
    y = np.array([p[2] for p in pts], dtype=np.float64)
    sc = np.array([p[3] for p in pts], dtype=np.float64)
    t = t - t[0]

    deg = 2 if n >= 5 else 1
    try:
        cx = np.polyfit(t, x, deg)
        cy = np.polyfit(t, y, deg)
    except Exception:
        return None, 1e9
    rx = x - np.polyval(cx, t)
    ry = y - np.polyval(cy, t)
    resid = float(np.sqrt(np.mean(rx * rx + ry * ry)) / max(1.0, height))
    if resid > max_resid:
        return None, resid

    dt = np.diff(t)
    dist = np.hypot(np.diff(x), np.diff(y))
    speeds = dist / np.maximum(dt, 1e-6) / max(1.0, height)
    mean_speed = float(speeds.mean()) if speeds.size else 0.0
    obs_max_speed = float(speeds.max()) if speeds.size else 0.0
    # mean speed below the lower bound => a static white dot (court line, light) rather than a shuttlecock
    if mean_speed < min_speed or mean_speed > max_speed * 1.5:
        return None, resid

    span = float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / max(1.0, height))
    if span < min_span:
        return None, resid

    # Fitted acceleration (quadratic term 2a): the shuttle must have been "bent" by gravity
    accel = 0.0
    if deg == 2:
        accel = float(np.hypot(2.0 * cx[0], 2.0 * cy[0]) / max(1.0, height))
    if min_accel > 0 and accel < min_accel:
        return None, resid
    # An absurdly large acceleration (most likely the fit was dragged off by noise) is also dropped
    if accel > 12.0:
        return None, resid

    score_mean = float(sc.mean())
    n_f = min(1.0, n / (2.0 * min_points))
    fit_q = max(0.0, 1.0 - resid / max(1e-6, max_resid))
    plaus = float(np.clip((speeds <= max_speed).mean() if speeds.size else 0.0, 0.0, 1.0))
    acc_q = 1.0 if min_accel <= 0 else float(np.clip(accel / (2.0 * min_accel), 0.0, 1.0))
    conf = float(np.clip(0.28 * score_mean + 0.22 * n_f + 0.28 * fit_q
                         + 0.12 * plaus + 0.10 * acc_q, 0.0, 1.0))

    track = ShuttleTrack(
        points=[
            ShuttlePoint(frame=int(p[0]), time=float(p[0] / fps),
                         x=float(p[1] / width), y=float(p[2] / height), score=float(p[3]))
            for p in pts
        ],
        start=float(pts[0][0] / fps),
        end=float(pts[-1][0] / fps),
        mean_speed=mean_speed,
        max_speed=obs_max_speed,
        span=span,
        confidence=conf,
    )
    return track, resid


# ---------------------------------------------------------------- main entry point


def analyze_shuttle(
    video_path: str,
    sample_fps: float = 30.0,
    roi: tuple[float, float, float, float] | None = None,
    max_seconds: float = 0.0,
    sensitivity: float = 0.5,
    on_progress=None,
    cancel=None,
    *,
    work_width: int = 0,
    window: int = DEFAULT_WINDOW,
    min_points: int = DEFAULT_MIN_POINTS,
    max_gap: int = DEFAULT_MAX_GAP,
    min_speed: float = DEFAULT_MIN_SPEED,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_resid: float = DEFAULT_MAX_RESID,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
    court_gex: float | None = None,
) -> ShuttleSignal:
    """Analyze the video, returning shuttlecock candidate trajectories and per-frame auxiliary signals.

    ``work_width`` and the parameters after it are optional tuning knobs (all have defaults and do not affect the call style given in the specification).

    Args:
        video_path: Input video. **The camera must be static**, otherwise the results are meaningless.
        sample_fps: Target sampling frame rate; when the source frame rate is lower it will not upsample.
        roi: Normalized ``(x0, y0, x1, y1)``; only look for the shuttle inside it; None means the whole frame.
        max_seconds: Only analyze the first N seconds; 0 means the whole video.
        sensitivity: 0~1; larger is more sensitive (lower thresholds, more candidates, more noise).
        on_progress: ``callable(progress: float, stage: str)``.
        cancel: ``callable() -> bool``; when it returns True, stop as soon as possible and return the results so far.
        work_width: Working resolution width; 0 means automatic (not exceeding :data:`DEFAULT_WORK_WIDTH`).
        window: Temporal window radius K; the reference uses K frames before and after.
        min_points: Minimum number of points for a track; tracks with fewer are dropped.
        max_gap: Maximum number of frames allowed to be lost during association.
        min_speed / max_speed: Speed gate (frame-heights/second).
        max_resid: Maximum RMS residual allowed for the parabola fit (frame heights).
        min_span: Minimum track span (frame heights).
        min_accel: Minimum fitted acceleration of a track (frame-heights/second²), used to exclude
            uniform straight-line motion (a walking person); 0 means disabled.
        court_gex: Court prior threshold (lower bound on the median green excess of the background ring around the candidate).
            None means it is taken automatically from sensitivity (45 when s=0.5). Set to -999 to disable
            this prior and let the module look for the shuttle over the whole frame (suited to a camera
            where the shuttle often flies against a dark ceiling/wall).

    Returns:
        :class:`ShuttleSignal`. ``fps`` is the **actual sampling frame rate** (lower than ``sample_fps`` when the source frame rate is low).
    """
    sig, _dbg = _run(
        video_path, sample_fps, roi, max_seconds, sensitivity, on_progress, cancel,
        work_width=work_width, window=window, min_points=min_points, max_gap=max_gap,
        min_speed=min_speed, max_speed=max_speed, max_resid=max_resid,
        min_span=min_span, min_accel=min_accel, court_gex=court_gex, want_debug=False,
    )
    return sig


def analyze_shuttle_debug(
    video_path: str,
    sample_fps: float = 30.0,
    roi: tuple[float, float, float, float] | None = None,
    max_seconds: float = 0.0,
    sensitivity: float = 0.5,
    on_progress=None,
    cancel=None,
    **kw,
) -> tuple[ShuttleSignal, dict]:
    """Diagnostic helper entry point (an extra function beyond the specification): additionally returns per-frame candidate points and track membership.

    The debug dict contains ``candidates`` (per-frame normalized ``(x, y, score)``), ``tracked``
    (per-frame normalized ``(x, y)`` belonging to valid tracks), ``width``/``height``/``step``/``frames``.
    The behavior and return value of ``analyze_shuttle`` are unaffected.
    """
    return _run(
        video_path, sample_fps, roi, max_seconds, sensitivity, on_progress, cancel,
        want_debug=True, **kw,
    )


def _run(
    video_path: str,
    sample_fps: float,
    roi: tuple[float, float, float, float] | None,
    max_seconds: float,
    sensitivity: float,
    on_progress,
    cancel,
    *,
    work_width: int = 0,
    window: int = DEFAULT_WINDOW,
    min_points: int = DEFAULT_MIN_POINTS,
    max_gap: int = DEFAULT_MAX_GAP,
    min_speed: float = DEFAULT_MIN_SPEED,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_resid: float = DEFAULT_MAX_RESID,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
    court_gex: float | None = None,
    want_debug: bool = False,
) -> tuple[ShuttleSignal, dict]:
    import cv2

    def emit(p: float, s: str) -> None:
        if on_progress is not None:
            try:
                on_progress(float(np.clip(p, 0.0, 1.0)), s)
            except Exception:
                pass

    empty = ShuttleSignal(fps=0.0, duration=0.0)
    if not video_path:
        return empty, {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return empty, {}

    # ---- working resolution
    ww = int(work_width) if work_width and work_width > 0 else min(src_w, DEFAULT_WORK_WIDTH)
    ww = max(64, min(ww, src_w))
    scale = ww / float(src_w)
    wh = max(2, int(round(src_h * scale)))
    wf = float(wh) / REF_HEIGHT            # scaling relative to the reference height
    thr = _sensitivity_thresholds(sensitivity, wf, wh, court_gex)

    step = max(1, int(round(src_fps / max(1e-3, float(sample_fps)))))
    eff_fps = src_fps / step
    limit_frames = int(max_seconds * src_fps) if max_seconds and max_seconds > 0 else (total or 10 ** 9)

    # ---- ROI mask
    mask_roi = None
    if roi is not None:
        x0, y0, x1, y1 = roi
        ax0 = int(np.clip(round(min(x0, x1) * ww), 0, ww))
        ax1 = int(np.clip(round(max(x0, x1) * ww), 0, ww))
        ay0 = int(np.clip(round(min(y0, y1) * wh), 0, wh))
        ay1 = int(np.clip(round(max(y0, y1) * wh), 0, wh))
        if ax1 <= ax0 or ay1 <= ay0:
            cap.release()
            return empty, {}
        mask_roi = np.zeros((wh, ww), bool)
        mask_roi[ay0:ay1, ax0:ax1] = True

    K = max(1, int(window))
    WIN = 2 * K + 1
    emit(0.02, tr("shuttle.detect_candidates"))

    # The ring buffer stores only uint8: W = min(R,G,B) is naturally in 0..255, and GEX with a 128
    # offset also fits (the court prior only cares whether gex is greater than ~35, so clipping to
    # [-128,127] does not affect the decision).
    wq: collections.deque = collections.deque(maxlen=WIN)
    gq: collections.deque = collections.deque(maxlen=WIN)
    bright_acc = np.zeros((wh, ww), np.float32)             # static white-region accumulation
    n_read = 0          # number of source frames read
    n_samp = 0          # number of frames sampled (= output timeline length)
    per_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    top_frames: list[tuple[int, int]] = []

    def process(target_local: int, frame_no: int) -> None:
        """Extract candidates for the target_local-th frame in the window, attributing the result to frame_no on the timeline."""
        stack = np.stack(wq).astype(np.float32)
        gimg = gq[target_local].astype(np.float32) - 128.0
        # Global brightness alignment to cancel auto-exposure drift
        gm = np.median(stack.reshape(stack.shape[0], -1), axis=1)
        off = gm[target_local]
        if np.any(np.abs(gm - off) > 0.5):
            stack -= (gm - off)[:, None, None]
        w_now = stack[target_local]
        w_ref = _pctl_axis0(stack, 85.0)

        frac = n_samp / max(1.0, float(n_read))
        if frac > 0.15:
            sw = bright_acc > (0.35 * n_read)
            static_white = cv2.dilate(sw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        else:
            static_white = None

        pts = _detect_candidates(w_now, w_ref, gimg, static_white, mask_roi, thr)
        if pts:
            per_frame[frame_no] = pts
            top_frames.append((frame_no, len(pts)))

    idx = 0
    while True:
        if cancel is not None and cancel():
            break
        if not cap.grab():
            break
        if idx % step != 0 and idx != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue
        if n_read >= limit_frames:
            break
        n_read += 1

        if frame.shape[1] != ww or frame.shape[0] != wh:
            frame_w = cv2.resize(frame, (ww, wh), interpolation=cv2.INTER_AREA)
        else:
            frame_w = frame
        w, gex = _white_map(frame_w)
        # GEX is stored as uint8 after adding 128 (saves memory); the court prior only cares whether it exceeds ~35
        gq.append(np.clip(gex + 128.0, 0, 255).astype(np.uint8))
        bright_acc += (w > 150.0)
        wq.append(np.clip(w, 0, 255).astype(np.uint8))

        if len(wq) < WIN:
            # Warm-up: the temporal reference has not been established yet, so any "novelty" is
            # untrustworthy. The footage measurably has an auto-exposure rise at the start (+22 gray
            # levels / 8 frames), and forcing candidates during warm-up would explode hundreds of
            # false points at once, so simply wait for the window to fill before starting.
            n_samp += 1
            continue

        cur = n_samp                      # index of the current frame on the timeline
        n_samp += 1
        # Window full: the target is the center frame, i.e. the frame K frames back
        process(K, cur - K)

        if n_read % 16 == 0:
            p = n_read / float(min(total, limit_frames)) if total else 0.5
            emit(0.02 + 0.73 * min(1.0, p), tr("shuttle.detect_candidates"))

    # Wrap-up: once the window is full, the last K frames have not been processed; finish them in the final window from the center onward
    done_all = cancel is None or not cancel()
    if done_all and len(wq) == WIN and n_samp > K:
        for r in range(K + 1, WIN):
            process(r, n_samp - (WIN - 1 - r))
    cap.release()
    emit(0.78, tr("shuttle.associate_tracks"))

    n_frames = n_samp
    duration = n_frames / eff_fps if eff_fps > 0 else 0.0
    presence = np.zeros(n_frames, np.float32)
    cand_count = np.zeros(n_frames, np.float32)
    max_speed_arr = np.zeros(n_frames, np.float32)
    for f, pts in per_frame.items():
        if 0 <= f < n_frames:
            cand_count[f] = len(pts)

    # ---- per-frame "maximum candidate speed": nearest-neighbor displacement from the previous sampled frame's candidates
    # Only counts within a physically reasonable search radius (anything beyond is treated as "cannot
    # associate" and recorded as 0 rather than infinity), so it is a **lower-bound** upper-bound
    # estimate: when the real shuttle speed exceeds the gate it is truncated here.
    gate = max(float(max_speed), 1e-3)
    dt = 1.0 / max(eff_fps, 1e-6)
    radius = gate * float(wh) * dt
    prev_pts: list[tuple[float, float, float, float]] | None = None
    for f in range(n_frames):
        pts = per_frame.get(f)
        if pts and prev_pts:
            pa = np.array([(p[0], p[1]) for p in pts], np.float32)
            pb = np.array([(p[0], p[1]) for p in prev_pts], np.float32)
            d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
            dmin = d.min(axis=1)
            ok_d = dmin[dmin <= radius]
            if ok_d.size:
                max_speed_arr[f] = float(ok_d.max()) / float(wh) / dt
        if pts:
            prev_pts = pts

    # ---- association + parabola fitting
    wf_h = float(wh)
    chains = _associate(per_frame, eff_fps, wh, min_points, max_gap, max_speed)
    tracks: list[ShuttleTrack] = []
    tracked_by_frame: dict[int, list[tuple[float, float]]] = {}
    for chain in chains:
        trk, _resid = _fit_track(chain, eff_fps, wh, ww, min_speed, max_speed,
                                 max_resid, min_points, min_span, min_accel)
        if trk is None:
            continue
        tracks.append(trk)
        for p in chain:
            tracked_by_frame.setdefault(p[0], []).append((p[1] / ww, p[2] / wf_h))

    tracks.sort(key=lambda t: (-t.confidence, t.start))
    for tk in tracks:
        for p in tk.points:
            if 0 <= p.frame < n_frames:
                presence[p.frame] = max(presence[p.frame], float(np.clip(tk.confidence, 0.0, 1.0)))

    emit(1.0, tr("shuttle.done"))
    sig = ShuttleSignal(
        fps=float(eff_fps),
        duration=float(duration),
        tracks=tracks,
        presence=presence,
        candidate_count=cand_count,
        max_candidate_speed=max_speed_arr,
    )
    if not want_debug:
        return sig, {}

    dbg = {
        "candidates": {f: [(p[0] / ww, p[1] / wf_h, p[2]) for p in pts]
                       for f, pts in per_frame.items()},
        "tracked": tracked_by_frame,
        "width": ww,
        "height": wh,
        "step": step,
    }
    return sig, dbg


__all__ = [
    "ShuttlePoint",
    "ShuttleTrack",
    "ShuttleSignal",
    "analyze_shuttle",
    "analyze_shuttle_debug",
]
