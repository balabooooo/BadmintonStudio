"""Court calibration and camera-angle recognition: stop the analysis from assuming "the camera is behind one particular court".

Problem
-------
In the original analysis the camera angle was **hard-coded**: the players are the two largest
person boxes in the frame, positioned in a horizontal band at 0.40~0.65 of the frame height, the
larger the box the closer to the camera, and the court is green. These priors hold for one
particular setup — "ultra-wide fisheye, low camera, mounted behind the court" — and any other angle
silently produces wrong results (no error, just wrong segmentation and the wrong person tracked).

This module provides a **neutral geometric foundation** so downstream modules no longer have to guess:

1. :func:`court_mask`: find the "court" region from color priors (green/blue/gray/wood floors all supported),
   and turn it into a mask;
2. :func:`find_court_poly`: fit the court boundary from the mask (a **polygon**, 4~16 points);
3. :func:`estimate_viewpoint`: determine which kind of shot this is
   (``rear`` behind the court / ``side`` from the sideline / ``elevated`` high camera / ``overhead`` top-down);
4. :func:`CourtCalibration`: provide a **homography** that projects frame coordinates onto the "court plane",
   so "who is closer to the camera", "how far apart are the two players", and "how fast is the shuttle"
   can all be answered in a coordinate system independent of the camera angle.

Why a **polygon** instead of a quadrilateral
--------------------------------------------
Before the 2025 version, calibration had only one representation: "four corners". That convention
baked in an assumption: **the court boundary is a straight line in the frame**. The court boundary
shot by a panoramic camera / fisheye lens is curved (barrel distortion, curving more the closer to
the frame edge), so:

* Framing a curved-boundary court with four corners either cuts off a corner (the two near corners
  fall outside the frame) or encloses the spectator area outside the court as well;
* Determining "is the player inside the court" used a **rectangular** bounding box, and the error
  caused by the curved boundary at the frame edges gets amplified to a whole stand — which is
  exactly where panoramic footage has the most background people.

So the core representation of this module is changed to an **N-point polygon** (``polygon``, normalized coordinates, 4~24 points):

* The polygon is used directly for "inside the court" (:func:`point_in_poly`, per-point ray casting)
  and the ROI bounding box (:func:`CourtCalibration.court_mask_roi`);
* When a homography is needed (camera-angle determination, distance comparison), the **largest-area
  quadrilateral** is **fit from the polygon** (:func:`fit_quad_from_poly`) and then the homography is
  computed — a homography requires four-point correspondence, and a multi-point boundary can only be
  approximated this way;
* The "area the polygon has beyond the quadrilateral" is recorded as the **distortion degree**
  (``CourtCalibration.distortion``), so the UI can hint "the edges of this court are strongly curved;
  consider adding more points".

A quadrilateral is the special case with point count = 4, so the old manual calibration (``court_quad``)
and automatic calibration remain fully compatible: if the fitted quadrilateral is already accurate
enough, the polygon is that quadrilateral.

Coordinate system convention
---------------------------
Calibrated court coordinates use **real badminton-court metric units**: ``u`` along the court's short
side (6.10 m between the doubles sidelines), ``v`` along the court's long side (13.40 m between the
baselines), origin at the court center. The four corners are numbered clockwise starting from the
"corner closest to the camera" (``c0..c3``), so the ``near`` side is ``v < 0`` (the half closer to
the origin). Even without a real court (e.g. unclear lines in a training hall), as long as a
quadrilateral is fitted, downstream can at least get camera-independent information such as "where
in the frame are the two people and along which direction are they separated".

All computation depends only on numpy / opencv; no GPU is needed, and no existing signal is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..i18n import tr

EPS = 1e-9

#: Standard badminton court dimensions (meters) — doubles sideline width 6.10, baseline-to-baseline distance 13.40
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

#: Allowed point-count range for the court polygon. The lower bound is 4 because a homography
#: requires four-point correspondence (three points give only an affine map, which cannot express
#: perspective); the upper bound 24 is a compromise for "enough and not letting the fitted
#: quadrilateral degenerate into noise" — no matter how curved the edges, there will not be more
#: inflection points than this.
MIN_POLY_POINTS = 4
MAX_POLY_POINTS = 24

Viewpoint = Literal["rear", "side", "elevated", "overhead", "unknown"]

_VIEWPOINT_CODES = ("rear", "side", "elevated", "overhead", "unknown")


def _viewpoint_label(vp: str) -> str:
    """Localized display label for a viewpoint code (falls back to the code itself)."""
    return tr(f"viewpoint.{vp}") if vp in _VIEWPOINT_CODES else vp


# ------------------------------------------------------------------ color / court


@dataclass
class CourtColor:
    """Court color statistics (used to build the mask)."""

    #: Center of the dominant color in HSV
    hue: float = 60.0
    sat: float = 120.0
    val: float = 140.0
    #: Fraction of pixels with the dominant color
    ratio: float = 0.0
    #: Human-readable name of the dominant color
    name: str = "unknown"


def _hsv_of(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def estimate_court_color(frames: list[np.ndarray], max_side: int = 240) -> CourtColor:
    """Estimate the dominant color of the **ground** from several sampled frames.

    The key is the "sampling location", not the histogram itself: the gym walls and ceiling are
    often in the same color family as the ground (in measured footage the wall is cyan wood paneling
    and the floor is teal-green matting, only a few hue bins apart), and whole-frame statistics would
    pick the wall as the dominant color, turning the fitted "court" into one long horizontal band.

    So sampling is done in the region at the **bottom 45%, horizontal middle 60%** of each frame:
    whether the camera is behind the baseline, at the sideline, or a high overhead angle, that
    region necessarily falls on the court. The resulting hue/saturation/value serves as the center
    values for :func:`court_mask`.
    """
    import cv2

    if not frames:
        return CourtColor()
    hist = np.zeros(180, dtype=np.float64)
    sat_acc = np.zeros(180, dtype=np.float64)
    val_acc = np.zeros(180, dtype=np.float64)
    total = 0
    for fr in frames:
        h, w = fr.shape[:2]
        scale = max_side / max(1, max(h, w))
        if scale < 1.0:
            fr = cv2.resize(fr, (max(2, int(w * scale)), max(2, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
        h, w = fr.shape[:2]
        patch = fr[int(h * 0.55):, int(w * 0.20):int(w * 0.80)]
        if patch.size == 0:
            patch = fr
        hsv = _hsv_of(patch)
        hh = hsv[:, :, 0].astype(np.int32).ravel()
        ss = hsv[:, :, 1].astype(np.float64).ravel()
        vv = hsv[:, :, 2].astype(np.float64).ravel()
        # Only look at colored, normally-bright pixels (white court lines and shadows are excluded).
        # The thresholds are deliberately very low: the far end of the floor gets washed out by the
        # lights, with saturation dropping to a few tens; a higher threshold would leave only the
        # small near-end patch and make the hue estimate more biased instead.
        sel = (ss > 30) & (vv > 32)
        if not np.any(sel):
            continue
        idx = hh[sel]
        wgt = ss[sel] / 255.0
        np.add.at(hist, idx, wgt)
        np.add.at(sat_acc, idx, ss[sel])
        np.add.at(val_acc, idx, vv[sel])
        total += int(sel.sum())
    if hist.sum() <= EPS or total == 0:
        return CourtColor()
    # Sliding-window sum over a 12-bin-wide window, weighted average inside -> insensitive to hue jitter
    win = 12
    kernel = np.ones(win, dtype=np.float64)
    ext = np.concatenate([hist, hist[: win - 1]])
    scores = np.convolve(ext, kernel[::-1], mode="valid")
    best = int(np.argmax(scores))
    idxs = np.arange(best, best + win) % 180
    wsum = float(hist[idxs].sum())
    if wsum <= EPS:
        return CourtColor()
    hue = float((idxs * hist[idxs]).sum() / wsum)
    sat = float(sat_acc[idxs].sum() / wsum)
    val = float(val_acc[idxs].sum() / wsum)
    ratio = float(wsum / max(hist.sum(), EPS))
    return CourtColor(hue=hue % 180.0, sat=sat, val=val, ratio=ratio,
                      name=_hue_name(hue % 180.0))


def _hue_name(hue: float) -> str:
    if 35 <= hue < 46:
        return "yellow"
    if 46 <= hue < 78:
        return "green"
    if 78 <= hue < 100:
        return "cyan"
    if 100 <= hue < 130:
        return "blue"
    if hue < 12 or hue >= 168:
        return "red"
    if 12 <= hue < 35:
        return "orange"
    return "unknown"


def court_mask(frame: np.ndarray, color: CourtColor, tol: float = 12.0,
               sat_min: float = 0.45, val_min: float = 0.62,
               sat_abs: float = 60.0) -> np.ndarray:
    """Generate a binary mask of the court region from the estimated ground dominant color (uint8 0/255).

    ``sat_min`` / ``val_min`` are **relative** to the dominant color, not absolute values: the
    matting is bright or dark under different lighting and distances (measured saturation 215 at the
    near end but only 81 at the far end), but the relative relation "saturation not below some
    fraction of the dominant color, brightness not below some fraction of the dominant color" is
    fairly stable, while also keeping the darker, grayer wall out. An additional absolute lower
    bound ``sat_abs`` is added: white court lines and gray floors have very low saturation, and
    without an absolute lower bound they would be counted in as well.
    """
    import cv2

    hsv = _hsv_of(frame)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    dh = np.abs(h - float(color.hue))
    dh = np.minimum(dh, 180.0 - dh)          # hue is circular
    m = ((dh <= tol)
         & (s >= max(sat_abs, float(color.sat) * sat_min))
         & (v >= max(25.0, float(color.val) * val_min)))
    mask = (m.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def _bottom_anchor(mask: np.ndarray) -> float:
    """Coverage of the mask in the bottom-center region of the frame (used to judge "has the ground been found")."""
    h, w = mask.shape[:2]
    band = mask[int(h * 0.90):, int(w * 0.25):int(w * 0.75)]
    return float(np.mean(band > 0)) if band.size else 0.0


def _union_mask(frames: list[np.ndarray], color: CourtColor, tol: float,
                sat_min: float, val_min: float, sat_abs: float,
                keep_ratio: float = 0.45) -> np.ndarray | None:
    """Build masks on several sampled frames and take the overlap region that is "selected in every frame".

    The meaning of overlap: the real ground is selected in every frame, while players, spectators,
    and other courts only hit occasionally, so taking the overlap lets these distractors disappear
    on their own.
    """
    import cv2

    acc: np.ndarray | None = None
    for fr in frames:
        m = court_mask(fr, color, tol=tol, sat_min=sat_min, val_min=val_min, sat_abs=sat_abs)
        # Keep only large-enough connected components, filtering out scattered noise
        con, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(m)
        for c in con:
            if cv2.contourArea(c) > 0.01 * m.size:
                cv2.drawContours(keep, [c], -1, 255, -1)
        m = keep
        if acc is None:
            acc = m.astype(np.float32)
        else:
            if m.shape != acc.shape:
                m = cv2.resize(m, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_NEAREST)
            acc += m.astype(np.float32)
    if acc is None:
        return None
    acc /= float(len(frames))
    mask = (acc > keep_ratio).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return mask


def _seed_component(mask: np.ndarray, seed_y: float = 0.93, seed_x: float = 0.5,
                    min_area_ratio: float = 0.02) -> np.ndarray | None:
    """Take the connected component containing "the point at the bottom-center of the frame" in the mask.

    That point is certainly the court, so using it as a seed is more reliable than "take the largest
    connected component": the largest color blob in a gym is not necessarily the court (in measured
    footage the ceiling and wood-paneled wall are brighter than the floor). To avoid this row falling
    exactly on a white court line, scan a horizontal band at the bottom and take a majority vote.
    """
    import cv2

    h, w = mask.shape[:2]
    band_y0 = int(h * max(0.0, seed_y - 0.03))
    band_x0, band_x1 = int(w * 0.35), int(w * 0.65)
    band = mask[band_y0:, band_x0:band_x1]
    seed = None
    if band.size:
        ys, xs = np.nonzero(band)
        if xs.size:
            k = int(np.argmax(np.bincount(xs)))
            rows = ys[xs == k]
            seed = (band_x0 + int(k), band_y0 + int(np.median(rows)))
    if seed is None:
        return None
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    lbl = labels[seed[1], seed[0]]
    if lbl == 0:
        return None
    comp = (labels == lbl).astype(np.uint8) * 255
    if float(np.mean(comp > 0)) < min_area_ratio:
        return None
    return comp


def measure_court_band(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Judge row by row "how much of this row is court" to get the court's top and bottom boundaries.

    This is the most critical step of the whole calibration, and the approach comes from measurement:
    analyze row by row from the **very bottom** of the frame upward; a row counts as court if and only if

    * the row has a sufficient fraction of pixels within the "ground color range measured from the bottom region";
    * and those pixels **form one connected run** (covering a large part of the row's horizontal span);
      scattered hits do not count (those are speckles on the wall, not the ground).

    The measured row profile of the footage (480 wide, 270 high, pixels with hue close to the ground dominant color):

    | Row position | Saturation | Brightness | Verdict |
    | --- | --- | --- | --- |
    | 0.00–0.44 | 60–95 | 45–107 | Ceiling / wall: low saturation |
    | 0.50 | 82 | 142 | Transition band: far court + wainscoting |
    | 0.56–0.94 | 194–220 | 160–180 | Near court: extremely high saturation |

    As can be seen, **saturation** separates the ground from the wall very cleanly (more than 2x
    difference), while the hue differs by only a dozen bins (the wall's wood paneling and the matting
    are both teal-green tones). So the criterion uses saturation rather than hue.

    Returns:
        ``(boolean array of whether each row is court, statistics info)``; array length equals the frame height.
    """
    if not frames:
        return np.zeros(0, dtype=bool), {}
    fr = frames[0]
    h, w = fr.shape[:2]
    hsv = _hsv_of(fr)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # Take the mode of the ground hue from the bottom 12% region (that is certainly the court)
    band = slice(int(h * 0.88), h)
    hs = H[band].ravel()
    ss = S[band].ravel()
    sel0 = ss > 60
    if int(sel0.sum()) < 50:
        return np.zeros(h, dtype=bool), {"reason": tr("court.note.band_low_samples")}
    hue_hist = np.bincount(hs[sel0].astype(np.int32), minlength=180)
    # Find the peak with a 12-bin sliding window, insensitive to hue jitter
    win = 12
    ext = np.concatenate([hue_hist, hue_hist[: win - 1]])
    scores = np.convolve(ext, np.ones(win), mode="valid")
    best = int(np.argmax(scores))
    idxs = np.arange(best, best + win) % 180
    wsum = float(hue_hist[idxs].sum())
    hue = float((idxs * hue_hist[idxs]).sum() / max(wsum, EPS)) if wsum > 0 else float(best)
    sat_ref = float(np.median(ss[sel0]))

    dh = np.abs(H - hue)
    dh = np.minimum(dh, 180.0 - dh)
    sat_thr = max(60.0, sat_ref * 0.62)
    cand = (dh <= 20.0) & (S >= sat_thr) & (V >= 40.0)

    # Per row: hit ratio + longest run
    row_hit = np.zeros(h, dtype=np.float32)
    row_span = np.zeros(h, dtype=np.float32)
    for y in range(h):
        row = cand[y]
        hit = float(row.mean())
        row_hit[y] = hit
        if hit < 0.05:
            continue
        xs = np.nonzero(row)[0]
        best_run = 1
        run = 1
        for k in range(1, xs.size):
            run = run + 1 if xs[k] == xs[k - 1] + 1 else 1
            if run > best_run:
                best_run = run
        row_span[y] = best_run / float(w)

    # The criterion must allow two kinds of "breaks": court lines splitting a row into several
    # segments, and net posts or players occluding parts. So both the hit ratio and the **longest
    # run** are checked, and both thresholds are fairly loose (in measured footage the near-end
    # matting's longest run is only about 40% of the whole row because of the court lines).
    is_court = (row_hit >= 0.60) & (row_span >= 0.30)
    # Search from the very bottom upward: stop as soon as several consecutive rows are not court (avoid interference from scattered rows above)
    court_rows = np.zeros(h, dtype=bool)
    miss = 0
    for y in range(h - 1, -1, -1):
        if is_court[y]:
            court_rows[y] = True
            miss = 0
        else:
            miss += 1
            if miss > max(2, int(0.02 * h)):
                break
    info = {
        "hue": round(hue, 1), "sat_ref": round(sat_ref, 1), "sat_thr": round(sat_thr, 1),
        "top_ratio": round(float(np.nonzero(court_rows)[0].min()) / h, 3)
        if court_rows.any() else 1.0,
        "cover": round(float(np.mean(court_rows)), 3),
    }
    return court_rows, info


def court_mask_from_rows(frame: np.ndarray, court_rows: np.ndarray,
                         hue: float, sat_thr: float) -> np.ndarray:
    """Generate this frame's court mask from the "per-row court determination" plus color conditions."""
    import cv2

    h, w = frame.shape[:2]
    if court_rows.size != h:
        court_rows = np.resize(court_rows, h)
    hsv = _hsv_of(frame)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    dh = np.abs(H - hue)
    dh = np.minimum(dh, 180.0 - dh)
    cand = ((dh <= 26.0) & (S >= max(45.0, sat_thr * 0.55))).astype(np.uint8)
    cand[~court_rows, :] = 0
    mask = cand * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    # Keep only the connected component containing the bottom-center
    comp = _seed_component(mask, min_area_ratio=0.03)
    return comp if comp is not None else mask


def refine_court_color(frames: list[np.ndarray],
                       base: CourtColor) -> tuple[CourtColor, np.ndarray | None]:
    """Search a small range around the base color to find the parameter set that "most looks like one whole ground".

    Scoring criterion: it must enclose a connected component **containing the bottom-center of the
    frame**, the larger the area the better, but exceeding 0.8 means the wall / ceiling was counted
    in too, which is heavily penalized.
    """
    grid = [
        (dh, sm)
        for dh in (-6.0, -3.0, 0.0, 3.0, 6.0, 10.0)
        for sm in (0.08, 0.20, 0.35)
    ]
    best_score = -1.0
    best_color = base
    best_mask: np.ndarray | None = None
    for dh, sm in grid:
        cand = CourtColor(hue=(base.hue + dh) % 180.0, sat=base.sat, val=base.val,
                          ratio=base.ratio, name=_hue_name((base.hue + dh) % 180.0))
        union = _union_mask(frames, cand, tol=12.0, sat_min=sm, val_min=0.35, sat_abs=60.0)
        if union is None:
            continue
        comp = _seed_component(union)
        if comp is None:
            continue
        cover = float(np.mean(comp > 0))
        if cover < 0.05 or cover > 0.98:
            continue
        score = min(cover, 0.80) - max(0.0, cover - 0.80) * 3.0
        if score > best_score:
            best_score, best_color, best_mask = score, cand, comp
    return best_color, best_mask


def _contour_area(pts: np.ndarray) -> float:
    """Shoelace area of the polygon (pixels², absolute value)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def find_court_quad(mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray | None:
    """Fit a quadrilateral from the court mask, returning 4×2 pixel coordinates (order not fixed).

    Prefer ``approxPolyDP`` to find a true quadrilateral; on degeneration use ``minAreaRect``; if
    that still fails, use the four extreme points of the convex hull. An area that is too small
    (under 5% of the frame) counts as not found.

    .. note::
       This function is kept for old callers that "must have exactly four points" (and regression
       tests). New code should use :func:`find_court_poly` — the court boundary in panoramic /
       fisheye footage is curved, and forcing it into a quadrilateral needlessly loses boundary accuracy.
    """
    import cv2

    if mask is None or mask.size == 0:
        return None
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < min_area_ratio * w * h:
        return None
    peri = cv2.arcLength(big, True)
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        ap = cv2.approxPolyDP(big, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(big)
    return cv2.boxPoints(rect).astype(np.float32)


def _resample_ring(pts: np.ndarray, n: int, phase: float = 0.0) -> np.ndarray:
    """Resample a closed contour uniformly by **arc length** into n points (``phase`` offsets the start).

    Why resample by arc length instead of continuing to use ``approxPolyDP``: ``approxPolyDP``'s
    tolerance is "relative to the perimeter", and on a curved-edge shape the same tolerance jumps
    right over the intermediate states — a slightly smaller tolerance keeps dozens of points, a
    slightly larger one squeezes the whole arc into one chord (the curve is flattened). So it cannot
    answer "how many points describe this curved edge". Arc-length resampling can ask precisely:
    "with n points, how accurately can this boundary be described?"

    ``phase`` is necessary: where the start of uniform sampling falls determines whether vertices
    land exactly on the "corners" (if the start falls in the middle of an edge, a corner gets cut and
    the area is short a piece). The caller tries several phases for the same n and takes the best.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return p[:n]
    # Remove adjacent duplicate points: very common on contours, and they create zero-length segments that make interpolation non-monotonic
    keep = np.ones(p.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
    p = p[keep]
    if p.shape[0] < 3:
        return p[:n]
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return p[:n]
    t = ((np.arange(n) / float(n) + float(phase) % 1.0) % 1.0) * total
    xs = np.interp(t, s, closed[:, 0])
    ys = np.interp(t, s, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def find_court_poly(mask: np.ndarray, min_area_ratio: float = 0.05,
                    max_points: int = 16, iou_min: float = 0.985,
                    phases: int = 12) -> np.ndarray | None:
    """Fit a **polygonal** boundary from the court mask, returning N×2 pixel coordinates (4 ≤ N ≤ max_points).

    The criterion is "**describe this region accurately with as few points as possible**": start at
    4 points and add one at a time, taking the first point count whose "IoU with the original contour"
    reaches ``iou_min`` or above.

    * When the boundary is straight (ordinary camera angle) 4 points suffice — the IoU is almost 1,
      so the returned result is the traditional quadrilateral and behavior matches the old version;
    * When the boundary is curved (panoramic / fisheye) 4 points cut off the corners and the IoU
      drops, so points are automatically added to 6, 8, ... until the curved edge is described clearly.

    For each point count two candidate families are compared (taking the higher IoU):

    * **RDP** (``approxPolyDP``, tolerance binary-searched to "just n points left"). It is good at
      preserving "corners": a rectangle with unequal edge lengths can be described exactly with 4
      points, which arc-length uniform sampling cannot do (uniform sampling places vertices evenly by
      perimeter, and with unequal edges they do not land on the corners);
    * **Arc-length uniform sampling** (trying ``phases`` start phases). It is good at describing
      smooth curved edges **without corners**: RDP's start point is given by the contour and fixed,
      and on a curved edge that fixed start may take up a vertex.

    Computing both families is cheap (a contour is usually a few hundred points, a dozen milliseconds),
    and in exchange "straight uses a quadrilateral, curved automatically adds points" without the user
    having to tell us which kind of footage it is.

    Why IoU rather than "area difference": area difference cannot see "the shape is right but shifted
    as a whole". IoU constrains position and shape at once, and what we ultimately care about is
    exactly "is this region enclosed by the polygon correct" — the sole purpose of the court ROI.
    Rasterization is done at 256 wide, so the criterion itself has about half a percentage point of
    quantization error.
    """
    import cv2

    if mask is None or mask.size == 0:
        return None
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    raw_area = float(cv2.contourArea(big))
    if raw_area < min_area_ratio * w * h:
        return None
    contour = big.reshape(-1, 2).astype(np.float64)
    if contour.shape[0] < 4:
        return None

    # Compute IoU on a low-resolution grid: the criterion needs far less precision than pixel
    # level, 256 wide is enough, and it allows computing "every point count × every candidate family" (a dozen milliseconds).
    scale = 256.0 / max(1, w)
    sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    ref = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(ref, [np.round(contour * scale).astype(np.int32).reshape(-1, 1, 2)], 1)
    if int(np.count_nonzero(ref)) == 0:
        return None

    def _iou(cand: np.ndarray) -> float:
        m = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(cand * scale).astype(np.int32).reshape(-1, 1, 2)], 1)
        inter = int(np.count_nonzero(np.logical_and(m, ref)))
        union = int(np.count_nonzero(np.logical_or(m, ref)))
        return float(inter / union) if union else 0.0

    diag = float(np.hypot(w, h))
    cnt32 = contour.astype(np.float32).reshape(-1, 1, 2)

    def _rdp(n: int) -> np.ndarray | None:
        """Binary-search the tolerance: find the smallest tolerance that leaves "just n points" and return that polygon.

        RDP's point count is a step function of the tolerance (a slightly larger tolerance means one
        fewer point), so binary-searching the tolerance is more reliable than supplying a list of
        fixed tolerances: a fixed-tolerance grid easily skips over the desired point count entirely.
        """
        lo, hi = 0.0, diag
        ap = cv2.approxPolyDP(cnt32, hi, True)
        if len(ap) > n:
            return None                     # even the largest tolerance cannot reduce it to n points
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            cur = cv2.approxPolyDP(cnt32, mid, True)
            if len(cur) > n:
                lo = mid
            else:
                hi = mid
        ap = cv2.approxPolyDP(cnt32, hi, True).reshape(-1, 2).astype(np.float64)
        return ap if 3 <= ap.shape[0] <= n else None

    lo_n = max(MIN_POLY_POINTS, 3)
    hi_n = int(max(lo_n, min(max_points, MAX_POLY_POINTS)))
    best_any: tuple[float, np.ndarray] | None = None
    for n in range(lo_n, hi_n + 1):
        cands: list[np.ndarray] = []
        rdp = _rdp(n)
        if rdp is not None:
            cands.append(rdp)
        for k in range(max(1, int(phases))):
            cands.append(_resample_ring(contour, n, phase=k / float(max(1, int(phases)))))
        best_iou = -1.0
        best_pts = cands[0]
        for cand in cands:
            if cand.shape[0] < 3:
                continue
            v = _iou(cand)
            if v > best_iou:
                best_iou, best_pts = v, cand
        if best_any is None or best_iou > best_any[0]:
            best_any = (best_iou, best_pts)
        if best_iou >= iou_min:
            return best_pts.astype(np.float32)
    # Still below target at the upper limit (the boundary is very curved, or the court is irregular):
    # fall back to "the best one" so at least downstream has a calibration available — "it is not
    # accurate enough" is recorded into notes by the caller.
    if best_any is not None and best_any[0] >= 0.90:
        return best_any[1].astype(np.float32)
    return find_court_quad(mask, min_area_ratio=min_area_ratio)


def poly_area_norm(poly: np.ndarray) -> float:
    """Area of a normalized polygon (0~1, total frame area = 1)."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def order_poly(poly: np.ndarray, aspect: float = 1.7778) -> np.ndarray:
    """Arrange a polygon with arbitrary point order into a ring order of "near-left start, going right across the frame".

    All three kinds of input must be handled:

    * points casually clicked by the user (possibly messy, even self-crossing back and forth);
    * output of ``approxPolyDP`` (already ordered, but the start is arbitrary);
    * program-generated.

    Approach: sort by "polar angle relative to the centroid" to get the ring order, then rotate the
    start to the **nearest vertex** (largest y, i.e. the bottom of the frame; on ties take the more
    leftward one, consistent with the convention of :func:`order_quad`). The polar angle uses
    ``atan2(dy, dx * aspect)``: only after multiplying x by the frame aspect ratio is "being close"
    comparable across different frame proportions; otherwise on portrait footage the polar angle gets
    horizontally flattened and the ring order easily swaps the two near-end points.

    After sorting, a quick "convexity check" is done: a shoelace area clearly smaller than the convex
    hull area means the point order self-intersects (the user clicked a bow tie), in which case the
    convex hull order is used — a self-intersecting polygon makes :func:`point_in_poly`'s decision
    completely counterintuitive and also cannot be used for homography fitting.
    """
    import cv2

    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    n = p.shape[0]
    if n < 3:
        return p
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    ang = np.arctan2(p[:, 1] - cy, (p[:, 0] - cx) * max(0.2, float(aspect)))
    order = np.argsort(-ang)                    # descending = wrap from the near end toward the right
    ring = p[order]
    # Self-intersection check (only meaningful with more than 4 points; four points are always a deformation of a simple polygon)
    if n > 4:
        try:
            hull = cv2.convexHull(ring.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) >= 3 and _contour_area(ring) < 0.90 * _contour_area(hull):
                hcx, hcy = float(hull[:, 0].mean()), float(hull[:, 1].mean())
                hang = np.arctan2(hull[:, 1] - hcy, (hull[:, 0] - hcx) * max(0.2, float(aspect)))
                ring = hull[np.argsort(-hang)]
                n = ring.shape[0]
        except Exception:
            pass
    # Start point: the leftmost among the largest y (nearest)
    y_max = float(ring[:, 1].max())
    tol = max(1e-6, 0.06 * float(ring[:, 1].max() - ring[:, 1].min()))
    near = np.nonzero(ring[:, 1] >= y_max - tol)[0]
    start = int(near[np.argmin(ring[near, 0])]) if near.size else 0
    return np.roll(ring, -start, axis=0).astype(np.float32)


def fit_quad_from_poly(poly: np.ndarray) -> np.ndarray | None:
    """Pick the **largest-area quadrilateral** from the polygon (4×2, pixel coordinates).

    A homography requires four-point correspondence, while the polygon gives N boundary points, so
    here we must "choose four". The choice uses a universal property of court boundaries: **the four
    corners of a real court are necessarily the four most convex points on the boundary**, so "the
    inscribed quadrilateral formed by the vertices has the largest area" is equivalent to "the four
    corners were chosen", and it does not need camera parameters.

    With ≤ 4 points it returns directly (fewer than 4 returns ``None``). With many points it
    exhaustively enumerates C(N,4): N is capped at :data:`MAX_POLY_POINTS` (24), worst case 10626
    combinations, each requiring one shoelace area, microsecond-level, not worth writing a smarter
    heuristic for.
    """
    from itertools import combinations

    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    n = p.shape[0]
    if n < 4:
        return None
    if n == 4:
        return order_quad(p)
    if n > MAX_POLY_POINTS:
        # Uniformly thin out to the upper limit, preserving ring order (thinning loses a bit of
        # boundary precision but not the corners: the sample points on both sides of a corner remain,
        # so the exhaustive search can still pick near them)
        idx = np.linspace(0, n - 1, MAX_POLY_POINTS).round().astype(int)
        p = p[np.unique(idx)]
        n = p.shape[0]
    best_area = -1.0
    best: np.ndarray | None = None
    for combo in combinations(range(n), 4):
        q = p[list(combo)]
        a = _contour_area(q)
        if a > best_area:
            best_area = a
            best = q
    if best is None or best_area <= EPS:
        return None
    return order_quad(best)


def offset_poly(poly: np.ndarray, dist: float) -> np.ndarray:
    """Translate the polygon **outward** by ``dist`` (same coordinate-system unit, usually normalized coordinates).

    The method is the standard geometric "polygon offset": translate each edge along its outward
    normal by ``dist``, then take the intersection of two adjacent translated edges as the new vertex.
    For a convex polygon (the court boundary is convex) this gives an exact equidistant expansion —
    every edge gives way outward by the same distance.

    Why not simply "scale up about the centroid": that would let edges far from the centroid give
    more and near edges give less, while the court polygon in the frame is often a flat horizontal
    strip (the near half-court); the centroid-to-far-edge distance is much smaller than the distance
    to the left/right sides, so scaling up would leave the far boundary almost unmoved and push the
    left/right boundaries far out — exactly opposite to the direction that most needs relaxing, "a
    real player standing on the far boundary".

    On self-intersection or degeneration (too few points, area not increased) it returns the original
    polygon: better to relax a little less than to produce a weirdly shaped ROI that puts everyone outside.
    """
    q = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = q.shape[0]
    if n < 3 or dist <= 0:
        return q.astype(np.float32)
    # The winding direction determines the sign of the outward normal: in the image coordinate
    # system (y downward), when the shoelace area is positive the normal (e_y, -e_x) points outward
    x, y = q[:, 0], q[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    sign = 1.0 if signed > 0 else -1.0

    pts: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for i in range(n):
        a = q[i]
        b = q[(i + 1) % n]
        e = b - a
        L = float(np.linalg.norm(e))
        if L < EPS:
            continue
        d = e / L
        nvec = sign * np.array([d[1], -d[0]], dtype=np.float64)
        dirs.append(d)
        pts.append(a + nvec * float(dist))
    m = len(pts)
    if m < 3:
        return q.astype(np.float32)

    out: list[np.ndarray] = []
    for i in range(m):
        p1, d1 = pts[i], dirs[i]
        p2, d2 = pts[(i + 1) % m], dirs[(i + 1) % m]
        # Intersect two lines: p1 + t*d1 = p2 + s*d2
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:            # parallel (collinear) edges: take the translated endpoint directly
            out.append(p1)
            continue
        rhs = p2 - p1
        t = (rhs[0] * d2[1] - rhs[1] * d2[0]) / denom
        out.append(p1 + d1 * t)
    # out[j] is "the intersection of edge j and edge j+1", i.e. the new position of original vertex
    # j+1; roll back one slot so output point i still corresponds to input point i (start unchanged)
    cand = np.roll(np.asarray(out, dtype=np.float64), 1, axis=0) if m == n else np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(cand)):
        return q.astype(np.float32)
    # After expansion the area must **increase** (using absolute value only: the winding may be
    # clockwise or counterclockwise, and using signed area to judge direction would misjudge a
    # successful expansion as failure on a clockwise ring); otherwise the offset was computed wrong,
    # and it is better to return as is than use a weirdly shaped polygon as the ROI.
    a_cand = _poly_area_signed(cand)
    if not np.isfinite(a_cand) or abs(a_cand) <= abs(signed):
        return q.astype(np.float32)
    return cand.astype(np.float32)


def _poly_area_signed(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_poly(pts: np.ndarray, poly: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Per-point test of "whether it falls inside the polygon" (ray casting, vectorized).

    Args:
        pts: ``(N, 2)`` points to test (same coordinate system as ``poly``, usually normalized coordinates).
        poly: ``(M, 2)`` polygon vertices (ring order, M ≥ 3).
        margin: Outward relaxation distance (same unit as the coordinates, 0 = exact test).

    Returns:
        A bool array of length N.

    Why not approximate with a "bounding rectangle": in full-court footage the error caused by a
    curved boundary is concentrated at the frame corners, which is exactly where "people on other
    courts / spectators" are most numerous — judging with a rectangle amounts to putting all of them
    into the candidate pool, leaving only the size threshold to hard-filter later.

    ``margin`` is for **filtering players**. Two reasons:

    * Ray casting gives a half-open interval result for points exactly on the boundary (upper boundary
      counts as outside, lower as inside), and a player's foot landing exactly on the drawn boundary
      line is entirely possible;
    * The "court" enclosed by the calibration is a conservative near half-court to begin with, and a
      real player at the far end often stands exactly on that boundary row.

    The cost of erring in one direction is asymmetric: keeping one extra person is just a bit more
    noise (size filtering and activity scoring filter again), while missing one person makes the
    downstream activity curve and crop tracking drop out completely. Exact determination (drawing,
    statistics) uses the default ``margin=0``.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if margin > 0 and q.shape[0] >= 3:
        q = offset_poly(q, float(margin)).astype(np.float64)
    m = q.shape[0]
    if p.shape[0] == 0 or m < 3:
        return np.ones(p.shape[0], dtype=bool)
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(p.shape[0], dtype=bool)
    x1, y1 = q[:, 0], q[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for i in range(m):
        ax, ay, bx, by = x1[i], y1[i], x2[i], y2[i]
        if abs(by - ay) < EPS and abs(bx - ax) < EPS:
            continue
        # Whether this edge crosses the horizontal ray (half-open interval, to avoid counting a vertex twice)
        crosses = ((ay > y) != (by > y))
        if not np.any(crosses):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (y - ay) / np.where(by == ay, EPS, by - ay)
        xint = ax + t * (bx - ax)
        inside ^= crosses & (x <= xint)
    return inside


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Arrange the four corner points as [near-left, near-right, far-right, far-left].

    "Near" = lower in the frame (larger y), which holds under any camera angle: a point on the
    ground closer to the bottom of the frame is closer to the camera.
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    order = np.argsort(-q[:, 1])          # y from large to small = from near to far
    near = q[order[:2]]
    far = q[order[2:]]
    near = near[np.argsort(near[:, 0])]   # sort the near edge by x -> near-left, near-right
    far = far[np.argsort(far[:, 0])]      # same for the far edge
    return np.array([near[0], near[1], far[1], far[0]], dtype=np.float32)


# ------------------------------------------------------------------ calibration


@dataclass
class CourtCalibration:
    """Court calibration result for one analysis.

    The core representation is the **polygon** (``polygon`` / ``polygon_norm``): with 4 points it is
    the traditional four-corner quadrilateral, and with more points it describes the curved court
    boundary in panoramic / fisheye footage. ``quad`` / ``quad_norm`` is the quadrilateral fitted
    from the polygon, serving only the homography.
    """

    #: Whether the court polygon was successfully calibrated (success means a homography can be built)
    ok: bool = False
    #: Fitted quadrilateral (pixel coordinates, [near-left, near-right, far-right, far-left]) — for homography only
    quad: np.ndarray | None = None
    #: Normalized (0~1) quadrilateral, convenient for the frontend to draw boxes (old field, kept for compatibility)
    quad_norm: list[list[float]] = field(default_factory=list)
    #: Court boundary polygon (pixel coordinates, ring order: near-left start, wrapping rightward across the frame)
    polygon: np.ndarray | None = None
    #: Normalized (0~1) court boundary polygon — used by the frontend for drawing and by in-court determination
    polygon_norm: list[list[float]] = field(default_factory=list)
    #: Where the polygon comes from: ``auto`` automatic recognition / ``manual`` user calibration / ``none``
    source: str = "none"
    #: Ratio 0~1 of "area the polygon has beyond the fitted quadrilateral"; larger for stronger edge curvature (distortion)
    distortion: float = 0.0
    #: Frame coordinates -> court coordinates (unit square [0,1]×[0,1], u horizontal, v near-to-far)
    homography: np.ndarray | None = None
    #: Inverse transform
    homography_inv: np.ndarray | None = None
    #: Recognized camera angle
    viewpoint: Viewpoint = "unknown"
    #: Confidence of the camera-angle recognition 0~1
    confidence: float = 0.0
    #: Court color
    color: CourtColor = field(default_factory=CourtColor)
    #: Fraction of the frame area the court occupies
    area_ratio: float = 0.0
    #: Frame aspect ratio
    aspect: float = 1.7778
    #: "How much shorter the far edge is than the near edge" — measures perspective compression from near-large/far-small (1 = no compression)
    foreshortening: float = 1.0
    #: Fraction of players (normalized box bottom-edge center) falling inside/outside the court
    in_court_ratio: float = 0.0
    #: Explanatory information
    notes: list[str] = field(default_factory=list)

    # ---- coordinate transform ----
    def to_court(self, pts: np.ndarray) -> np.ndarray:
        """Frame pixel coordinates -> court coordinates (unit square; valid when ``ok`` is true)."""
        if not self.ok or self.homography is None:
            return np.zeros((0, 2), dtype=np.float32)
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        import cv2

        return cv2.perspectiveTransform(p, self.homography).reshape(-1, 2)

    def to_court_norm(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        """Normalized frame coordinates -> court coordinates."""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return self.to_court(np.stack([p[:, 0] * w, p[:, 1] * h], axis=1))

    def contains(self, pts_norm: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Which points in normalized frame coordinates fall **inside the court polygon** (bool array).

        If a polygon exists judge by the polygon; if only a quadrilateral exists (old data) judge by
        the quadrilateral — both are the same path, so the caller needs no branching.
        """
        poly = self.polygon_norm or self.quad_norm
        p = np.asarray(pts_norm, dtype=np.float32).reshape(-1, 2)
        if not poly:
            return np.ones(p.shape[0], dtype=bool)
        return point_in_poly(p, np.asarray(poly, dtype=np.float32), margin=margin)

    def contains_box_bottom(self, boxes: "list | np.ndarray", margin: float = 0.02) -> np.ndarray:
        """Whether the **bottom-edge center** of a batch of normalized boxes (xyxy) falls inside the court.

        The bottom-edge center approximates "the person's landing point on court", and that is what
        player-detection ROI filtering uses. Default 2% outward expansion: a player standing on the
        boundary line must not be judged off-court (the cost of missing a real player far exceeds
        keeping one extra background person).
        """
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if arr.size == 0:
            return np.zeros(0, dtype=bool)
        pts = np.stack([0.5 * (arr[:, 0] + arr[:, 2]), arr[:, 3]], axis=1)
        return self.contains(pts, margin=margin)

    def court_mask_roi(self, margin: float = 0.06) -> tuple[float, float, float, float] | None:
        """Bounding rectangle of the court extent (normalized, with expansion), usable directly as the analysis ROI.

        The bounding rectangle is only for "coarse filtering" (narrowing the decode range and
        excluding people clearly at the other end of the frame); for fine in-court determination use
        :meth:`contains`.
        """
        src = self.polygon_norm or self.quad_norm
        if not src:
            return None
        arr = np.asarray(src, dtype=np.float32)
        x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
        x1, y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
        return (max(0.0, x0 - margin), max(0.0, y0 - margin),
                min(1.0, x1 + margin), min(1.0, y1 + margin))

    def poly_norm(self) -> list[list[float]]:
        """Polygon (normalized coordinates) for the frontend to draw lines."""
        return self.polygon_norm or self.quad_norm

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def as_payload(self) -> dict[str, Any]:
        """Serializable summary for the frontend / statistics."""
        poly = self.polygon_norm or self.quad_norm
        return {
            "ok": self.ok,
            "viewpoint": self.viewpoint,
            "viewpoint_label": _viewpoint_label(self.viewpoint),
            "confidence": round(float(self.confidence), 3),
            # polygon is the primary representation; quad is kept for old frontends / projects that "only recognize four corners"
            "polygon": poly,
            "quad": self.quad_norm,
            "point_count": len(poly),
            "source": self.source,
            "distortion": round(float(self.distortion), 4),
            "court_color": self.color.name,
            "court_area_ratio": round(float(self.area_ratio), 3),
            "foreshortening": round(float(self.foreshortening), 3),
            "in_court_ratio": round(float(self.in_court_ratio), 3),
            "roi": list(self.court_mask_roi()) if poly else None,
            "notes": list(self.notes),
        }


#: Target coordinate system of the court plane: unit square, u horizontal 0~1 (left->right), v vertical 0~1 (near->far)
_COURT_TARGET = np.array([
    [0.0, 0.0],   # near-left
    [1.0, 0.0],   # near-right
    [1.0, 1.0],   # far-right
    [0.0, 1.0],   # far-left
], dtype=np.float32)


def build_calibration(poly: np.ndarray, frame_w: int, frame_h: int,
                      color: CourtColor | None = None,
                      source: str = "auto") -> CourtCalibration:
    """Build a calibration from the court boundary (an **N-point polygon**, N ≥ 4; 4 points is the traditional quadrilateral).

    A homography requires four-point correspondence, so :func:`fit_quad_from_poly` first picks the
    largest-area inscribed quadrilateral from the polygon, then uses it to compute the homography.
    The polygon itself is used for in-court determination and ROI — together they keep curved-boundary
    footage from being forced into straight lines, while downstream's ``to_court()`` /
    ``in_court_ratio`` / camera-angle determination stay completely unchanged.

    Args:
        poly: ``(N, 2)`` pixel coordinates, point order arbitrary (normalized by :func:`order_poly`).
        frame_w / frame_h: Original frame size (the space the polygon coordinates live in).
        color: Court color estimate (may be empty).
        source: ``auto`` / ``manual``, sent out with the payload.

    **Note**: the target coordinate system is a unit square, not real metric units. The reason is
    very practical: often only part of the court is visible in the frame (measured footage shows only
    the near half-court), and forcing it to map to 13.40 m × 6.10 m would magnify all vertical
    distances by more than 2x, which is worse than no calibration. Under the unit square, relative
    quantities like "the two players are 0.4 court-widths apart" are reliable, and judging the camera
    angle, judging who is at the near end, and crop tracking all do not need real metric units.
    """
    import cv2

    cal = CourtCalibration(ok=False, color=color or CourtColor(),
                           aspect=float(frame_w) / max(1, frame_h), source=source)
    if poly is None:
        cal.note(tr("court.note.no_boundary"))
        return cal

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < MIN_POLY_POINTS:
        cal.note(tr("court.note.poly_too_few", n=pts.shape[0], min=MIN_POLY_POINTS))
        return cal

    ring = order_poly(pts, aspect=cal.aspect)
    cal.polygon = ring
    cal.polygon_norm = [[round(float(x) / frame_w, 5), round(float(y) / frame_h, 5)]
                        for x, y in ring]
    poly_area = poly_area_norm(np.asarray(cal.polygon_norm, dtype=np.float64))
    if poly_area < 0.02:
        cal.note(tr("court.note.poly_too_small"))
        return cal

    quad = fit_quad_from_poly(ring)
    if quad is None:
        cal.note(tr("court.note.quad_fit_failed"))
        return cal
    q = order_quad(quad)
    cal.quad = q
    cal.quad_norm = [[round(float(x) / frame_w, 5), round(float(y) / frame_h, 5)] for x, y in q]

    quad_area = float(abs(cv2.contourArea(q.reshape(-1, 1, 2))) / max(1, frame_w * frame_h))
    cal.area_ratio = poly_area
    # The "extra area ratio of the polygon over the quadrilateral" = the degree of boundary curvature (lens distortion / court matting bulging)
    cal.distortion = float(np.clip(1.0 - quad_area / max(poly_area, EPS), 0.0, 1.0))
    if ring.shape[0] > 4:
        cal.note(tr("court.note.poly_points", count=ring.shape[0],
                    pct=f"{cal.distortion:.1%}"))
    if cal.distortion > 0.06:
        cal.note(tr("court.note.distortion", pct=f"{cal.distortion:.0%}"))

    near_w = float(np.linalg.norm(q[1] - q[0]))
    far_w = float(np.linalg.norm(q[2] - q[3]))
    cal.foreshortening = float(min(near_w, far_w) / max(near_w, far_w, EPS))
    H = cv2.getPerspectiveTransform(q, _COURT_TARGET)
    if not np.all(np.isfinite(H)) or abs(float(H[2, 2])) < EPS:
        cal.note(tr("court.note.homography_degenerate"))
        return cal
    HH = H / float(H[2, 2])
    cal.homography = HH.astype(np.float32)
    cal.homography_inv = np.linalg.inv(cal.homography).astype(np.float32)
    cal.ok = True
    return cal


# ------------------------------------------------------------------ camera-angle determination


def _as_xyxy(item: Any) -> tuple[float, float, float, float] | None:
    """Unify a "per-frame box" into ``(x1, y1, x2, y2)``.

    Both representations exist in this project: ``PlayerSignal.frame_boxes`` is
    ``(track_id, x1, y1, x2, y2)``, while per-frame boxes from the detection stage are
    ``(x1, y1, x2, y2)``. Camera-angle determination only cares about geometry, so both must be
    accepted — early versions recognized only 5-tuples, and 4-tuples were skipped entirely (showing
    up as "too few player boxes to determine the camera angle").
    """
    try:
        arr = np.asarray(item, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size >= 5:
        return (float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]))
    if arr.size >= 4:
        return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    return None


def estimate_viewpoint(
    cal: CourtCalibration,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    frame_w: int = 1000,
    frame_h: int = 1000,
    hint: Viewpoint | None = None,
) -> CourtCalibration:
    """Determine the camera-angle type and write the basis into ``notes``.

    Everything used is **independent of frame orientation**:

    * ``foreshortening``: far-end width / near-end width of the court. When viewed at an angle the
      far end is clearly narrower (near-large/far-small); for an overhead shot it is close to 1. It
      directly reflects "how high is the camera above the ground and how far from the court";
    * The distribution of players' **frame coordinates**: spread horizontally vs spread vertically;
    * ``size_ratio``: box area ratio of the two players on screen (a proxy for depth difference).

    Thus:

    * Players mainly separated horizontally in the frame and of similar box size -> **side** (the
      typical signature of a sideline view);
    * Players mainly separated vertically in the frame, one near and one far with an obvious size
      difference -> **rear** (camera behind the baseline);
    * The court quadrilateral is almost undistorted (``foreshortening`` close to 1) -> **overhead**;
    * In between and the players are distributed near the frame center -> **elevated** (high camera, angled down).
    """
    if player_boxes is None or player_fps <= 0 or not player_boxes:
        cal.viewpoint = hint or "unknown"
        cal.confidence = 0.9 if hint else 0.0
        cal.note(tr("court.note.no_players")
                 + (tr("court.note.hint_specified") if hint else tr("court.note.use_generic")))
        return cal

    # ---- player-box statistics (computable whether or not calibration succeeded)
    xs: list[float] = []
    ys: list[float] = []
    heights: list[float] = []
    ratios: list[float] = []
    bottoms: list[float] = []          # y of the box bottom edge (the "person's position on court")
    parsed: list[tuple[float, float, float, float]] = []
    for fr in player_boxes:
        fr_boxes: list[tuple[float, float, float, float]] = []
        for item in fr or ():
            b = _as_xyxy(item)
            if b is None:
                continue
            x1, y1, x2, y2 = b
            fr_boxes.append(b)
            xs.append(0.5 * (x1 + x2))
            ys.append(0.5 * (y1 + y2))
            bottoms.append(y2)
            heights.append(abs(y2 - y1))
        if fr_boxes:
            parsed.extend(fr_boxes)
        if len(fr_boxes) >= 2:
            h = sorted((abs(b[3] - b[1]) for b in fr_boxes), reverse=True)
            if len(h) >= 2 and h[1] > EPS:
                ratios.append(float(h[0] / h[1]))
    if len(xs) < 8:
        cal.viewpoint = "unknown"
        cal.note(tr("court.note.too_few_boxes"))
        return cal

    asp = float(frame_w) / max(1, frame_h)
    spread_x = float(np.percentile(xs, 92) - np.percentile(xs, 8)) * asp
    spread_y = float(np.percentile(ys, 92) - np.percentile(ys, 8))
    med_h = float(np.median(heights)) if heights else 0.0
    size_ratio = float(np.median(ratios)) if ratios else 1.0
    area_ratio = cal.area_ratio
    foreshort = float(cal.foreshortening)

    # ---- when calibration succeeded, also look at the distribution after projecting players into court coordinates
    spread_u = spread_v = -1.0
    if cal.ok:
        pts = np.asarray([(x, y) for x, y in zip(xs, ys)], dtype=np.float32)
        court = cal.to_court_norm(pts, frame_w, frame_h)
        good = (np.isfinite(court).all(axis=1)
                & (np.abs(court[:, 0]) < 3.0) & (np.abs(court[:, 1]) < 3.0))
        cal.in_court_ratio = float(np.mean(good))
        if good.sum() >= 8:
            spread_u = float(np.percentile(court[good, 0], 92) - np.percentile(court[good, 0], 8))
            spread_v = float(np.percentile(court[good, 1], 92) - np.percentile(court[good, 1], 8))
        # Another measure of "in court": judge the polygon directly with the **box bottom-edge center**
        # (without going through the homography). The bottom-edge center = the player's landing point
        # on court, which is the criterion player-detection ROI actually uses; judging with the box
        # center underestimates (the upper half of a person box is often outside the court: raised arm,
        # jump, or the box also captured the scoreboard), giving a measure nobody uses.
        if (cal.polygon_norm or cal.quad_norm) and parsed:
            arr = np.asarray(parsed, dtype=np.float32)
            ratio = float(np.mean(cal.contains_box_bottom(arr)))
            if abs(ratio - float(np.mean(good))) > 0.05:
                cal.note(tr("court.note.in_court_poly", poly_pct=f"{ratio:.0%}",
                            homo_pct=f"{float(np.mean(good)):.0%}"))
            cal.in_court_ratio = ratio

    cal.note(tr("court.note.spread", spread_x=f"{spread_x:.2f}", spread_y=f"{spread_y:.2f}",
                med_h=f"{med_h:.3f}", size_ratio=f"{size_ratio:.2f}"))
    if cal.ok and spread_u >= 0:
        cal.note(tr("court.note.court_spread", spread_u=f"{spread_u:.2f}",
                    spread_v=f"{spread_v:.2f}", foreshort=f"{foreshort:.2f}",
                    area_pct=f"{area_ratio:.0%}"))
    elif not cal.ok:
        cal.note(tr("court.note.no_calib_rough"))

    ratios_uv = spread_u / max(spread_v, 1e-3) if spread_u >= 0 else 0.0
    horiz = spread_x / max(spread_y, 1e-3)
    ratio_hint = hint if hint in ("rear", "side", "elevated", "overhead") else None

    # The height of player boxes in the frame is the most direct measure of camera height: the higher
    # the camera, the smaller the players. Thresholds are drawn by "how tall people are in the frame":
    # overhead/high camera is usually < 0.12, a normal angled camera 0.12~0.5, close-up > 0.5. The
    # measured footage is 0.18~0.22 (low ultra-wide camera).
    tall = med_h >= 0.16
    short = med_h < 0.11

    conf = 0.45
    if not cal.ok:
        # No calibration: can only use "which direction players spread in" and "size difference / box height"
        if short and horiz < 1.6 and size_ratio < 2.0:
            cal.viewpoint, conf = "elevated", 0.45
        elif horiz > 1.8 and size_ratio < 2.2:
            cal.viewpoint, conf = "side", 0.5
        elif size_ratio > 1.8 and horiz < 1.6:
            cal.viewpoint, conf = "rear", 0.55
        else:
            cal.viewpoint, conf = "unknown", 0.3
    elif ratio_hint:
        # The user/upper layer explicitly specified the camera angle: trust it, but record the two measured quantities for verification
        cal.viewpoint = ratio_hint
        conf = 0.9
        cal.note(tr("court.note.hint_viewpoint", viewpoint=_viewpoint_label(ratio_hint)))
    elif short and foreshort > 0.80 and area_ratio > 0.30:
        cal.viewpoint = "overhead"
        conf = min(0.9, 0.5 + area_ratio * 0.5)
    elif short:
        cal.viewpoint = "elevated"
        conf = 0.6
    elif ratios_uv > 1.5 and size_ratio < 2.2:
        cal.viewpoint = "side"
        conf = min(0.9, 0.5 + 0.12 * min(ratios_uv - 1.5, 3.0))
    elif ratios_uv < 1.0 and size_ratio > 1.35:
        cal.viewpoint = "rear"
        conf = min(0.9, 0.5 + 0.15 * min(size_ratio - 1.35, 3.0))
    elif horiz > 1.8 and size_ratio < 2.2:
        cal.viewpoint = "side"
        conf = 0.5
    elif size_ratio > 1.6 and tall:
        cal.viewpoint = "rear"
        conf = 0.55
    elif tall:
        # Players being big enough means the camera is not high and not far from the court; when
        # unable to distinguish side from rear, treat it as rear (rear is the most common setup, and
        # the prior parameters are tuned for it)
        cal.viewpoint = "rear"
        conf = 0.4
    else:
        cal.viewpoint = "unknown"
        conf = 0.3
    cal.confidence = float(np.clip(conf, 0.0, 1.0))
    cal.note(tr("court.note.viewpoint_result", viewpoint=_viewpoint_label(cal.viewpoint),
                conf=f"{cal.confidence:.2f}"))
    return cal


# ------------------------------------------------------------------ main entry point


def _sample_frames(video_path: str, count: int = 24, max_side: int = 320) -> list[np.ndarray]:
    """Uniformly sample several frames over the whole video (for color/court estimation); results are cached to disk."""
    cached = _load_sampled(video_path, count, max_side)
    if cached is not None:
        return cached
    frames = _read_sampled(video_path, count, max_side)
    _save_sampled(video_path, count, max_side, frames)
    return frames


def _sample_key(video_path: str, count: int, max_side: int) -> str:
    import hashlib
    from pathlib import Path

    p = Path(video_path)
    #: Version number of the sampling logic. Change it whenever the frame-sampling / calibration
    #: algorithm changes, otherwise old caches will be read, showing up as "the code changed but the
    #: results did not change at all", which is very hard to debug.
    version = "2"
    try:
        st = p.stat()
        sig = f"{version}|{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{count}|{max_side}"
    except OSError:
        sig = f"{version}|{video_path}|{count}|{max_side}"
    return hashlib.sha1(sig.encode("utf-8", "replace")).hexdigest()[:16]


def _sample_cache_path(video_path: str, count: int, max_side: int):
    from pathlib import Path

    try:
        from ..config import CACHE_DIR

        root = Path(CACHE_DIR) / "calib"
    except Exception:  # pragma: no cover
        root = Path(__file__).resolve().parents[3] / "data" / "cache" / "calib"
    return root / f"{_sample_key(video_path, count, max_side)}.npz"


def _load_sampled(video_path: str, count: int, max_side: int) -> list[np.ndarray] | None:
    """Read the cache. Sampling requires decoding 16 times sequentially, which takes one or two seconds on long 4K footage, so caching is worthwhile."""
    p = _sample_cache_path(video_path, count, max_side)
    if not p.is_file():
        return None
    try:
        z = np.load(p)
        arr = z["frames"]
        return [arr[i] for i in range(arr.shape[0])]
    except Exception:
        return None


def _save_sampled(video_path: str, count: int, max_side: int,
                  frames: list[np.ndarray]) -> None:
    if not frames:
        return
    p = _sample_cache_path(video_path, count, max_side)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            return
        np.savez_compressed(p, frames=np.stack(frames, axis=0))
    except Exception:
        pass


def _read_sampled(video_path: str, count: int, max_side: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        for i in range(count):
            pos = int(total * (i + 0.5) / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fr = cap.read()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                s = max_side / max(1, max(h, w))
                if s < 1.0:
                    fr = cv2.resize(fr, (max(2, int(w * s)), max(2, int(h * s))),
                                    interpolation=cv2.INTER_AREA)
                frames.append(fr)
    else:
        while len(frames) < count:
            ok, fr = cap.read()
            if not ok:
                break
            if len(frames) % 10 == 0:
                frames.append(fr)
    cap.release()
    return frames


def calibrate(
    video_path: str,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    frame_size: tuple[int, int] | None = None,
    samples: int = 24,
    hint: Viewpoint | None = None,
) -> CourtCalibration:
    """Calibrate the court and camera angle from the video.

    Args:
        video_path: Video path (the proxy is fine; sampled frames get scaled to within 320 pixels).
        player_boxes: ``PlayerSignal.frame_boxes`` (normalized boxes); used to determine the camera angle.
        player_fps: Frame rate of the above sequence.
        frame_size: Original frame (w, h); if not given, inferred from the sampled frames.
        hint: The camera angle specified by the user in the UI; when given it takes precedence (the measured metrics are still computed and recorded).

    Returns:
        :class:`CourtCalibration`; on failure ``ok=False``, and the caller should fall back to generic mode.
    """
    frames = _sample_frames(video_path, count=samples)
    if not frames:
        cal = CourtCalibration()
        cal.note(tr("court.note.read_frames_failed"))
        return cal
    h0, w0 = frames[0].shape[:2]
    if frame_size is None:
        frame_size = (w0, h0)
    fw, fh = int(frame_size[0]), int(frame_size[1])

    base = estimate_court_color(frames)
    if base.ratio < 0.35:
        cal = CourtCalibration(color=base, aspect=float(fw) / max(1, fh))
        cal.note(tr("court.note.ground_unclear", color=base.name,
                    pct=f"{base.ratio:.0%}"))
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    color, mask = refine_court_color(frames, base)
    cal = CourtCalibration(color=color, aspect=float(fw) / max(1, fh))
    if mask is None:
        cal.note(tr("court.note.ground_no_region", color=color.name))
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    # Determine the court's top and bottom boundaries row by row (this step separates ground from wall by saturation, see measure_court_band)
    rows, band_info = measure_court_band(frames)
    if not rows.any():
        cal.note(tr("court.note.band_failed",
                    reason=band_info.get("reason", tr("court.note.band_rows_default"))))
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    # Build masks over multiple frames with "row determination + color" and take the overlap, then take the connected component from the bottom seed
    acc: np.ndarray | None = None
    for fr in frames:
        m = court_mask_from_rows(fr, rows, float(band_info["hue"]), float(band_info["sat_thr"]))
        acc = m.astype(np.float32) if acc is None else acc + m.astype(np.float32)
    assert acc is not None
    acc /= float(len(frames))
    union = (acc > 0.5).astype(np.uint8) * 255
    seed_comp = _seed_component(union, min_area_ratio=0.03)
    comp = seed_comp if seed_comp is not None else union
    cover = float(np.mean(comp > 0))
    cal.note(tr("court.note.region_cover", pct=f"{cover:.0%}"))

    quad_small = find_court_poly(comp, min_area_ratio=0.03)
    if quad_small is None:
        cal.note(tr("court.note.poly_from_mask_failed"))
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est
    sx = float(fw) / comp.shape[1]
    sy = float(fh) / comp.shape[0]
    poly = quad_small.astype(np.float32) * np.array([sx, sy], dtype=np.float32)
    cal = build_calibration(poly, fw, fh, color, source="auto")
    cal.note(tr("court.note.ground_summary", color=color.name,
                hue=band_info.get("hue"), pct=f"{cover:.0%}"))
    return estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)


def refine_viewpoint(cal: CourtCalibration, player_boxes: list | None,
                     player_fps: float, frame_size: tuple[int, int],
                     hint: Viewpoint | None = None) -> CourtCalibration:
    """Determine the camera angle once more after player detection has run (this time real player boxes are available)."""
    if player_boxes is None or player_fps <= 0 or not player_boxes:
        return cal
    drop = tuple(tr(k).split("{", 1)[0] for k in (
        "court.note.no_players", "court.note.too_few_boxes",
        "court.note.spread", "court.note.court_spread", "court.note.viewpoint_result"))
    cal.notes = [n for n in cal.notes if not n.startswith(drop)]
    return estimate_viewpoint(cal, player_boxes, player_fps,
                              int(frame_size[0]), int(frame_size[1]), hint=hint)


def detect_viewpoint_only(player_boxes: list | None, player_fps: float,
                          frame_size: tuple[int, int]) -> Viewpoint:
    """Without court calibration, give a **rough** camera-angle judgment based only on the relative distribution of player boxes.

    Uses only "the horizontal/vertical distribution of players in the frame" and "the box area ratio on screen":

    * The two players spread far apart horizontally and of similar box size -> sideline side;
    * One near and one far with an obvious box size difference -> behind the court;
    * Players squeezed into a small central patch with all boxes tiny -> high camera / overhead.

    This judgment is less reliable than the calibrated version, so it is only used to "pick prior
    parameters"; when it cannot decide, return ``unknown`` (downstream runs with generic parameters).
    """
    if not player_boxes or player_fps <= 0:
        return "unknown"
    xs: list[float] = []
    ys: list[float] = []
    ratios: list[float] = []
    sizes: list[float] = []
    for fr in player_boxes:
        if not fr:
            continue
        for item in fr:
            try:
                _, x1, y1, x2, y2 = item
            except Exception:
                continue
            xs.append(0.5 * (float(x1) + float(x2)))
            ys.append(0.5 * (float(y1) + float(y2)))
            sizes.append(abs(float(y2) - float(y1)))
        if len(fr) >= 2:
            a = sorted((abs(float(b[3]) - float(b[1])) for b in fr if len(b) >= 5), reverse=True)
            if len(a) >= 2 and a[1] > EPS:
                ratios.append(a[0] / a[1])
    if len(xs) < 8:
        return "unknown"
    asp = float(frame_size[0]) / max(1, float(frame_size[1]))
    spread_x = float(np.percentile(xs, 92) - np.percentile(xs, 8)) * asp
    spread_y = float(np.percentile(ys, 92) - np.percentile(ys, 8))
    size_ratio = float(np.median(ratios)) if ratios else 1.0
    med_h = float(np.median(sizes)) if sizes else 0.0
    if med_h < 0.10 and spread_x < 0.35 and spread_y < 0.25:
        return "elevated"
    if spread_x > 1.5 * max(spread_y, 0.05) and size_ratio < 2.0:
        return "side"
    if size_ratio > 1.6:
        return "rear"
    return "unknown"


__all__ = [
    "COURT_LENGTH_M",
    "COURT_WIDTH_M",
    "MAX_POLY_POINTS",
    "MIN_POLY_POINTS",
    "CourtCalibration",
    "CourtColor",
    "build_calibration",
    "calibrate",
    "court_mask",
    "detect_viewpoint_only",
    "estimate_court_color",
    "estimate_viewpoint",
    "find_court_poly",
    "find_court_quad",
    "fit_quad_from_poly",
    "order_poly",
    "order_quad",
    "offset_poly",
    "point_in_poly",
    "poly_area_norm",
    "refine_viewpoint",
]
