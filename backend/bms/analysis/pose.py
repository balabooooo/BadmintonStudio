"""Pose assistance: extract keypoints from player boxes to produce the "swing" signal.

Why it is needed
----------------
In a multi-court gym, audio hits are unreliable (measured on 30 minutes of footage: ``audio_reliability = 0.04``):
hit sounds from the neighboring court fill the "hit sequence" densely, therefore

* :func:`bms.analysis.rally.split_by_hit_gaps` cannot find gaps and fails to split what should be split;
* :func:`bms.analysis.rally.refine_with_hits` cannot tell that "no one is playing after the last shot",
  and dares not tighten the end point.

The measured result is that 0~48.6 seconds still get glued into a single "rally". **Audio alone cannot
provide the information of "whether this shot is ours"**, but pose can: our players only swing when they
actually hit, and the neighboring court's hit sounds have no corresponding motion in our frames.

Two key design choices
----------------------
1. **Do not re-detect, do not re-track.** Directly reuse ``PlayerSignal.frame_boxes``
   (already tracked and already filtered to match players). This saves half the compute and, more
   importantly, avoids the hardest-to-debug bugs like "two tracking results disagree".
2. **Crop the box, upscale it, then feed it to the pose model.** In the measured footage players are
   only about 100 pixels tall, and running keypoints on the full frame is unstable (wrists and elbows
   are lost first); after cropping to a square window and upscaling to 192, the keypoints become stably
   usable, and it also conveniently keeps people from the neighboring court out of the window.
   The window must be shifted **upward** with some margin — the wrist goes outside the box on an overhead hit.

Output
------
The most important part of :class:`PoseSignal` is ``swing``: the displacement speed of the wrist
**relative to the midpoint of the two shoulders**, divided by body height. Subtracting the shoulder
midpoint removes overall displacement (the wrist also moves when the player runs, which is not a swing);
dividing by body height makes it independent of how near or far the player is.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..i18n import tr

EPS = 1e-9

#: Npz cache layout version. v1 stores only the derived swing/ok/overhead curves;
#: v2 additionally stores the full 17-keypoint skeleton of every detected slot;
#: v3 fixes the crop->frame keypoint mapping (v2 skeletons projected far off the player and its
#: swing amplitudes were attenuated).
POSE_CACHE_VERSION = 3
#: Quantization of normalized keypoint coordinates / confidences for compact npz storage
_KP_XY_QUANT = 65535.0
_KP_CONF_QUANT = 255.0
#: Number of COCO keypoints emitted by yolo-pose models
KP_COUNT = 17

#: COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16

#: Keypoint confidence threshold. Joints below it are treated as "not measured" and excluded from computation.
KP_CONF = 0.35


# ------------------------------------------------------------------ Data structures


@dataclass
class PoseSignal:
    """Per-frame pose signal (frame rate matches ``PlayerSignal.frame_boxes``)."""

    fps: float
    duration: float
    #: Per-frame "swing strength": wrist displacement speed relative to the shoulder / body height (unit: body-heights/second)
    swing: np.ndarray
    #: Whether a usable pose exists per frame (0/1)
    ok: np.ndarray
    #: Whether the swinging hand is above the shoulder line per frame (0/1) — used to distinguish overhead shots from underhand shots
    overhead: np.ndarray
    #: Fraction of frames with a usable pose
    coverage: float = 0.0
    #: "Resting level" of swing strength (used to decide whether a peak is significant enough)
    quiet: float = 0.0
    #: Diagnostic info, fed directly into stats for the user
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.swing.size)


# ------------------------------------------------------------------ Cropping


def _map_keypoints(kp_raw: np.ndarray, x0: float, y0: float,
                   sx: float, sy: float) -> np.ndarray:
    """Map 192px patch-space keypoints back to full-frame pixels.

    The patch is ``cv2.resize(crop, (192, 192))`` (a stretch), so patch pixel p is frame
    ``origin + p * crop_size / 192`` — MULTIPLY by the scale. (A previous version divided, which
    projects the skeleton far outside the crop: keypoints must stay within the crop window.)
    Separate x/y scales are required because the crop window may be non-square after frame-edge
    clamping even though the patch is square.
    """
    return np.stack([x0 + kp_raw[:, 0] * sx, y0 + kp_raw[:, 1] * sy], axis=1)


def _crop_spec(box: tuple[float, float, float, float], w: int, h: int,
               margin: float, up_shift: float,
               normalized: bool = True) -> tuple[int, int, int, int]:
    """Compute a square crop window ``(x0, y0, x1, y1)`` (pixels, already clamped to the frame) from the player box.

    ``PlayerSignal.frame_boxes`` uses **normalized** coordinates (0~1), so by default they must be
    restored to pixels using the frame size — treating them directly as pixels yields a 4x4 window
    and the pose model detects nothing.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    # Adaptive fallback: even if the caller passes pixel coordinates (x2/y2 within the frame are necessarily > 1), it will not miscalculate
    if normalized and max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1.5:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    bh = max(4.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - up_shift * bh          # shift the whole window upward
    side = bh * (1.0 + 2.0 * margin)
    a0 = int(round(cx - side / 2.0))
    b0 = int(round(cy - side / 2.0))
    a1 = int(round(cx + side / 2.0))
    b1 = int(round(cy + side / 2.0))
    return (max(0, a0), max(0, b0), min(w, a1), min(h, b1))


def analyze_pose(
    video_path: str,
    frame_boxes: list,
    boxes_fps: float,
    model_name: str = "yolo11n-pose.pt",
    crop_size: int = 192,
    margin: float = 0.35,
    up_shift: float = 0.05,
    confidence: float = 0.25,
    device: str = "cuda",
    max_players_per_frame: int = 4,
    smoothing: float = 0.2,
    cache_dir: str | Path | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PoseSignal | None:
    """Run pose once and return :class:`PoseSignal`; return ``None`` when no keypoints are available.

    Args:
        video_path: video path (the proxy video is fine; the player boxes come from it anyway).
        frame_boxes: ``PlayerSignal.frame_boxes``, per-frame ``(track_id, x1,y1,x2,y2)``
            normalized coordinates.
        boxes_fps: frame rate of ``frame_boxes``.
        crop_size: side length the crop window is upscaled to (square). 192 is the measured
            sweet spot: smaller and keypoints start dropping, larger gives little benefit while
            compute rises linearly.
        margin: outward expansion ratio of the crop window relative to box height.
        up_shift: ratio by which the crop-window center is shifted up relative to the box center
            (the wrist is above the box on an overhead hit).
        smoothing: moving-average time of the swing signal (seconds).
        cache_dir: if given, cache the result as npz (hashed by video/params/weights), so the user
            does not have to re-run pose when repeatedly tuning segmentation parameters.
        on_progress: ``callable(progress 0~1, description)``.
        cancel: ``callable() -> bool``.

    Returns:
        PoseSignal, or ``None`` (no player boxes / no GPU / missing weights / no pose detected in the whole clip).
    """
    n = len(frame_boxes)
    if n == 0 or boxes_fps <= 0:
        return None

    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, video_path, boxes_fps, crop_size,
                                 margin, up_shift, model_name,
                                 boxes_sig=_boxes_signature(frame_boxes))
        cached = _load_cache(cache_path, n)
        if cached is not None:
            cached.trace["cached"] = True
            cached.trace["cache_tag"] = cache_path.stem
            if on_progress:
                on_progress(1.0, tr("pose.cached"))
            return cached

    weights = _resolve_weights(model_name)
    if weights is None:
        return None
    dev = _pick_device(device)

    swing = np.zeros(n, dtype=np.float32)
    overhead = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=np.float32)

    try:
        import cv2
        from ultralytics import YOLO
    except Exception:
        return None

    try:
        model = YOLO(weights)
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or boxes_fps) or boxes_fps
    step = max(1e-6, boxes_fps / max(src_fps, 1e-6))    # source-frame -> sampled-frame step

    # The per-frame cropped patches are sent to the GPU in "batches". A single frame has only
    # 2~4 people, so sending one frame at a time leaves the GPU idle (batching measured over 3x faster).
    batch_patches: list[np.ndarray] = []
    batch_slots: list[tuple[int, int]] = []             # (frame index, which person in that frame)
    kp_store: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    # Full-skeleton records retained for visualization / boundary templates (one per detected
    # slot, independent of whether the swing gates below accept the pose): quantized normalized
    # coordinates, accumulated as Python lists then stacked once at save time.
    kp_rec_frames: list[int] = []
    kp_rec_fis: list[int] = []
    kp_rec_tracks: list[int] = []
    kp_rec_xy: list[np.ndarray] = []                    # uint16 (17, 2), normalized frame coords
    kp_rec_conf: list[np.ndarray] = []                  # uint8 (17,)

    def _flush() -> None:
        if not batch_patches:
            return
        try:
            res = model.predict(batch_patches, imgsz=crop_size, conf=confidence,
                                verbose=False, device=dev)
        except Exception:
            batch_patches.clear()
            batch_slots.clear()
            return
        for slot, r in zip(batch_slots, res):
            if r.keypoints is None or len(r.keypoints) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            if b is None or len(b) == 0:
                continue
            pick = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            kp_store[slot] = (r.keypoints.xy.cpu().numpy()[pick],
                              r.keypoints.conf.cpu().numpy()[pick])
        batch_patches.clear()
        batch_slots.clear()

    read_idx = 0
    kept = 0
    total = int(n)
    misses = 0
    w = h = 0                       # proxy frame size, set on the first decoded frame
    pos = np.full((n, 4), np.nan, dtype=np.float32)     # shoulder-midpoint x,y + both wrists x,y
    body = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if cancel and cancel():
            break
        # Skip to the source frame corresponding to this sampled frame
        target = int(round(i / max(step, 1e-6)))
        while read_idx < target:
            if not cap.grab():
                break
            read_idx += 1
        ok_read, frame = cap.read()
        read_idx += 1
        if not ok_read or frame is None:
            # An occasional decode failure should not invalidate the whole pose pass, but
            # consecutive failures mean we have reached the end of the clip.
            # Note: do **not** clear frame_boxes[i] — that is the caller's list
            # (PlayerSignal.frame_boxes), and modifying it would pollute the caller's later use.
            misses += 1
            if misses > 30:
                break
            continue
        misses = 0
        h, w = frame.shape[:2]
        boxes = list(frame_boxes[i] or [])[:max_players_per_frame]
        if not boxes:
            continue
        fi = 0
        for item in boxes:
            if len(item) < 5:
                continue
            spec = _crop_spec(item[1:5], w, h, margin, up_shift)
            x0, y0, x1, y1 = spec
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            patch = frame[y0:y1, x0:x1]
            interp = cv2.INTER_CUBIC if (x1 - x0) < crop_size else cv2.INTER_AREA
            patch = cv2.resize(patch, (crop_size, crop_size), interpolation=interp)
            batch_patches.append(patch)
            batch_slots.append((i, fi))
            # Record the crop-window geometry alongside the slot: keypoints must be mapped back to
            # frame coordinates (per-axis scale — the clamped crop window need not be square).
            kp_store.setdefault(("spec", i, fi), (np.asarray([x0, y0], dtype=np.float32),
                                                  np.asarray([(x1 - x0) / crop_size,
                                                              (y1 - y0) / crop_size],
                                                             dtype=np.float32)))
            # Keep the player track id of this slot so cached skeletons stay associated with boxes
            kp_store[("tid", i, fi)] = int(item[0])
            fi += 1
            if len(batch_patches) >= 32:
                _flush()
        if (i % 60) == 0 and on_progress:
            on_progress(min(0.99, i / max(total, 1)), tr("pose.analyzing"))
    _flush()
    cap.release()

    # ---- Keypoints -> swing signal ----
    # For each frame, use the "fastest wrist among all people in that frame" as that frame's swing strength.
    for i in range(n):
        for fi in range(max_players_per_frame):
            key = (i, fi)
            if key not in kp_store:
                continue
            spec = kp_store.get(("spec", i, fi))
            if spec is None:
                continue
            kp_raw, kc = kp_store[key]
            x0, y0 = float(spec[0][0]), float(spec[0][1])
            sx, sy = float(spec[1][0]), float(spec[1][1])
            kp = _map_keypoints(kp_raw, x0, y0, sx, sy)
            # Retain the full 17-keypoint skeleton (normalized frame coords) for every detected
            # slot — including poses the swing computation below rejects (e.g. no wrist visible).
            # Visualization must show what the model saw, not only what fed the gate.
            if kp_raw.shape[0] >= KP_COUNT:
                nxy = np.stack([kp[:, 0] / max(1, w), kp[:, 1] / max(1, h)], axis=1)
                kp_rec_frames.append(i)
                kp_rec_fis.append(fi)
                kp_rec_tracks.append(int(kp_store.get(("tid", i, fi), -1)))
                kp_rec_xy.append(np.round(np.clip(nxy[:KP_COUNT], 0.0, 1.0)
                                          * _KP_XY_QUANT).astype(np.uint16))
                kp_rec_conf.append(np.round(np.clip(kc[:KP_COUNT], 0.0, 1.0)
                                            * _KP_CONF_QUANT).astype(np.uint8))
            sh = [kp[j] for j in (L_SHOULDER, R_SHOULDER) if kc[j] > KP_CONF]
            if len(sh) < 1:
                continue
            mid = np.mean(np.asarray(sh, dtype=np.float64), axis=0)
            # Body height: shoulder to ankle. Without ankles, fall back to a rough estimate of "shoulder to bottom of frame"
            ys = [kp[j, 1] for j in (L_ANKLE, R_ANKLE) if kc[j] > KP_CONF]
            if ys:
                bh = float(max(ys)) - float(mid[1])
            else:
                bh = float(mid[1]) * 0.9
            if bh < 0.02 * h:
                continue
            # Take the wrist with the higher confidence
            cand = [(kc[j], j) for j in (L_WRIST, R_WRIST) if kc[j] > KP_CONF]
            if not cand:
                continue
            cand.sort(reverse=True)
            kc_w, j_w = cand[0]
            if not np.isfinite(pos[i, 0]):
                pos[i, 0], pos[i, 1] = mid
                body[i] = bh
            else:
                body[i] = max(body[i], bh) if np.isfinite(body[i]) else bh
            px, py = float(kp[j_w, 0]), float(kp[j_w, 1])
            if not np.isfinite(pos[i, 2]):
                pos[i, 2], pos[i, 3] = px, py
            else:
                # Multiple people in the same frame: keep "the wrist farther from the shoulder" = the one swinging wider
                d_old = np.hypot(pos[i, 2] - pos[i, 0], pos[i, 3] - pos[i, 1])
                d_new = np.hypot(px - pos[i, 0], py - pos[i, 1])
                if d_new > d_old:
                    pos[i, 2], pos[i, 3] = px, py
            overhead[i] = 1.0 if py < pos[i, 1] else 0.0
            ok[i] = 1.0
            kept += 1

    coverage = float(np.mean(ok > 0)) if n else 0.0
    if coverage < 0.05:
        return None

    # Wrist displacement speed relative to the shoulder (normalized by frame height), then divided by body height
    rel = np.stack([pos[:, 2] - pos[:, 0], pos[:, 3] - pos[:, 1]], axis=1)
    good = np.isfinite(rel).all(axis=1)
    if good.sum() < 4:
        return None
    idx = np.arange(n)
    for j in range(2):
        rel[:, j] = np.interp(idx, idx[good], rel[good, j])
    d = np.hypot(np.diff(rel[:, 0]), np.diff(rel[:, 1]))
    d = np.concatenate([[0.0], d]) * boxes_fps
    # Per-frame displacement divided by "that frame's body height" — a nearer person has a taller body and larger displacement, so only the ratio is comparable
    bh = np.where(np.isfinite(body) & (body > 1.0), body, np.nan)
    if np.isfinite(bh).sum() > 2:
        bh = np.interp(idx, idx[np.isfinite(bh)], bh[np.isfinite(bh)])
    else:
        bh = np.full(n, max(1.0, 0.2 * h), dtype=np.float64)
    swing = (d / np.maximum(bh, 1e-6)).astype(np.float32)
    # An excessively large single-frame jump is basically keypoint jitter, so drop it
    swing = np.clip(swing, 0.0, 6.0)
    win = max(1, int(round(smoothing * boxes_fps)))
    if win > 1:
        k = np.ones(win, dtype=np.float32) / win
        swing = np.convolve(swing, k, mode="same").astype(np.float32)
    swing[ok <= 0] = 0.0
    quiet = float(np.percentile(swing[swing > 0], 25)) if np.any(swing > 0) else 0.0

    sig = PoseSignal(
        fps=float(boxes_fps), duration=float(n / max(boxes_fps, EPS)),
        swing=swing, ok=ok, overhead=overhead.astype(np.float32),
        coverage=coverage, quiet=quiet,
        trace={"coverage": round(coverage, 3), "frames_with_pose": int(kept),
               "players": int(kept / max(1, int(np.sum(ok > 0)))) if np.any(ok > 0) else 0,
               "swing_p50": round(float(np.percentile(swing, 50)), 3),
               "swing_p90": round(float(np.percentile(swing, 90)), 3),
               "crop_size": crop_size, "model": Path(weights).name},
    )
    if cache_path is not None:
        keypoints = None
        if kp_rec_frames:
            keypoints = {
                "frame": np.asarray(kp_rec_frames, dtype=np.int32),
                "fi": np.asarray(kp_rec_fis, dtype=np.uint8),
                "track": np.asarray(kp_rec_tracks, dtype=np.int32),
                "xy": np.stack(kp_rec_xy, axis=0),
                "conf": np.stack(kp_rec_conf, axis=0),
            }
        _save_cache(cache_path, sig, keypoints)
        sig.trace["cache_tag"] = cache_path.stem
    if on_progress:
        on_progress(1.0, tr("pose.analyzing"))
    return sig


# ------------------------------------------------------------------ Hit attribution


def swing_peaks(
    pose: PoseSignal,
    min_distance: float = 0.26,
    prominence_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Find "swing moments" on the swing signal; return ``(frame index of the peak, prominence 0~1)``.

    Why find peaks instead of directly taking "the maximum within a window": during a rally there is
    a shot every 1.0~1.5 seconds, while the hit time itself has ±0.05 second precision. If you take
    the maximum within a ±0.35 second window, that window already covers half a shot interval —
    **some wrist motion can be found near almost every hit**, and the evidence score loses its
    discriminative power (measured p50 as high as 0.72, the gating might as well not exist).

    Peaks, by contrast, turn "swings" into sparse events: only a few dozen in 90 seconds, while there
    are over a hundred hits, so "one peak can explain only one hit" becomes the strongest constraint.
    """
    if pose is None or pose.n < 8:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    from scipy.signal import find_peaks

    sw = pose.swing
    hi = float(np.percentile(sw, 95))
    span = max(hi - pose.quiet, 1e-6)
    dist = max(1, int(round(min_distance * pose.fps)))
    idx, props = find_peaks(sw, distance=dist, prominence=prominence_ratio * span)
    if idx.size == 0:
        return idx.astype(np.int64), np.zeros(0, dtype=np.float32)
    prom = props.get("prominences")
    if prom is None:
        prom = np.full(idx.size, span, dtype=np.float64)
    return idx.astype(np.int64), np.clip(prom / span, 0.0, 1.0).astype(np.float32)


def hit_swing_evidence(
    pose: PoseSignal,
    times: np.ndarray,
    strengths: np.ndarray | None = None,
    window: float = 0.30,
    one_to_one: bool = True,
) -> np.ndarray:
    """Swing evidence (0~1) matched by each audio hit within ±``window`` seconds.

    With ``one_to_one=True``, **one swing peak explains only one hit**: if there are multiple hit
    candidates near the same peak (the same sound detected repeatedly, or a neighboring-court sound
    happening to coincide), only the one closest to the peak is kept (on a tie, the stronger one),
    and the rest are set to 0. This constraint is the main reason this gating works.

    The evidence score is further multiplied by a "how close to the peak" weight: the shot right at
    the peak gets full marks, and one 0.3 seconds off gets only half.
    """
    if pose is None or times is None or len(times) == 0:
        return np.zeros(0, dtype=np.float32)
    ev = np.zeros(len(times), dtype=np.float32)
    idx, prom = swing_peaks(pose)
    if idx.size == 0:
        return ev
    peak_t = idx.astype(np.float64) / max(pose.fps, EPS)
    order = np.argsort(peak_t)
    peak_t, prom = peak_t[order], prom[order]

    # For each hit, first find the nearest peak
    pos = np.searchsorted(peak_t, np.asarray(times, dtype=np.float64))
    best_j = np.full(len(times), -1, dtype=np.int64)
    best_dt = np.full(len(times), np.inf, dtype=np.float64)
    for k in range(len(times)):
        for j in (pos[k] - 1, pos[k]):
            if 0 <= j < peak_t.size:
                dt = abs(peak_t[j] - float(times[k]))
                if dt < best_dt[k]:
                    best_dt[k], best_j[k] = dt, j
    ok = (best_j >= 0) & (best_dt <= window)
    if one_to_one and np.any(ok):
        # For multiple candidates on the same peak, keep only the nearest one (on a tie, keep the stronger)
        strengths = (np.asarray(strengths, dtype=np.float64)
                     if strengths is not None and len(strengths) == len(times)
                     else np.zeros(len(times)))
        for j in np.unique(best_j[ok]):
            group = np.nonzero(ok & (best_j == j))[0]
            if group.size <= 1:
                continue
            key = best_dt[group] - 1e-3 * strengths[group]     # proximity first, then strength
            keep = int(group[int(np.argmin(key))])
            for k in group:
                if k != keep:
                    ok[k] = False
    ev[ok] = prom[best_j[ok]] * (1.0 - 0.5 * (best_dt[ok] / max(window, EPS)))
    return np.clip(ev, 0.0, 1.0).astype(np.float32)


def local_strength_pct(times: np.ndarray, strength: np.ndarray,
                       half_win: float = 15.0) -> np.ndarray:
    """Local percentile of each hit's strength among hits within ±``half_win`` seconds.

    AGC-robust "our court is nearer the mic" cue. On the measured multi-court footage this is the
    most discriminative single feature (AUC 0.79 on clip2 vs pose evidence 0.54), because our hits
    are consistently stronger than the neighboring court's.

    One degenerate case is neutralized: when every hit in the neighborhood has (nearly) the same
    strength — synthetic or heavily normalized audio — the cue carries no attribution information,
    so it yields 0 and the composite gate degrades to pure pose evidence instead of letting the
    strength branch rescue pose-less hits. An isolated hit (no local competition) keeps the legacy
    neutral-high 1.0 so recall is preserved.
    """
    n = times.size
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    order = np.argsort(times)
    ts, ss = times[order], strength[order]
    for i in range(n):
        lo = np.searchsorted(ts, ts[i] - half_win)
        hi = np.searchsorted(ts, ts[i] + half_win, side="right")
        nb = ss[lo:hi]
        if nb.size <= 1:
            out[order[i]] = 1.0 if nb.size else 0.5
            continue
        if float(nb.max() - nb.min()) < 1e-6:
            continue
        out[order[i]] = float(np.mean(nb <= ss[i]))
    return out


def composite_gate(
    hits,
    pose: PoseSignal | None,
    threshold: float = 0.45,
    w_pose: float = 1.0,
    w_str: float = 0.6,
    min_keep_ratio: float = 0.12,
    max_keep_ratio: float = 0.97,
    window: float = 0.35,
    one_to_one: bool = True,
    force: bool = False,
):
    """Composite hit attribution gate: pose swing evidence OR local strength percentile.

    The two cues are complementary:
    * pose evidence says "someone in our frame really swung at this moment";
    * local strength percentile says "this sound is among the loudest in its neighborhood",
      which on a fixed camera/mic setup means "our court, not the neighboring one".

    Taking the max of the two (rather than the product) keeps recall high: a real hit that the pose
    gate missed (occlusion, player too small) can still be rescued by its strength, and a real hit
    that is soft (a drop shot) can still be rescued by a visible swing.
    """
    trace: dict[str, Any] = {}
    if hits is None or hits.times.size == 0:
        return None, trace
    ev = np.zeros(hits.times.size, dtype=np.float32)
    if pose is not None and pose.coverage >= 0.35:
        ev = hit_swing_evidence(pose, hits.times, strengths=hits.strength,
                                window=window, one_to_one=one_to_one)
    sp = local_strength_pct(hits.times, hits.strength)
    score = np.maximum(w_pose * ev, w_str * sp)
    trace["evidence_p50"] = round(float(np.percentile(ev, 50)), 3)
    trace["strength_p50"] = round(float(np.percentile(sp, 50)), 3)
    if pose is not None:
        trace["peaks"] = int(swing_peaks(pose)[0].size)
    mask = score >= threshold
    ratio = float(np.mean(mask)) if mask.size else 0.0
    trace["keep_ratio"] = round(ratio, 3)
    if ratio < min_keep_ratio or ratio > max_keep_ratio:
        if not force:
            trace["gate_skipped"] = tr(
                "pose.gate_bad_ratio",
                ratio=ratio, min_ratio=min_keep_ratio, max_ratio=max_keep_ratio)
            return None, trace
        trace["forced"] = True
    trace["kept"] = int(np.count_nonzero(mask))
    trace["dropped"] = int(mask.size - np.count_nonzero(mask))
    return mask, trace


def gate_hits(
    hits,
    pose: PoseSignal | None,
    threshold: float = 0.22,
    min_keep_ratio: float = 0.12,
    max_keep_ratio: float = 0.97,
    window: float = 0.35,
    one_to_one: bool = True,
    force: bool = False,
):
    """Filter audio hits into "those hit in our match" based on pose evidence.

    Returns ``(mask, trace)``: ``mask`` is a boolean array (True = keep).

    **It only acts when there is enough evidence**; if any condition is not met it passes everything
    through unchanged — the pose path itself can also fail (player too small, severe occlusion,
    non-match footage), and in that case "use all hits as before" is always better than
    "chop rallies apart with half the hits":

    * pose coverage too low (< 0.35) -> pass through;
    * keep ratio outside [12%, 97%] -> means the threshold did nothing (kept all) or rejected most
      hits (most likely the pose signal itself is problematic) -> pass through.

    ``force=True`` overrides both guards (the user explicitly insists on filtering). The trace then
    carries ``forced=True`` so the UI can show that the safety net was bypassed.
    """
    trace: dict[str, Any] = {}
    if hits is None or hits.times.size == 0 or pose is None:
        return None, trace
    forced_guard = False
    if pose.coverage < 0.35:
        if not force:
            trace["gate_skipped"] = tr("pose.gate_low_coverage", coverage=pose.coverage)
            return None, trace
        forced_guard = True
    ev = hit_swing_evidence(pose, hits.times, strengths=hits.strength,
                            window=window, one_to_one=one_to_one)
    mask = ev >= threshold
    ratio = float(np.mean(mask)) if mask.size else 0.0
    trace["evidence_p50"] = round(float(np.percentile(ev, 50)), 3)
    trace["peaks"] = int(swing_peaks(pose)[0].size)
    trace["keep_ratio"] = round(ratio, 3)
    if ratio < min_keep_ratio or ratio > max_keep_ratio:
        if not force:
            trace["gate_skipped"] = tr(
                "pose.gate_bad_ratio",
                ratio=ratio, min_ratio=min_keep_ratio, max_ratio=max_keep_ratio)
            return None, trace
        forced_guard = True
    if forced_guard:
        trace["forced"] = True
    trace["kept"] = int(np.count_nonzero(mask))
    trace["dropped"] = int(mask.size - np.count_nonzero(mask))
    return mask, trace


def filter_hits(hits, mask: np.ndarray):
    """Filter :class:`~bms.analysis.audio_hits.HitDetection` by ``mask``."""
    if mask is None or hits is None or hits.times.size == 0:
        return hits
    from .audio_hits import HitDetection

    return HitDetection(
        times=hits.times[mask],
        strength=hits.strength[mask],
        confidence=hits.confidence[mask],
        envelope=hits.envelope,
        env_fps=hits.env_fps,
        threshold=hits.threshold,
        noise_floor_db=hits.noise_floor_db,
    )


# ------------------------------------------------------------------ Cache


#: Must match players._BOX_QUANT: the box cache stores coords as uint16 on this grid, and the
#: signature has to be identical for in-memory float boxes and boxes loaded back from that cache.
_BOXES_SIG_QUANT = 65535.0


def _quantize_box_coord(x: float) -> int:
    """Clip + round one normalized box coordinate onto the uint16 cache grid."""
    q = int(round(float(x) * _BOXES_SIG_QUANT))
    if q < 0:
        return 0
    if q > int(_BOXES_SIG_QUANT):
        return int(_BOXES_SIG_QUANT)
    return q


def _boxes_signature(frame_boxes: list) -> str:
    """Compute a cheap fingerprint of the per-frame player boxes to use as a cache key.

    It must be included: the pose result is cropped from the **player boxes**, so changing the size
    filter / camera setup changes the boxes while the video stays the same. Hashing only by video
    would hit a stale cache, causing users to see the hardest-to-debug problem of "changed parameters
    but the result did not change at all". Sampling is enough — take one frame every 37 frames.

    Coordinates are hashed on the same uint16 grid as the compact box npz so that boxes reloaded
    from the cache (a ~1.5e-5 dequantization away from the floats) produce the same key; otherwise
    every cache-rebuild run would miss and rerun GPU pose estimation.
    """
    h = hashlib.sha1()
    n = len(frame_boxes)
    h.update(str(n).encode())
    for i in range(0, n, 37):
        for item in (frame_boxes[i] or ()):
            try:
                h.update(("%d:%d,%d,%d,%d;" % (
                    int(item[0]),
                    _quantize_box_coord(item[1]), _quantize_box_coord(item[2]),
                    _quantize_box_coord(item[3]), _quantize_box_coord(item[4]))).encode())
            except (TypeError, ValueError, IndexError):
                continue
    return h.hexdigest()[:10]


def _cache_path(cache_dir, video_path: str, fps: float, crop: int,
                margin: float, up_shift: float, model_name: str,
                boxes_sig: str = "") -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        st = Path(video_path).stat()
        stamp = f"{video_path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        stamp = str(video_path)
    key = (f"{stamp}|{fps:.3f}|{crop}|{margin:.3f}|{up_shift:.3f}|"
           f"{model_name}|{boxes_sig}")
    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return d / f"pose_{tag}.npz"


def pose_cache_path(cache_dir, video_path: str, boxes_fps: float, frame_boxes: list,
                    crop_size: int = 192, margin: float = 0.35, up_shift: float = 0.05,
                    model_name: str = "yolo11n-pose.pt") -> Path:
    """Public resolver for the pose npz of a given (video, boxes) pair."""
    return _cache_path(cache_dir, video_path, boxes_fps, crop_size,
                       margin, up_shift, model_name,
                       boxes_sig=_boxes_signature(frame_boxes))


def _load_cache(path: Path, n: int) -> PoseSignal | None:
    """Load the derived swing curves from a v1/v2 npz; keypoints (v2) are left on disk.

    Old caches without a ``version`` key keep working for segmentation: forcing a GPU rerun on
    every upgraded install just to refresh visualization would be a bad trade. Skeleton access
    goes through :func:`load_keypoints`, which treats a v1 file as a miss.
    """
    if path is None or not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if int(z["swing"].size) != n:
            return None
        return PoseSignal(
            fps=float(z["fps"]), duration=float(z["duration"]),
            swing=z["swing"].astype(np.float32),
            ok=z["ok"].astype(np.float32),
            overhead=z["overhead"].astype(np.float32),
            coverage=float(z["coverage"]), quiet=float(z["quiet"]),
            trace={"coverage": round(float(z["coverage"]), 3)},
        )
    except Exception:
        return None


def _save_cache(path: Path, sig: PoseSignal, keypoints: dict | None = None) -> None:
    """Write a v2 npz: derived curves plus, when available, every detected slot's skeleton.

    Keypoint payload (``keypoints``): ``frame`` int32 (K,), ``fi`` uint8 (K,), ``track`` int32
    (K,), ``xy`` uint16 (K,17,2) normalized, ``conf`` uint8 (K,17). Never raises.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")     # .npz suffix keeps savez from appending another one
        payload: dict[str, Any] = {
            "version": np.int64(POSE_CACHE_VERSION),
            "fps": sig.fps, "duration": sig.duration,
            "swing": sig.swing, "ok": sig.ok, "overhead": sig.overhead,
            "coverage": sig.coverage, "quiet": sig.quiet,
        }
        if keypoints is not None:
            payload["kp_frame"] = keypoints["frame"]
            payload["kp_fi"] = keypoints["fi"]
            payload["kp_track"] = keypoints["track"]
            payload["kp_xy"] = keypoints["xy"]
            payload["kp_conf"] = keypoints["conf"]
        else:
            # Still a v2 file, just with zero skeleton records
            payload["kp_frame"] = np.zeros(0, dtype=np.int32)
            payload["kp_fi"] = np.zeros(0, dtype=np.uint8)
            payload["kp_track"] = np.zeros(0, dtype=np.int32)
            payload["kp_xy"] = np.zeros((0, KP_COUNT, 2), dtype=np.uint16)
            payload["kp_conf"] = np.zeros((0, KP_COUNT), dtype=np.uint8)
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    except Exception:
        pass


def load_keypoints(path: str | Path | None) -> dict | None:
    """Load per-slot skeletons from a v2 pose npz.

    Returns ``{"n", "fps", "duration", "frame", "fi", "track", "xy", "conf"}`` where ``xy`` is a
    float32 (K,17,2) array in normalized frame coordinates and ``conf`` float32 (K,17); returns
    ``None`` for a missing/corrupt/v1 file so callers degrade to "skeleton unavailable".
    """
    if path is None:
        return None
    try:
        p = Path(str(path))
        if not p.is_file():
            return None
        z = np.load(p, allow_pickle=False)
        if "version" not in z.files or int(z["version"]) < POSE_CACHE_VERSION:
            return None
        needed = ("kp_frame", "kp_fi", "kp_track", "kp_xy", "kp_conf")
        if any(k not in z.files for k in needed):
            return None
        xy = z["kp_xy"]
        conf = z["kp_conf"]
        if xy.ndim != 3 or xy.shape[1] != KP_COUNT or xy.shape[2] != 2:
            return None
        return {
            "n": int(z["swing"].size),
            "fps": float(z["fps"]),
            "duration": float(z["duration"]),
            "frame": z["kp_frame"].astype(np.int32),
            "fi": z["kp_fi"].astype(np.int16),
            "track": z["kp_track"].astype(np.int32),
            "xy": (xy.astype(np.float32) / _KP_XY_QUANT),
            "conf": (conf.astype(np.float32) / _KP_CONF_QUANT),
        }
    except Exception:
        return None


# ------------------------------------------------------------------ Environment


def _data_paths() -> tuple[Path, Path]:
    try:
        from ..config import MODELS_DIR  # type: ignore

        return MODELS_DIR.parent / "data" / "yolo", MODELS_DIR
    except Exception:
        root = Path(__file__).resolve().parents[3]
        return root / "data" / "yolo", root / "models"


def default_cache_dir() -> Path | None:
    """Default pose cache directory (``data/cache/pose``)."""
    try:
        from ..config import CACHE_DIR  # type: ignore

        return CACHE_DIR / "pose"
    except Exception:
        return None


def _resolve_weights(model_name: str) -> str | None:
    """Prefer local weights under the repo's ``models/``; if absent, let ultralytics download them itself."""
    _, models_dir = _data_paths()
    p = Path(model_name)
    if p.is_absolute() and p.exists():
        return str(p)
    local = models_dir / p.name
    if local.exists():
        return str(local)
    return model_name


def _pick_device(device: str) -> str:
    if device not in ("cuda", "cpu"):
        return device
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device


__all__ = [
    "PoseSignal",
    "analyze_pose",
    "filter_hits",
    "gate_hits",
    "hit_swing_evidence",
    "swing_peaks",
    "pose_cache_path",
    "load_keypoints",
]
