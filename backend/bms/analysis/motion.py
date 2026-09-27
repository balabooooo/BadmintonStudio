"""Visual motion analysis: inter-frame motion energy, global shake, sharpness, scene cuts, activity hotspots.

These signals do not depend on any neural network, are extremely fast, and can provide
evidence complementary to audio for rally segmentation (audio suffers from echo in a quiet
gym, vision suffers from crowd movement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.media import proxy_source
from ..core.models import MediaInfo
from ..i18n import tr

Progress = Callable[[float, str], None]


@dataclass
class MotionSignal:
    fps: float
    duration: float
    #: Global inter-frame motion energy (0~1)
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Motion energy within the court region (0~1); equals global when there is no calibration
    court_motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Mean brightness
    brightness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Sharpness (Laplacian variance, normalized)
    sharpness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Handheld shake magnitude (pixels/frame, normalized)
    shake: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Histogram difference between adjacent sampled frames (scene cut detection)
    cut: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Activity hotspot (small 2D array)
    activity_map: np.ndarray | None = None
    #: Automatically estimated court region (normalized x0,y0,x1,y1)
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def resample(self, fps_target: float) -> dict[str, np.ndarray]:
        """Resample to a unified time axis for signal fusion."""
        if self.fps <= 0 or self.motion.size == 0:
            return {}
        n = max(1, int(round(self.duration * fps_target)))
        out: dict[str, np.ndarray] = {}
        src_t = np.arange(self.motion.size) / self.fps
        dst_t = np.arange(n) / fps_target
        for name in ("motion", "court_motion", "brightness", "sharpness", "shake", "cut"):
            arr = getattr(self, name)
            if arr is None or arr.size == 0:
                continue
            out[name] = np.interp(dst_t, src_t, arr).astype(np.float32)
        return out


def analyze_motion(
    media: MediaInfo,
    sample_fps: float = 15.0,
    work_width: int = 256,
    max_seconds: float = 0.0,
    on: Progress | None = None,
    cancel=None,
) -> MotionSignal:
    import cv2

    src = str(proxy_source(media))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(tr("analysis.motion.open_failed", src=src))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or media.fps or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(1e-3, sample_fps))))
    eff_fps = src_fps / step
    limit = int(max_seconds * src_fps) if max_seconds > 0 else (total or 10**9)

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    window: np.ndarray | None = None
    motion, court_motion, bright, sharp, shake, cuts = [], [], [], [], [], []
    activity: np.ndarray | None = None

    idx = 0
    read = 0
    while True:
        if cancel and cancel():
            break
        ok = cap.grab()
        if not ok:
            break
        read += 1
        if idx % step != 0 and idx != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue
        if read > limit:
            break

        h, w = frame.shape[:2]
        scale = work_width / max(1, w)
        small = cv2.resize(frame, (work_width, max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grayf = gray.astype(np.float32)

        bright.append(float(grayf.mean()) / 255.0)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharp.append(float(lap.var()))

        if prev_gray is not None and prev_gray.shape == grayf.shape:
            d = np.abs(grayf - prev_gray)
            motion.append(float(d.mean()) / 255.0)
            diff = cv2.GaussianBlur(d, (0, 0), 1.2)
            activity = diff if activity is None else 0.985 * activity + 0.015 * diff
            # Global displacement (shake + panning)
            if window is None:
                wy = np.hanning(grayf.shape[0]).astype(np.float32)
                wx = np.hanning(grayf.shape[1]).astype(np.float32)
                window = np.outer(wy, wx).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray, grayf, window)
            shake.append(float(np.hypot(dx, dy)))
        else:
            motion.append(0.0)
            shake.append(0.0)

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-9)
        if prev_hist is not None:
            cuts.append(float(0.5 * np.abs(hist - prev_hist).sum()))
        else:
            cuts.append(0.0)

        prev_gray, prev_hist = grayf, hist

        if on and total:
            on(min(0.99, read / min(total, limit)), tr("analysis.stage.motion"))

    cap.release()

    sig = MotionSignal(
        fps=eff_fps,
        duration=(len(motion) / eff_fps) if eff_fps > 0 else 0.0,
        motion=np.asarray(motion, dtype=np.float32),
        brightness=np.asarray(bright, dtype=np.float32),
        sharpness=_norm(np.asarray(sharp, dtype=np.float32)),
        shake=_norm(np.asarray(shake, dtype=np.float32)),
        cut=np.asarray(cuts, dtype=np.float32),
        activity_map=activity,
    )
    sig.motion = _norm(sig.motion)
    sig.roi = _auto_roi(activity)
    sig.court_motion = sig.motion.copy()  # without players/court calibration, equate to global for now
    return sig


def _norm(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _auto_roi(activity: np.ndarray | None, thr: float = 0.35) -> tuple[float, float, float, float]:
    """Estimate the rough court extent from the time-accumulated activity hotspot (normalized coords)."""
    if activity is None or activity.size == 0:
        return (0.0, 0.0, 1.0, 1.0)
    a = activity / (activity.max() + 1e-9)
    m = a > thr
    if m.sum() < 0.02 * m.size:
        m = a > np.percentile(a, 85)
    ys, xs = np.where(m)
    if ys.size == 0:
        return (0.0, 0.0, 1.0, 1.0)
    h, w = a.shape
    x0, x1 = xs.min() / w, (xs.max() + 1) / w
    y0, y1 = ys.min() / h, (ys.max() + 1) / h
    # Expand outward a bit to avoid cropping off players' hands and feet
    px, py = 0.04, 0.04
    return (
        float(max(0.0, x0 - px)),
        float(max(0.0, y0 - py)),
        float(min(1.0, x1 + px)),
        float(min(1.0, y1 + py)),
    )


def motion_events(sig: MotionSignal, fps: float = 10.0) -> dict[str, np.ndarray]:
    """Output a dict of motion signals resampled to the target frame rate."""
    return sig.resample(fps)
