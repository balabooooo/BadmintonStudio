"""视觉运动分析：帧间运动能量、全局抖动、清晰度、场景切换、活动热区。

这些信号本身不依赖任何神经网络，速度极快，并且能给回合分割提供
与音频互补的证据（音频怕安静球馆里的回声，视觉怕观众走动）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.models import MediaInfo

Progress = Callable[[float, str], None]


@dataclass
class MotionSignal:
    fps: float
    duration: float
    #: 全局帧间运动能量（0~1）
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 场地区域内运动能量（0~1），无标定时等于全局
    court_motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 亮度均值
    brightness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 清晰度（Laplacian 方差，归一化）
    sharpness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 手持抖动幅度（像素/帧，归一化）
    shake: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 相邻采样帧直方图差异（场景切换检测）
    cut: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 活动热区（小尺寸 2D 数组）
    activity_map: np.ndarray | None = None
    #: 自动估计的场地区域（归一化 x0,y0,x1,y1）
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def resample(self, fps_target: float) -> dict[str, np.ndarray]:
        """重采样到统一时间轴，供信号融合使用。"""
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

    src = str(media.proxy_path or media.path)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")

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
            # 全局位移（抖动 + 摇镜）
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
            on(min(0.99, read / min(total, limit)), "分析画面运动")

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
    sig.court_motion = sig.motion.copy()  # 无球员/场地标定时先等同全局
    return sig


def _norm(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _auto_roi(activity: np.ndarray | None, thr: float = 0.35) -> tuple[float, float, float, float]:
    """从时间累积的活动热区里估计场地大致范围（归一化坐标）。"""
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
    # 适当外扩，避免裁掉球员手脚
    px, py = 0.04, 0.04
    return (
        float(max(0.0, x0 - px)),
        float(max(0.0, y0 - py)),
        float(min(1.0, x1 + px)),
        float(min(1.0, y1 + py)),
    )


def motion_events(sig: MotionSignal, fps: float = 10.0) -> dict[str, np.ndarray]:
    """按目标帧率输出重采样后的运动信号字典。"""
    return sig.resample(fps)
