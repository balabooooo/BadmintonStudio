"""音频击球检测。

羽毛球回合的本质是「一串球拍击球声」。球拍击球在中高频段（约 2–8 kHz）
表现为极短（< 15 ms）的宽带瞬态，与场馆里的低频人声、环境噪声区分明显。
本模块用「高频带谱通量 + 自适应阈值 + 峰值拾取」检测每一次触球，
再按静音间隔聚类成回合。

只依赖 numpy / scipy / 标准库 wave，因此不需要 librosa / numba。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d, median_filter

# ------------------------------------------------------------------ 常量

HOP = 64               # STFT 跳步（样点）@16k -> 4 ms
NFFT = 512
HIGH_BAND = (1800.0, 7800.0)   # 球拍击球的主能量带
LOW_BAND = (80.0, 900.0)       # 人声 / 脚步 / 场馆噪声
MIN_HIT_GAP = 0.055            # 两次独立击球最短间隔（秒）


@dataclass
class HitDetection:
    times: np.ndarray                     # 击球时刻（秒）
    strength: np.ndarray                  # 归一化强度 0~1
    confidence: np.ndarray                # 击球置信度 0~1
    envelope: np.ndarray                  # 击球响应包络（用于前端画波形）
    env_fps: float                        # 包络每秒采样数
    threshold: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise_floor_db: float = -60.0


# ------------------------------------------------------------------ 读音频


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """读取 PCM WAV 为 float32 单声道（用标准库，无需额外依赖）。"""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"不支持的位深: {sw * 8} bit")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


# ------------------------------------------------------------------ 检测


def _bandpass_env(x: np.ndarray, sr: int, lo: float, hi: float, smooth_ms: float = 8.0) -> np.ndarray:
    nyq = sr / 2.0
    hi = min(hi, nyq * 0.98)
    if lo >= hi:
        return np.abs(x)
    sos = sps.butter(3, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    y = sps.sosfiltfilt(sos, x)
    env = np.abs(y)
    n = max(1, int(sr * smooth_ms / 1000.0))
    kernel = np.hanning(n)
    kernel /= kernel.sum()
    return np.convolve(env, kernel, mode="same")


def _spectral_flux(x: np.ndarray, sr: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray, int]:
    f, t, Z = sps.stft(x, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann", padded=False)
    mag = np.abs(Z)
    band = (f >= lo) & (f <= hi)
    # 对频带内做能量加权，抑制窄带纯音
    m = mag[band]
    flat = np.exp(np.mean(np.log(m + 1e-10), axis=0)) / (np.mean(m, axis=0) + 1e-10)
    d = np.diff(m, axis=1, prepend=m[:, :1])
    flux = np.maximum(d, 0).sum(axis=0)
    flux = flux * (0.35 + 0.65 * np.clip(flat * 4.0, 0, 1))  # 越接近噪声型越像击球
    hop_s = HOP / sr
    return flux.astype(np.float32), flat.astype(np.float32), max(1, int(round(1.0 / hop_s)))


def _adaptive_threshold(env: np.ndarray, fps: float, sensitivity: float) -> np.ndarray:
    """基于长窗中位数 + 局部 MAD 的鲁棒自适应阈值。

    灵敏度映射到 k：
      0.0 -> k≈6.2（只留最响亮的击球）
      0.5 -> k≈4.4（默认）
      1.0 -> k≈2.6（连轻挑也抓，代价是噪声变多）
    """
    win = max(5, int(round(fps * 2.5)) | 1)
    med = median_filter(env, size=win, mode="nearest")
    mad = median_filter(np.abs(env - med), size=win, mode="nearest")
    k = 6.2 - 3.6 * float(np.clip(sensitivity, 0.0, 1.0))
    return med + k * (mad * 1.4826 + 1e-6)


def detect_hits(
    wav_path: str | Path,
    sensitivity: float = 0.5,
    max_hits: int = 60000,
) -> HitDetection:
    x, sr = load_wav_mono(wav_path)
    if x.size < sr // 4:
        return HitDetection(
            times=np.zeros(0), strength=np.zeros(0), confidence=np.zeros(0),
            envelope=np.zeros(0), env_fps=1.0,
        )

    # 去直流 + 轻降噪
    x = x - float(np.mean(x))
    sos_hp = sps.butter(2, 60.0 / (sr / 2), btype="highpass", output="sos")
    x = sps.sosfiltfilt(sos_hp, x)

    flux, flatness, env_fps = _spectral_flux(x, sr, *HIGH_BAND)
    env_smooth = maximum_filter1d(flux, size=3)
    thr = _adaptive_threshold(env_smooth, env_fps, sensitivity)

    # 低带能量：用于排除「人声喊叫 / 脚步 / 拖地」这类低频为主的伪触发
    low_env = _bandpass_env(x, sr, *LOW_BAND, smooth_ms=60.0)
    low_ds = _resample_to(low_env, sr, env_fps)

    distance = max(1, int(round(MIN_HIT_GAP * env_fps)))
    peaks, props = sps.find_peaks(
        env_smooth, height=thr, distance=distance, prominence=thr * 0.35
    )
    if peaks.size == 0:
        return HitDetection(
            times=np.zeros(0), strength=np.zeros(0), confidence=np.zeros(0),
            envelope=flux, env_fps=env_fps, threshold=thr,
        )

    heights = props["peak_heights"]
    # ---- 置信度：高频瞬态强度 vs 低频背景，以及尖锐度
    lo_ref = np.percentile(low_ds, 60) + 1e-9
    low_at = low_ds[np.clip(peaks, 0, low_ds.size - 1)]
    sharp = np.zeros(peaks.size, dtype=np.float32)
    w = max(1, int(round(0.02 * env_fps)))
    for i, p in enumerate(peaks):
        a, b = max(0, p - w), min(env_smooth.size, p + w + 1)
        seg = env_smooth[a:b]
        base = float(np.percentile(seg, 20)) + 1e-9
        sharp[i] = float(env_smooth[p]) / base
    sharp_n = np.clip((sharp - 1.4) / 3.6, 0.0, 1.0)
    snr = np.clip(heights / (thr[peaks] + 1e-9), 0.0, 4.0) / 4.0
    low_pen = np.clip(low_at / (lo_ref * 6.0), 0.0, 1.0)
    conf = np.clip(0.45 * snr + 0.40 * sharp_n + 0.15 * (1.0 - low_pen), 0.0, 1.0).astype(np.float32)

    # 强度归一（分位数，抗离群）
    h = heights.astype(np.float32)
    p5, p95 = np.percentile(h, 5), np.percentile(h, 97)
    strength = np.clip((h - p5) / max(p95 - p5, 1e-9), 0.0, 1.0).astype(np.float32)

    # ---- 亚帧精修：向原始高频包络的上升沿对齐
    hi_env = _bandpass_env(x, sr, *HIGH_BAND, smooth_ms=3.0)
    times = peaks.astype(np.float64) / env_fps
    times = _refine_onsets(hi_env, sr, times)

    order = np.argsort(times)
    times, strength, conf = times[order], strength[order], conf[order]
    if times.size > max_hits:
        sel = np.linspace(0, times.size - 1, max_hits).astype(int)
        times, strength, conf = times[sel], strength[sel], conf[sel]

    return HitDetection(
        times=times, strength=strength, confidence=conf,
        envelope=flux.astype(np.float32), env_fps=env_fps, threshold=thr.astype(np.float32),
        noise_floor_db=float(20 * np.log10(np.median(np.abs(x)) + 1e-12)),
    )


def _resample_to(env: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    if abs(src_rate - dst_rate) < 1e-6 or env.size == 0:
        return env
    n = max(1, int(round(env.size / src_rate * dst_rate)))
    idx = np.linspace(0, env.size - 1, n)
    return np.interp(idx, np.arange(env.size), env).astype(np.float32)


def _refine_onsets(env: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """把峰值时刻对齐到该瞬态的起始上升点。"""
    if times.size == 0:
        return times
    out = times.copy()
    search = int(0.02 * sr)
    for i, t in enumerate(times):
        c = int(t * sr)
        a = max(0, c - search)
        b = min(env.size, c + search // 2)
        if b - a < 4:
            continue
        seg = env[a:b]
        peak = int(np.argmax(seg)) + a
        # 从峰值向前找局部极小 -> 视作起始
        j = peak
        floor = env[peak] * 0.18
        while j > a and env[j] > floor:
            j -= 1
        out[i] = j / sr
    return np.maximum.accumulate(out)  # 保证单调


# ------------------------------------------------------------------ 回合聚类


@dataclass
class HitCluster:
    start: float
    end: float
    hits: list[int]  # 索引

    @property
    def duration(self) -> float:
        return self.end - self.start


def cluster_rallies(
    det: HitDetection,
    gap_seconds: float = 3.2,
    min_seconds: float = 2.0,
    min_hits: int = 2,
    max_seconds: float = 120.0,
) -> list[HitCluster]:
    """把击球时刻按静音间隔聚类为回合候选。"""
    t = det.times
    if t.size == 0:
        return []
    clusters: list[HitCluster] = []
    cur = [0]
    for i in range(1, t.size):
        if t[i] - t[i - 1] <= gap_seconds:
            cur.append(i)
        else:
            clusters.append(HitCluster(float(t[cur[0]]), float(t[cur[-1]]), cur))
            cur = [i]
    clusters.append(HitCluster(float(t[cur[0]]), float(t[cur[-1]]), cur))

    out: list[HitCluster] = []
    for c in clusters:
        if len(c.hits) < min_hits or c.duration < min_seconds:
            continue
        if c.duration > max_seconds:
            c = _split_long(c, det, max_seconds, gap_seconds)
            out.extend(c)
        else:
            out.append(c)
    return out


def _split_long(c: HitCluster, det: HitDetection, max_seconds: float, gap_seconds: float) -> list[HitCluster]:
    """超长簇按内部最大间隔递归切分，避免把两个回合粘在一起。"""
    if c.duration <= max_seconds or len(c.hits) < 4:
        return [c]
    t = det.times
    gaps = np.diff(t[c.hits])
    order = np.argsort(gaps)[::-1]
    for k in order:
        if gaps[k] < 0.35:
            break
        left = c.hits[: k + 1]
        right = c.hits[k + 1 :]
        l = HitCluster(float(t[left[0]]), float(t[left[-1]]), left)
        r = HitCluster(float(t[right[0]]), float(t[right[-1]]), right)
        return _split_long(l, det, max_seconds, gap_seconds) + _split_long(r, det, max_seconds, gap_seconds)
    mid = len(c.hits) // 2
    left, right = c.hits[:mid], c.hits[mid:]
    return [
        HitCluster(float(t[left[0]]), float(t[left[-1]]), left),
        HitCluster(float(t[right[0]]), float(t[right[-1]]), right),
    ]


__all__ = ["HitDetection", "HitCluster", "load_wav_mono", "detect_hits", "cluster_rallies"]
