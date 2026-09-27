"""Audio hit detection.

The essence of a badminton rally is "a string of racket hit sounds". Racket hits in the
mid-high frequency band (about 2-8 kHz) appear as extremely short (< 15 ms) broadband
transients, clearly distinguishable from the low-frequency voices and ambient noise in the
venue. This module uses "high-band spectral flux + adaptive threshold + peak picking" to
detect every touch, then clusters hits into rallies by silence gaps.

It only depends on numpy / scipy / the standard-library wave, so librosa / numba are not needed.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d, median_filter

from ..i18n import tr

# ------------------------------------------------------------------ Constants

HOP = 64               # STFT hop (samples) @16k -> 4 ms
NFFT = 512
HIGH_BAND = (1800.0, 7800.0)   # main energy band of racket hits
LOW_BAND = (80.0, 900.0)       # voices / footsteps / venue noise
MIN_HIT_GAP = 0.055            # minimum gap between two independent hits (seconds)


@dataclass
class HitDetection:
    times: np.ndarray                     # hit times (seconds)
    strength: np.ndarray                  # normalized strength 0~1
    confidence: np.ndarray                # hit confidence 0~1
    envelope: np.ndarray                  # hit response envelope (used by the frontend to draw the waveform)
    env_fps: float                        # envelope samples per second
    threshold: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise_floor_db: float = -60.0


@dataclass
class HitEnvelope:
    """Everything derivable from the WAV ahead of thresholding (the expensive STFT pass).

    Holding this lets the annotation optimizer re-detect hits over a ``hit_sensitivity`` grid cheaply:
    the threshold / peak-picking step is milliseconds, while the STFT is only computed once.
    """

    flux: np.ndarray
    env_fps: float
    env_smooth: np.ndarray
    low_ds: np.ndarray
    hi_env: np.ndarray
    sr: int
    noise_floor_db: float = -60.0


# ------------------------------------------------------------------ Reading audio


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as float32 mono (using the standard library, no extra dependency)."""
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
        raise ValueError(tr("analysis.audio.unsupported_bit_depth", bits=sw * 8))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


# ------------------------------------------------------------------ Detection


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
    # Weight energy within the band to suppress narrow-band pure tones
    m = mag[band]
    flat = np.exp(np.mean(np.log(m + 1e-10), axis=0)) / (np.mean(m, axis=0) + 1e-10)
    d = np.diff(m, axis=1, prepend=m[:, :1])
    flux = np.maximum(d, 0).sum(axis=0)
    flux = flux * (0.35 + 0.65 * np.clip(flat * 4.0, 0, 1))  # the closer to noise-like, the more it looks like a hit
    hop_s = HOP / sr
    return flux.astype(np.float32), flat.astype(np.float32), max(1, int(round(1.0 / hop_s)))


def _adaptive_threshold(env: np.ndarray, fps: float, sensitivity: float) -> np.ndarray:
    """Robust adaptive threshold based on a long-window median + local MAD.

    Sensitivity maps to k:
      0.0 -> k≈6.2 (keep only the loudest hits)
      0.5 -> k≈4.4 (default)
      1.0 -> k≈2.6 (catch even light touches, at the cost of more noise)
    """
    win = max(5, int(round(fps * 2.5)) | 1)
    med = median_filter(env, size=win, mode="nearest")
    mad = median_filter(np.abs(env - med), size=win, mode="nearest")
    k = 6.2 - 3.6 * float(np.clip(sensitivity, 0.0, 1.0))
    return med + k * (mad * 1.4826 + 1e-6)


def build_hit_envelope(wav_path: str | Path) -> HitEnvelope | None:
    """Compute the expensive per-frame envelope once (STFT / band energies), before thresholding.

    Returns ``None`` when the audio is too short to analyze, so callers can short-circuit.
    """
    x, sr = load_wav_mono(wav_path)
    if x.size < sr // 4:
        return None
    # Remove DC + light denoising
    x = x - float(np.mean(x))
    sos_hp = sps.butter(2, 60.0 / (sr / 2), btype="highpass", output="sos")
    x = sps.sosfiltfilt(sos_hp, x)
    flux, _flatness, env_fps = _spectral_flux(x, sr, *HIGH_BAND)
    env_smooth = maximum_filter1d(flux, size=3)
    # Low-band energy: reject false triggers dominated by "shouts / footsteps / scraping"
    low_env = _bandpass_env(x, sr, *LOW_BAND, smooth_ms=60.0)
    low_ds = _resample_to(low_env, sr, env_fps)
    # Raw high-band envelope for sub-frame onset refinement
    hi_env = _bandpass_env(x, sr, *HIGH_BAND, smooth_ms=3.0)
    return HitEnvelope(
        flux=flux.astype(np.float32), env_fps=env_fps,
        env_smooth=np.asarray(env_smooth, dtype=np.float32),
        low_ds=np.asarray(low_ds, dtype=np.float32),
        hi_env=hi_env, sr=sr,
        noise_floor_db=float(20 * np.log10(np.median(np.abs(x)) + 1e-12)),
    )


def pick_hits(env: HitEnvelope, sensitivity: float = 0.5,
              max_hits: int = 60000) -> HitDetection:
    """Threshold / peak-pick a precomputed :class:`HitEnvelope` into hits (cheap, re-runnable)."""
    env_smooth = env.env_smooth
    env_fps = env.env_fps
    thr = _adaptive_threshold(env_smooth, env_fps, sensitivity)
    if env_smooth.size == 0:
        return HitDetection(
            times=np.zeros(0), strength=np.zeros(0), confidence=np.zeros(0),
            envelope=env.flux, env_fps=env_fps, threshold=thr,
            noise_floor_db=env.noise_floor_db,
        )

    distance = max(1, int(round(MIN_HIT_GAP * env_fps)))
    peaks, props = sps.find_peaks(
        env_smooth, height=thr, distance=distance, prominence=thr * 0.35
    )
    if peaks.size == 0:
        return HitDetection(
            times=np.zeros(0), strength=np.zeros(0), confidence=np.zeros(0),
            envelope=env.flux, env_fps=env_fps, threshold=thr,
            noise_floor_db=env.noise_floor_db,
        )

    heights = props["peak_heights"]
    # ---- Confidence: high-frequency transient strength vs low-frequency background, plus sharpness
    low_ds = env.low_ds
    lo_ref = np.percentile(low_ds, 60) + 1e-9 if low_ds.size else 1e-9
    low_at = low_ds[np.clip(peaks, 0, low_ds.size - 1)] if low_ds.size else np.zeros(peaks.size)
    sharp = np.zeros(peaks.size, dtype=np.float32)
    w = max(1, int(round(0.02 * env_fps)))
    for i, p in enumerate(peaks):
        a, b = max(0, p - w), min(env_smooth.size, p + w + 1)
        seg = env_smooth[a:b]
        base = float(np.percentile(seg, 20)) + 1e-9
        sharp[i] = float(env_smooth[p]) / base
    sharp_n = np.clip((sharp - 1.4) / 3.6, 0.0, 1.0)
    snr = np.clip(heights / (thr[peaks] + 1e-9), 0.0, 4.0) / 4.0
    low_pen = np.clip(low_at / (lo_ref * 6.0), 0.0, 1.0) if low_ds.size else np.zeros(peaks.size)
    conf = np.clip(0.45 * snr + 0.40 * sharp_n + 0.15 * (1.0 - low_pen), 0.0, 1.0).astype(np.float32)

    # Strength normalization (quantiles, outlier-resistant)
    h = heights.astype(np.float32)
    p5, p95 = np.percentile(h, 5), np.percentile(h, 97)
    strength = np.clip((h - p5) / max(p95 - p5, 1e-9), 0.0, 1.0).astype(np.float32)

    # ---- Sub-frame refinement: align to the rising edge of the raw high-frequency envelope
    times = peaks.astype(np.float64) / env_fps
    times = _refine_onsets(env.hi_env, env.sr, times)

    order = np.argsort(times)
    times, strength, conf = times[order], strength[order], conf[order]
    if times.size > max_hits:
        sel = np.linspace(0, times.size - 1, max_hits).astype(int)
        times, strength, conf = times[sel], strength[sel], conf[sel]

    return HitDetection(
        times=times, strength=strength, confidence=conf,
        envelope=env.flux.astype(np.float32), env_fps=env_fps,
        threshold=thr.astype(np.float32), noise_floor_db=env.noise_floor_db,
    )


def detect_hits(
    wav_path: str | Path,
    sensitivity: float = 0.5,
    max_hits: int = 60000,
) -> HitDetection:
    env = build_hit_envelope(wav_path)
    if env is None:
        return HitDetection(
            times=np.zeros(0), strength=np.zeros(0), confidence=np.zeros(0),
            envelope=np.zeros(0), env_fps=1.0,
        )
    return pick_hits(env, sensitivity=sensitivity, max_hits=max_hits)


def _resample_to(env: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    if abs(src_rate - dst_rate) < 1e-6 or env.size == 0:
        return env
    n = max(1, int(round(env.size / src_rate * dst_rate)))
    idx = np.linspace(0, env.size - 1, n)
    return np.interp(idx, np.arange(env.size), env).astype(np.float32)


def _refine_onsets(env: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """Align peak times to the initial rising point of that transient."""
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
        # Walk forward from the peak to find the local minimum -> treat as the onset
        j = peak
        floor = env[peak] * 0.18
        while j > a and env[j] > floor:
            j -= 1
        out[i] = j / sr
    return np.maximum.accumulate(out)  # ensure monotonicity


# ------------------------------------------------------------------ Rally clustering


@dataclass
class HitCluster:
    start: float
    end: float
    hits: list[int]  # indices

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
    """Cluster hit times into rally candidates by silence gaps."""
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
    """Recursively split over-long clusters at the largest internal gap, to avoid gluing two rallies together."""
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


__all__ = ["HitDetection", "HitEnvelope", "HitCluster", "load_wav_mono", "build_hit_envelope",
           "pick_hits", "detect_hits", "cluster_rallies"]
