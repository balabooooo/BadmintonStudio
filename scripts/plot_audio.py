"""音频信号诊断图：波形 / 高频通量 / 自适应阈值 / 检出的击球 / 频谱图。

用法:
    python scripts/plot_audio.py <wav> [--from 0] [--to 60] [--out out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from bms.analysis import audio_hits as AH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--to", dest="end", type=float, default=60.0)
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    x, sr = AH.load_wav_mono(args.wav)
    a, b = int(args.start * sr), int(args.end * sr)
    seg = x[a:b]
    dur = len(seg) / sr

    det = AH.detect_hits(args.wav, sensitivity=args.sensitivity)
    m = (det.times >= args.start) & (det.times < args.end)
    ht = det.times[m] - args.start
    hs = det.strength[m]

    fig, axes = plt.subplots(5, 1, figsize=(22, 16), sharex=True)

    t = np.arange(len(seg)) / sr
    axes[0].plot(t, seg, lw=0.3, color="#4a9")
    axes[0].set_ylabel("waveform")
    axes[0].set_ylim(-0.6, 0.6)

    env = det.envelope
    et = np.arange(env.size) / det.env_fps
    mm = (et >= args.start) & (et < args.end)
    axes[1].plot(et[mm] - args.start, env[mm], lw=0.6, color="#39f", label="HF flux")
    axes[1].plot(et[mm] - args.start, det.threshold[mm], lw=0.8, color="#f33", label="threshold")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("spectral flux")

    axes[2].vlines(ht, 0, hs, color="#e63", lw=0.8)
    axes[2].set_ylabel("hits")
    axes[2].set_ylim(0, 1)

    low = AH._bandpass_env(seg, sr, *AH.LOW_BAND, smooth_ms=80.0)
    axes[3].plot(t, low, lw=0.6, color="#a3e")
    axes[3].set_ylabel("LF energy")
    axes[3].set_yscale("log")

    f, tt, Z = sps.stft(seg, fs=sr, nperseg=1024, noverlap=768)
    axes[4].pcolormesh(tt, f, 20 * np.log10(np.abs(Z) + 1e-6), shading="auto", cmap="magma", vmin=-90, vmax=-20)
    axes[4].set_ylim(0, 8000)
    axes[4].set_ylabel("Hz")
    axes[4].set_xlabel("seconds")

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"hits in [{args.start},{args.end})s : {int(m.sum())}   (total {det.times.size})")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.wav).with_suffix(".diag.png")
    fig.savefig(out, dpi=90)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
