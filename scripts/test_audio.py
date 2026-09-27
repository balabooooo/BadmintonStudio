"""Verification script: audio hit detection + rally clustering.

Usage:
    python scripts/test_audio.py "<video path>" [--from 0] [--to 300]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from bms.analysis import audio_hits as AH  # noqa: E402
from bms.config import AUDIO_DIR, CACHE_DIR, ensure_dirs  # noqa: E402
from bms.core import ffmpeg as ff  # noqa: E402


def extract_audio_range(video: str, start: float, end: float, sr: int = 16000) -> Path:
    ensure_dirs()
    out = AUDIO_DIR / f"_test_{Path(video).stem}_{int(start)}_{int(end)}.wav"
    cmd = [ff.find_ffmpeg(), "-hide_banner", "-y", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video]
    if end > start:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-loglevel", "error", str(out)]
    r = ff.run(cmd)
    if not r.ok:
        raise RuntimeError(r.stderr[-2000:])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--to", dest="end", type=float, default=300.0)
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=3.2)
    args = ap.parse_args()

    t0 = time.time()
    wav = extract_audio_range(args.video, args.start, args.end)
    print(f"[1/3] 提取音频 {wav.name}  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    det = AH.detect_hits(wav, sensitivity=args.sensitivity)
    print(f"[2/3] 击球检测: {det.times.size} 次触球, 包络 {det.envelope.size} 点 @ {det.env_fps}Hz  ({time.time() - t1:.1f}s)")
    if det.times.size:
        print("      首次/末次击球: %.2fs / %.2fs" % (det.times[0] + args.start, det.times[-1] + args.start))
        gaps = np.diff(det.times)
        print("      间隔统计: min %.3f  p25 %.3f  中位 %.3f  p75 %.3f  max %.3f" % (
            gaps.min(), np.percentile(gaps, 25), np.median(gaps), np.percentile(gaps, 75), gaps.max()))
        q = np.argsort(-det.strength[:6])
        print("      最强击球 (t, 强度, 置信):")
        for i in q:
            print(f"        {det.times[i] + args.start:8.2f}s  str={det.strength[i]:.2f}  conf={det.confidence[i]:.2f}")

    clusters = AH.cluster_rallies(det, gap_seconds=args.gap, min_seconds=2.0, min_hits=2)
    print(f"[3/3] 回合聚类: {len(clusters)} 个回合  ({time.time() - t0:.1f}s 总计)")
    print("      序号  起(s)     止(s)    时长   拍数   平均间隔")
    for i, c in enumerate(clusters[:60], 1):
        h = det.times[c.hits]
        iv = float(np.mean(np.diff(h))) if len(h) > 1 else 0.0
        print(f"      {i:4d}  {c.start + args.start:8.2f}  {c.end + args.start:8.2f}  {c.duration:6.2f}  {len(c.hits):5d}   {iv:6.3f}")
    if len(clusters) > 60:
        print(f"      ... 其余 {len(clusters) - 60} 个省略")

    # Export the envelope so the frontend / a human can verify it
    out_npz = CACHE_DIR / "_test_audio_signal.npz"
    np.savez_compressed(out_npz, env=det.envelope, thr=det.threshold, fps=det.env_fps,
                        hits=det.times + args.start, strength=det.strength, conf=det.confidence)
    print(f"      信号已存 -> {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
