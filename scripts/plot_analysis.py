"""Plot analysis results as diagnostic charts: signals, thresholds, rally intervals, scores.

Usage:
    python scripts/plot_analysis.py [json] [--from 0] [--to 300] [--out x.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bms.config import FRAMES_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", default=str(ROOT / "data" / "cache" / "last_analysis.json"))
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--to", dest="end", type=float, default=0.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    sig = data.get("signals", {})
    dur = float((sig.get("duration") or [0])[0])
    fps = float((sig.get("fps") or [15])[0])
    end = args.end or dur
    a0, a1 = args.start, end

    act = np.array(sig.get("activity") or [])
    if act.size == 0:
        print("没有 activity 信号")
        return 1
    t = np.linspace(0, dur, act.size)
    m = (t >= a0) & (t <= a1)
    hi = float((sig.get("threshold_hi") or [0.5])[0])
    lo = float((sig.get("threshold_lo") or [0.3])[0])

    has_players = bool(sig.get("active_speed"))
    has_shuttle = bool(sig.get("shuttle_presence"))
    comp_names = [k for k in ("component_players", "component_motion", "component_audio_hits",
                              "component_shuttle", "component_roi") if sig.get(k)]
    nrows = 2 + len(comp_names) + (1 if has_players else 0) + (1 if has_shuttle else 0)

    fig, axes = plt.subplots(nrows, 1, figsize=(24, 2.6 * nrows), sharex=True)
    i = 0

    ax = axes[i]; i += 1
    ax.plot(t[m], act[m], lw=1.0, color="#1f9", label="fused activity")
    ax.axhline(hi, color="#e33", ls="--", lw=1.0, label=f"hi={hi:.3f}")
    ax.axhline(lo, color="#e93", ls=":", lw=1.0, label=f"lo={lo:.3f}")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylabel("activity"); ax.grid(alpha=0.2)

    colors = {"component_players": "#39f", "component_motion": "#a3e",
              "component_audio_hits": "#fa3", "component_shuttle": "#3fa",
              "component_roi": "#888"}
    for name in comp_names:
        va = np.array(sig[name])
        ax = axes[i]; i += 1
        tt = t[m] if va.size == act.size else np.linspace(0, dur, va.size)
        vv = va[m] if va.size == act.size else va
        ax.plot(tt, vv, lw=0.8, color=colors.get(name, "#666"))
        ax.set_ylabel(name.replace("component_", "")); ax.grid(alpha=0.2)

    if has_players:
        ax = axes[i]; i += 1
        cnt = np.array(sig["active_count"]); spd = np.array(sig["active_speed"])
        pf = float((sig.get("player_fps") or [fps])[0])
        tp = np.linspace(0, dur, cnt.size)
        ax.plot(tp, cnt, lw=0.9, color="#2a8", label="active_count")
        ax2 = ax.twinx()
        ax2.plot(tp, spd, lw=0.9, color="#c33", label="active_speed")
        ax.set_ylabel("count", color="#2a8"); ax2.set_ylabel("speed", color="#c33")
        ax.grid(alpha=0.2)

    if has_shuttle:
        ax = axes[i]; i += 1
        sp = np.array(sig["shuttle_presence"])
        ax.plot(np.linspace(0, dur, sp.size), sp, lw=0.9, color="#3fa")
        ax.set_ylabel("shuttle"); ax.grid(alpha=0.2)

    ax = axes[i]; i += 1
    ht = np.array(sig.get("hit_times") or [])
    hs = np.array(sig.get("hit_strength") or [])
    hc = np.array(sig.get("hit_confidence") or [])
    mm = (ht >= a0) & (ht <= a1)
    if ht.size:
        sc = ax.scatter(ht[mm], hs[mm], c=hc[mm] if hc.size == ht.size else None,
                        s=6, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, pad=0.01, label="conf")
    ax.set_ylabel("hits"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.2); ax.set_xlabel("seconds")

    # Highlight rally intervals + scores
    rallies = data.get("rallies", [])
    for ax in axes:
        for r in rallies:
            if r["end"] < a0 or r["start"] > a1:
                continue
            ax.axvspan(r["start"], r["end"], color="#2c8", alpha=0.10)
            ax.axvline(r["start"], color="#0a6", lw=0.6, alpha=0.5)
        ax.set_xlim(a0, a1)

    top = axes[0]
    for r in rallies:
        if r["end"] < a0 or r["start"] > a1:
            continue
        top.text((r["start"] + r["end"]) / 2, 1.02, f"#{r['index']} {r['scores']['total']:.0f}",
                 ha="center", va="bottom", fontsize=7, color="#063")
    top.set_ylim(0, 1.35)

    fig.suptitle(f"{data.get('media_id')}  rallies={len(rallies)}  weights={sig.get('weights')}")
    fig.tight_layout()
    out = Path(args.out) if args.out else FRAMES_DIR / "analysis_diag.png"
    fig.savefig(out, dpi=88)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
