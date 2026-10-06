"""Runtime synthesis of the guided-tour sample media.

The demo clips are not shipped as binary assets: every frame is drawn with
OpenCV (dark-green hall, white court lines, two smoothly moving players and a
shuttlecock on parabolic arcs) and piped raw into ffmpeg. This keeps the
repository binary-free while still letting the first-run tour seed real,
playable media.

Filenames embed a version tag (``samples_v1_*``) so a future content change
can bump the tag and generate fresh files instead of silently overwriting
media the user already has.

Unlike the analysis pipeline, failures are **not** swallowed here: a broken
ffmpeg path or a failed encode must raise so the caller can surface it.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .core.ffmpeg import CREATE_NO_WINDOW

__all__ = ["ensure_sample_files"]

_VERSION = "v1"
_MATCH_NAME = f"samples_{_VERSION}_match.mp4"
_RALLY_NAME = f"samples_{_VERSION}_rally.mp4"

# BGR colors: dark-green hall, slightly lighter court surface, white lines.
_BG_COLOR = (30, 54, 34)
_COURT_COLOR = (44, 84, 52)
_LINE_COLOR = (225, 225, 225)
_NET_COLOR = (185, 185, 185)
_P1_COLOR = (70, 70, 225)   # left side, red-ish
_P2_COLOR = (225, 140, 50)  # right side, blue-ish
_SHUTTLE_COLOR = (255, 255, 255)
_SHUTTLE_EDGE = (40, 40, 40)

# Standard court dimensions in meters (doubles court).
_COURT_LEN = 13.4
_COURT_WID = 6.1
_SHORT_SERVICE = 1.98
_LONG_SERVICE = 0.76
_SINGLES_SIDE = 0.46

# Fixed seed: the demos are deterministic, so a regenerated clip looks the same.
_RNG_SEED = 20261006


# ------------------------------------------------------------------ Clip specs


@dataclass(frozen=True)
class _Spec:
    """One demo clip: geometry/fps plus the rally/pause timeline it renders."""

    name: str
    width: int
    height: int
    fps: int
    seconds: float
    # ("rally" | "pause", start, end) in seconds, ordered, covering [0, seconds].
    segments: tuple[tuple[str, float, float], ...]


def _small_specs() -> list[_Spec]:
    # 2s @ 12fps, 320x180, a single short rally each: must finish in seconds.
    segs = (("rally", 0.0, 2.0),)
    return [
        _Spec(_MATCH_NAME, 320, 180, 12, 2.0, segs),
        _Spec(_RALLY_NAME, 320, 180, 12, 2.0, segs),
    ]


def _full_specs() -> list[_Spec]:
    # Match: 3 x ~10s rallies with 4-6s pauses between (plus lead-in / tail) = 48s.
    segs = (
        ("pause", 0.0, 3.0),
        ("rally", 3.0, 13.0),
        ("pause", 13.0, 19.0),
        ("rally", 19.0, 29.0),
        ("pause", 29.0, 35.0),
        ("rally", 35.0, 45.0),
        ("pause", 45.0, 48.0),
    )
    match = _Spec(_MATCH_NAME, 960, 540, 30, 48.0, segs)
    rally = _Spec(_RALLY_NAME, 960, 540, 30, 16.0, (("rally", 0.0, 16.0),))
    return [match, rally]


# ------------------------------------------------------------------ Geometry / plan


@dataclass(frozen=True)
class _Geom:
    """Court placement in frame pixels; ``s`` is the pixels-per-meter scale."""

    x0: float
    y0: float
    x1: float
    y1: float
    s: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def net_x(self) -> float:
        return self.cx


@dataclass
class _Plan:
    """Everything the per-frame renderer needs, precomputed once per clip."""

    geom: _Geom
    background: np.ndarray
    # (t0, t1, (x0, y0), (x1, y1), peak_px) — shuttle flights, ordered by t0.
    flights: list[tuple[float, float, tuple[float, float], tuple[float, float], float]]
    player_w: float
    player_h: float
    shuttle_r: float


def _geom(width: int, height: int) -> _Geom:
    s = min(width * 0.80 / _COURT_LEN, height * 0.80 / _COURT_WID)
    w, h = _COURT_LEN * s, _COURT_WID * s
    return _Geom((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2, s)


def _player_home(g: _Geom, side: int) -> tuple[float, float]:
    x = g.x0 + 3.0 * g.s if side == 0 else g.x1 - 3.0 * g.s
    return (x, g.cy)


def _target_point(g: _Geom, rng: random.Random, side: int) -> tuple[float, float]:
    """A landing spot inside ``side``'s half of the court."""
    if side == 0:
        x = rng.uniform(g.x0 + 0.8 * g.s, g.net_x - 0.7 * g.s)
    else:
        x = rng.uniform(g.net_x + 0.7 * g.s, g.x1 - 0.8 * g.s)
    y = rng.uniform(g.y0 + 0.55 * g.s, g.y1 - 0.55 * g.s)
    return (x, y)


def _peak_for(g: _Geom, p0: tuple[float, float], p1: tuple[float, float],
              rng: random.Random, *, low: bool = False) -> float:
    """Arc height in px, clamped so the shuttle never leaves the frame top."""
    base = rng.uniform(0.5, 0.9) * g.s if low else rng.uniform(1.4, 2.4) * g.s
    limit = max(8.0, min(p0[1], p1[1]) - g.y0 - 10.0)
    return min(base, limit)


def _build_plan(spec: _Spec) -> _Plan:
    g = _geom(spec.width, spec.height)
    rng = random.Random(_RNG_SEED)
    flights: list[tuple[float, float, tuple[float, float], tuple[float, float], float]] = []
    pos = _player_home(g, 0)
    rally_idx = 0
    for kind, start, end in spec.segments:
        if kind == "rally":
            # Continuous exchange: each flight departs from where the last one
            # landed, so the shuttle never teleports.
            t = start + 0.3
            while True:
                dur = rng.uniform(0.7, 1.1)
                if t + dur > end - 0.1:
                    break
                tgt_side = 1 if pos[0] < g.net_x else 0
                p1 = _target_point(g, rng, tgt_side)
                flights.append((t, t + dur, pos, p1, _peak_for(g, pos, p1, rng)))
                pos = p1
                t += dur + rng.uniform(0.05, 0.15)
            rally_idx += 1
        else:  # pause: gentle pick-up flight to the next server + one serve toss
            gap = end - start
            server = _player_home(g, rally_idx % 2)
            if gap > 1.2 and pos != server:
                dur = min(0.7, gap * 0.3)
                flights.append((start, start + dur, pos, server,
                                _peak_for(g, pos, server, rng, low=True)))
                pos = server
            if gap > 2.0:
                toss_t = end - 1.1
                flights.append((toss_t, toss_t + 0.6, pos, pos, 0.55 * g.s))
    return _Plan(
        geom=g,
        background=_render_background(spec.width, spec.height, g),
        flights=flights,
        player_w=0.55 * g.s,
        player_h=0.85 * g.s,
        shuttle_r=0.10 * g.s,
    )


# ------------------------------------------------------------------ Drawing


def _render_background(width: int, height: int, g: _Geom) -> np.ndarray:
    """Static part of every frame: hall backdrop, court surface and lines."""
    img = np.full((height, width, 3), _BG_COLOR, dtype=np.uint8)
    x0, y0 = int(round(g.x0)), int(round(g.y0))
    x1, y1 = int(round(g.x1)), int(round(g.y1))
    cv2.rectangle(img, (x0, y0), (x1, y1), _COURT_COLOR, -1)
    th = max(1, int(round(g.s * 0.05)))
    cy = int(round(g.cy))
    net_x = int(round(g.net_x))
    # Outer boundary + singles side lines.
    cv2.rectangle(img, (x0, y0), (x1, y1), _LINE_COLOR, th)
    for y in (y0 + _SINGLES_SIDE * g.s, y1 - _SINGLES_SIDE * g.s):
        yy = int(round(y))
        cv2.line(img, (x0, yy), (x1, yy), _LINE_COLOR, th)
    # Doubles long service lines, short service lines, center lines.
    for x in (x0 + _LONG_SERVICE * g.s, x1 - _LONG_SERVICE * g.s):
        xx = int(round(x))
        cv2.line(img, (xx, y0), (xx, y1), _LINE_COLOR, th)
    for x in (net_x - _SHORT_SERVICE * g.s, net_x + _SHORT_SERVICE * g.s):
        xx = int(round(x))
        cv2.line(img, (xx, y0), (xx, y1), _LINE_COLOR, th)
    for xa, xb in ((x0, net_x - _SHORT_SERVICE * g.s), (net_x + _SHORT_SERVICE * g.s, x1)):
        cv2.line(img, (int(round(xa)), cy), (int(round(xb)), cy), _LINE_COLOR, th)
    return img


def _draw_net(img: np.ndarray, g: _Geom) -> None:
    th = max(1, int(round(g.s * 0.05)))
    nx = int(round(g.net_x))
    top = int(round(g.y0 - 0.35 * g.s))
    bot = int(round(g.y1 + 0.35 * g.s))
    cv2.line(img, (nx, top), (nx, bot), _NET_COLOR, max(2, th * 2))


def _draw_player(img: np.ndarray, cx: float, cy: float, w: float, h: float,
                 color: tuple[int, int, int]) -> None:
    """A capsule (rounded rectangle) standing in for a player, seen top-down."""
    x0 = int(round(cx - w / 2))
    x1 = int(round(cx + w / 2))
    y0 = int(round(cy - h / 2))
    y1 = int(round(cy + h / 2))
    r = max(2, int(round(w / 2)))
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    for px, py in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (px, py), r, color, -1)


def _player_pos(g: _Geom, side: int, t: float, env: float) -> tuple[float, float]:
    """Smooth sine sway around the home spot, scaled by the rally envelope."""
    hx, hy = _player_home(g, side)
    wx, wy = (1.05, 1.55) if side == 0 else (0.95, 1.65)
    px, py = (0.0, 1.1) if side == 0 else (0.6, 2.0)
    ax, ay = 1.8 * g.s, 1.5 * g.s
    return (hx + ax * math.sin(wx * t + px) * env,
            hy + ay * math.sin(wy * t + py) * env)


def _rally_env(segments: tuple[tuple[str, float, float], ...], t: float) -> float:
    """1 inside rallies, 0 in pauses, with a short cosine ramp at the seams."""
    env = 0.0
    d = 0.45
    for kind, start, end in segments:
        if kind != "rally":
            continue
        r_in = min(1.0, max(0.0, (t - start) / d))
        r_out = min(1.0, max(0.0, (end - t) / d))
        e = _smoothstep(r_in) * _smoothstep(r_out)
        env = max(env, e)
    return env


def _smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def _shuttle_pos(plan: _Plan, t: float) -> tuple[float, float]:
    """Shuttle position: parabolic arc while airborne, rest at the last landing
    point between flights (or at the first flight's origin before it starts)."""
    prev_end: tuple[float, float] | None = None
    for t0, t1, p0, p1, peak in plan.flights:
        if t < t0:
            break
        if t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            x = p0[0] + (p1[0] - p0[0]) * u
            y = p0[1] + (p1[1] - p0[1]) * u - peak * math.sin(math.pi * u)
            return (x, y)
        prev_end = p1
    if prev_end is None:
        return plan.flights[0][2]
    return prev_end


def _render_frame(spec: _Spec, plan: _Plan, t: float) -> np.ndarray:
    img = plan.background.copy()
    g = plan.geom
    env = _rally_env(spec.segments, t)
    for side, color in ((0, _P1_COLOR), (1, _P2_COLOR)):
        cx, cy = _player_pos(g, side, t, env)
        _draw_player(img, cx, cy, plan.player_w, plan.player_h, color)
    _draw_net(img, g)
    sx, sy = _shuttle_pos(plan, t)
    r = max(2, int(round(plan.shuttle_r)))
    center = (int(round(sx)), int(round(sy)))
    cv2.circle(img, center, r, _SHUTTLE_COLOR, -1)
    if r >= 3:
        cv2.circle(img, center, r, _SHUTTLE_EDGE, 1)
    return img


# ------------------------------------------------------------------ Encoding


def _encode(spec: _Spec, target: Path, ffmpeg: str) -> None:
    """Draw every frame and pipe it raw into ffmpeg, writing to a temp file
    that only replaces ``target`` once the encode succeeded."""
    plan = _build_plan(spec)
    total = int(round(spec.seconds * spec.fps))
    tmp = target.with_name(target.name + ".part")
    cmd = [
        str(ffmpeg), "-y", "-nostats", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{spec.width}x{spec.height}", "-r", str(spec.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-f", "mp4", str(tmp),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )
    try:
        assert proc.stdin is not None and proc.stderr is not None
        for i in range(total):
            proc.stdin.write(_render_frame(spec, plan, i / spec.fps).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read()
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg exited with code {code} while encoding {target.name}: {stderr[-2000:]}"
        )
    os.replace(tmp, target)


# ------------------------------------------------------------------ Public API


def ensure_sample_files(data_dir: Path, ffmpeg: str, *, small: bool = False) -> list[Path]:
    """Return the two demo clips under ``data_dir/samples``, generating them if
    missing.

    Idempotent: existing non-empty files are returned as-is. ``small=True``
    renders tiny 2s/12fps/320x180 stand-ins (used by tests). Raises on any
    failure — a broken ffmpeg path or failed encode must not degrade silently.
    """
    specs = _small_specs() if small else _full_specs()
    out_dir = Path(data_dir) / "samples"
    targets = [out_dir / s.name for s in specs]
    if all(p.is_file() and p.stat().st_size > 0 for p in targets):
        return targets
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec, target in zip(specs, targets):
        if target.is_file() and target.stat().st_size > 0:
            continue
        _encode(spec, target, ffmpeg)
    return targets
