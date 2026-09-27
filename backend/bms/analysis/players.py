"""Player detection and tracking: pick out "the players currently in a match" from the frame.

Scene characteristics (confirmed by measurement):
    * A completely static ultra-wide-angle fisheye low camera, mounted behind one court;
    * The players currently in a match are the **largest** two person boxes in the frame; they run/stride/jump,
      and their positions fall in a horizontal band at roughly 0.40~0.65 of the frame height;
    * The rest are a large number of background people (people on other courts, spectators, staff): smaller boxes,
      mostly stationary, including distractors such as "staff who keep walking back and forth over a small area nearby".

So this module does three things:
    1. **Detection**: ultralytics YOLO (``classes=[0]`` persons only), sampling frames at ``sample_fps``
       and sending them to the GPU in **batches** (never frame by frame);
    2. **Tracking**: ultralytics' built-in **ByteTrack** (Kalman motion model + high/low score
       two-stage association), using low-score boxes to reconnect trajectories during occlusion, trying to avoid splitting one player into two trajectories;
    3. **Identifying match players**: score by "large box + fast speed + long presence", and penalize
       features such as "slow for a long time", "nearly fixed horizontal position", "long pinned to the frame edge",
       then take the 1~4 trajectories with the highest combined score (usually 2, or 4 for doubles).

Only depends on ultralytics (including its built-in ByteTrack and the ``lap`` needed for association) / opencv / numpy /
torch; automatically falls back to CPU when there is no GPU.
"""

from __future__ import annotations

import bisect
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..i18n import tr
from .court_calib import point_in_poly

# ------------------------------------------------------------------ constants

#: Boxes below this confidence are ignored outright (for tracking)
LOW_CONF = 0.15

# ---- ByteTrack (ultralytics built-in implementation) parameters ----
#: Stage one only associates "high-score boxes"; boxes below it go to stage two, used to reconnect trajectories during occlusion.
#: Works together with :data:`LOW_CONF`: detection keeps boxes with ≥0.15, so 0.15~0.25 precisely goes through stage two.
BT_HIGH_THRESH = 0.25
#: Lower bound for stage-two "low-score boxes"; below it they do not participate in association at all.
BT_LOW_THRESH = 0.10
#: When a detection matches no track, its score must reach this to start a new track (blocks new tracks from low-score noise).
BT_NEW_THRESH = 0.25
#: Number of **sampled frames** a lost track is kept before deletion. Counted on the sampled timeline:
#: with the default sample_fps=15, 30 frames ≈ 2 seconds, so a track can reconnect after one or two seconds of occlusion.
BT_TRACK_BUFFER = 30
#: Upper bound on association cost (1 - IoU). Default 0.8 (i.e. IoU lower bound 0.2).
BT_MATCH_THRESH = 0.8
#: Whether to fuse detection scores into the association cost (enabled by default in official ByteTrack).
BT_FUSE_SCORE = True

#: Upper bound on normalized box area: beyond it is basically an abnormally large box from "a person right up against the lens"
MAX_BOX_AREA = 0.12
#: Lower bound on normalized box height: smaller boxes are certainly noise
MIN_BOX_HEIGHT = 0.02
#: Reasonable range for the box aspect ratio (w/h), used to drop horizontal/vertical-strip false detections
ASPECT_RANGE = (0.12, 1.8)
#: IoU threshold for near-overlapping duplicate boxes within the same frame
DUP_IOU = 0.75
#: Track merging: upper bound on the gap duration (seconds). A player may disappear for a few seconds when running out of frame or fully occluded,
#: and a measured clip has a 3.2-second gap, so this is left at 5 seconds; the position and scale checks are still very strict.
MERGE_MAX_GAP = 5.0
#: Track merging: upper bound on position difference (in frame-height units)
MERGE_MAX_DIST = 0.10
#: Track merging: upper bound on the box area ratio
MERGE_AREA_RATIO = 3.5
#: Track merging: when overlapping in time, the lower bound on the fraction of overlapping frames that look like "the same person"
MERGE_OVERLAP_RATIO = 0.5

#: Size filtering: histogram bin count and upper bound (box height in frame-height units; above 0.5 is basically a close-up)
SIZE_HIST_BINS = 24
SIZE_HIST_MAX = 0.50
#: Size filtering: upper bound on the number of samples sent with the statistics (the UI uses them to preview threshold drags in real time)
SIZE_SAMPLE_MAX = 600
#: Outward expansion of the court polygon ROI (normalized units; every edge gives way outward by this much).
#: Matches the old rectangular ROI expansion of 0.04. A player standing exactly on the boundary line must count as "in court":
#: the cost of erring in one direction is asymmetric — keeping one extra person is just noise (size filtering and activity scoring come later),
#: while missing a real player makes the downstream activity curve and crop tracking drop out completely.
ROI_POLY_MARGIN = 0.04

Progress = Callable[[float, str], None]

#: Default path relative to the project root (fallback when importing bms.config fails)
_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ public data structures


@dataclass
class PlayerTrack:
    """A person who is being tracked continuously."""

    track_id: int
    frames: list[int]           # Frame indices where it appeared (relative to the analysis start, from 0)
    times: list[float]          # Corresponding seconds
    boxes: list[tuple[float, float, float, float]]   # Normalized xyxy (0~1)
    speeds: list[float]         # Normalized speed at each appearance frame (box-center displacement/second, normalized by frame height)
    confidences: list[float]
    # Summary statistics
    mean_area: float = 0.0      # Mean normalized box area
    max_area: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0
    active_score: float = 0.0   # Combined "activity" 0~1, used to pick out match players
    total_travel: float = 0.0   # Normalized cumulative travel

    @property
    def n(self) -> int:
        """Number of sampled frames where it appeared."""
        return len(self.frames)


@dataclass
class PlayerSignal:
    fps: float
    duration: float
    tracks: list[PlayerTrack] = field(default_factory=list)
    #: Number of "active players on court" per frame (counted by active_player_ids)
    active_count: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Mean speed of active players per frame (normalized/second)
    active_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Maximum speed per frame
    max_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Mean speed of all people (including background) per frame
    crowd_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: List of track_ids judged to be "match players" (usually 2, possibly 4 for doubles)
    active_player_ids: list[int] = field(default_factory=list)
    #: Per-frame boxes of each person, used for automatic crop tracking: [frame] -> [(track_id, x1,y1,x2,y2), ...] (active only)
    frame_boxes: list[list[tuple[int, float, float, float, float]]] = field(default_factory=list)
    #: Statistics for person-box size filtering (pre-filter box-height distribution, how many were dropped, reference scale), for UI parameter tuning
    size_stats: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ small utilities


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two normalized xyxy boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_height(box: Sequence[float]) -> float:
    return abs(float(box[3]) - float(box[1]))


@dataclass
class SizeFilter:
    """Person-box size filtering: drop boxes that are "clearly not players in this match" during detection.

    Why it is needed: the original size threshold is **adaptive** (it uses the "90th percentile
    of the largest box per frame" as the reference scale, then cuts tracks that are clearly too
    small), which works well on footage where "the player is the largest person in frame". But it
    fails on two kinds of footage:

    * **Stands / spectators closer to the camera than the players** (side camera angle, stands right
      behind the lens): the reference scale gets hijacked by the largest spectator box, and the real
      players end up being the "too small" group;
    * **Panoramic / fisheye**: the same player's box height at the frame center and at the frame
      corner can differ by more than a factor of two, so "how big counts as a player" is simply not
      the same number at different frame positions.

    These two cases can only be resolved by user indication, so two filtering modes are offered here:

    * ``absolute``: box height as an **absolute** fraction of the frame height. Suited to a fixed
      camera and stable person-box sizes;
    * ``relative``: the ratio of box height to the **largest box height in the same frame**. Suited
      to heavily distorted footage / closer spectators — it renormalizes in every frame and is
      naturally immune to "different position, different box size".

    The area upper/lower bounds apply under both modes (area is more sensitive to "huge false
    detections right up against the lens", because distortion widens the box rather than heightens it).

    When ``mode == "off"`` no filtering is done, and behavior is exactly the same as the old version.
    """

    mode: str = "off"
    min_height: float = 0.05
    max_height: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0

    @property
    def active(self) -> bool:
        if self.mode not in ("absolute", "relative"):
            return False
        return bool(self.min_height > 0 or self.max_height > 0
                    or self.min_area > 0 or self.max_area > 0)

    def thresholds(self, ref_h: float) -> tuple[float, float]:
        """Given the "largest box height in the same frame", compute the effective box-height bounds (0 = unlimited)."""
        if self.mode == "relative":
            if ref_h <= 0.02:            # no trustworthy reference in this frame (all people are very small)
                return 0.0, 0.0
            return (self.min_height * ref_h,
                    self.max_height * ref_h if self.max_height > 0 else 0.0)
        return float(self.min_height), float(self.max_height)

    def keep(self, box: Sequence[float], ref_h: float = 0.0) -> bool:
        """Whether this box passes the filter. ``ref_h`` only takes effect under the ``relative`` mode."""
        if not self.active:
            return True
        lo, hi = self.thresholds(ref_h)
        h = _box_height(box)
        if lo > 0 and h < lo:
            return False
        if hi > 0 and h > hi:
            return False
        area = _box_area(box)
        if self.min_area > 0 and area < self.min_area:
            return False
        if self.max_area > 0 and area > self.max_area:
            return False
        return True

    def as_dict(self) -> dict[str, float | str]:
        return {"mode": self.mode, "min_height": round(float(self.min_height), 4),
                "max_height": round(float(self.max_height), 4),
                "min_area": round(float(self.min_area), 5),
                "max_area": round(float(self.max_area), 5)}


def size_filter_from(raw: "SizeFilter | dict | None") -> SizeFilter:
    """Normalize the parameter (model / dict / instance of this class) into a :class:`SizeFilter`.

    pipeline passes a few fields from ``AnalysisParams``, while UI debugging may pass a dict
    directly, so a fault-tolerant normalization is done here to keep type issues from crashing
    the entire player-detection chain.
    """
    if raw is None:
        return SizeFilter()
    if isinstance(raw, SizeFilter):
        return raw
    if isinstance(raw, dict):
        def _f(key: str, default: float) -> float:
            try:
                v = float(raw.get(key, default))
                return v if np.isfinite(v) else default
            except Exception:
                return default

        mode = str(raw.get("mode") or "off")
        return SizeFilter(
            mode=mode if mode in ("off", "absolute", "relative") else "off",
            min_height=max(0.0, _f("min_height", 0.05)),
            max_height=max(0.0, _f("max_height", 0.0)),
            min_area=max(0.0, _f("min_area", 0.0)),
            max_area=max(0.0, _f("max_area", 0.0)),
        )
    return SizeFilter()


def _sanitize_box(x1: float, y1: float, x2: float, y2: float,
                  viewpoint: str = "unknown") -> tuple[float, float, float, float] | None:
    """Normalize/clip the detection box to [0,1] and drop clearly unreasonable boxes.

    ``viewpoint`` affects the geometric thresholds: from above (overhead/elevated), players look
    "short and wide" and the aspect ratio exceeds 1.8, while distant people occupy only a few
    percent of the frame height — the hard-coded thresholds for a low camera would directly drop
    real players. So here the thresholds are relaxed per camera angle, or these hard clippings are
    simply not applied when information is insufficient (confidence is left to the detector).
    """
    w = max(1e-9, float(x2) - float(x1))
    h = max(1e-9, float(y2) - float(y1))
    box = (max(0.0, float(x1)), max(0.0, float(y1)), min(1.0, float(x2)), min(1.0, float(y2)))
    if box[2] - box[0] <= 1e-6 or box[3] - box[1] <= 1e-6:
        return None
    top_view = viewpoint in ("overhead", "elevated")
    lo_h = 0.008 if top_view else MIN_BOX_HEIGHT
    lo_r, hi_r = (0.08, 6.0) if top_view else ASPECT_RANGE
    hi_area = 0.30 if top_view else MAX_BOX_AREA
    if box[3] - box[1] < lo_h:
        return None
    ratio = w / h
    if ratio < lo_r or ratio > hi_r:
        return None
    if _box_area(box) > hi_area:      # huge false detection right up against the lens
        return None
    return box


def _same_person(b1: Sequence[float], b2: Sequence[float], aspect: float = 1.78) -> bool:
    """Determine whether two boxes are "duplicate detections of the same person".

    Measurements show that on 480p small targets YOLO often outputs two slightly different boxes
    for the same person (IoU 0.4~0.7; during a stride one is a "vertical body strip" and the other
    is a "horizontal strip including the racket and person"). If not cleaned up, the same person
    sprouts two tracks that compete for detections, which is the main source of track
    fragmentation. So in addition to IoU, duplicates are allowed to be identified by
    "centers almost coincide + scales close".

    ``aspect`` must be the **runtime** frame aspect ratio. This used to be hard-coded to 1.78,
    which on portrait footage (9:16) magnifies horizontal distances by more than 3x: two different
    players standing at the same height but some distance apart horizontally would be judged "the
    same person" and merged away.
    """
    iou = _iou(b1, b2)
    if iou > DUP_IOU:
        return True
    h1 = max(1e-6, b1[3] - b1[1])
    h2 = max(1e-6, b2[3] - b2[1])
    if iou <= 0.30 and abs(_box_center(b1)[1] - _box_center(b2)[1]) > 0.5 * min(h1, h2):
        return False
    c1, c2 = _box_center(b1), _box_center(b2)
    dist = math.hypot((c1[0] - c2[0]) * max(0.2, float(aspect)), c1[1] - c2[1])
    if dist > 0.45 * min(h1, h2):
        return False
    a1 = _box_area(b1)
    a2 = _box_area(b2)
    ratio = a1 / max(a2, 1e-9)
    return 0.30 <= ratio <= 3.2


def _dedup_dets(
    dets: list[tuple[tuple[float, float, float, float], float]],
    aspect: float = 1.78,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Remove duplicate boxes of "the same person" within one frame, keeping the highest-confidence one."""
    ordered = sorted(dets, key=lambda z: -z[1])
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in ordered:
        if any(_same_person(box, kb, aspect) for kb, _ in kept):
            continue
        kept.append((box, conf))
    return kept


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


# ------------------------------------------------------------------ environment setup


def _data_paths() -> tuple[Path, Path]:
    """Return (a writable ultralytics config directory, the weights directory)."""
    try:  # prefer reusing the project config so it is written to the same place as other backend modules
        from ..config import DATA_DIR, MODELS_DIR  # type: ignore

        return Path(DATA_DIR) / "yolo", Path(MODELS_DIR)
    except Exception:  # pragma: no cover - fallback for standalone runs
        return _ROOT / "data" / "yolo", _ROOT / "models"


def _prepare_yolo_env(cfg_dir: Path) -> None:
    """Explicitly set YOLO_CONFIG_DIR to keep ultralytics from writing to a non-writable user directory."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cfg_dir)


def _resolve_weights(model_name: str, models_dir: Path) -> str:
    """Locate weights: explicit path > models/ > project root > let ultralytics download into models/."""
    p = Path(model_name)
    if p.is_file():
        return str(p)
    models_dir.mkdir(parents=True, exist_ok=True)
    cand = models_dir / model_name
    if cand.is_file():
        return str(cand)
    root_cand = _ROOT / model_name
    if root_cand.is_file():                     # already downloaded, don't download again
        return str(root_cand)
    return str(cand)


def _pick_device(device: str) -> str:
    """Automatically fall back to CPU when there is no GPU."""
    want = (device or "cpu").strip()
    if want.lower().startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return want


# ------------------------------------------------------------------ tracker


class _TrackBuf:
    """Internal track buffer: stores the observation sequence and maintains the state needed for constant-velocity prediction."""

    __slots__ = (
        "track_id", "aspect", "frames", "times", "boxes", "speeds", "confs",
        "last_box", "last_center", "last_time", "vx", "vy", "missed",
    )

    @property
    def n(self) -> int:
        """Number of observation frames recorded so far."""
        return len(self.frames)

    def __init__(self, track_id: int, aspect: float) -> None:
        self.track_id = track_id
        self.aspect = aspect
        self.frames: list[int] = []
        self.times: list[float] = []
        self.boxes: list[tuple[float, float, float, float]] = []
        self.speeds: list[float] = []
        self.confs: list[float] = []
        self.last_box: tuple[float, float, float, float] | None = None
        self.last_center: tuple[float, float] | None = None
        self.last_time: float = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0

    # ---- prediction (constant-velocity extrapolation, used when merging tracks to judge the seam position) ----
    def predict(self, time: float) -> tuple[float, float] | None:
        """Extrapolate the center point to time ``time`` using the most recent velocity.

        The longer the track has been lost, the more the velocity decays (a player often already
        stopped or changed direction after being occluded), to avoid the predicted point flying
        too far and "stealing" other people nearby.
        """
        if self.last_center is None or self.last_box is None:
            return None
        gap = max(0.0, time - self.last_time)
        decay = 0.80 ** min(self.missed, 6)
        cx = self.last_center[0] + self.vx * gap * decay
        cy = self.last_center[1] + self.vy * gap * decay
        return (cx, cy)

    # ---- observation update ----
    def observe(self, frame_idx: int, time: float, box: tuple[float, float, float, float], conf: float) -> None:
        cx, cy = _box_center(box)
        speed = 0.0
        if self.last_center is not None:
            dt = time - self.last_time
            if dt > 1e-6:
                # Frame aspect conversion: dx is normalized by frame width; multiplying by aspect puts it on the same scale as dy (both in frame-height units)
                dx = (cx - self.last_center[0]) * self.aspect
                dy = cy - self.last_center[1]
                speed = math.hypot(dx, dy) / dt
                vx = (cx - self.last_center[0]) / dt
                vy = (cy - self.last_center[1]) / dt
                # Exponential smoothing to suppress single-frame jitter; also clamp to keep false detections from sending the speed wild
                self.vx = 0.5 * self.vx + 0.5 * float(np.clip(vx, -4.0, 4.0))
                self.vy = 0.5 * self.vy + 0.5 * float(np.clip(vy, -4.0, 4.0))
        self.frames.append(int(frame_idx))
        self.times.append(float(time))
        self.boxes.append(box)
        self.speeds.append(float(speed))
        self.confs.append(float(conf))
        self.last_box = box
        self.last_center = (cx, cy)
        self.last_time = float(time)
        self.missed = 0

    # ---- absorb another track (when the same person is tracked as two tracks) ----
    def absorb(self, other: "_TrackBuf") -> None:
        """Merge another track's observations in (sorted by time to keep the timeline monotonic)."""
        self.frames.extend(other.frames)
        self.times.extend(other.times)
        self.boxes.extend(other.boxes)
        self.speeds.extend(other.speeds)
        self.confs.extend(other.confs)
        order = sorted(range(len(self.times)), key=lambda k: self.times[k])
        self.frames = [self.frames[k] for k in order]
        self.times = [self.times[k] for k in order]
        self.boxes = [self.boxes[k] for k in order]
        self.speeds = [self.speeds[k] for k in order]
        self.confs = [self.confs[k] for k in order]
        if other.times and other.times[-1] >= self.times[-1]:
            self.last_box = other.last_box
            self.last_center = other.last_center
            self.last_time = other.last_time
            self.vx, self.vy = other.vx, other.vy
            self.missed = other.missed


class _ByteTracker:
    """Adapter for ByteTrack (ultralytics built-in implementation).

    Why an adapter instead of calling ``model.track()`` directly: inference in this module is
    **batched** (see :func:`analyze_players`), while ``model.track()`` is a frame-by-frame
    streaming call. The adapter feeds each sampled frame's detections to
    ``ultralytics.trackers.byte_tracker.BYTETracker``, keeping batched GPU inference while still
    using ByteTrack's Kalman motion model and "high/low score two-stage" association.

    Coordinate convention: internally ByteTrack does IoU / Kalman in **isometric** coordinates.
    This project uses normalized boxes (x divided by frame width, y by frame height); the two
    scales differ, and feeding them directly would distort IoU and invalidate Kalman's x/y noise
    assumptions. So here x is multiplied by the frame aspect ratio ``aspect`` to convert to
    isometric coordinates "in frame-height units" (equivalent to dividing pixel coordinates by the
    frame height; IoU then matches pixel space exactly), and converted back to normalized on output.

    Track state is still stored in :class:`_TrackBuf`, so downstream
    :func:`_merge_tracks` / :func:`_summarize` / match-player identification need no changes.

    Concurrency safety: ByteTrack's track id counter is **process-global**
    (``BaseTrack._count``), and ``BYTETracker.__init__`` resets it to 0. If a second analysis runs
    concurrently, the reset would make newly started tracks in the first analysis receive already
    used ids, merging two people into the same ``_TrackBuf`` (silently corrupting the tracks). So
    the reset is temporarily disabled during construction, letting the counter only increase:
    ids are globally unique, and each adapter isolates by id via its own ``_bufs``, so they
    naturally do not interfere.
    """

    def __init__(
        self,
        fps: float,
        aspect: float,
        track_buffer: int = BT_TRACK_BUFFER,
    ) -> None:
        from types import SimpleNamespace

        from ultralytics.trackers import basetrack as _basetrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.fps = max(1e-6, fps)
        self.aspect = max(1e-6, float(aspect))
        args = SimpleNamespace(
            track_high_thresh=BT_HIGH_THRESH,
            track_low_thresh=BT_LOW_THRESH,
            new_track_thresh=BT_NEW_THRESH,
            track_buffer=int(track_buffer),
            match_thresh=BT_MATCH_THRESH,
            fuse_score=BT_FUSE_SCORE,
        )
        _orig_reset = _basetrack.BaseTrack.reset_id
        _basetrack.BaseTrack.reset_id = staticmethod(lambda: None)
        try:
            self._bt = BYTETracker(args)
        finally:
            _basetrack.BaseTrack.reset_id = _orig_reset
        self.tracks: list[_TrackBuf] = []
        self._bufs: dict[int, _TrackBuf] = {}

    def _buf(self, track_id: int) -> _TrackBuf:
        buf = self._bufs.get(track_id)
        if buf is None:
            buf = _TrackBuf(track_id, self.aspect)
            self._bufs[track_id] = buf
            self.tracks.append(buf)
        return buf

    def _to_boxes(self, dets: list[tuple[tuple[float, float, float, float], float]]) -> Any:
        """Convert normalized detections into the ``Boxes`` ByteTrack needs (isometric coordinates, cls always 0)."""
        import torch
        from ultralytics.engine.results import Boxes

        a = self.aspect
        if dets:
            data = torch.tensor(
                [[b[0] * a, b[1], b[2] * a, b[3], c, 0.0] for b, c in dets],
                dtype=torch.float32,
            )
        else:
            data = torch.zeros((0, 6), dtype=torch.float32)
        return Boxes(data, orig_shape=(1, 1))

    def update(
        self,
        frame_idx: int,
        time: float,
        dets: list[tuple[tuple[float, float, float, float], float]],
    ) -> None:
        """Update ByteTrack with the current sampled frame's detections and write the results into :class:`_TrackBuf`."""
        out = self._bt.update(self._to_boxes(dets), None)
        if out is None or len(out) == 0:
            return
        a = self.aspect
        for row in out:
            box = (
                _clip01(float(row[0]) / a), _clip01(float(row[1])),
                _clip01(float(row[2]) / a), _clip01(float(row[3])),
            )
            if box[2] - box[0] <= 1e-6 or box[3] - box[1] <= 1e-6:
                continue
            self._buf(int(row[4])).observe(frame_idx, time, box, float(row[5]))


# ------------------------------------------------------------------ track merging


def _junction_ok(prev: _TrackBuf, nxt: _TrackBuf, aspect: float) -> bool:
    """Determine whether the "seam" between two tracks looks like the same person.

    Two cases:
      * **Separated in time**: extrapolate the earlier segment to the start time of the later one
        using velocity; the position must fall within :data:`MERGE_MAX_DIST` and the scales must be close;
      * **Overlapping in time**: over the overlapping period, most boxes of the two tracks must
        "look like the same person" (this is the case when one player is simultaneously tracked as two tracks).
    """
    if prev.times[-1] < nxt.times[0]:
        if nxt.times[0] - prev.times[-1] > MERGE_MAX_GAP:
            return False
        c_prev = prev.predict(nxt.times[0]) or _box_center(prev.boxes[-1])
        c_new = _box_center(nxt.boxes[0])
        dist = math.hypot((c_prev[0] - c_new[0]) * aspect, c_prev[1] - c_new[1])
        if dist > MERGE_MAX_DIST:
            return False
        a_prev = _box_area(prev.boxes[-1])
        a_new = _box_area(nxt.boxes[0])
        return min(a_prev, a_new) > 1e-9 and max(a_prev, a_new) / min(a_prev, a_new) <= MERGE_AREA_RATIO

    # Time overlap: sample the observations of the shorter track that fall within the overlap interval
    o0 = max(prev.times[0], nxt.times[0])
    o1 = min(prev.times[-1], nxt.times[-1])
    if o1 < o0:
        return False
    short, other = (prev, nxt) if prev.n <= nxt.n else (nxt, prev)
    # Use binary search to locate the nearest neighbor. The original `min(range(other.n), key=...)` was O(n),
    # and applied to "every track compared against every observation of the short track" it becomes O(n²);
    # 30 minutes of footage (hundreds of tracks, each with thousands of observations) would peg the CPU for tens of minutes.
    other_times = other.times
    hit = 0
    total = 0
    for k, t in enumerate(short.times):
        if t < o0 or t > o1:
            continue
        total += 1
        j = bisect.bisect_left(other_times, t)
        best = -1
        best_d = 1e9
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < other.n:
                d = abs(other_times[cand] - t)
                if d < best_d:
                    best_d, best = d, cand
        if best < 0 or best_d > 0.2:
            continue
        box = short.boxes[k]
        o_box = other.boxes[best]
        c1, c2 = _box_center(box), _box_center(o_box)
        dist = math.hypot((c1[0] - c2[0]) * aspect, c1[1] - c2[1])
        a1, a2 = _box_area(box), _box_area(o_box)
        ratio = max(a1, a2) / max(1e-9, min(a1, a2))
        if dist <= MERGE_MAX_DIST and ratio <= MERGE_AREA_RATIO:
            hit += 1
    return total > 0 and hit / total >= MERGE_OVERLAP_RATIO


def _merge_tracks(bufs: list[_TrackBuf], aspect: float) -> list[_TrackBuf]:
    """Merge "two track segments of the same person that were split apart" into one.

    This is the last safeguard against "a player being tracked as two tracks", handling two cases:

    1. The player briefly leaves the frame / is occluded and then reappears (a gap in time);
    2. The same player is tracked as two parallel tracks at the same moment (caused by the box shape changing back and forth).

    Every merge must pass :func:`_junction_ok`'s seam check: if position and scale don't match, don't merge,
    to avoid gluing two different people nearby into one "mishmash".

    Note: already "ended" tracks (which stopped updating after a long mismatch) also participate in merging —
    their data is valid, and they are exactly the "player leaves the frame and comes back" cases that need reconnection.

    Performance: first coarse-filter by "time interval + spatial bounding box" to avoid pairwise seam checks
    on all tracks (hundreds of thousands of comparisons are very slow on long videos).
    """
    live = [t for t in bufs if t.n > 0]
    live.sort(key=lambda t: (t.times[0], -t.n))

    # Precompute each track's spatio-temporal extent for coarse filtering
    meta: list[tuple[float, float, float, float, float, float]] = []
    for t in live:
        xs = [b[0] for b in t.boxes]
        ys = [b[1] for b in t.boxes]
        ws = [b[2] for b in t.boxes]
        hs = [b[3] for b in t.boxes]
        meta.append((t.times[0], t.times[-1], min(xs), min(ys), max(xs), max(ys)))
        del ws, hs

    merged: list[_TrackBuf] = []
    merged_meta: list[tuple[float, float, float, float, float, float]] = []
    for tr, mt in zip(live, meta):
        target: _TrackBuf | None = None
        for idx, prev in enumerate(merged):
            pm = merged_meta[idx]
            # Gap in time too long -> skip
            if pm[1] < tr.times[0] and tr.times[0] - pm[1] > MERGE_MAX_GAP:
                continue
            # Entirely later than the other in time -> cannot possibly connect (live is already sorted by start time)
            if pm[0] > mt[1]:
                continue
            # Spatial bounding boxes too far apart -> skip (leaving 3x tolerance)
            tol = MERGE_MAX_DIST * 3.0
            if (mt[2] - pm[4] > tol or pm[2] - mt[4] > tol or
                    mt[3] - pm[5] > tol or pm[3] - mt[5] > tol):
                continue
            if _junction_ok(prev, tr, aspect):
                target = prev
                break
        if target is None:
            merged.append(tr)
            merged_meta.append(mt)
        else:
            target.absorb(tr)
            i = merged.index(target)
            old = merged_meta[i]
            merged_meta[i] = (
                min(old[0], mt[0]), max(old[1], mt[1]),
                min(old[2], mt[2]), min(old[3], mt[3]),
                max(old[4], mt[4]), max(old[5], mt[5]),
            )
    return merged


# ------------------------------------------------------------------ summarization and identification


class _Feat:
    """Derived features used to identify "match players" (internal use, not part of the public data structures)."""

    __slots__ = ("area_p90", "cx_mean", "x_range", "duration", "static_ratio")

    def __init__(self, area_p90: float, cx_mean: float, x_range: float,
                 duration: float, static_ratio: float) -> None:
        self.area_p90 = area_p90
        self.cx_mean = cx_mean
        self.x_range = x_range
        self.duration = duration
        self.static_ratio = static_ratio


def _track_feat(track: PlayerTrack, fps: float) -> _Feat:
    """Compute the derived features needed for identification from a track.

    * ``area_p90``: 90th percentile of box area (more stable than mean/max, see :func:`_select_active_players`);
    * ``cx_mean`` / ``x_range``: mean and spread of the horizontal position (identifies "people who stay in the same small area");
    * ``static_ratio``: fraction of low-speed frames (identifies "people who stand still for a long time").
    """
    areas = np.asarray([_box_area(b) for b in track.boxes], dtype=np.float64)
    cxs = np.asarray([0.5 * (b[0] + b[2]) for b in track.boxes], dtype=np.float64)
    sp = np.asarray(track.speeds[1:], dtype=np.float64) if len(track.speeds) > 1 else np.zeros(0)
    return _Feat(
        area_p90=float(np.percentile(areas, 90)) if areas.size else 0.0,
        cx_mean=float(cxs.mean()) if cxs.size else 0.5,
        x_range=float(cxs.max() - cxs.min()) if cxs.size else 0.0,
        duration=float(track.times[-1] - track.times[0]) if track.times else 0.0,
        static_ratio=float(np.mean(sp < 0.05)) if sp.size else 1.0,
    )


def _summarize(bufs: list[_TrackBuf], fps: float) -> list[PlayerTrack]:
    """Convert internal track buffers into public PlayerTrack objects and compute various statistics."""
    out: list[PlayerTrack] = []
    for tk in bufs:
        if tk.n == 0:
            continue
        # Merging may leave multiple observations for the same frame: keep only the highest-confidence one,
        # ensuring each frame appears only once in frames / times.
        best: dict[int, int] = {}
        for k, f in enumerate(tk.frames):
            j = best.get(f)
            if j is None or tk.confs[k] > tk.confs[j]:
                best[f] = k
        order = sorted(best.values(), key=lambda k: tk.times[k])
        frames = [tk.frames[k] for k in order]
        times = [tk.times[k] for k in order]
        boxes = [tk.boxes[k] for k in order]
        speeds = [tk.speeds[k] for k in order]
        confs = [tk.confs[k] for k in order]

        areas = np.asarray([_box_area(b) for b in boxes], dtype=np.float64)
        track = PlayerTrack(
            track_id=tk.track_id,
            frames=frames,
            times=times,
            boxes=boxes,
            speeds=speeds,
            confidences=confs,
        )
        track.mean_area = float(areas.mean())
        track.max_area = float(areas.max())
        # Speed statistics ignore frame 0 (no history, always 0) to avoid dragging the mean down
        sp = np.asarray(speeds[1:], dtype=np.float64) if len(speeds) > 1 else np.zeros(0)
        track.mean_speed = float(sp.mean()) if sp.size else 0.0
        track.max_speed = float(sp.max()) if sp.size else 0.0
        # Cumulative travel: sum of box-center distances between adjacent appearance frames (normalized by frame height)
        travel = 0.0
        for k in range(1, len(times)):
            p0, p1 = _box_center(boxes[k - 1]), _box_center(boxes[k])
            dt = times[k] - times[k - 1]
            if dt > 4.0 / fps:      # frames were skipped in between, so it does not count as continuous travel
                continue
            travel += math.hypot((p1[0] - p0[0]) * tk.aspect, p1[1] - p0[1])
        track.total_travel = float(travel)
        out.append(track)
    return out


def _box_xyxy(item: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Unify the two representations of a "per-frame box" into ``(x1, y1, x2, y2)``.

    Two exist in this project at the same time:

    * ``PlayerSignal.frame_boxes``: ``(track_id, x1, y1, x2, y2)`` (with a track id,
      since downstream needs to get boxes per person);
    * per-frame boxes from the detection stage: ``(x1, y1, x2, y2)``.

    Both must be accepted here. **This is a real pitfall we hit**: ``_box_size_stats`` originally
    only recognized 5-tuples (using ``b[4] - b[2]`` for box height), while ``analyze_players``
    passed in 4-tuples, so "roughly how big are the players in this match" could never be computed
    (``ref`` was always 0), and the adaptive size hard threshold **never took effect** in the real
    pipeline — unit tests fed 5-tuples, so it went unnoticed all along. Without that threshold,
    spectators and people from neighboring courts keep squeezing into the candidates.
    """
    n = len(item)
    try:
        if n >= 5:
            return (float(item[1]), float(item[2]), float(item[3]), float(item[4]))
        if n >= 4:
            return (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
    except (TypeError, ValueError):
        return None
    return None


def _box_size_stats(boxes: list, aspect: float = 1.7778) -> dict[str, float]:
    """Estimate "roughly how big the players are in this match" from per-frame player boxes.

    Match players are the largest group of people in the frame, so the 90th percentile of the
    **largest box per frame** is used as the reference scale, rather than the global median
    (background people far outnumber players, so the median is dragged down by them).
    This approach is insensitive to camera angle: no matter how high the camera is or where the
    court is in the frame, the largest box is always the person closest to the camera.

    Returns:
        ``{"ref": reference box height, "min_abs": absolute lower bound, "max_abs": absolute upper bound}``
        (all in frame-height units).
    """
    per_frame_max: list[float] = []
    for fr in boxes:
        if not fr:
            continue
        hs = []
        for item in fr:
            b = _box_xyxy(item)
            if b is not None:
                hs.append(abs(b[3] - b[1]))
        if hs:
            per_frame_max.append(max(hs))
    if len(per_frame_max) < 5:
        return {"ref": 0.0, "min_abs": 0.0, "max_abs": 1.0}
    arr = np.asarray(per_frame_max, dtype=np.float64)
    ref = float(np.percentile(arr, 90))
    # The reference scale must be at least 6% of the frame height; otherwise this whole segment
    # detected nobody who looks like a player, and it should not be used for hard filtering (it would drop everyone)
    if ref < 0.06:
        return {"ref": 0.0, "min_abs": 0.0, "max_abs": 1.0}
    return {
        "ref": ref,
        "min_abs": max(0.03, ref * 0.42),
        # The upper bound is set very loose (4x the reference scale): in close-ups a player occupies a large area,
        # here we only want to block false detections of "a whole person right up against the lens"
        "max_abs": min(1.0, ref * 4.0),
    }


def _build_size_stats(size_filter: SizeFilter,
                      samples: list[list[float]],
                      dropped: int,
                      frames: int,
                      ref_scale: float) -> dict[str, Any]:
    """Organize "detected boxes" into statistics the UI can use directly (histogram + samples + counts).

    Args:
        samples: one entry per box, ``[box height, box area, largest box height in the same frame]``.
        dropped: number of boxes cut by size filtering.
        frames: number of sampled frames contributing to the statistics.
        ref_scale: adaptive reference scale (90th percentile of the largest box height per frame).

    The statistics cover **all boxes before filtering** (only geometric thresholds and ROI applied).
    That is the only way the UI can depict "how much a threshold drag will cut" — if only filtered
    boxes were counted, the user would never see the dropped part and would have no basis for tuning.

    The samples include "largest box height in the same frame" and area, so the frontend can
    **precisely** simulate any threshold under both modes locally: dragging the slider does not
    require re-running detection on every move.
    """
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, 3) if samples else np.zeros((0, 3))
    h = arr[:, 0] if arr.size else np.zeros(0)
    total = int(h.size)
    edges = np.linspace(0.0, SIZE_HIST_MAX, SIZE_HIST_BINS + 1)
    if total:
        hist, _ = np.histogram(np.clip(h, 0.0, SIZE_HIST_MAX), bins=edges)
        overflow = int(np.sum(h > SIZE_HIST_MAX))
    else:
        hist = np.zeros(SIZE_HIST_BINS, dtype=np.int64)
        overflow = 0
    step = max(1, total // SIZE_SAMPLE_MAX) if total else 1
    sample = [[round(float(v), 5) for v in arr[i]] for i in range(0, total, step)][:SIZE_SAMPLE_MAX]
    return {
        **size_filter.as_dict(),
        "active": bool(size_filter.active),
        "total": total,
        "kept": int(total - dropped),
        "dropped": int(dropped),
        "frames": int(frames),
        # Adaptive reference scale (90th percentile of the largest box per frame): the "1.0" of the relative mode is here
        "ref": round(float(ref_scale), 5),
        "bins": SIZE_HIST_BINS,
        "hist_max": SIZE_HIST_MAX,
        "hist": [int(v) for v in hist],
        "overflow": overflow,
        "hist_median": round(float(np.median(h)), 5) if total else 0.0,
        "hist_p90": round(float(np.percentile(h, 90)), 5) if total else 0.0,
        #: Each entry [box height, box area, largest box height in the same frame]
        "sample": sample,
    }


def _median_box(track: PlayerTrack) -> tuple[float, float, float, float]:
    """Representative box of a track: the per-coordinate median of the per-frame boxes.

    Median rather than mean: a player's box suddenly stretches when striding / lunging to the net,
    and the mean gets pulled off by those instants, whereas size filtering cares about "how big
    this person usually is". Taking the median per coordinate also guarantees the result is still a
    valid rectangle (taking the median of areas would produce a box that does not exist).
    """
    if not track.boxes:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(track.boxes, dtype=np.float64).reshape(-1, 4)
    med = np.median(arr, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]), float(med[3]))


def _track_median_size(track: PlayerTrack, aspect: float = 1.7778) -> float:
    """Typical size of a track: the larger of box height and box width (in frame-height units).

    Taking the larger of "height / width" adapts to overhead shots: seen from above, people are
    "short and wide", and looking only at box height would misjudge these players as small targets
    and filter them out.
    """
    if not track.boxes:
        return 0.0
    vals: list[float] = []
    for b in track.boxes:
        h = abs(float(b[3]) - float(b[1]))
        w = abs(float(b[2]) - float(b[0])) * max(0.2, float(aspect))
        vals.append(max(h, w))
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _select_active_players(
    tracks: list[PlayerTrack],
    fps: float,
    total_duration: float,
    max_players: int = 4,
    viewpoint: str = "unknown",
    boxes: list | None = None,
    aspect: float = 1.7778,
    size_filter: SizeFilter | None = None,
) -> list[PlayerTrack]:
    """Pick out "the players currently in a match" and write each one's ``active_score``.

    Criteria (each corresponds to a phenomenon observed in measurements):

    1. **Large box** (highest weight for a low camera): match players are close to the camera and
       are the two largest person boxes in the frame. Uses the "90th percentile of box area" rather
       than the mean — the box suddenly grows at the instant of a stride/lunge (measured area
       differences of 5~10x), the mean gets flattened by the many "ordinary standing frames", and
       max is too easily thrown off by a single-frame false detection. Finally it is divided by the
       90th percentile over all candidate tracks for robust normalization.
    2. **Moves fast**: running/striding/jumping makes their mean_speed, max_speed far higher than
       people standing and watching and spectators; again normalized robustly by the 90th percentile,
       with sample-size shrinkage by observation count (an estimate from a fragment lasting only a
       second or two is unreliable and must be discounted).
    3. **Presence duration** (weak weight): background passers-by may flash by, but spectators also
       persist for a long time, so duration is given only a small weight and is not used alone.
    4. **Penalties**:
       - "Slow for a long time": people who just stand there (spectators, resting players) are
         penalized even if their box is not small;
       - "Nearly fixed horizontal position": staff who keep walking back and forth over a small
         area at the sideline are penalized;
       - "Long pinned to the left/right frame edge": fisheye edges magnify distant people, so
         people near the edge (mostly spectators sitting courtside / against the wall) are penalized.

    **"Box big enough" is a hard threshold, not a bonus.** Match players must be clearly larger than
    everyone else: in measured footage a real player's box height is about 0.16~0.22 (16%~22% of
    the frame height), while spectators and people on neighboring courts are only 0.02~0.06. With a
    weighted sum alone, a track with a "tiny box but always moving" (spectators walking, warm-ups on
    a neighboring court) could squeeze into the top few on speed score. When ``boxes`` (per-frame
    boxes) is given, a "typical player size" is computed — the median of the largest cluster of box
    heights across all tracks — and tracks that are clearly too small are excluded outright.

    **Camera dependent**: "largest box = closest to camera" only holds for a low rear/side camera.
    With a high camera or overhead shot, everyone is roughly the same size in the frame, and then
    **motion** is the only reliable criterion, so the weights switch to "motion dominant, area only
    a minor adjustment"; the "edge penalty" is also disabled then (in an overhead shot the court
    may naturally be off to one side of the frame).

    Finally take by score from high to low: usually 2 tracks; if the 3rd and 4th are in the same
    tier as the 1st and clearly above the rest, take 4 (supports doubles). **When one player is
    split into multiple tracks, they count as one person**, leaving the slot for another player;
    a fragment is included only when it clearly extends this player's time coverage (otherwise the
    player would "vanish" during some periods and downstream cropping would have nothing to follow).
    """
    if not tracks:
        return []
    top_view = viewpoint in ("overhead", "elevated")
    w_area, w_speed, w_dur = ((0.20, 0.65, 0.15) if top_view else (0.45, 0.40, 0.15))

    # Tracks that are too short (under about 0.4 seconds) are not candidates at all
    min_frames = max(3, int(round(0.4 * fps)))
    cand = [t for t in tracks if t.n >= min_frames]
    if not cand:
        cand = sorted(tracks, key=lambda t: t.n, reverse=True)[:2]
    if not cand:
        return []

    feats: dict[int, _Feat] = {t.track_id: _track_feat(t, fps) for t in cand}

    # ---- Size hard threshold: exclude "too small to be a match player" early.
    # The reference size uses "the largest cluster of box heights among candidates" — match players
    # are the largest people in the frame, so this reference points at the real players' scale under any camera angle.
    size = _box_size_stats(boxes, aspect) if boxes else None
    size_ref = float(size["ref"]) if size else 0.0
    if size_filter is not None and size_filter.active:
        # The user explicitly specified size filtering: it takes priority over the adaptive threshold
        # (the adaptive threshold gets hijacked by "spectators closer to the camera", which is exactly
        # the scenario the user wants to intervene in manually).
        # The track-level box is the median of the per-frame boxes, so judge it with the same thresholds directly.
        kept = [t for t in cand if size_filter.keep(_median_box(t), size_ref)]
        if kept:
            cand = kept
            feats = {t.track_id: _track_feat(t, fps) for t in cand}
    elif size is not None and size_ref > 0:
        lo = max(size["min_abs"], size_ref * 0.42)
        sized = [t for t in cand if _track_median_size(t, aspect) >= lo]
        if len(sized) >= 1:
            dropped = len(cand) - len(sized)
            if dropped:
                cand = sized
                feats = {t.track_id: _track_feat(t, fps) for t in cand}

    areas = np.asarray([feats[t.track_id].area_p90 for t in cand], dtype=np.float64)
    speeds = np.asarray([t.mean_speed for t in cand], dtype=np.float64)
    peaks = np.asarray([t.max_speed for t in cand], dtype=np.float64)
    area_ref = max(_pct(areas, 90), 1e-6)
    speed_ref = max(_pct(speeds, 90), 1e-6)
    peak_ref = max(_pct(peaks, 90), 1e-6)
    dur_ref = max(total_duration, 1e-6)

    for t in cand:
        f = feats[t.track_id]
        # Sample-size shrinkage: the fewer times a track is seen, the less trustworthy its speed/area estimates
        shrink = t.n / (t.n + 2.0 * fps)
        # --- positive score ---
        area_score = _clip01(f.area_p90 / area_ref)
        speed_score = shrink * _clip01(
            0.65 * t.mean_speed / speed_ref + 0.35 * t.max_speed / peak_ref
        )
        duration_score = _clip01(f.duration / dur_ref)
        # --- penalties ---
        penalty_static = 0.35 * _clip01(f.static_ratio)                  # slow for a long time
        penalty_fixed = 0.25 if (f.x_range < 0.06 and f.duration > 0.5 * dur_ref) else 0.0
        # With a high camera / overhead shot the court may naturally be off to one side of the frame, so being at the edge does not mean spectator
        edge = _clip01((abs(f.cx_mean - 0.5) - 0.42) / 0.08)             # pinned to the frame edge
        penalty_edge = 0.0 if top_view else 0.30 * edge

        score = (
            w_area * area_score
            + w_speed * speed_score
            + w_dur * duration_score
            - penalty_static
            - penalty_fixed
            - penalty_edge
        )
        t.active_score = _clip01(score)

    ranked = sorted(cand, key=lambda t: t.active_score, reverse=True)

    # Post-hoc size check: drop any whole track whose "size differs too much from the top one".
    # A weighted sum can always let a "tiny box but always moving" track squeeze in (spectators
    # walking, warm-ups on a neighboring court), so cut once more by size — match players do not
    # differ from each other by more than 2x in size.
    if size is not None and size["ref"] > 0 and len(ranked) > 1:
        ref_size = _track_median_size(ranked[0], aspect)
        if ref_size > 0:
            keep = [t for t in ranked
                    if _track_median_size(t, aspect) >= max(size["min_abs"], ref_size * 0.45)]
            if keep:
                ranked = keep

    best = ranked[0].active_score
    if best <= 1e-6:
        return ranked[:1]

    def _same_player(a: PlayerTrack, b: PlayerTrack) -> bool:
        """Whether two tracks are "the same player" (overlap in time + close in position/scale)."""
        fa, fb = feats[a.track_id], feats[b.track_id]
        overlap = min(a.times[-1], b.times[-1]) - max(a.times[0], b.times[0])
        if overlap <= 0.0:
            return False
        if abs(fa.cx_mean - fb.cx_mean) > 0.08:
            return False
        return max(fa.area_p90, fb.area_p90) / max(1e-9, min(fa.area_p90, fb.area_p90)) <= 2.5

    # "Player groups": the same player may be split into multiple tracks, but they still count as
    # one person. A fragment is included only when it is mostly "new time coverage" (periods when
    # this player was not tracked before) — so the player does not "vanish" during some period and
    # downstream crop tracking does not lose them midway.
    EXTEND_MIN = 2.0
    groups: list[dict[str, Any]] = []
    picked: list[PlayerTrack] = []
    for t in ranked:
        if len(picked) >= max_players:
            break
        f = feats[t.track_id]
        t0, t1 = t.times[0], t.times[-1]
        grp = next((g for g in groups if _same_player(t, g["rep"])), None)
        if grp is not None:
            gain = max(0.0, grp["t0"] - t0) + max(0.0, t1 - grp["t1"])
            span = max(1e-6, t1 - t0)
            # It must add appreciable time and also cover more than half of this track's own
            # duration; otherwise it is just a duplicate fragment of the same player and should not
            # take up a "player slot".
            if gain < EXTEND_MIN or gain < 0.5 * span:
                continue
        # Moving + non-small box is a necessary condition for a "match player"
        moving = (t.mean_speed / speed_ref) >= 0.20 and (f.area_p90 / area_ref) >= 0.20
        if not moving:
            continue
        # From the 2nd on: must be in the same tier as the 1st (0.55x or more) and the absolute score must not be too low
        if picked and t.active_score < max(0.55 * best, 0.30):
            break
        picked.append(t)
        if grp is None:
            groups.append({"rep": t, "t0": t0, "t1": t1})
        else:
            grp["t0"] = min(grp["t0"], t0)
            grp["t1"] = max(grp["t1"], t1)
    return picked


#: Window length and step (seconds) when re-picking match players per time window. The window must
#: be much longer than "one substitution / one occlusion" so the player has enough observations
#: inside it; the step is half of it so there is no seam at the boundary where nobody is selected.
_ACTIVE_WINDOW_S = 180.0
_ACTIVE_WINDOW_STEP_S = 90.0
#: Maximum number of tracks allowed to remain after per-window merging (safety valve to keep background false detections from blowing up frame_boxes).
_ACTIVE_MAX_TRACKS = 64


def _select_active_players_windowed(
    tracks: list[PlayerTrack],
    fps: float,
    duration: float,
    max_players: int = 4,
    viewpoint: str = "unknown",
    boxes: list | None = None,
    aspect: float = 1.7778,
    size_filter: SizeFilter | None = None,
) -> list[PlayerTrack]:
    """Pick match players **window by window in time**, then merge the results.

    Why not pick only once for the whole video: a measured 30-minute clip has 285 tracks — the
    player gets occluded midway, walks out of frame, or overlaps with a background person, and the
    tracker gives them a new id. Meanwhile :func:`_select_active_players` ranks by **whole-video**
    statistics, so the top 4 are very likely all in the middle of the video. The consequences are
    severe: ``PlayerSignal.frame_boxes`` is empty for the rest of the time, so

    * ``active_count`` / ``player_motion`` are 0 for entire segments (a measured first 372 seconds were constantly 0);
    * at segmentation time player coverage is only 0.48, which kills the "player-motion segmentation"
      path outright and falls back to "whole-frame activity" — and the latter never collapses in a
      multi-court gym, so rallies get glued into segments dozens of seconds long.

    Approach: slice into windows of ``_ACTIVE_WINDOW_S`` (step half, so nothing is missed at the
    boundaries); each window ranks only the tracks that **overlap in time with that window**, then
    the selected ids from all windows are unioned. The whole-video pass still runs: its ``size_ref``
    is the global reference of "how big the largest person in the frame is", used as the final size
    gate to keep a window from selecting distant spectators.

    When ``duration`` is shorter than one window, behavior is exactly the same as the old implementation.
    """
    if not tracks:
        return []
    if duration <= _ACTIVE_WINDOW_S:
        return _select_active_players(tracks, fps, duration, max_players=max_players,
                                      viewpoint=viewpoint, boxes=boxes, aspect=aspect,
                                      size_filter=size_filter)

    # Whole-video pass: get the "reference size" and a set of baseline ids
    base = _select_active_players(tracks, fps, duration, max_players=max_players,
                                  viewpoint=viewpoint, boxes=boxes, aspect=aspect,
                                  size_filter=size_filter)
    base_ids = {t.track_id for t in base}

    gsize = _box_size_stats(boxes, aspect) if boxes else None
    g_ref = float(gsize["ref"]) if gsize else 0.0
    g_min = float(gsize["min_abs"]) if gsize else 0.0
    # Tracks picked per window must also pass the same "big enough" gate: discard only when they are
    # clearly smaller than the whole-video reference size (meaning distant spectators / people on a neighboring court).
    size_floor = max(g_min, g_ref * 0.45) if g_ref > 0 else 0.0

    n_win = max(1, int(np.ceil((duration - _ACTIVE_WINDOW_S)
                               / _ACTIVE_WINDOW_STEP_S))) + 1
    chosen: dict[int, PlayerTrack] = {t.track_id: t for t in base}
    for w in range(n_win):
        t0 = w * _ACTIVE_WINDOW_STEP_S
        t1 = min(duration, t0 + _ACTIVE_WINDOW_S)
        if t1 - t0 < 8.0:
            continue
        sub = [t for t in tracks
               if t.n > 0 and t.times[-1] >= t0 and t.times[0] <= t1]
        if not sub:
            continue
        sub_boxes = None
        if boxes:
            i0 = max(0, int(t0 * fps))
            i1 = min(len(boxes), max(i0 + 1, int(t1 * fps)))
            sub_boxes = boxes[i0:i1]
        try:
            picked = _select_active_players(sub, fps, t1 - t0,
                                            max_players=max_players, viewpoint=viewpoint,
                                            boxes=sub_boxes, aspect=aspect,
                                            size_filter=size_filter)
        except Exception:
            continue
        for t in picked:
            if t.track_id in chosen:
                continue
            if size_floor > 0 and _track_median_size(t, aspect) < size_floor:
                continue
            chosen[t.track_id] = t

    out = sorted(chosen.values(), key=lambda t: (t.times[0] if t.n else 0.0))
    if len(out) > _ACTIVE_MAX_TRACKS:
        # Too many means background false detections mixed in: prefer keeping the whole-video baseline + the longest-appearing ones
        out = sorted(out, key=lambda t: (t.track_id not in base_ids, -t.n))
        out = out[: _ACTIVE_MAX_TRACKS]
    return out


# ------------------------------------------------------------------ main entry point


def analyze_players(
    video_path: str,
    sample_fps: float = 15.0,
    roi: tuple[float, float, float, float] | None = None,   # normalized x0,y0,x1,y1; only person-box bottom-center inside counts as candidate
    max_seconds: float = 0.0,      # 0 = whole video
    model_name: str = "yolo11n.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    batch_hint: int = 16,
    viewpoint: str = "unknown",
    roi_poly: list[list[float]] | None = None,   # normalized court polygon; more precise than roi
    size_filter: "SizeFilter | dict | None" = None,
    on_progress: Progress | None = None,
    cancel: Any = None,            # callable() -> bool
) -> PlayerSignal:
    """Detect and track people in the video and pick out the players in a match.

    Args:
        video_path: Video path (may be a low-resolution proxy video, which is faster).
        sample_fps: Frame rate for sampled-frame analysis; the output timeline also uses it.
        roi: Normalized (x0, y0, x1, y1). When given, only boxes whose **bottom-edge center** falls
            inside it are candidates (used to coarsely filter out the stands / people on other courts).
            The pipeline passes this automatically when court calibration succeeds, equivalent to
            "only look for people on this court".
        roi_poly: Normalized polygon (4~24 points). When given it **replaces** ``roi`` for in-court
            determination: panoramic / fisheye footage has a curved court boundary, and judging with
            the bounding rectangle would count the large area beyond the curve (often the stands) as
            "in court".
        size_filter: :class:`SizeFilter` or an equivalent dict; during detection, filter out boxes
            that are "clearly not players in this match" by person-box height/area. With ``None`` /
            ``mode="off"`` behavior is exactly the same as the old version.
        max_seconds: Only analyze the first few seconds; 0 means the entire video.
        model_name: YOLO weight name or path; prefers weights under the local ``models/``.
        imgsz: Inference input size.
        conf: Detection confidence threshold (tracking automatically relaxes it to 0.15 to pass through brief occlusion).
        device: ``cuda`` / ``cpu``; automatically falls back when unavailable.
        batch_hint: Number of frames sent to the GPU at a time.
        viewpoint: Camera angle type (``rear`` / ``side`` / ``elevated`` / ``overhead`` /
            ``unknown``). Affects two things: the geometric thresholds (in an overhead shot people
            are "short and wide") and the weights for "picking match players" (with a high camera
            everyone is the same size, so only motion can distinguish them).
        on_progress: ``callable(progress: float, stage: str)`` progress callback.
        cancel: ``callable() -> bool``; when it returns True, abort as soon as possible and return the part already analyzed.

    Returns:
        PlayerSignal (``size_stats`` carries the pre-filter box-height distribution, which the UI uses for tuning)
    """
    import cv2

    if not video_path:
        raise ValueError(tr("players.empty_video_path"))
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    sf = size_filter_from(size_filter)

    cfg_dir, models_dir = _data_paths()
    _prepare_yolo_env(cfg_dir)
    weights = _resolve_weights(model_name, models_dir)
    dev = _pick_device(device)

    def _report(p: float, stage: str) -> None:
        if on_progress is not None:
            on_progress(_clip01(p), stage)

    def _cancelled() -> bool:
        try:
            return bool(cancel is not None and cancel())
        except Exception:
            return False

    _report(0.0, tr("players.prepare_model"))

    # ---- open the video, compute the sampling step ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0 or not np.isfinite(src_fps):
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, probe = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"无法读取视频帧: {video_path}")
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    aspect = float(width) / float(max(1, height))

    step = max(1, int(round(src_fps / max(1e-3, sample_fps))))
    eff_fps = src_fps / step                      # actual sampling frame rate
    step_dt = 1.0 / eff_fps
    limit = int(round(max_seconds * src_fps)) if max_seconds > 0 else (total_frames or 10 ** 9)
    batch_size = max(1, int(batch_hint))

    # ---- load the model ----
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        cap.release()
        raise RuntimeError(f"无法导入 ultralytics: {exc}") from exc

    model = YOLO(weights)
    # Tracking needs looser boxes (to pass through brief occlusion), but anything below 0.15 is not trusted
    det_conf = float(min(conf, LOW_CONF))

    tracker = _ByteTracker(fps=eff_fps, aspect=aspect)

    def _predict(frames: list[np.ndarray]) -> list[Any]:
        """Batch inference; on CUDA failure, fall back to CPU and retry once."""
        nonlocal dev
        try:
            return model.predict(
                frames, classes=[0], conf=det_conf, imgsz=imgsz,
                device=dev, verbose=False,
            )
        except Exception:
            if dev != "cpu":
                dev = "cpu"
                return model.predict(
                    frames, classes=[0], conf=det_conf, imgsz=imgsz,
                    device="cpu", verbose=False,
                )
            raise

    # ---- frame sampling + batch inference ----
    roi_arr = None
    if roi is not None:
        roi_arr = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
    poly_arr = None
    if roi_poly:
        p = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
        p = p[np.all(np.isfinite(p), axis=1)]
        if p.shape[0] >= 3:
            poly_arr = p
            # When both polygon and rectangle are given, the polygon wins: it is the true shape of the court
            roi_arr = None

    def _passes_roi(box: tuple[float, float, float, float]) -> bool:
        """Whether the box's bottom-edge center falls inside the court.

        If a court polygon is available use it (mandatory for curved-boundary footage), otherwise
        fall back to the bounding rectangle. Bottom-edge center rather than box center: the upper
        half of a person box is often outside the court (raised arm, jump, or the box also captured
        the scoreboard), and judging by center would exclude real players.
        """
        bcx = 0.5 * (box[0] + box[2])
        by = box[3]
        if poly_arr is not None:
            # ROI_POLY_MARGIN: a player standing exactly on the boundary line counts as "in court" (see point_in_poly)
            return bool(point_in_poly(np.asarray([[bcx, by]], dtype=np.float32),
                                      poly_arr, margin=ROI_POLY_MARGIN)[0])
        if roi_arr is None:
            return True
        return (roi_arr[0] <= bcx <= roi_arr[2]) and (roi_arr[1] <= by <= roi_arr[3])

    def _dets_of(result: Any) -> list[tuple[tuple[float, float, float, float], float]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cf = boxes.conf.cpu().numpy()
        out: list[tuple[tuple[float, float, float, float], float]] = []
        for k in range(len(boxes)):
            c = float(cf[k])
            if c < LOW_CONF:             # boxes with too low confidence are ignored outright
                continue
            b = _sanitize_box(
                float(xyxy[k, 0]) / width, float(xyxy[k, 1]) / height,
                float(xyxy[k, 2]) / width, float(xyxy[k, 3]) / height,
                viewpoint,
            )
            if b is None or not _passes_roi(b):
                continue
            out.append((b, c))
        # Near-overlapping duplicate boxes within the same frame: keep only the highest-confidence one
        return _dedup_dets(out, aspect)

    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    proc = 0                     # number of sampled frames processed
    read = 0                     # number of raw frames grabbed
    idx = 0
    # Raw detection boxes per sampled frame (normalized, including background people). Used to
    # adaptively estimate "roughly how big the players are in this match", see `_box_size_stats`.
    det_frames: list[list[tuple[float, float, float, float]]] = []
    # ---- size-filter statistics (counting **pre-filter** boxes, so the UI can depict "what was cut")
    #: Each entry [box height, box area, this frame's reference box height]
    size_samples: list[list[float]] = []
    size_dropped = 0
    size_frames = 0

    def _apply_size_filter(dets: list[tuple[tuple[float, float, float, float], float]],
                           ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Filter out unsuitable person boxes by size and record the statistics (updating the accumulators above in place).

        The ``relative`` mode needs "the largest box height in the same frame" as the denominator:
        here the maximum over **all candidate boxes in this frame** is used. It must be this frame's,
        not the whole video's: detection box size varies with a person's distance, and using a
        whole-video reference would pass everyone in a frame where "the camera sweeps across the stands".
        """
        nonlocal size_dropped, size_frames
        if not dets:
            return dets
        size_frames += 1
        heights = [_box_height(b) for b, _c in dets]
        ref_h = max(heights) if heights else 0.0
        size_samples.extend([[_box_height(b), _box_area(b), ref_h] for b, _c in dets])
        if not sf.active:
            return dets
        kept = [(b, c) for (b, c) in dets if sf.keep(b, ref_h)]
        size_dropped += len(dets) - len(kept)
        # When everything is filtered out, keep it as is: better to give the tracker a bit more noise
        # than to have "no box at all for a whole stretch" — that would make downstream activity and
        # crop tracking drop out completely.
        return kept or dets

    while True:
        if _cancelled():
            break
        ok = cap.grab()
        if not ok:
            break
        read += 1
        if idx % step != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue
        if read > limit:
            break
        buf.append(frame)
        buf_idx.append(proc)
        proc += 1

        if len(buf) >= batch_size:
            results = _predict(buf)
            for k, res in enumerate(results):
                dets = _apply_size_filter(_dets_of(res))
                det_frames.append([b for b, _c in dets])
                tracker.update(buf_idx[k], buf_idx[k] * step_dt, dets)
            buf.clear()
            buf_idx.clear()
            if total_frames:
                _report(min(0.95, read / min(total_frames, limit)), tr("players.detect_players"))

    if buf and not _cancelled():
        results = _predict(buf)
        for k, res in enumerate(results):
            dets = _apply_size_filter(_dets_of(res))
            det_frames.append([b for b, _c in dets])
            tracker.update(buf_idx[k], buf_idx[k] * step_dt, dets)
        buf.clear()
        buf_idx.clear()
    cap.release()

    duration = proc * step_dt                 # seconds actually analyzed
    _report(0.97, tr("players.track_stats"))

    # ---- track merging + summarization ----
    bufs = _merge_tracks(tracker.tracks, aspect)
    tracks = _summarize(bufs, eff_fps)

    # ---- identify match players ----
    # Window-by-window identification: in long videos a player gets split into many tracks, and
    # picking only once for the whole video leaves "time outside the middle" with no player signal
    # at all (measured: frame_boxes is empty 58% of the time).
    active = _select_active_players_windowed(tracks, eff_fps, duration, viewpoint=viewpoint,
                                             boxes=det_frames, aspect=aspect,
                                             size_filter=sf)
    active_ids = [t.track_id for t in active]

    # ---- assemble the timeline (length = ceil(duration * sample_fps); index i corresponds to i/sample_fps) ----
    n = max(1, int(math.ceil(duration * sample_fps - 1e-9)))

    def _ti(t: float) -> int:
        return min(n - 1, max(0, int(round(t * sample_fps))))

    active_present: list[set[int]] = [set() for _ in range(n)]
    active_sp: list[list[float]] = [[] for _ in range(n)]
    crowd_sp: list[list[float]] = [[] for _ in range(n)]
    max_sp = np.zeros(n, dtype=np.float32)
    boxes_map: list[dict[int, tuple[int, float, float, float, float]]] = [dict() for _ in range(n)]
    active_set = set(active_ids)

    for tk in tracks:
        for j in range(tk.n):
            i = _ti(tk.times[j])
            s = float(tk.speeds[j])
            crowd_sp[i].append(s)
            if s > max_sp[i]:
                max_sp[i] = s
            if tk.track_id in active_set:
                active_present[i].add(tk.track_id)
                active_sp[i].append(s)
                b = tk.boxes[j]
                boxes_map[i][tk.track_id] = (tk.track_id, float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    active_count = np.asarray([len(s) for s in active_present], dtype=np.float32)
    active_speed = np.asarray(
        [float(np.mean(v)) if v else 0.0 for v in active_sp], dtype=np.float32
    )
    crowd_speed = np.asarray(
        [float(np.mean(v)) if v else 0.0 for v in crowd_sp], dtype=np.float32
    )
    frame_boxes = [list(m.values()) for m in boxes_map]

    size_stats = _build_size_stats(
        sf, size_samples, size_dropped, size_frames,
        float((_box_size_stats(det_frames, aspect) or {}).get("ref", 0.0)) if det_frames else 0.0,
    )

    _report(1.0, tr("players.done"))
    return PlayerSignal(
        fps=float(sample_fps),
        duration=float(duration),
        tracks=tracks,
        active_count=active_count,
        active_speed=active_speed,
        max_speed=max_sp,
        crowd_speed=crowd_speed,
        active_player_ids=active_ids,
        frame_boxes=frame_boxes,
        size_stats=size_stats,
    )


# ------------------------------------------------------------------ box-size probing


#: Model cache for probing: ``weight path -> YOLO instance``.
#: Why cache: when tuning the size threshold in the UI, the user **repeatedly grabs different
#: frames of the same video**, and ``YOLO(weights)`` itself takes 1~2 seconds to load the weights
#: (slower than inferring one frame). Only for probing: a full analysis loads once and runs
#: thousands of frames, so caching yields nothing, and sharing one model object would let
#: "analyzing" and "grabbing frames" interfere with each other.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
#: ultralytics' ``predict`` mutates the model object's internal state, so calling the same cached
#: instance from multiple threads is unsafe; frame grabbing is a short call, so serializing is an
#: acceptable cost.
_PREDICT_LOCK = threading.Lock()


def _load_model_cached(weights: str) -> Any:
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(weights)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(weights)
            _MODEL_CACHE[weights] = model
        return model


def _probe_frame_path(video_path: str, t: float, index: int = 0) -> Path:
    """On-disk path for a probe frame image (placed under the cache directory; served by ``/api/asset``)."""
    import hashlib

    try:
        from ..config import FRAMES_DIR

        root = Path(FRAMES_DIR) / "probe"
    except Exception:  # pragma: no cover - fallback for standalone runs
        root = Path(__file__).resolve().parents[3] / "data" / "cache" / "frames" / "probe"
    key = hashlib.sha1(
        f"{Path(video_path).resolve()}|{t:.3f}|{index}".encode("utf-8", "replace")
    ).hexdigest()[:16]
    return root / f"{key}.jpg"


def _read_frame_at(cap: Any, src_fps: float, t: float, tries: int = 6) -> np.ndarray | None:
    """Seek to second ``t`` and read out a frame, skipping the black frames common after a seek.

    After seeking in long-GOP footage, a few all-black/garbled frames are often returned first.
    The probe frame image is for the user to **look at**, and a black frame makes it impossible to
    verify "whether the box positions are right", so read a few more frames here and take the first
    "not solid-color" frame; if they are all solid color, hand over the last frame (at least not empty).
    """
    import cv2

    idx = int(round(max(0.0, t) * max(src_fps, 1e-6)))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    except Exception:
        pass
    last: np.ndarray | None = None
    for _ in range(max(1, int(tries))):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        last = fr
        try:
            if float(fr.std()) > 2.0:      # a solid-color frame has std≈0
                return fr
        except Exception:
            return fr
    return last


def probe_boxes(
    video_path: str,
    count: int = 10,
    model_name: str = "yolo11n.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    viewpoint: str = "unknown",
    roi: tuple[float, float, float, float] | None = None,
    roi_poly: list[list[float]] | None = None,
    max_side: int = 960,
    times: Sequence[float] | None = None,
    save_frames: bool = False,
    on_progress: Progress | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """**Detection only** (no tracking) on several frames, returning person boxes and the per-frame box-size distribution.

    Use: when tuning "person-box size filtering" in the UI, waiting for a full analysis to finish
    to see the result means a parameter change takes minutes — yet the only thing the user really
    wants to know is: **"how big are the players' boxes, and how big are the boxes of spectators and
    people on other courts"**.

    Two ways to take frames:

    * By default **uniformly sample** ``count`` frames over the whole video (one batch inference,
      1~3 seconds), giving the overall distribution of this footage;
    * If ``times`` is given, **take only those moments** (user-selected frames / grab the current
      playback position); in that case each frame also carries ``image`` (the on-disk JPEG path),
      which the UI can draw to place "which boxes were selected and which were filtered out"
      directly on the picture.

    Args:
        video_path: Video path (prefer passing the proxy video, which decodes faster).
        count: Number of frames when sampling uniformly (1~40); ignored when ``times`` is given.
        roi / roi_poly: Same meaning as in :func:`analyze_players`; when given, filter by court,
            so the resulting distribution matches the real analysis.
        max_side: Scale the frame to within this side length before inference (affects only speed, not normalized coordinates).
        times: Specified moments (seconds); at most 40.
        save_frames: Whether to write the captured frames as JPEG and return their paths (for UI display).

    Returns:
        ``{"frames": [{"t", "boxes", "confs", "ref", "image"}...],
        "points": [[box height, box area, largest box height in the same frame]...], ...}``
    """
    import cv2
    import time

    if not video_path:
        raise ValueError(tr("players.empty_video_path"))
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    t0 = time.time()
    want_times: list[float] = []
    if times:
        for v in list(times)[:40]:
            try:
                want_times.append(max(0.0, float(v)))
            except (TypeError, ValueError):
                continue
    count = len(want_times) if want_times else int(max(1, min(40, count)))

    cfg_dir, models_dir = _data_paths()
    _prepare_yolo_env(cfg_dir)
    weights = _resolve_weights(model_name, models_dir)
    dev = _pick_device(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0 or not np.isfinite(src_fps):
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[np.ndarray] = []
    used_times: list[float] = []
    for i in range(count):
        if cancel is not None and cancel():
            break
        if want_times:
            fr = _read_frame_at(cap, src_fps, want_times[i])
            t_used = want_times[i]
        elif total_frames > 0:
            pos = int(total_frames * (i + 0.5) / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fr = cap.read()
            if not ok:
                fr = None
            t_used = pos / src_fps
        else:
            # read sequentially when the frame count is unknown (some containers)
            ok, fr = cap.read()
            if not ok:
                fr = None
            t_used = len(frames) * 1.0 / max(src_fps, 1e-3)
        if fr is None:
            continue
        used_times.append(float(t_used))
        h, w = fr.shape[:2]
        s = max_side / max(1, max(h, w))
        if s < 1.0:
            fr = cv2.resize(fr, (max(2, int(w * s)), max(2, int(h * s))),
                            interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        return {"frames": [], "points": [], "count": 0, "duration": 0.0,
                "width": 0, "height": 0, "aspect": 1.7778,
                "viewpoint": viewpoint, "error": tr("players.read_frame_failed")}

    # Normalized coordinates must be computed against the **actually decoded frame** (not the size
    # reported by the container): the probe boxes must be drawable directly on the returned image,
    # so both must come from the same pixel space.
    height, width = frames[0].shape[:2]
    aspect = float(width) / float(max(1, height))

    # Write the frame image to disk: it is the same frame used for inference, so boxes and picture align exactly
    images: list[str | None] = [None] * len(frames)
    if save_frames:
        for i, fr in enumerate(frames):
            p = _probe_frame_path(video_path, used_times[i] if i < len(used_times) else 0.0, i)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(p), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
                    images[i] = str(p)
            except Exception:
                pass

    if on_progress is not None:
        on_progress(0.3, tr("players.load_model"))
    di = float(min(conf, LOW_CONF))
    try:
        model = _load_model_cached(weights)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"无法加载 ultralytics 模型: {exc}") from exc

    poly_arr = None
    if roi_poly:
        p = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
        p = p[np.all(np.isfinite(p), axis=1)]
        if p.shape[0] >= 3:
            poly_arr = p
    roi_arr = tuple(float(v) for v in roi) if (roi is not None and poly_arr is None) else None

    def _predict_all() -> list[Any]:
        nonlocal dev
        with _PREDICT_LOCK:
            try:
                return model.predict(frames, classes=[0], conf=di, imgsz=imgsz,
                                     device=dev, verbose=False)
            except Exception:
                if dev != "cpu":
                    dev = "cpu"
                    return model.predict(frames, classes=[0], conf=di, imgsz=imgsz,
                                         device="cpu", verbose=False)
                raise

    if on_progress is not None:
        on_progress(0.6, tr("players.detect_boxes"))
    results = _predict_all()

    out_frames: list[dict[str, Any]] = []
    points: list[list[float]] = []
    for k, res in enumerate(results):
        fh = int(frames[k].shape[0]) if k < len(frames) else height
        fw = int(frames[k].shape[1]) if k < len(frames) else width
        boxes = getattr(res, "boxes", None)
        dets: list[tuple[tuple[float, float, float, float], float]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cf = boxes.conf.cpu().numpy()
            for j in range(len(boxes)):
                c = float(cf[j])
                if c < LOW_CONF:
                    continue
                b = _sanitize_box(float(xyxy[j, 0]) / fw, float(xyxy[j, 1]) / fh,
                                  float(xyxy[j, 2]) / fw, float(xyxy[j, 3]) / fh,
                                  viewpoint)
                if b is None:
                    continue
                bcx, by = 0.5 * (b[0] + b[2]), b[3]
                if poly_arr is not None:
                    if not bool(point_in_poly(np.asarray([[bcx, by]], dtype=np.float32),
                                              poly_arr, margin=ROI_POLY_MARGIN)[0]):
                        continue
                elif roi_arr is not None:
                    if not (roi_arr[0] <= bcx <= roi_arr[2] and roi_arr[1] <= by <= roi_arr[3]):
                        continue
                dets.append((b, c))
        dets = _dedup_dets(dets, aspect)
        ref = max((_box_height(b) for b, _c in dets), default=0.0)
        out_frames.append({
            "t": round(float(used_times[k]) if k < len(used_times) else 0.0, 3),
            "boxes": [[round(float(v), 5) for v in b] for b, _c in dets],
            "confs": [round(float(c), 3) for _b, c in dets],
            "ref": round(float(ref), 5),
            # The frame image matching boxes (JPEG path; the UI fetches it via /api/asset)
            "image": images[k] if k < len(images) else None,
        })
        # [box height, box area, largest box height in the same frame]: the UI uses these to simulate any threshold locally
        points.extend([[_box_height(b), _box_area(b), ref] for b, _c in dets])

    if on_progress is not None:
        on_progress(1.0, tr("players.done"))
    return {
        "frames": out_frames,
        "points": points,
        "count": len(out_frames),
        # Time of the last captured frame (not the total video duration)
        "duration": round(float(max(used_times)) if used_times else 0.0, 3),
        "video_duration": round(float(total_frames / src_fps) if total_frames else 0.0, 3),
        "width": int(width),
        "height": int(height),
        "aspect": round(aspect, 5),
        "viewpoint": viewpoint,
        "elapsed": round(time.time() - t0, 2),
        "model": Path(weights).name,
        "device": dev,
        "frames_saved": bool(save_frames),
    }
