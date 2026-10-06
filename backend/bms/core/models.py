"""Domain data models (Pydantic v2)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


# ------------------------------------------------------------------ Media


class MediaInfo(BaseModel):
    id: str = Field(default_factory=lambda: _uid("m_"))
    path: str
    name: str = ""
    size: int = 0
    duration: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    rotation: int = 0
    vcodec: str = ""
    acodec: str | None = None
    has_audio: bool = False
    created_at: int = Field(default_factory=now_ms)
    # Derived assets (for analysis)
    proxy_path: str | None = None
    proxy_fps: float | None = None
    proxy_width: int | None = None
    proxy_height: int | None = None
    audio_path: str | None = None
    poster: str | None = None

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 16 / 9


# ------------------------------------------------------------------ Rallies and shots

PlayerSide = Literal["near", "far", "unknown"]
ShotKind = Literal["serve", "receive", "clear", "drop", "smash", "drive", "net", "lift", "unknown"]


class ShotEvent(BaseModel):
    """A single shot (the moment the racket touches the shuttle)."""

    time: float
    player: PlayerSide = "unknown"
    kind: ShotKind = "unknown"
    confidence: float = 0.5
    #: Estimated shuttle speed after contact (court coordinates, m/s), None if unknown
    speed: float | None = None
    #: Whether this shot was accompanied by a clear jump (smash/jump smash)
    airborne: bool = False
    #: Height of the contact point relative to the court (m), None if unknown
    height: float | None = None


class RallyFeatures(BaseModel):
    """Objective feature quantities used for scoring."""

    duration: float = 0.0
    shot_count: int = 0
    #: Shots per second
    tempo: float = 0.0
    #: Shuttle speed percentile (pixels/second, normalized to court scale)
    shuttle_speed_p50: float = 0.0
    shuttle_speed_p95: float = 0.0
    #: Mean court motion energy within the rally (normalized 0~1)
    motion_energy: float = 0.0
    motion_peak: float = 0.0
    #: Running distance of each side (meters)
    travel_near: float = 0.0
    travel_far: float = 0.0
    #: Number of smashes
    smash_count: int = 0
    #: Longest consecutive exchange (= shot_count)
    longest_exchange: int = 0
    #: Whether the last 2 seconds before the rally ended were intense
    finish_intensity: float = 0.0
    #: Whether an extreme retrieval occurred (peak instantaneous player acceleration)
    scramble: float = 0.0
    #: Score closeness 0~1 (available when scoreboard recognition is enabled)
    closeness: float | None = None
    #: Whether it is a key point (game point/match point/deuce)
    clutch: bool = False
    #: Analysis confidence
    confidence: float = 0.5

    # ---- The following are intermediate quantities used directly by scoring.
    # Storing only the per-category scores cannot produce scores for "a different weighting": switching the
    # weighting requires recomputing from the same raw features, so all quantities used for scoring are kept
    # here (in older analysis results they are the default values).
    #: Hit strength 90th percentile
    hit_strength_p90: float = 0.0
    #: Player movement speed mean / peak (pixels per second, normalized to court scale)
    player_speed_mean: float = 0.0
    player_speed_max: float = 0.0
    #: Fraction of frames in which the shuttle appears 0~1
    shuttle_presence: float = 0.0
    #: Image quality: sharpness / shake / subject size
    quality_sharpness: float = 0.6
    quality_shake: float = 0.3
    quality_subject_size: float = 0.25
    #: Voice command bonus: added directly to the total score when a configured phrase is detected (0 = no hit).
    #: It is an **absolute bonus** rather than a category, so it is not part of the five-way weighting and is
    #: capped at 100 on its own.
    speech_bonus: float = 0.0
    #: Voice phrases hit in this rally (at most 2); the UI uses them to tag / explain the bonus
    speech_phrases: list[str] = Field(default_factory=list)


class RallyScores(BaseModel):
    """Category scores (0~100)."""

    total: float = 0.0
    length: float = 0.0
    intensity: float = 0.0
    technique: float = 0.0
    excitement: float = 0.0
    production: float = 0.0  # image/framing quality (sharpness, shake, occlusion)


class Rally(BaseModel):
    id: str = Field(default_factory=lambda: _uid("r_"))
    index: int = 0
    #: The rally itself (serve start -> dead ball)
    start: float = 0.0
    end: float = 0.0
    #: Suggested clip range (including pre-action padding)
    clip_start: float = 0.0
    clip_end: float = 0.0
    #: Serve / receive
    serve_time: float | None = None
    serve_player: PlayerSide = "unknown"
    receive_time: float | None = None
    receive_player: PlayerSide = "unknown"
    shots: list[ShotEvent] = Field(default_factory=list)
    features: RallyFeatures = Field(default_factory=RallyFeatures)
    scores: RallyScores = Field(default_factory=RallyScores)
    tags: list[str] = Field(default_factory=list)
    #: User intervention
    keep: bool = True
    starred: bool = False
    note: str = ""
    #: Automatically determined rally result (when a scoreboard is present)
    winner: PlayerSide | None = None
    #: Duration of the rally itself (= end - start); sent along with serialization for direct use by the frontend
    duration: float = 0.0


class AnalysisParams(BaseModel):
    """Tunable analysis parameters."""

    #: Hit detection sensitivity 0~1 (larger = more sensitive)
    hit_sensitivity: float = 0.5
    #: Silence duration that marks the end of a rally (seconds)
    gap_seconds: float = 3.2
    #: Minimum rally duration (seconds); shorter ones are dropped
    min_rally_seconds: float = 2.0
    #: Maximum rally duration (seconds); longer ones are truncated
    max_rally_seconds: float = 120.0
    #: Typical rally duration (seconds). During continuous training / multi-shuttle drills, players pause only
    #: a few seconds between rallies, the activity curve does not collapse, and a threshold alone would glue
    #: several rallies into one. Intervals longer than this are recursively split at the "lowest activity point".
    target_rally_seconds: float = 28.0
    #: Segmentation aggressiveness 0~1: larger values favor finer splits (mapped to a shorter target duration and an earlier exit threshold)
    split_sensitivity: float = 0.5
    #: Rallies below this confidence are grayed out by default in the UI (adjustable there)
    confidence_min: float = 0.0
    #: Padding kept before and after each rally when editing.
    #: ``pre_roll`` is the serve preparation (player walks into position, pause before the toss);
    #: ``post_roll`` is a little extra breathing room after the end point, which is what makes the final
    #: exported ``clip_end``. Because ``hit_tail_seconds`` already accounts for "the shuttle landing", the
    #: default here is very small -- the old default of 1.6 added a second and a half of dead-ball footage
    #: to every rally out of thin air.
    pre_roll: float = 1.2
    post_roll: float = 0.5
    #: **How long to keep after the last shot** (seconds), used to anchor the rally end to "the shuttle landing".
    #: After a shot is hit the shuttle still flies for a while before landing, so the end cannot simply equal
    #: the last shot. In amateur footage this flight segment is mostly 0.4~1.2 seconds; 0.9 covers the common
    #: cases and simultaneously removes both "lingering long after the shuttle lands" and "eating into the
    #: next rally". Larger = safer but looser, smaller = tighter but may eat the landing moment of a powerful
    #: clear.
    hit_tail_seconds: float = 0.9
    #: Enable individual analysis modules
    use_audio: bool = True
    use_motion: bool = True
    use_players: bool = True
    #: Pose assistance (YOLO-pose): attributes audio hits to "whether it was our match".
    #: In a multi-court gym, audio cannot distinguish "who is hitting", while our players swing only on real
    #: hits, so this is the key evidence for solving "rallies glued too long / eating into the next rally".
    #: Failures (no GPU / missing weights / players too small) degrade automatically, matching the behavior
    #: when this switch is off.
    use_pose: bool = True
    #: ---- Hit attribution gate (cross-court rejection) ----
    #: Pose evidence threshold 0~1: an audio hit is kept only when a swing peak explains it. Larger =
    #: stricter (drops more neighboring-court sounds); smaller = keeps more. This is the "cross-court
    #: suppression strength" slider in the rally panel and can be re-applied instantly via resegment.
    pose_gate_threshold: float = 0.45
    #: Matching window (seconds) around a swing peak that can explain a hit.
    pose_gate_window: float = 0.35
    #: One swing can explain only one hit. This one-to-one constraint is the main reason the gate works
    #: (it removes the extra neighboring-court sounds that happen to coincide with our swing).
    pose_gate_one_to_one: bool = True
    #: ---- Boundary refinement (local evidence tie-breaker) ----
    #: Off by default: segmentation only moves a quiet-span boundary when a locally-calibrated
    #: motion/hit/swing template clearly prefers a nearby valley (``boundary.py``). Disabled, old
    #: projects and pose-unavailable videos produce byte-identical intervals to the current pipeline.
    use_boundary_refine: bool = False
    #: How far (seconds) a boundary is allowed to move from the quiet-span edge.
    boundary_max_move: float = 3.0
    #: Required score advantage of the best candidate over the current boundary.
    boundary_score_margin: float = 0.12
    #: ---- Fusion base weights (rally.fuse) ----
    #: Multipliers of the five activity components BEFORE the per-signal discriminative-power /
    #: audio-reliability factors. Defaults equal rally.DEFAULT_BASE_WEIGHTS; the annotation
    #: optimizer can calibrate them offline, and resegment re-fuses the stored full-rate component
    #: curves (no AI rerun) when any value differs from the default.
    fuse_weight_players: float = 1.35
    fuse_weight_motion: float = 1.0
    fuse_weight_audio: float = 0.95
    fuse_weight_shuttle: float = 0.9
    fuse_weight_roi: float = 0.8

    #: Force-apply the gate even when the retention ratio falls outside the safe band. By default an
    #: out-of-band ratio makes the gate silently pass all hits through (protection against a broken pose
    #: signal); turning this on lets the user insist on filtering anyway.
    pose_gate_force: bool = False
    #: Shuttle trajectory tracking: the compute cost is proportional to "frames x pixels x time window", so
    #: 30 minutes of 4K footage can take hours; disabled by default, enable on demand for long videos.
    use_shuttle: bool = False
    use_scoreboard: bool = False
    #: Sampling frame rate for shuttle tracking (higher than player detection to catch fast-flying shuttles)
    shuttle_fps: float = 10.0
    #: Time budget for shuttle tracking (seconds, 0 = full clip). If coverage is below 60% the whole signal
    #: is dropped, to avoid "data only in the first half" skewing the fusion result.
    shuttle_budget_seconds: float = 420.0
    #: Voice command bonus (optional): recognizes short phrases shouted by spectators / fellow players
    #: (e.g. "nice shot") with faster-whisper and raises the corresponding rally's score on a hit.
    #: Missing dependency / models (and a bad audio track) degrade silently, matching the behavior
    #: when this switch is off.
    use_speech: bool = False
    #: Voice phrases to detect, at most 2, at most 3 characters each; extra items are truncated / dropped.
    speech_phrases: list[str] = Field(default_factory=list)
    #: How many points to add to the total per phrase hit (added after the confidence discount, total capped at 100)
    speech_bonus_points: float = 10.0
    #: faster-whisper model size (larger = more accurate but slower and a bigger download).
    speech_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    #: Tolerate near-homophone mis-recognition (e.g. Whisper hears 到球/倒球 for 好球). Needs pypinyin;
    #: without it only exact matches are kept.
    speech_fuzzy: bool = True
    #: Court orientation: auto | landscape | portrait
    court_orientation: Literal["auto", "landscape", "portrait"] = "auto"
    #: Camera setup / shooting style. ``auto`` = detect automatically; the rest skip guessing when the user
    #: knows their own setup.
    #:   rear   = behind the court (behind the baseline, most common)
    #:   side   = to the side of the sideline
    #:   elevated = high angled downward view (stands / second floor)
    #:   overhead = straight top-down shot
    #: Under different camera setups "who is closer to the camera" and "how large the player box should be"
    #: are completely different; detecting it lets us pick the right prior.
    viewpoint: Literal["auto", "rear", "side", "elevated", "overhead"] = "auto"
    #: Whether to auto-calibrate the court (color + polygon + homography). Turning it off uses the full frame
    #: throughout, which amounts to falling back to the old behavior (useful with multiple courts / abnormal colors).
    auto_calibrate: bool = True
    #: **Manually calibrated court boundary** (normalized 0~1, 4~24 points, order irrelevant).
    #: The user just clicks around it on the preview image; when provided, no guessing is done and it is used
    #: directly to build the calibration. Color calibration fails with multiple courts / abnormal mat colors /
    #: a court occupying only a corner of the frame; in those cases calibrating once manually is far more
    #: effective than continuing to tune the algorithm.
    #: **For panoramic / fisheye footage, add more points**: describing a curved boundary with four corners
    #: cuts off the edges, and the edges are exactly where background people are densest.
    court_poly: list[list[float]] | None = None
    #: Legacy field: the manually calibrated four corners (equivalent to ``court_poly`` with only 4 points).
    #: Kept so that already-saved old projects keep working; new code should write ``court_poly``.
    court_quad: list[list[float]] | None = None
    #: ---- Person box size filtering ----
    #: Mode for filtering detected person boxes by size:
    #:   ``off``      no filtering (keep only the existing geometric thresholds)
    #:   ``absolute`` filter by "box height as a fraction of frame height" (min/max are absolute fractions)
    #:   ``relative`` filter by "box height / largest box height in the same frame" (a ratio)
    #: In panoramic / fisheye footage the box height of the same person can differ by more than 2x between
    #: the center and the edges of the frame; ``absolute`` easily filters out real players near the edges,
    #: while ``relative`` is more robust.
    player_size_mode: Literal["off", "absolute", "relative"] = "off"
    #: Box height lower bound (absolute = fraction of frame height; relative = ratio to the largest box in the frame)
    player_min_height: float = 0.05
    #: Box height upper bound (0 = unlimited)
    player_max_height: float = 0.0
    #: Box area lower bound (normalized area 0~1, 0 = unlimited). Under distortion boxes get wider, so area
    #: is more accurate than height when trying to tightly exclude "people right in front of the lens".
    player_min_area: float = 0.0
    #: Box area upper bound (0 = unlimited)
    player_max_area: float = 0.0
    #: Rally segmentation method: ``auto`` prefers player-motion segmentation (recommended), ``activity``
    #: uses the old fused activity + hysteresis state machine, ``hybrid`` runs both and picks the better one.
    segment_mode: Literal["auto", "activity", "hybrid"] = "auto"
    #: ---- Quiet-segment segmentation scales (target params for manual-annotation calibration) ----
    #: These are the internal scales for finding quiet valleys on the "player motion x hit density" evidence
    #: curve. They were not in the old parameters because there was no ground truth to rely on (see HANDOVER
    #: 8.6). With annotations available they are the quantities most worth calibrating, so they were promoted
    #: to persistable parameters that the optimizer can write.
    #: Minimum width of a single quiet valley: too short treats curve jitter as a pause.
    seg_min_quiet: float = 0.6
    #: Prominence threshold of a quiet valley (relative to p95-p20); larger requires a deeper valley.
    seg_prominence: float = 0.10
    #: Minimum separation between two quiet segments to count as a "rally end".
    seg_min_rest: float = 0.6
    #: Minimum allowed continuous movement segment; shorter candidates are dropped.
    seg_min_core: float = 2.5
    #: Maximum number of frames for per-frame AI analysis (to rate-limit long videos; 0 = unlimited)
    max_frames: int = 0
    #: Analysis frame rate
    sample_fps: float = 15.0

    def fuse_weight_base(self) -> dict[str, float]:
        """Component-key -> base weight (component keys are rally.fuse canonical names)."""
        return {
            "players": float(self.fuse_weight_players),
            "motion": float(self.fuse_weight_motion),
            "audio_hits": float(self.fuse_weight_audio),
            "shuttle": float(self.fuse_weight_shuttle),
            "roi": float(self.fuse_weight_roi),
        }

    @field_validator("speech_phrases", mode="before")
    @classmethod
    def _clean_speech_phrases(cls, v: object) -> list[str]:
        # Lazy import to avoid core depending on the analysis subpackage at import time
        from ..analysis.speech import sanitize_phrases

        return sanitize_phrases(v)

    @field_validator("speech_bonus_points")
    @classmethod
    def _clamp_speech_bonus(cls, v: float) -> float:
        return float(min(30.0, max(0.0, v)))

    @field_validator("speech_model", mode="before")
    @classmethod
    def _clean_speech_model(cls, v: object) -> str:
        # Lazy import to avoid core depending on the analysis subpackage at import time
        from ..analysis.speech import DEFAULT_MODEL, WHISPER_SIZES

        s = str(v or "").strip().lower()
        return s if s in WHISPER_SIZES else DEFAULT_MODEL

    @field_validator("pose_gate_threshold")
    @classmethod
    def _clamp_gate_threshold(cls, v: float) -> float:
        return float(min(0.95, max(0.0, v)))

    @field_validator("pose_gate_window")
    @classmethod
    def _clamp_gate_window(cls, v: float) -> float:
        return float(min(2.0, max(0.05, v)))


class AnalysisResult(BaseModel):
    media_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"] = "pending"
    stage: str = ""
    message: str = ""
    progress: float = 0.0
    error: str | None = None
    params: AnalysisParams = Field(default_factory=AnalysisParams)
    started_at: int | None = None
    finished_at: int | None = None
    #: Time-series signals (downsampled before returning to the frontend for waveform plotting)
    signals: dict[str, list[float]] = Field(default_factory=dict)
    signal_fps: float = 0.0
    hits: list[ShotEvent] = Field(default_factory=list)
    rallies: list[Rally] = Field(default_factory=list)
    court: dict[str, Any] | None = None
    #: Court calibration and camera setup detection result (``court_calib.CourtCalibration.as_payload()``)
    calibration: dict[str, Any] | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ Project / timeline


class Transform(BaseModel):
    scale: float = 1.0
    x: float = 0.0  # normalized offset
    y: float = 0.0
    rotation: float = 0.0


class Clip(BaseModel):
    id: str = Field(default_factory=lambda: _uid("c_"))
    media_id: str
    #: In point / out point within the source media (seconds)
    src_in: float = 0.0
    src_out: float = 0.0
    #: Start point on the timeline; when None, clips are arranged in order
    tl_start: float = 0.0
    speed: float = 1.0
    volume: float = 1.0
    transform: Transform = Field(default_factory=Transform)
    #: Associated rally (a segment cut out by the AI)
    rally_id: str | None = None
    label: str = ""
    #: Automatic vertical crop
    vertical_crop: bool = False
    #: Portion kept at original speed (protected during speed changes; empty when unused)
    protected: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, (self.src_out - self.src_in) / max(self.speed, 1e-6))


class Track(BaseModel):
    id: str = Field(default_factory=lambda: _uid("t_"))
    #: Language-neutral default; callers fill the display name via ``tr("timeline.track_default", ...)``.
    name: str = ""
    kind: Literal["video", "audio", "overlay"] = "video"
    muted: bool = False
    locked: bool = False
    clips: list[Clip] = Field(default_factory=list)


class Timeline(BaseModel):
    tracks: list[Track] = Field(default_factory=list)
    duration: float = 0.0
    fps: float = 30.0
    width: int = 1920
    height: int = 1080


class ExportPreset(BaseModel):
    id: str
    name: str
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    vcodec: Literal["h264", "hevc"] = "h264"
    encoder: Literal["auto", "nvenc", "x264", "qsv"] = "auto"
    video_bitrate: str = "12M"
    audio_bitrate: str = "192k"
    crf: int | None = None
    container: str = "mp4"
    #: Automatic vertical follow crop
    auto_reframe: bool = False


class Project(BaseModel):
    id: str = Field(default_factory=lambda: _uid("p_"))
    #: Language-neutral default; ``store.create_project`` fills it via ``tr("project.untitled")``.
    name: str = ""
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)
    media: list[MediaInfo] = Field(default_factory=list)
    analyses: dict[str, AnalysisResult] = Field(default_factory=dict)
    timeline: Timeline = Field(default_factory=Timeline)
    #: Player/UI state
    ui: dict[str, Any] = Field(default_factory=dict)
    version: int = 1


class ProjectSummary(BaseModel):
    id: str
    name: str
    created_at: int
    updated_at: int
    media_count: int = 0
    duration: float = 0.0
    poster: str | None = None
    rally_count: int = 0
    analyzed: bool = False


# ------------------------------------------------------------------ Jobs


class JobInfo(BaseModel):
    id: str = Field(default_factory=lambda: _uid("j_"))
    kind: str
    title: str = ""
    #: The media this job serves (used by media-triggered jobs like prepare), so the UI can attach progress to the right card
    media_id: str | None = None
    status: Literal["queued", "running", "done", "error", "cancelled"] = "queued"
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    error: str | None = None
    result: Any = None
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)
