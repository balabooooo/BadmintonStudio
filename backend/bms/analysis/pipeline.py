"""Analysis pipeline: wire the modules together and produce an editable rally list.

Flow
----
1. Prepare derived resources (proxy video / audio track)
2. Audio hit detection        -> hit moments
3. Frame motion analysis      -> motion energy / shake / sharpness / court hot region
4. Player detection & tracking -> on-court activity (optional; auto-degrades when missing)
5. Shuttlecock trajectory tracking -> shuttle presence and speed (optional; auto-degrades when missing)
6. Multimodal fusion + state machine -> rally intervals
7. Audio re-anchoring of boundaries -> serve / receive moments
8. Feature extraction + scoring -> rally scores and tags
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import numpy as np
from loguru import logger

from ..config import ensure_dirs
from ..i18n import tr
from ..core.models import (
    AnalysisParams,
    AnalysisResult,
    MediaInfo,
    Rally,
    RallyFeatures,
    RallyScores,
    ShotEvent,
)
from ..core import media as M
from . import audio_hits as AH
from . import court_calib as CC
from . import motion as MO
from . import rally as RA
from . import rally_vision as RV
from . import scoring as SC
from . import speech as SP

Progress = Callable[[float, str, str], None]  # (progress, stage, message)


def _noop(p: float, s: str, m: str = "") -> None:
    pass


def _try_import(name: str):
    try:
        mod = __import__(f"bms.analysis.{name}", fromlist=[name])
        return mod
    except Exception as e:  # noqa: BLE001
        logger.warning("optional analysis module {} unavailable: {}: {}", name, type(e).__name__, e)
        return None


def attribute_sides(rallies: list[Rally], player_sig, fps: float,
                    viewpoint: str = "unknown") -> None:
    """Assign "serve / receive" to specific players.

    Approach: among match players, **the one with the larger box must be closer to the camera**;
    based on that the two players are split into near / far; then look at who is moving around the
    serve moment — the one who just swung is the server. This judgment uses only detection boxes, not
    a pose model, so it mostly works even when players are small.

    **"Larger box = nearer" only holds for a low rear/side camera.** With a high camera or overhead
    shot the two are at roughly the same distance from the camera and their box areas are therefore
    similar, so continuing to separate them by area only assigns near/far to noise. Under these two
    camera angles this function simply writes nothing (leaving ``unknown``).

    When unsure, keep ``unknown``; better to write nothing than to write it wrong.
    """
    if viewpoint in ("overhead", "elevated"):
        return
    if player_sig is None:
        return
    tracks = getattr(player_sig, "tracks", None) or []
    ids = list(getattr(player_sig, "active_player_ids", None) or [])
    boxes = getattr(player_sig, "frame_boxes", None) or []
    pfps = float(getattr(player_sig, "fps", fps) or fps)
    if not tracks or not ids or not boxes or pfps <= 0:
        return

    by_id = {t.track_id: t for t in tracks}
    actives = [by_id[i] for i in ids if i in by_id]
    if len(actives) < 2:
        return

    # the larger the box in the frame = the closer to the camera
    biggest = max(actives, key=lambda t: t.mean_area)
    others = [t for t in actives if t.track_id != biggest.track_id]
    if not others:
        return
    second = max(others, key=lambda t: t.mean_area)
    if biggest.mean_area <= second.mean_area * 1.05:
        return  # the two are at about the same distance, cannot tell near/far, give up
    near_id, far_id = biggest.track_id, second.track_id

    def center_at(track_id: int, t: float) -> tuple[float, float] | None:
        idx = int(round(t * pfps))
        if idx < 0 or idx >= len(boxes):
            return None
        for item in boxes[idx]:
            if item[0] == track_id:
                _, x1, y1, x2, y2 = item
                return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return None

    def movement(track_id: int, t0: float, t1: float) -> float:
        """Cumulative displacement of this player within the time interval (normalized by frame height)."""
        a = max(0, int(round(t0 * pfps)))
        b = min(len(boxes), max(a + 1, int(round(t1 * pfps))))
        total = 0.0
        prev: tuple[float, float] | None = None
        for i in range(a, b):
            cur = None
            for item in boxes[i]:
                if item[0] == track_id:
                    _, x1, y1, x2, y2 = item
                    cur = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    break
            if cur is not None and prev is not None:
                total += float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
            if cur is not None:
                prev = cur
        return total

    for r in rallies:
        t = r.serve_time
        if t is None:
            continue
        # the serve motion happens about 0.7 seconds before the serve moment
        mv_near = movement(near_id, t - 0.7, t + 0.25)
        mv_far = movement(far_id, t - 0.7, t + 0.25)
        if max(mv_near, mv_far) < 0.006:
            continue  # neither moved, cannot tell
        server_is_near = mv_near >= mv_far
        r.serve_player = "near" if server_is_near else "far"  # type: ignore[assignment]
        r.receive_player = "far" if server_is_near else "near"  # type: ignore[assignment]
        # Reassign shot by shot as "serving side alternates", closer to reality than fixing the near side as the start
        first_side = r.serve_player
        for k, s in enumerate(r.shots):
            s.player = first_side if k % 2 == 0 else ("far" if first_side == "near" else "near")  # type: ignore[assignment]


def run_analysis(
    media: MediaInfo,
    params: AnalysisParams | None = None,
    on_progress: Progress = _noop,
    cancel: Callable[[], bool] | None = None,
    weights_key: str = "balanced",
    roi: tuple[float, float, float, float] | None = None,
) -> AnalysisResult:
    ensure_dirs()
    params = params or AnalysisParams()
    res = AnalysisResult(media_id=media.id, params=params, status="running")
    res.started_at = int(__import__("time").time() * 1000)
    log = logger.bind(media=media.id)
    log.info("analysis start: name={!r} duration={:.1f}s size={}x{} weights={}",
             media.name, media.duration, media.width, media.height, weights_key)

    def emit(p: float, stage: str, msg: str = "") -> None:
        res.progress = float(max(0.0, min(1.0, p)))
        res.stage = stage
        res.message = msg
        on_progress(res.progress, stage, msg)

    def cancelled() -> bool:
        return bool(cancel and cancel())

    try:
        # ---------------------------------------------------------- 1. derived resources
        emit(0.01, "prepare", tr("analysis.stage.prepare"))
        M.ensure_proxy(media, on=lambda p, m: emit(0.01 + 0.14 * p, "proxy", m), cancel=cancel)
        M.ensure_audio(media, on=lambda p, m: emit(0.15 + 0.05 * p, "audio", m), cancel=cancel)
        M.ensure_poster(media)
        if cancelled():
            res.status = "cancelled"
            return res

        duration = media.duration
        fw = int(media.proxy_width or media.width or 0)
        fh = int(media.proxy_height or media.height or 0)
        if fw <= 0 or fh <= 0:
            fw, fh = 1920, 1080

        # ---------------------------------------------------------- 1.5 court calibration
        # Calibrate before detecting players: knowing "which part of the frame the court is in"
        # lets player detection be restricted to this court, no longer relying on the
        # camera-coupled guess "the two largest boxes are the players"; the camera-angle type also
        # determines which prior parameter set is used afterward.
        calib: CC.CourtCalibration | None = None
        hint = params.viewpoint if params.viewpoint != "auto" else None
        # Manually calibrated court boundary: prefer the new multi-point polygon field; a legacy
        # project's court_quad (only four corners) still works — it is the special case of court_poly
        # given only four points.
        manual = _manual_poly(params.court_poly)
        manual_source = "court_poly"
        if manual is None and params.court_quad:
            manual = _manual_poly(params.court_quad)
            manual_source = "court_quad"
        if manual is not None:
            # The user manually calibrated the court: use it directly, no more guessing.
            # Manual calibration is more reliable than any automatic method, and it can be reused once done.
            calib = CC.build_calibration(manual * np.array([fw, fh], dtype=np.float32),
                                         fw, fh, source="manual")
            n_pts = int(manual.shape[0])
            calib.note(f"使用手动标定的场地边界（{n_pts} 个点，来自 params.{manual_source}）")
            if not calib.ok:
                calib.note("手动标定退化（点太少、面积为零或共线），回退到自动标定")
                manual = None
        if manual is None and params.auto_calibrate:
            emit(0.205, "calibrate", tr("analysis.stage.calibrate"))
            try:
                calib = CC.calibrate(
                    str(M.proxy_source(media)),
                    frame_size=(fw, fh),
                    hint=hint,      # type: ignore[arg-type]
                )
            except Exception as e:      # a calibration failure must not interrupt the analysis
                calib = CC.CourtCalibration()
                calib.note(f"标定异常：{type(e).__name__}: {e}")
        elif manual is None:
            calib = CC.CourtCalibration()
            calib.note("已关闭自动标定，且没有手动标定（使用全画幅与通用参数）")
        if cancelled():
            res.status = "cancelled"
            return res

        # If the user explicitly gave an ROI use it, otherwise use the calibrated court extent.
        # Note this is the **bounding rectangle** (only for coarse filtering and decode range); the
        # real in-court determination is delegated to the player module via eff_poly below and done by polygon.
        eff_roi = roi
        eff_poly: list[list[float]] | None = None
        if calib is not None and calib.ok:
            if eff_roi is None:
                eff_roi = calib.court_mask_roi(margin=0.04)
            eff_poly = calib.poly_norm()

        # ---------------------------------------------------------- 2. audio hits
        hits: AH.HitDetection | None = None
        #: Pre-gate hit sequence. The gate below replaces ``hits`` in place, so this reference is what
        #: lets the fast resegmentation path re-apply a *different* gate threshold from the raw
        #: candidates instead of double-filtering the already-gated sequence.
        hits_raw: AH.HitDetection | None = None
        hit_trace: dict[str, Any] = {}
        if params.use_audio and media.audio_path:
            emit(0.22, "hits", tr("analysis.stage.hits"))
            hits = AH.detect_hits(media.audio_path, sensitivity=params.hit_sensitivity)
            hits_raw = hits
            hit_trace = {
                "count": int(hits.times.size),
                "env_fps": float(hits.env_fps),
                "noise_floor_db": round(float(hits.noise_floor_db), 1),
            }

        # ---------------------------------------------------------- 2.5 speech phrases
        # Optional: recognize short phrases shouted by players / spectators (e.g. "nice shot"). It
        # does **not participate in segmentation**, it only adds points to rallies that it hits, so
        # it just needs to finish before segmentation. Any failure degrades: a missing faster-whisper
        # dependency / model is only recorded in speech_trace, and the analysis continues as usual.
        speech_res: SP.SpeechDetection | None = None
        speech_trace: dict[str, Any] = {}
        if params.use_speech and params.speech_phrases and media.audio_path:
            emit(0.24, "speech", tr("analysis.stage.speech"))
            speech_res = SP.detect_phrases(
                media.audio_path,
                params.speech_phrases,
                on_progress=lambda p, m: emit(0.24 + 0.01 * p, "speech", m),
                cancel=cancel,
                model=params.speech_model,
                fuzzy=params.speech_fuzzy,
            )
            speech_trace = dict(speech_res.trace)
            speech_trace["available"] = bool(speech_res.available)
            speech_trace["count"] = int(speech_res.times.size)
        elif not params.use_speech:
            speech_trace = {"disabled": "params.use_speech = false"}
        elif not params.speech_phrases:
            speech_trace = {"disabled": tr("analysis.trace.speech.no_phrases")}
        else:
            speech_trace = {"disabled": tr("analysis.trace.speech.no_audio")}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 3. frame motion
        motion_sig: MO.MotionSignal | None = None
        motion_dict: dict[str, np.ndarray] | None = None
        if params.use_motion:
            emit(0.26, "motion", tr("analysis.stage.motion"))
            motion_sig = MO.analyze_motion(
                media,
                sample_fps=min(params.sample_fps, 15.0),
                work_width=320,
                on=lambda p, m: emit(0.26 + 0.14 * p, "motion", m),
                cancel=cancel,
            )
            motion_dict = {"fps": np.array([motion_sig.fps], dtype=np.float32), "motion": motion_sig.motion,
                           "court_motion": motion_sig.court_motion, "sharpness": motion_sig.sharpness,
                           "shake": motion_sig.shake}
            motion_dict["fps"] = float(motion_sig.fps)
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 4. players
        player_sig = None
        players_dict: dict[str, np.ndarray] | None = None
        player_trace: dict[str, Any] = {}
        # Person-box size filtering: the threshold specified by the user in parameters (off is
        # equivalent to the old behavior). It and the "court ROI" are two complementary gates: ROI
        # governs "where to look", size governs "how big to look for".
        size_filter = {
            "mode": params.player_size_mode,
            "min_height": params.player_min_height,
            "max_height": params.player_max_height,
            "min_area": params.player_min_area,
            "max_area": params.player_max_area,
        }
        if params.use_players:
            mod = _try_import("players")
            if mod is not None and hasattr(mod, "analyze_players"):
                try:
                    emit(0.40, "players", tr("analysis.stage.players"))
                    player_sig = mod.analyze_players(
                        str(M.proxy_source(media)),
                        sample_fps=min(params.sample_fps, 12.0),
                        roi=eff_roi,
                        roi_poly=eff_poly,
                        size_filter=size_filter,
                        max_seconds=0.0,
                        viewpoint=(calib.viewpoint if calib is not None else "unknown"),
                        on_progress=lambda p, s: emit(0.40 + 0.24 * p, "players", s),
                        cancel=cancel,
                    )
                    players_dict = {
                        "fps": float(getattr(player_sig, "fps", params.sample_fps)),
                        "active_count": getattr(player_sig, "active_count", np.zeros(0)),
                        "active_speed": getattr(player_sig, "active_speed", np.zeros(0)),
                        "max_speed": getattr(player_sig, "max_speed", np.zeros(0)),
                    }
                    player_trace = {
                        "tracks": len(getattr(player_sig, "tracks", []) or []),
                        "active_ids": list(getattr(player_sig, "active_player_ids", []) or []),
                        # Measured distribution of size filtering: the UI uses it to draw a histogram and preview threshold effects in real time
                        "size_filter": dict(getattr(player_sig, "size_stats", {}) or {}),
                        "roi_poly_points": len(eff_poly or []),
                    }
                except Exception as e:  # a player-module failure should not interrupt the whole analysis
                    player_trace = {"error": f"{type(e).__name__}: {e}"}
                    log.opt(exception=e).warning("player detection degraded")
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 4.5 pose
        # Pose **does not change which signal is used for segmentation**, it only answers one
        # question: was this audio hit made in our match. In a multi-court gym audio cannot answer
        # this by itself (measured audio_reliability = 0.04), and without it one can only guess rally
        # boundaries from "hit intervals" — sounds from neighboring courts fill the intervals, so
        # rallies get glued longer. Any failure degrades: when pose_sig is None downstream behavior
        # matches having the switch off.
        pose_sig = None
        pose_trace: dict[str, Any] = {}
        gate_mask: np.ndarray | None = None
        if params.use_pose and hits is not None and player_sig is not None:
            boxes = getattr(player_sig, "frame_boxes", None)
            if boxes:
                mod = _try_import("pose")
                if mod is not None and hasattr(mod, "analyze_pose"):
                    try:
                        emit(0.64, "pose", tr("analysis.stage.pose"))
                        pose_sig = mod.analyze_pose(
                            str(M.proxy_source(media)),
                            boxes,
                            float(getattr(player_sig, "fps", 12.0) or 12.0),
                            on_progress=lambda p, s: emit(0.64 + 0.01 * p, "pose", s),
                            cancel=cancel,
                        )
                        if pose_sig is not None:
                            pose_trace = dict(pose_sig.trace)
                            gate_mask, g_trace = mod.gate_hits(
                                hits, pose_sig,
                                threshold=params.pose_gate_threshold,
                                window=params.pose_gate_window,
                                one_to_one=params.pose_gate_one_to_one,
                                force=params.pose_gate_force,
                            )
                            pose_trace.update(g_trace)
                            if gate_mask is not None:
                                hits = mod.filter_hits(hits, gate_mask)
                    except Exception as e:  # a pose failure can only degrade, never interrupt the whole analysis
                        pose_trace = {"error": f"{type(e).__name__}: {e}"}
                        log.opt(exception=e).warning("pose gating degraded")
                else:
                    pose_trace = {"disabled": tr("analysis.trace.pose.module_unavailable")}
            else:
                pose_trace = {"skipped": tr("analysis.trace.pose.no_boxes")}
        elif not params.use_pose:
            pose_trace = {"disabled": "params.use_pose = false"}
        else:
            pose_trace = {"skipped": tr("analysis.trace.pose.no_signals")}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 5. shuttlecock
        shuttle_sig = None
        shuttle_dict: dict[str, np.ndarray] | None = None
        shuttle_trace: dict[str, Any] = {}
        if params.use_shuttle:
            mod = _try_import("shuttle")
            if mod is not None and hasattr(mod, "analyze_shuttle"):
                try:
                    budget = float(params.shuttle_budget_seconds or 0.0)
                    limit = min(duration, budget) if budget > 0 else 0.0
                    coverage = (limit / duration) if (limit > 0 and duration > 0) else 1.0
                    emit(0.66, "shuttle", tr("analysis.stage.shuttle"))
                    shuttle_sig = mod.analyze_shuttle(
                        str(M.proxy_source(media)),
                        sample_fps=min(params.shuttle_fps, params.sample_fps),
                        roi=eff_roi,
                        max_seconds=limit,
                        on_progress=lambda p, s: emit(0.66 + 0.18 * p * max(coverage, 0.05), "shuttle", s),
                        cancel=cancel,
                    )
                    shuttle_trace = {"tracks": len(getattr(shuttle_sig, "tracks", []) or [])}
                    if coverage < 0.6:
                        # only part of the timeline is covered; keeping it would badly skew the fusion weights, so drop it
                        shuttle_trace["skipped"] = tr(
                            "analysis.trace.shuttle.coverage_low",
                            coverage=f"{coverage * 100:.0f}",
                            budget=f"{budget:.0f}",
                        )
                        shuttle_dict = None
                    else:
                        shuttle_dict = {
                            "fps": float(getattr(shuttle_sig, "fps", params.sample_fps)),
                            "presence": getattr(shuttle_sig, "presence", np.zeros(0)),
                            "max_candidate_speed": getattr(shuttle_sig, "max_candidate_speed", np.zeros(0)),
                        }
                        shuttle_trace["coverage"] = round(coverage, 3)
                except Exception as e:
                    shuttle_trace = {"error": f"{type(e).__name__}: {e}"}
                    log.opt(exception=e).warning("shuttle tracking degraded")
        else:
            shuttle_trace = {"disabled": tr("analysis.trace.shuttle.disabled")}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 6. fusion + segmentation
        emit(0.85, "segment", tr("analysis.stage.segment"))
        ana_fps = float(min(params.sample_fps, 15.0)) or 15.0
        fused = RA.fuse(
            fps=ana_fps,
            duration=duration,
            hits=hits,
            motion=motion_dict,
            players=players_dict,
            shuttle=shuttle_dict,
            roi_activity=motion_sig.motion if motion_sig is not None else None,
        )

        # Player detection has finished; determine the camera angle once more with real player boxes (this time backed by data)
        if calib is not None and player_sig is not None:
            try:
                CC.refine_viewpoint(
                    calib,
                    getattr(player_sig, "frame_boxes", None),
                    float(getattr(player_sig, "fps", 0.0) or 0.0),
                    (fw, fh),
                    hint=hint,      # type: ignore[arg-type]
                )
            except Exception as e:
                calib.note(f"机位二次判定失败：{type(e).__name__}: {e}")

        intervals, seg_trace = _segment_rallies(
            fused=fused,
            params=params,
            duration=duration,
            player_sig=player_sig,
            shuttle_sig=shuttle_sig,
            hits=hits,
        )

        # ---------------------------------------------------------- 7. boundary anchoring
        # Regardless of the segmentation method, pin the boundaries once with the **hit sequence**:
        #   1. split at large gaps in the hit sequence — removes "the next rally mixed into one rally";
        #   2. anchor the end to the last shot + the flight time needed for the shuttle to land —
        #      removes "a long stretch remaining after the shuttle lands". The old code used max()
        #      here, so the end could only be pushed later and could never come back, meaning even a
        #      69-second, 72-shot interval could not be fixed.
        # The **start** given by player-motion segmentation is "the player starts moving", which
        # already includes the serve preparation, so for it only the end is trimmed and the start is
        # not delayed.
        player_path = seg_trace.get("method") == "player_motion"
        if hits is not None:
            intervals = RA.refine_with_hits(
                intervals, hits,
                pre_roll=params.pre_roll,
                post_roll=params.post_roll,
                tail_seconds=params.hit_tail_seconds,
                trim_start=not player_path,
            )
        else:
            for iv in intervals:
                iv.start = max(0.0, iv.start - params.pre_roll)
                iv.end = min(duration, iv.end + params.post_roll)

        if not intervals and hits is not None and hits.times.size:
            # when vision has no signal at all, fall back to audio only
            intervals = RA.fallback_from_audio(hits, params)
            seg_trace["method"] = "audio_fallback"

        # The padding before/after adjacent rallies overlaps, which would make the same frame appear twice when concatenating clips
        intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=fused.fps, activity=fused.activity)

        # Filter out too-short rallies again
        intervals = [iv for iv in intervals if (iv.end - iv.start) >= params.min_rally_seconds]
        intervals.sort(key=lambda v: v.start)

        # Wrap-up: merge intervals that are still "abutting" into one. A real match necessarily has
        # a pause after a point, so two "rallies" within 0.3 seconds of each other must be one rally
        # split in two; leaving a fake boundary only makes people think "a new shuttle started here".
        intervals = _join_abutting(intervals)

        # ---------------------------------------------------------- 7.5 speech recall pass
        # The full-file decode is unstable on some 30 s chunks and can drop a clear short shout
        # (see speech.refine_phrases). When near-homophone matching is on, re-decode tight windows
        # over the rally spans and merge the hits before the bonus is applied. A failure or a
        # cancellation is ignored: the events from the main pass are kept.
        speech_events = _speech_events(speech_res)
        if (params.use_speech and params.speech_fuzzy and speech_res is not None
                and speech_res.available and media.audio_path and intervals):
            emit(0.90, "speech", tr("analysis.stage.speech_refine"))
            refined = SP.refine_phrases(
                media.audio_path,
                params.speech_phrases,
                _speech_regions(intervals),
                base_events=speech_events,
                on_progress=lambda p, m: emit(0.90 + 0.02 * p, "speech", m),
                cancel=cancel,
                model=params.speech_model,
                fuzzy=True,
            )
            if refined.available:
                speech_res = refined
                speech_events = _speech_events(refined)
                speech_trace.update(refined.trace.get("refine", {}))
                speech_trace["count"] = len(speech_events)

        # ---------------------------------------------------------- 8. features + scoring
        emit(0.92, "score", tr("analysis.stage.score"))
        RA.attach_features(intervals, fused, hits=hits, motion=motion_dict,
                           players=players_dict, shuttle=shuttle_dict)
        # The speech bonus is written after features and before scoring so that resegmentation uses
        # the same logic (see resegment) and both paths give the same scores.
        _apply_speech_bonus(intervals, speech_events, params)

        quality = _quality_per_rally(intervals, motion_sig, player_sig, fused.fps)
        weights = SC.PRESETS.get(weights_key, SC.PRESETS["balanced"])
        feats = [iv.features for iv in intervals]
        scores = SC.score_rallies(feats, weights, quality)

        rallies: list[Rally] = []
        for i, iv in enumerate(intervals):
            sc = scores[i] if i < len(scores) else {}
            q = quality[i] if i < len(quality) else None
            rallies.append(_build_rally(i, iv, hits, media, sc, params, q))

        # Assign serve/receive to specific players (depends on player-tracking results)
        attribute_sides(rallies, player_sig, fused.fps,
                        viewpoint=(calib.viewpoint if calib is not None else "unknown"))

        # Use "the midpoint of the highest-scoring rally" as the poster, far more reliable than a fixed 25% of the video length
        if rallies:
            best = max(rallies, key=lambda r: r.scores.total)
            try:
                M.ensure_poster(media, at=(best.start + best.end) / 2.0)
            except Exception:
                pass

        res.rallies = rallies
        res.hits = _hits_to_events(hits, intervals) if hits is not None else []
        res.signals = _pack_signals(fused, motion_sig, player_sig, shuttle_sig, hits, duration,
                                    hits_raw=hits_raw)
        # Pose signals are stored along with the "hit attribution" mask: fast resegmentation
        # (resegment) does not re-run AI, but it must reproduce the same gating results, otherwise
        # "resegment" and "re-run" would give different rally counts — which is exactly where users
        # get confused most.
        if pose_sig is not None:
            res.signals["pose_swing"] = _downsample(pose_sig.swing)
            res.signals["pose_ok"] = _downsample(pose_sig.ok)
            res.signals["pose_overhead"] = _downsample(pose_sig.overhead)
            res.signals["pose_fps"] = [round(float(pose_sig.fps), 3)]
            res.signals["pose_coverage"] = [round(float(pose_sig.coverage), 4)]
            # Full-rate swing + quiet level: the re-gating path at resegment time needs the same
            # resolution as the original ±window match. The downsampled copy above is display-only
            # and far too coarse (about one sample per 0.75 s) for a ±0.35 s window.
            res.signals["pose_swing_full"] = [round(float(v), 4) for v in pose_sig.swing]
            res.signals["pose_quiet"] = [round(float(pose_sig.quiet), 5)]
        # Note: what is stored here is the **post-gating** hit sequence, so the mask does not need to
        # be stored separately. Fast resegmentation (resegment) does not re-run AI; it reads the same
        # hits, so the gating result is naturally consistent. Storing the mask is instead risky: once
        # the mask length and the stored hit count mismatch (e.g. after re-running with different
        # parameters), applying it again would filter the hits twice.
        res.signal_fps = fused.fps
        res.stats = SC.derive_stats(feats, scores)
        res.stats["hit_trace"] = hit_trace
        res.stats["player_trace"] = player_trace
        res.stats["shuttle_trace"] = shuttle_trace
        res.stats["pose_trace"] = pose_trace
        res.stats["speech_trace"] = speech_trace
        # Speech events go into stats (not signals): the convention for signals is "pure float
        # curves", whereas these carry phrase strings; resegmentation reads them back to recompute the bonus.
        res.stats["speech"] = {
            "engine": speech_res.engine if speech_res is not None else "none",
            "available": bool(speech_res.available) if speech_res is not None else False,
            "events": speech_events,
        }
        res.stats["weights"] = weights_key
        res.stats["audio_reliability"] = round(float(fused.audio_reliability), 3)
        res.stats["component_weights"] = {k: round(v, 3) for k, v in fused.weights.items()}
        res.stats["roi"] = list(roi) if roi else None
        res.stats["effective_roi"] = list(eff_roi) if eff_roi else None
        # The polygon actually used for "in-court determination" (the court boundary when calibration succeeded; otherwise None)
        res.stats["effective_poly"] = [list(p) for p in eff_poly] if eff_poly else None
        res.stats["player_size_filter"] = dict(size_filter)
        res.stats["duration"] = duration
        res.stats["segmentation"] = seg_trace
        if motion_sig is not None:
            res.stats["auto_roi"] = list(motion_sig.roi)
        res.calibration = calib.as_payload() if calib is not None else None
        res.court = _court_payload(motion_sig, player_sig, calib)
        res.status = "done"
        emit(1.0, "done", tr("analysis.stage.done", n=len(rallies)))
        log.info("analysis done: rallies={} hits={}", len(rallies), int(hits.times.size) if hits is not None else 0)
        return res

    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        log.opt(exception=e).error("analysis failed at stage={}", res.stage)
        emit(res.progress, "error", str(e))
        return res
    finally:
        import time as _t

        res.finished_at = int(_t.time() * 1000)


# ------------------------------------------------------------------ helpers


def _manual_poly(raw: object) -> "np.ndarray | None":
    """Convert a manually calibrated court boundary (normalized 0~1, 4~24 points) into an array; return None if invalid.

    The accepted point-count upper limit comes from :data:`CC.MAX_POLY_POINTS`: homography fitting
    must exhaustively enumerate quadrilaterals among the polygon vertices, and more points only make
    it slower, not more accurate (a real court boundary has only a few inflection points). The area
    lower bound 0.02 targets waste calibrations like "four points almost coincident" — it makes the
    ROI degenerate into a tiny point and filters everyone out, which is worse than no calibration, so
    it is better treated as not calibrated.
    """
    if not raw:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    n = arr.shape[0]
    if n < CC.MIN_POLY_POINTS or n > CC.MAX_POLY_POINTS:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if float(np.abs(arr).max()) > 1.5:      # clearly not normalized coordinates (e.g. pixels were passed)
        return None
    area = CC.poly_area_norm(arr)
    if area < 0.02:
        return None
    return arr


#: Old name (from the era of only four corners). Kept so existing callers / tests need no changes.
_manual_quad = _manual_poly


def _build_rally(i: int, iv: RA.RallyInterval, hits, media: MediaInfo,
                 sc: dict[str, float], params: AnalysisParams,
                 quality: dict[str, float] | None = None) -> Rally:
    f = iv.features
    r = Rally(
        index=i + 1,
        start=round(iv.start, 3),
        end=round(iv.end, 3),
        duration=round(max(0.0, iv.end - iv.start), 3),
        clip_start=round(max(0.0, iv.start - params.pre_roll), 3),
        clip_end=round(min(media.duration, iv.end + params.post_roll), 3),
        serve_time=round(iv.serve_time, 3) if iv.serve_time is not None else None,
        receive_time=round(iv.receive_time, 3) if iv.receive_time is not None else None,
        scores=RallyScores(
            total=float(sc.get("total", 0.0)),
            length=float(sc.get("length", 0.0)),
            intensity=float(sc.get("intensity", 0.0)),
            technique=float(sc.get("technique", 0.0)),
            excitement=float(sc.get("excitement", 0.0)),
            production=float(sc.get("production", 0.0)),
        ),
        tags=list(sc.get("tags", [])),  # type: ignore[arg-type]
    )
    r.features = RallyFeatures(
        duration=float(f.get("duration", 0.0)),
        shot_count=int(f.get("shot_count", 0)),
        tempo=float(f.get("tempo", 0.0)),
        motion_energy=float(f.get("motion_mean", 0.0)),
        motion_peak=float(f.get("motion_peak", 0.0)),
        travel_near=0.0,
        travel_far=0.0,
        smash_count=int(f.get("smash_count", 0)),
        longest_exchange=int(f.get("shot_count", 0)),
        finish_intensity=float(f.get("finish_tempo", 0.0)),
        shuttle_speed_p50=float(f.get("shuttle_speed_p50", 0.0)),
        shuttle_speed_p95=float(f.get("shuttle_speed_p90", 0.0)),
        confidence=float(f.get("confidence", 0.5)),
        # Intermediate quantities needed for scoring are also stored, so that a different scoring scheme can recompute from the same raw values
        hit_strength_p90=float(f.get("hit_strength_p90", 0.0)),
        player_speed_mean=float(f.get("player_speed_mean", 0.0)),
        player_speed_max=float(f.get("player_speed_max", 0.0)),
        shuttle_presence=float(f.get("shuttle_presence", 0.0)),
        quality_sharpness=float((quality or {}).get("sharpness", 0.6)),
        quality_shake=float((quality or {}).get("shake", 0.3)),
        quality_subject_size=float((quality or {}).get("subject_size", 0.25)),
        speech_bonus=float(f.get("speech_bonus", 0.0)),
        speech_phrases=list(f.get("speech_phrases", []) or []),
    )
    # Shot-by-shot events. **Do not write near/far here yet**: whether near and far can be
    # distinguished must be inferred by `attribute_sides()` from the player-tracking results, and
    # when it cannot it stays unknown. The old version unconditionally wrote alternating "near"/"far"
    # here, so "inference failed" and "inference succeeded" were completely identical in the data —
    # i.e. a guess was passed off as a measurement.
    if hits is not None and iv.hit_indices:
        idx = iv.hit_indices
        ts = hits.times[idx]
        cs = hits.confidence[idx]
        for k in range(len(idx)):
            kind = "serve" if k == 0 else ("receive" if k == 1 else "unknown")
            r.shots.append(ShotEvent(
                time=round(float(ts[k]), 3),
                player="unknown",     # type: ignore[arg-type]
                kind=kind,            # type: ignore[arg-type]
                confidence=round(float(cs[k]), 3),
                speed=None,
            ))
    return r


def _hits_to_events(hits: AH.HitDetection, intervals: list[RA.RallyInterval]) -> list[ShotEvent]:
    keep: set[int] = set()
    for iv in intervals:
        keep.update(iv.hit_indices)
    out = []
    for i in sorted(keep):
        out.append(ShotEvent(time=round(float(hits.times[i]), 3),
                             confidence=round(float(hits.confidence[i]), 3)))
    return out


#: How far outside a rally boundary a speech moment can fall and still count as a hit (seconds).
#: Recognition latency plus "shouting usually happens after the shuttle is dead" both make phrases
#: slightly exceed ``iv.end``; a 1-second tolerance is closer to the real meaning of "this rally was
#: well played" than insisting on the exact boundary.
SPEECH_MATCH_TOLERANCE = 1.0


def _speech_events(det: SP.SpeechDetection | None) -> list[dict[str, Any]]:
    """Flatten the detection results into a serializable event list (for stats storage and resegmentation reuse)."""
    if det is None:
        return []
    return [{"t": round(float(t), 3), "phrase": str(lab)}
            for t, lab in zip(det.times.tolist(), det.labels)]


def _speech_regions(intervals: list[RA.RallyInterval]) -> list[tuple[float, float]]:
    """Rally spans for the targeted speech recall pass.

    ``RallyInterval`` only carries ``start``/``end``; the padded ``clip_start``/``clip_end`` are
    built later on the final ``Rally``. ``speech._candidate_windows`` pads each region itself, so
    the raw span is the right input here.
    """
    return [(iv.start, iv.end) for iv in intervals]


def _apply_speech_bonus(intervals: list[RA.RallyInterval],
                        events: list[dict[str, Any]] | None,
                        params: AnalysisParams) -> None:
    """Convert matched phrases into a bonus for each rally and write it into ``iv.features``.

    The bonus scheme is **each distinct phrase hit** adds ``speech_bonus_points`` (at most 2
    phrases, naturally capped at 2×points); the same phrase shouted multiple times within a rally is
    not added repeatedly. Rallies with no hit explicitly get 0 — after resegmentation the old value
    must be cleared, otherwise the previous bonus would persist.
    """
    evs = list(events or [])
    pts = float(getattr(params, "speech_bonus_points", 10.0) or 0.0)
    tol = SPEECH_MATCH_TOLERANCE
    for iv in intervals:
        matched: list[str] = []
        for ev in evs:
            try:
                t = float(ev.get("t", 0.0))
                ph = str(ev.get("phrase", ""))
            except (TypeError, ValueError):
                continue
            if not ph:
                continue
            if iv.start - tol <= t <= iv.end + tol and ph not in matched:
                matched.append(ph)
        iv.features["speech_phrases"] = matched
        iv.features["speech_bonus"] = pts * len(matched)


def _quality_per_rally(intervals, motion_sig, player_sig, fps: float) -> list[dict[str, float]]:
    """Picture quality of each rally (sharpness / shake / subject size)."""
    out = []
    for iv in intervals:
        q = {"sharpness": 0.6, "shake": 0.3, "subject_size": 0.25}
        if motion_sig is not None:
            q["sharpness"] = _mean_range(motion_sig.sharpness, motion_sig.fps, iv.start, iv.end, 0.6)
            q["shake"] = _mean_range(motion_sig.shake, motion_sig.fps, iv.start, iv.end, 0.3)
        if player_sig is not None:
            boxes = getattr(player_sig, "frame_boxes", None)
            pfps = float(getattr(player_sig, "fps", fps))
            if boxes:
                a = max(0, int(iv.start * pfps))
                b = min(len(boxes), max(a + 1, int(iv.end * pfps)))
                heights = []
                for fr in boxes[a:b]:
                    for item in fr:
                        try:
                            _, x1, y1, x2, y2 = item
                            heights.append(abs(y2 - y1))
                        except Exception:
                            continue
                if heights:
                    q["subject_size"] = float(np.median(heights))
        out.append(q)
    return out


def _mean_range(arr, fps, t0, t1, default: float) -> float:
    if arr is None or len(arr) == 0 or fps <= 0:
        return default
    a = max(0, int(t0 * fps))
    b = min(len(arr), max(a + 1, int(t1 * fps)))
    if b <= a:
        return default
    return float(np.asarray(arr[a:b], dtype=np.float32).mean())


def _downsample(a: np.ndarray, target: int = 2400) -> list[float]:
    if a is None or len(a) == 0:
        return []
    a = np.asarray(a, dtype=np.float32)
    if a.size <= target:
        return [round(float(v), 4) for v in a]
    idx = np.linspace(0, a.size - 1, target).astype(np.int64)
    return [round(float(v), 4) for v in a[idx]]


def _pack_signals(fused, motion_sig, player_sig, shuttle_sig, hits, duration: float,
                  hits_raw=None) -> dict[str, list[float]]:
    """Package downsampled signals for the frontend to draw.

    The **full-frame-rate fused activity** is also stored (``activity_full``): this way, when the user
    tunes "segmentation granularity / gap determination", only the segmentation step needs to be
    re-run (millisecond-level), without waiting minutes for player detection and motion analysis.

    The **pre-gate hit sequence** (``hits_raw``) is stored too: the attribution gate that removes
    neighboring-court sounds runs during the full analysis, and without the candidates it produced,
    fast resegmentation could only re-read the already-gated list. Storing both lets resegment apply a
    different gate threshold to the raw candidates (idempotent, never double-filtered).
    """
    out: dict[str, list[float]] = {
        "activity": _downsample(fused.activity),
        "activity_full": [round(float(v), 4) for v in fused.activity],
        "duration": [round(duration, 3)],
        "fps": [round(fused.fps, 3)],
        "threshold_hi": [round(fused.threshold_hi, 4)],
        "threshold_lo": [round(fused.threshold_lo, 4)],
    }
    for k, v in fused.components.items():
        out[f"component_{k}"] = _downsample(v)
    out["weights"] = [round(fused.weights.get(k, 0.0), 3) for k in
                      ("players", "motion", "audio_hits", "shuttle", "roi")]
    if motion_sig is not None:
        out["motion"] = _downsample(motion_sig.motion)
        out["motion_fps"] = [round(motion_sig.fps, 3)]
    if player_sig is not None:
        out["active_count"] = _downsample(getattr(player_sig, "active_count", np.zeros(0)))
        out["active_speed"] = _downsample(getattr(player_sig, "active_speed", np.zeros(0)))
        out["player_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
        # Store the "player motion curve" at **full frame rate**: it is the most useful signal when
        # tuning segmentation parameters (quiet segments are the rally boundaries), and only storing
        # it at full frame rate lets fast resegmentation reproduce faithfully.
        boxes = getattr(player_sig, "frame_boxes", None)
        pfps = float(getattr(player_sig, "fps", 0.0) or 0.0)
        if boxes and pfps > 0:
            try:
                pm = RV.box_motion(list(boxes), pfps, window=1.0)
                pm = RV._robust_norm(RV._smooth(pm, max(1, int(pfps * 0.5))))
                out["player_motion_full"] = [round(float(v), 4) for v in pm]
                out["player_motion"] = _downsample(pm)
                # Per-frame "is player detection valid" is also stored at full frame rate: fast
                # resegmentation uses it to decide "for this stretch, trust the players or trust
                # activity", and to restrict boundaries to where detection is valid. Storing it lets
                # resegment fully reproduce "player-motion segmentation" without degrading to
                # activity only.
                cov = RV.detection_coverage(list(boxes), pfps)
                out["player_coverage_full"] = [1.0 if v > 0.5 else 0.0 for v in cov]
            except Exception:
                pass
        # Per-frame "subject horizontal center" + subject width, for automatic follow-crop in portrait
        if boxes:
            xs = np.full(len(boxes), np.nan, dtype=np.float32)
            ws = np.zeros(len(boxes), dtype=np.float32)
            for i, fr in enumerate(boxes):
                if not fr:
                    continue
                cx = [(float(b[1]) + float(b[3])) / 2.0 for b in fr]
                wd = [abs(float(b[3]) - float(b[1])) for b in fr]
                xs[i] = float(np.mean(cx))
                ws[i] = float(np.max(wd)) if wd else 0.0
            # Fill missing frames by forward fill, then apply a moving average to get a smooth follow path
            idx = np.arange(xs.size)
            good = ~np.isnan(xs)
            if good.sum() > 2:
                xs = np.interp(idx, idx[good], xs[good]).astype(np.float32)
                out["subject_x"] = _downsample(xs)
                out["subject_w"] = _downsample(ws)
                out["subject_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
    if player_sig is not None:
        # The **left/right span** of the two players in the frame: portrait cropping wants "both
        # people in frame", and using the mean center would aim the camera between the two (which
        # under a side camera means aiming at neither).
        left, right = _subject_span(getattr(player_sig, "frame_boxes", None) or [])
        if left is not None and right is not None:
            out["subject_left"] = _downsample(left)
            out["subject_right"] = _downsample(right)
            out["subject_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
    if shuttle_sig is not None:
        out["shuttle_presence"] = _downsample(getattr(shuttle_sig, "presence", np.zeros(0)))
        out["shuttle_fps"] = [round(float(getattr(shuttle_sig, "fps", 0.0)), 3)]
    if hits is not None and hits.times.size:
        out["hit_times"] = [round(float(v), 3) for v in hits.times]
        out["hit_strength"] = [round(float(v), 3) for v in hits.strength]
        out["hit_confidence"] = [round(float(v), 3) for v in hits.confidence]
        out["envelope"] = _downsample(hits.envelope)
        out["envelope_fps"] = [round(float(hits.env_fps), 3)]
    if hits_raw is not None and hits_raw.times.size:
        # Always persist the pre-gate candidates (even when the gate was skipped, raw == gated).
        # Otherwise a gate that was skipped by its safety guard could never be re-tuned or forced
        # later, because resegment would find nothing to re-gate from.
        out["hit_times_raw"] = [round(float(v), 3) for v in hits_raw.times]
        out["hit_strength_raw"] = [round(float(v), 3) for v in hits_raw.strength]
        out["hit_confidence_raw"] = [round(float(v), 3) for v in hits_raw.confidence]
    return out


def _subject_span(frame_boxes: list) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-frame "leftmost / rightmost boundary of all players on court", for portrait automatic cropping.

    The old implementation took the **mean** of all box centers: under a side camera the two players
    are at opposite ends of the frame, so the mean falls exactly between them and the crop shows
    neither clearly. With the left/right envelope, the crop can be sized to "fit all of [left, right]",
    so doubles can both be in frame.
    """
    n = len(frame_boxes)
    if n == 0:
        return None, None
    left = np.full(n, np.nan, dtype=np.float32)
    right = np.full(n, np.nan, dtype=np.float32)
    for i, fr in enumerate(frame_boxes):
        if not fr:
            continue
        xs0 = [float(b[1]) for b in fr if len(b) >= 5]
        xs1 = [float(b[3]) for b in fr if len(b) >= 5]
        if not xs0:
            continue
        left[i] = min(xs0)
        right[i] = max(xs1)
    idx = np.arange(n)
    for arr in (left, right):
        good = ~np.isnan(arr)
        if good.sum() > 2:
            arr[:] = np.interp(idx, idx[good], arr[good])
    # smooth to avoid crop-box jitter
    win = max(1, int(round(0.8 * 12)))
    if left.size > win:
        left = RV._smooth(left, win)
        right = RV._smooth(right, win)
    return left, right


def _court_payload(motion_sig, player_sig, calib: CC.CourtCalibration | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"source": "auto"}
    if motion_sig is not None:
        payload["roi"] = list(motion_sig.roi)
    if player_sig is not None and getattr(player_sig, "frame_boxes", None):
        payload["active_player_ids"] = list(getattr(player_sig, "active_player_ids", []) or [])
    if calib is not None:
        payload["calibration"] = calib.as_payload()
        if calib.ok:
            payload["source"] = "color+homography"
    return payload


# ------------------------------------------------------------------ segmentation dispatch


def _join_abutting(intervals: list[RA.RallyInterval],
                   gap: float = 0.3) -> list[RA.RallyInterval]:
    """Merge adjacent intervals whose gap is less than ``gap`` seconds into one.

    A real match necessarily has a pause for picking up the shuttle / changing serve after a point,
    so there is never no gap between two rallies. "The previous one's end exactly equals the next
    one's start" means this is actually one rally split in two (a player was briefly occluded, or
    activity twitched). Merging is better than leaving a fake boundary: a fake boundary makes the
    user think a new shuttle started here and cuts out an extra clip on the timeline.
    """
    if len(intervals) < 2:
        return intervals
    out: list[RA.RallyInterval] = []
    for iv in sorted(intervals, key=lambda v: v.start):
        if out and iv.start - out[-1].end < gap:
            prev = out[-1]
            prev.end = max(prev.end, iv.end)
            prev.confidence = max(prev.confidence, iv.confidence)
            # The hit indices are no longer contiguous after merging, but they are **still valid**
            # (they are the union of global indices). The old code cleared them here, so merged
            # rallies all showed "0 shots" — and since shot_count participates in scoring, the
            # "glued-together rallies" could not get technique points.
            prev.hit_indices = sorted(set(prev.hit_indices) | set(iv.hit_indices))
            if prev.features.get("hit_anchored") and iv.features.get("hit_anchored"):
                g = max(float(prev.features.get("tail_gap", 0.0)),
                        float(iv.features.get("tail_gap", 0.0)))
                prev.features["tail_gap"] = g
            if prev.serve_time is None:
                prev.serve_time = iv.serve_time
            continue
        out.append(iv)
    return out


def _segments_to_intervals(segs: list[RV.RawSegment]) -> list[RA.RallyInterval]:
    """Convert segmentation results into rally intervals (adjacent candidates are merged by :func:`_join_abutting`)."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, -s.end))
    groups: list[list[RV.RawSegment]] = [[segs[0]]]
    for s in segs[1:]:
        if s.start - groups[-1][-1].end < 0.3:
            groups[-1].append(s)
        else:
            groups.append([s])
    out: list[RA.RallyInterval] = []
    for g in groups:
        start = min(x.start for x in g)
        end = max(x.end for x in g)
        score = max(x.score for x in g)
        iv = RA.RallyInterval(start=start, end=end,
                              confidence=float(np.clip(0.25 + 0.75 * score, 0.05, 1.0)))
        for x in g:
            for k, v in (x.meta or {}).items():
                if isinstance(v, (int, float)):
                    iv.features[k] = float(v)
        out.append(iv)
    return out


def _seg_quality(segs: list[RV.RawSegment], fps: float = 15.0,
                 hit_density: np.ndarray | None = None) -> float:
    """Score how trustworthy a set of segmentation results is, used to pick the best under hybrid mode.

    Rewards "there really are pauses between rallies" and "a reasonable duration distribution", and
    penalizes two things:

    * **Abutting end to start**: the previous rally's end exactly equals the next one's start. A real
      match necessarily has shuttle-picking / serve-changing after a point, so abutting ends mean the
      gaps "between rallies" were not recognized at all.
    * **Too-uniform durations**: a bunch of nearly equal-length segments means mechanical equal division.

    Additionally (when ``hit_density`` is available) it rewards the contrast of "dense hits inside
    rallies, sparse between them". Without it, the selection would pick a neat-looking one out of two
    **both-inaccurate** paths: measured, after giving relaxed parameters to both the player path and
    the activity path, the activity path scored higher due to fewer segments and larger gaps, so the
    real "player motion × hit density" evidence path was discarded (the "overall drift" warned about
    in section 10 of HANDOVER).
    """
    if not segs:
        return 0.0
    segs = sorted(segs, key=lambda s: s.start)
    n = len(segs)
    durs = np.asarray([s.end - s.start for s in segs], dtype=np.float64)
    gaps = np.asarray([max(0.0, segs[i + 1].start - segs[i].end) for i in range(n - 1)],
                      dtype=np.float64) if n > 1 else np.zeros(0)
    rest_ratio = float(np.mean(gaps > 1.0)) if gaps.size else 0.0
    # Duration dispersion: too large (long rests mixed in) or too small (mechanical equal division) are both bad
    cv = float(durs.std() / max(durs.mean(), 1e-6))
    spread_score = float(np.clip(cv / 0.45, 0.0, 1.0)) if cv < 0.45 else float(np.clip((1.2 - cv) / 0.6, 0.0, 1.0))
    dur_ok = float(np.clip(np.mean((durs >= 4.0) & (durs <= 90.0)), 0.0, 1.0))
    mean_score = float(np.mean([s.score for s in segs]))
    q = float(np.clip(0.45 * rest_ratio + 0.25 * spread_score + 0.15 * dur_ok + 0.15 * mean_score,
                      0.0, 1.0))
    if hit_density is not None and hit_density.size and fps > 0:
        # An "F-score proxy" for hit evidence: p = mean hit density inside segments, r = the
        # fraction of total hit density covered by the segments. It directly measures "whether the
        # chosen segments enclose the moments of continuous play", closer to what is really wanted
        # than shape metrics like segment count / gaps. When evidence is available it dominates:
        # shape metrics alone would pick a mechanical equal-division result with "few segments, large
        # gaps" (measured: the activity path had 5 segments with a shape score of 0.86, but covered
        # only 19% of the hit evidence).
        m = hit_density.size
        mask = np.zeros(m, dtype=bool)
        for s in segs:
            a = max(0, min(m - 1, int(s.start * fps)))
            b = max(a + 1, min(m, int(s.end * fps)))
            mask[a:b] = True
        total = float(hit_density.sum())
        if mask.any() and total > 0.0:
            p = float(hit_density[mask].mean())
            r = float(hit_density[mask].sum() / total)
            hit_f = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
            q = 0.3 * q + 0.7 * float(np.clip(hit_f, 0.0, 1.0))
    return float(np.clip(q, 0.0, 1.0))


def _segment_rallies(
    fused: RA.FusedSignal,
    params: AnalysisParams,
    duration: float,
    player_sig=None,
    shuttle_sig=None,
    player_motion: np.ndarray | None = None,
    player_coverage: np.ndarray | None = None,
    hits: Any = None,
    opt_override: RV.SegmentOptions | None = None,
) -> tuple[list[RA.RallyInterval], dict[str, Any]]:
    """Choose the segmentation method by ``params.segment_mode``, returning (intervals, diagnostics).

    Priority:

    1. **Player-motion segmentation** (``player_motion``): the player is the only observed object
       that runs through the whole "serve -> rally -> shuttle dead" process and is not affected by
       neighboring courts. Its "quiet segments" are the rally boundaries; no manual threshold is
       needed, and abutting segments do not occur.
    2. **Activity-valley segmentation** (``activity_valleys``): use fused activity when there is no
       player signal, but only accept valleys with sufficient prominence, with no mechanical equal division.
    3. **The old hysteresis state machine** (``activity_state_machine``): the last fallback when the
       first two produce nothing, with behavior matching the old version.

    ``player_motion`` / ``player_coverage`` are the **already computed** player motion curve and
    detection coverage (both at ``fused.fps``). Fast resegmentation (:func:`resegment`) has no player
    boxes, but these two curves were stored during analysis, so it can also take the "player-motion
    segmentation" path instead of being forced to degrade to activity only — otherwise the user would
    see "the rally count after resegmentation does not match a full re-run".
    """
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds),
        max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll),
        post_roll=float(params.post_roll),
        # Quiet-segment scale: values calibrated from manual annotation take priority, falling back
        # to SegmentOptions defaults when absent from old parameters (getattr fallback keeps legacy
        # projects working after deserialization).
        min_quiet=float(getattr(params, "seg_min_quiet", 0.7)),
        prominence_ratio=float(getattr(params, "seg_prominence", 0.18)),
        min_rest=float(getattr(params, "seg_min_rest", 0.8)),
        min_core=float(getattr(params, "seg_min_core", 1.0)),
    )
    #: Allows the caller (annotation-calibration script / tests) to override the quiet-segment
    #: detection parameters. These are not in ``AnalysisParams`` (they are internal scales of the
    #: segmentation algorithm), but calibrating them with manual annotation is exactly the most valuable thing.
    if opt_override is not None:
        opt = opt_override
    shuttle_presence = getattr(shuttle_sig, "presence", None)
    shuttle_fps = float(getattr(shuttle_sig, "fps", 0.0) or 0.0) if shuttle_sig is not None else 0.0
    boxes = getattr(player_sig, "frame_boxes", None) if player_sig is not None else None
    player_fps = float(getattr(player_sig, "fps", 0.0) or 0.0) if player_sig is not None else 0.0

    # Hit density: the most discriminative evidence on this footage. It is used in both paths —
    # multiplied into player motion (`audio_visual_evidence`) and passed to `segment_visual`.
    # Fast resegmentation (resegment) reconstructs it via `hits` without re-running AI.
    hit_density = (RA.hit_density_signal(hits, fused.fps, fused.activity.size)
                   if hits is not None and getattr(hits, "times", np.zeros(0)).size else None)

    mode = params.segment_mode
    trace: dict[str, Any] = {"mode": mode}

    vis: list[RV.RawSegment] = []
    coverage = 0.0
    #: Per-frame coverage array — note not to confuse it with ``coverage`` above (the fraction of
    #: valid time): one feeds "per-time-window selection", the other is a diagnostic scalar for the user.
    cov_arr: np.ndarray | None = None
    #: Player motion curve (at ``fused.fps``) — used by "per-time-window selection" to judge whether
    #: "player segmentation gave no candidate" is a miss or a real pause.
    pm_arr: np.ndarray | None = None
    want_players = mode in ("auto", "hybrid")
    if want_players and player_motion is not None and player_motion.size:
        # Fast resegmentation path: the player motion curve and coverage were stored during analysis, so reuse them directly.
        pm = RV._resample_to(player_motion, player_fps or fused.fps, fused.fps,
                             fused.activity.size)
        cov_arr = (RV._resample_to(player_coverage, player_fps or fused.fps, fused.fps,
                                   fused.activity.size)
                   if player_coverage is not None and player_coverage.size else None)
        coverage = float(np.mean(cov_arr > 0.5)) if cov_arr is not None else 0.0
        pm_arr = pm
        # Segmentation uses "player motion × hit density"; `pm_arr` keeps the raw player motion for
        # `_merge_by_availability` to judge "did the player actually move" (that should not be
        # affected by hit density, otherwise missed hits would misjudge a "real pause" as "playing").
        ev = RV.audio_visual_evidence(pm, hit_density)
        vis = RV.segment_by_player_motion(RV.SegmentSignals(
            fps=fused.fps, duration=duration, player_motion=ev,
            activity=fused.activity,
            shuttle=(RV._resample_to(shuttle_presence, shuttle_fps, fused.fps,
                                     fused.activity.size)
                     if shuttle_presence is not None and shuttle_fps > 0 else None),
            has_players=True, coverage=cov_arr, coverage_ratio=coverage,
        ), opt)
        if vis:
            vis = RV.verify_with_shuttle(vis, RV.SegmentSignals(
                fps=fused.fps, duration=duration, player_motion=ev, has_players=True,
                shuttle=(RV._resample_to(shuttle_presence, shuttle_fps, fused.fps,
                                         fused.activity.size)
                         if shuttle_presence is not None and shuttle_fps > 0 else None),
            ), opt)
    elif want_players:
        vis, coverage = RV.segment_visual(
            fps=fused.fps, duration=duration,
            player_boxes=boxes, player_fps=player_fps,
            activity=fused.activity,
            shuttle_presence=shuttle_presence, shuttle_fps=shuttle_fps,
            hit_density=hit_density, opt=opt,
        )
        if boxes and player_fps > 0:
            # Only to judge whether the player moved during "selection"; normalization is not needed (the threshold is relative)
            try:
                pm_arr = RV._resample_to(RV.box_motion(list(boxes), player_fps, window=1.0),
                                         player_fps, fused.fps, fused.activity.size)
            except Exception:
                pm_arr = None
    trace["player_coverage"] = round(float(coverage), 3)
    if player_motion is not None and player_motion.size:
        trace["player_signal"] = "reused"

    # Activity segmentation is cheap (just a valley detection), so it is always computed once in auto
    # mode. Only once computed is "per-time-window selection" even possible; using a threshold like
    # "only compute when coverage is below some number" would instead drop the whole path near the
    # critical value (measured: both 0.49 and 0.53 occurred).
    need_activity = mode in ("activity", "hybrid", "auto")

    act: list[RV.RawSegment] = []
    if need_activity:
        act = RV.segment_activity(
            fused.activity, fused.fps, duration, opt,
            shuttle_presence=shuttle_presence, shuttle_fps=shuttle_fps,
        )
    trace["activity_count"] = len(act)

    # auto mode: decide by **time window** whom to trust. Where player detection is good, use player
    # segmentation (the boundary is where the player really stops, the most reliable); where player
    # detection fails, fill in with activity segmentation. This decision must be local: in the same
    # footage the player may be detectable in the first half and not in the second, and using a
    # whole-video average coverage to make a "global either/or" would drop an entire stretch.
    #
    # Note this does **not** overwrite ``vis``: the merged result participates in the later selection
    # as a **parallel candidate**. The old implementation did `vis = merged`, making the "merge"
    # action impossible to reject; measured, after handing the hit-density evidence path to the
    # merger, it would stitch a long candidate back together within a window (measured: stitched a
    # 19s interval, gluing two real rallies together), while `_seg_quality` thought it "neater"
    # because it only looks at segment count/gaps. Keeping it as a parallel candidate allows
    # evidence-based selection.
    merged_vis: list[RV.RawSegment] | None = None
    if (mode == "auto" and vis and act and (boxes or cov_arr is not None)
            and hit_density is None):
        merged_vis = _merge_by_availability(
            vis, act, boxes, player_fps, fused.fps, duration,
            coverage=cov_arr, player_motion=pm_arr,
            merge_trace=trace.setdefault("windows", {}))
        trace["merged_from"] = {"player_motion": len(vis), "activity_valleys": len(act)}

    legacy: list[RA.RallyInterval] = []
    if not vis and not merged_vis and not act:
        legacy = RA.segment(
            fused,
            gap_seconds=params.gap_seconds,
            min_seconds=params.min_rally_seconds,
            max_seconds=params.max_rally_seconds,
            target_seconds=params.target_rally_seconds,
            split_sensitivity=params.split_sensitivity,
        )
        trace.update(method="activity_state_machine", count=len(legacy))
        return legacy, trace

    if mode == "hybrid" and vis and act:
        qv, qa = (_seg_quality(vis, fused.fps, hit_density),
                  _seg_quality(act, fused.fps, hit_density))
        trace["quality"] = {"player_motion": round(qv, 3), "activity_valleys": round(qa, 3)}
        if qa > qv + 0.05:
            trace.update(method="activity_valleys", count=len(act))
            return _segments_to_intervals(act), trace
        trace.update(method="player_motion", count=len(vis))
        return _segments_to_intervals(vis), trace

    # Three parallel candidates are selected among: the unmerged evidence path / the path merged by
    # time window / pure activity. For the judgment see :func:`_seg_quality` (where "hit-evidence
    # alignment" is the objective basis for "which path really matches the fact of continuous play").
    cands: list[tuple[list[RV.RawSegment], str]] = []
    if vis:
        cands.append((vis, "player_motion"))
    if merged_vis:
        cands.append((merged_vis, "player_motion+activity"))
    if act:
        cands.append((act, "activity_valleys"))
    scored = [(_seg_quality(c, fused.fps, hit_density), c, name) for c, name in cands]
    trace["quality"] = {name: round(q, 3) for q, _, name in scored}
    best = max(scored, key=lambda t: t[0]) if scored else (0.0, [], "empty")
    trace.update(method=best[2], count=len(best[1]))
    return _segments_to_intervals(best[1]), trace


def _merge_by_availability(
    vis: list[RV.RawSegment],
    act: list[RV.RawSegment],
    player_boxes: list,
    player_fps: float,
    fps: float,
    duration: float,
    coverage: np.ndarray | None = None,
    player_motion: np.ndarray | None = None,
    merge_trace: dict[str, Any] | None = None,
) -> list[RV.RawSegment]:
    """Select between "player segmentation" and "activity segmentation" by time window.

    The approach is to split the candidates from both sides into non-overlapping time windows, and
    within each window:

    * compute the **valid fraction of player detection** in this window (``detection_coverage``);
    * high valid fraction -> use the rally from player segmentation (the boundary is where the
      player really stops, the most reliable);
    * low valid fraction -> use the rally from activity segmentation (player segmentation is
      untrustworthy here; its "quiet" may just be a failure to detect anyone).

    Why it must be a local decision: in the same footage the match players may be detectable in the
    first half and not in the second. Using a whole-video average coverage for a "global either/or"
    either drops rallies entirely during stretches with poor player signal, or gets skewed by the
    coarse boundaries of activity segmentation during stretches with good player signal. Measured on
    30-minute footage, player detection is valid only 30%~54% of the time, and the two approaches
    differ by 2~3x in rally count.

    **Each window trusts only one side**, with three cases:

    * player detection valid -> player segmentation decides;
    * player detection valid but player segmentation gave no candidate -> look at **whether the
      player actually moved**: if the player is moving it is a miss by player segmentation (it has
      thresholds on "movement duty cycle / core segment length", and mixed segments are easily
      dropped whole), so fall back to activity segmentation; if the player did not move it is a real
      pause, leave it empty;
    * player detection invalid -> use activity segmentation.

    The middle case is essential: writing only "no candidate from player segmentation => leave
    empty" would drop entire rallies (measured: recall drops 6 percentage points), while writing
    only "no candidate => fall back to activity" would glue back together rallies that player
    segmentation had managed to split (measured: the longest rally grew from 22.7 seconds back to
    28.7 seconds).
    """
    if not vis:
        return act
    if not act:
        return vis
    n = max(1, int(round(duration * fps)))
    if coverage is not None and len(coverage) == n:
        # Fast resegmentation path: coverage was stored during analysis, use it directly
        cov = np.asarray(coverage, dtype=np.float32)
    elif player_boxes:
        cov = RV._resample_to(RV.detection_coverage(player_boxes, player_fps),
                              player_fps, fps, n)
    else:
        # Neither boxes nor stored coverage: cannot assert "detection invalid" out of thin air, since
        # that would assign all candidates to activity segmentation. Treat it as "valid throughout".
        cov = np.ones(n, dtype=np.float32)
    thr = 0.5

    def valid_ratio(a: float, b: float) -> float:
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        return float(np.mean(cov[i0:i1] > thr))

    # Player motion curve: used to judge whether "player segmentation gave no candidate" is a miss or
    # a real pause. The threshold matches rally_vision.segment_by_player_motion (rest floor + 25% dynamic range).
    if player_motion is not None and player_motion.size:
        if player_motion.size == n:
            pm = np.asarray(player_motion, dtype=np.float32)
        else:
            # The caller has usually already resampled to fps; as a fallback here, derive its own
            # frame rate from "array length / total duration" and resample.
            src_fps = player_motion.size / max(duration, 1e-6)
            pm = RV._resample_to(player_motion, src_fps, fps, n)
        base = float(np.percentile(pm, 20))
        span = max(float(np.percentile(pm, 95)) - base, 1e-9)
        move_thr = base + 0.25 * span
        moving = pm >= move_thr
    else:
        moving = None

    def moving_ratio(a: float, b: float) -> float:
        if moving is None:
            return 0.0
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        return float(np.mean(moving[i0:i1]))

    def has_motion_burst(a: float, b: float, min_run: float = 1.2,
                         min_duty: float = 0.35) -> bool:
        """Whether this stretch contains a segment of "the player moving continuously for a while".

        Why not look at the average movement ratio of the whole stretch: a 20-second window may have
        only 8 seconds of play (the rest picking up the shuttle), so the average ratio is low, but
        **those 8 seconds are indeed a rally** — measured, this is exactly why the whole 29~51 second
        stretch was judged "nobody moved", losing 20 seconds of match at once.

        The threshold is **deliberately relaxed to 1.2 seconds** versus ``min_core`` (2.5 seconds) in
        ``rally_vision``: this function is a **recall safety net**, and it is better to be a bit
        suspicious than to drop an entire rally; while the judgment "truly nobody moved" itself has
        strong enough evidence — the shuttle
        """
        if moving is None:
            return False
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        idx = np.nonzero(moving[i0:i1])[0]
        if idx.size == 0:
            return False
        gap = max(1, int(round(0.35 * fps)))
        splits = np.nonzero(np.diff(idx) > gap)[0]
        bounds = np.concatenate([[0], splits + 1, [idx.size]])
        for k in range(len(bounds) - 1):
            run = int(idx[bounds[k + 1] - 1]) - int(idx[bounds[k]])
            if run < min_run * fps:
                continue
            seg = moving[i0 + int(idx[bounds[k]]): i0 + int(idx[bounds[k + 1] - 1]) + 1]
            if float(np.mean(seg)) >= min_duty:
                return True
        return False

    # The time windows are determined only by the boundaries of the first two paths. **The hit
    # sequence (``extra``) deliberately does not participate in the division** — it is used only to
    # "fill gaps". The reason is practical: windows are cut by sorting boundary points, and one more
    # boundary path would re-divide all windows, so the "fill gaps" action would incidentally change
    # the already-fixed windows, causing overall drift (measured: the first rally disappeared
    # entirely). Filling gaps should only fill gaps.
    marks = sorted({0.0, duration}
                   | {s.start for s in vis} | {s.end for s in vis}
                   | {s.start for s in act} | {s.end for s in act})
    out: list[RV.RawSegment] = []
    used = {"player_motion": 0, "activity_valleys": 0, "rescued": 0, "empty": 0}
    for k in range(len(marks) - 1):
        w0, w1 = marks[k], marks[k + 1]
        if w1 - w0 < 0.2:
            continue
        mid = (w0 + w1) / 2.0
        # Key: **choose whom to trust by whether player detection is actually valid in this window**,
        # not "whoever has a candidate wins". The old implementation wrote "if vis has a candidate use
        # vis, otherwise fall back to act", computing valid_ratio but never using it — so wherever
        # player segmentation **deliberately left a gap** (which is exactly the real pause between two
        # rallies), activity segmentation candidates filled it in, gluing back together rallies that
        # player segmentation had managed to split.
        if valid_ratio(w0, w1) >= thr:
            candidates = [s for s in vis if s.start <= mid < s.end]
            src = "player_motion"
            if not candidates and has_motion_burst(w0, w1):
                # The player really did move continuously -> player segmentation missed this window;
                # catch it with activity candidates rather than dropping a whole rally. Only when both
                # paths are silent does it count as "this window truly has no rally".
                candidates = [s for s in act if s.start <= mid < s.end]
                src = "rescued"
        else:
            candidates = [s for s in act if s.start <= mid < s.end]
            src = "activity_valleys"
        if not candidates:
            used["empty"] += 1
            continue
        used[src] += 1
        s = max(candidates, key=lambda x: x.end - x.start)
        out.append(RV.RawSegment(start=max(w0, s.start), end=min(w1, s.end),
                                 score=s.score, meta=dict(s.meta or {})))
    if merge_trace is not None:
        merge_trace.update(used)

    # Adjacent windows may come from the same candidate, so merge them directly by time
    merged: list[RV.RawSegment] = []
    for s in sorted(out, key=lambda x: x.start):
        if merged and s.start - merged[-1].end < 0.3:
            prev = merged[-1]
            prev.end = max(prev.end, s.end)
            prev.score = max(prev.score, s.score)
            continue
        merged.append(s)
    return [s for s in merged if s.end - s.start >= 1.0]


__all__ = ["run_analysis", "resegment", "regate_hits", "recover_pose_from_cache"]


# ------------------------------------------------------------------ fast resegmentation


def resegment(
    res: AnalysisResult,
    params: AnalysisParams | None = None,
    weights_key: str = "balanced",
) -> AnalysisResult:
    """Re-run only "segmentation + features + scoring", reusing the already computed activity curve.

    Use it when tuning "segmentation granularity / gap determination / padding / scoring scheme";
    results come out in milliseconds without waiting minutes for player detection and motion analysis.
    The prerequisite is that the analysis result stored ``signals["activity_full"]`` (new-version
    analyses always store it).

    Per-shot information is rebuilt from the stored audio hit moments, so boundaries such as
    "serve / receive" can still align with the real hit sounds.
    """
    params = params or res.params
    sig = res.signals or {}
    full = sig.get("activity_full") or []
    if not full:
        raise RuntimeError(tr("analysis.resegment.no_activity"))

    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or 0.0
    if duration <= 0:
        duration = len(full) / fps

    act = np.asarray(full, dtype=np.float32)
    med = float(np.median(act))
    fused = RA.FusedSignal(
        fps=fps,
        duration=duration,
        activity=act,
        components={},
        weights={},
        threshold_hi=max(float(np.percentile(act, 78)), med * 1.25),
        threshold_lo=max(float(np.percentile(act, 55)) * 0.92, med * 1.05),
        audio_reliability=float(res.stats.get("audio_reliability", 1.0) or 1.0),
    )

    # Resegmentation has no player boxes, but the player **motion curve** and detection coverage were
    # stored during analysis (``player_motion_full`` / ``player_coverage_full``), so here it can take
    # exactly the same :func:`_segment_rallies` as a full analysis instead of degrading to activity
    # only. It is precisely this degradation in the old version that made users see "the rally count
    # after fast resegmentation does not match a full re-run" — and resegmentation is exactly the most
    # common operation when tuning "padding / min-max rally length".
    # Re-apply the hit attribution gate from the **pre-gate** candidates using the current params.
    # This is what lets the cross-court suppression slider take effect instantly: the stored raw
    # sequence is gated once here, never the already-gated one, so repeated resegments are idempotent.
    hits, regate_trace = regate_hits(sig, params)
    shuttle_presence = None
    shuttle_fps = 0.0
    if sig.get("shuttle_presence"):
        shuttle_presence = np.asarray(sig["shuttle_presence"], dtype=np.float32)
        shuttle_fps = float((sig.get("shuttle_fps") or [10.0])[0])

    pm_full = sig.get("player_motion_full") or []
    cov_full = sig.get("player_coverage_full") or []
    player_fps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    player_motion = (np.asarray(pm_full, dtype=np.float32)
                     if pm_full and player_fps > 0 else None)
    player_coverage = (np.asarray(cov_full, dtype=np.float32)
                       if cov_full and len(cov_full) == len(pm_full) else None)
    if player_motion is None:
        player_fps = 0.0

    intervals, seg_trace = _segment_rallies(
        fused=fused, params=params, duration=duration,
        player_sig=None, shuttle_sig=None,
        player_motion=player_motion, player_coverage=player_coverage,
        hits=hits,
    )
    seg_method = str(seg_trace.get("method") or "")

    # Boundary anchoring: both the fast and slow paths must do it, otherwise the "the end cannot be
    # pulled back" problem resurfaces on the resegmentation path (the user would see "it got longer again after resegmentation").
    if hits is not None:
        intervals = RA.refine_with_hits(
            intervals, hits,
            pre_roll=params.pre_roll,
            post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds,
            # Kept consistent with run_analysis: the start from player segmentation is "the player
            # starts moving", which already includes the serve preparation, and should not be delayed
            # to the first shot. The old resegment missed this switch (default True), so for the same
            # project "fast resegmentation" shifted the start later overall, not matching the boundaries
            # a full re-run gives.
            trim_start=seg_method != "player_motion",
        )
    else:
        for iv in intervals:
            iv.start = max(0.0, iv.start - params.pre_roll)
            iv.end = min(duration, iv.end + params.post_roll)

    intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=fps, activity=act)
    intervals = [iv for iv in intervals if (iv.end - iv.start) >= params.min_rally_seconds]
    intervals.sort(key=lambda v: v.start)
    # Kept consistent with run_analysis: two abutting segments are the two halves of one rally split
    # apart, and leaving a fake boundary only makes people think "a new shuttle started here".
    # (The old resegment missed this step, so "fast resegmentation" and "full re-run" gave different
    #  results — the rally count would change after resegmentation for the same project.)
    intervals = _join_abutting(intervals)

    motion_dict = _dict_from(sig, "motion", "motion_fps")
    players_dict = _dict_from_multi(sig, ("active_count", "active_speed", "max_speed"), "player_fps")
    shuttle_dict = _dict_from_multi(sig, ("presence", "max_candidate_speed"), "shuttle_fps")
    if shuttle_dict and "presence" in shuttle_dict:
        shuttle_dict["presence"] = shuttle_dict.pop("presence")

    RA.attach_features(intervals, fused, hits=hits, motion=motion_dict,
                       players=players_dict, shuttle=shuttle_dict)
    # Speech events were stored in stats during analysis; read them back here to recompute the bonus,
    # ensuring "fast resegmentation" and "full re-run" give consistent scores (changing phrases /
    # bonus values also takes effect immediately).
    _apply_speech_bonus(intervals, (res.stats or {}).get("speech", {}).get("events") or [], params)

    quality = [_quality_default() for _ in intervals]
    weights = SC.PRESETS.get(weights_key, SC.PRESETS["balanced"])
    feats = [iv.features for iv in intervals]
    scores = SC.score_rallies(feats, weights, quality)

    media_stub = _MediaStub(duration)
    kept: dict[str, Rally] = {}
    for r in res.rallies:
        kept[r.id] = r

    rallies: list[Rally] = []
    for i, iv in enumerate(intervals):
        sc = scores[i] if i < len(scores) else {}
        r = _build_rally(i, iv, hits, media_stub, sc, params)
        # Preserve the user's edits on old rallies: migrate starred / note / keep by time overlap
        old = _best_overlap(r, res.rallies)
        if old is not None:
            r.starred = old.starred
            r.note = old.note
            r.keep = old.keep
        rallies.append(r)

    res.rallies = rallies
    res.hits = _hits_to_events(hits, intervals) if hits is not None else []
    res.params = params
    # Refresh the display hit sequence with the newly gated result, but never touch ``*_raw`` so the
    # next resegment can gate again from the original candidates.
    _store_gated_hits(res.signals, hits)
    # First set aside the part of the statistics that "resegmentation does not change", then replace wholesale
    keep_keys = ("hit_trace", "player_trace", "shuttle_trace", "pose_trace",
                 "audio_reliability", "component_weights", "duration", "auto_roi",
                 "roi", "effective_roi", "effective_poly", "player_size_filter",
                 "segmentation", "speech", "speech_trace")
    carried = {k: v for k, v in (res.stats or {}).items() if k in keep_keys}
    carried.setdefault("duration", duration)
    res.stats = SC.derive_stats(feats, scores)
    res.stats.update(carried)
    # Merge the latest gating diagnostics so the UI reflects the current cross-court suppression.
    if regate_trace:
        pose_trace = dict(res.stats.get("pose_trace") or {})
        pose_trace.update(regate_trace)
        res.stats["pose_trace"] = pose_trace
    # Preserve segmentation diagnostics: previously this wrote only method/count, discarding which
    # side "per-time-window selection" trusted in each window and which windows were judged "no rally"
    # — and that is exactly the only useful information when investigating "why did this stretch of
    # rallies disappear".
    seg_trace = dict(seg_trace or {})
    seg_trace.update({"mode": params.segment_mode, "method": seg_method,
                      "count": len(intervals), "resegmented": True})
    res.stats["segmentation"] = seg_trace
    res.stats["weights"] = weights_key
    res.stats["resegmented"] = True
    res.signal_fps = fps
    return res


def _quality_default() -> dict[str, float]:
    return {"sharpness": 0.6, "shake": 0.3, "subject_size": 0.25}


class _MediaStub:
    """``_build_rally`` only uses duration, so this gives a minimal stand-in."""

    def __init__(self, duration: float) -> None:
        self.duration = duration


def _hits_from_keys(sig: dict, t_key: str, s_key: str, c_key: str) -> AH.HitDetection | None:
    t = sig.get(t_key) or []
    if not t:
        return None
    s = sig.get(s_key) or [1.0] * len(t)
    c = sig.get(c_key) or [0.5] * len(t)
    return AH.HitDetection(
        times=np.asarray(t, dtype=np.float64),
        strength=np.asarray(s, dtype=np.float32),
        confidence=np.asarray(c, dtype=np.float32),
        envelope=np.zeros(0),
        env_fps=float((sig.get("envelope_fps") or [250.0])[0]),
    )


def _rebuild_hits(sig: dict) -> AH.HitDetection | None:
    """Rebuild the (post-gate) hit sequence stored for display / backward compatibility."""
    return _hits_from_keys(sig, "hit_times", "hit_strength", "hit_confidence")


def _rebuild_hits_raw(sig: dict) -> AH.HitDetection | None:
    """Rebuild the pre-gate hit candidates; fall back to the stored sequence when absent (old analyses)."""
    return _hits_from_keys(sig, "hit_times_raw", "hit_strength_raw", "hit_confidence_raw") \
        or _rebuild_hits(sig)


def _rebuild_pose_signal(sig: dict):
    """Reconstruct the full-rate pose signal needed for re-gating; ``None`` when unavailable.

    Only ``swing`` (and its frame rate / quiet level / coverage) is needed by
    :func:`bms.analysis.pose.hit_swing_evidence`; the ``ok`` / ``overhead`` curves are display-only.
    """
    sw = sig.get("pose_swing_full")
    if not sw:
        return None
    fps = float((sig.get("pose_fps") or [0.0])[0]) or 0.0
    if fps <= 0:
        return None
    try:
        from . import pose as POSE
    except Exception:
        return None
    arr = np.asarray(sw, dtype=np.float32)
    return POSE.PoseSignal(
        fps=fps,
        duration=float(arr.size) / fps,
        swing=arr,
        ok=np.ones(arr.size, dtype=np.float32),
        overhead=np.zeros(arr.size, dtype=np.float32),
        coverage=float((sig.get("pose_coverage") or [0.0])[0]) or 0.0,
        quiet=float((sig.get("pose_quiet") or [0.0])[0]) or 0.0,
        trace={},
    )


def _store_gated_hits(sig: dict, hits: AH.HitDetection | None) -> None:
    """Write the gated hit sequence into ``signals`` for display, leaving ``*_raw`` untouched."""
    if not isinstance(sig, dict):
        return
    if hits is None or hits.times.size == 0:
        sig["hit_times"] = []
        sig["hit_strength"] = []
        sig["hit_confidence"] = []
        return
    sig["hit_times"] = [round(float(v), 3) for v in hits.times]
    sig["hit_strength"] = [round(float(v), 3) for v in hits.strength]
    sig["hit_confidence"] = [round(float(v), 3) for v in hits.confidence]


def regate_hits(sig: dict, params: AnalysisParams,
                hits_raw: AH.HitDetection | None = None,
                pose=None) -> tuple[AH.HitDetection | None, dict[str, Any]]:
    """Re-apply the hit attribution gate to the **pre-gate** candidates with the given params.

    Shared by :func:`resegment` and the annotation optimizer so that "predicted" and "applied"
    results cannot diverge. Always starts from the raw candidates, so calling it repeatedly with
    different thresholds is idempotent (never double-filters).

    Degrades explicitly (see the ``regate`` trace key) instead of silently changing the result:
    no raw candidates -> ``no_raw``; pose switched off -> ``pose_off``; no pose signal / module ->
    ``no_pose``; audio switched off -> ``audio_off``.

    When ``params.use_audio`` is false the whole hit path is dropped here, not just the gate: fast
    resegmentation must mirror a full analysis run with audio disabled, so that toggling the audio
    switch and resegmenting is a valid A/B of "visual only vs. audio + visual". The raw candidates
    are left untouched in ``signals`` so switching back restores the gated sequence idempotently.
    """
    trace: dict[str, Any] = {}
    if not params.use_audio:
        trace["regate"] = "audio_off"
        return None, trace
    if hits_raw is not None:
        raw, has_raw = hits_raw, True
    else:
        raw = _rebuild_hits_raw(sig)
        has_raw = bool(sig.get("hit_times_raw"))
    if raw is None or raw.times.size == 0:
        return raw, trace
    if not has_raw:
        trace["regate"] = "no_raw"
        return raw, trace
    if not params.use_pose:
        trace["regate"] = "pose_off"
        return raw, trace
    if pose is None:
        pose = _rebuild_pose_signal(sig)
    if pose is None:
        trace["regate"] = "no_pose"
        return raw, trace
    mod = _try_import("pose")
    if mod is None:
        trace["regate"] = "no_pose"
        return raw, trace
    # The composite gate (pose evidence OR local strength percentile) is the measured best on this
    # multi-court footage: strength is the single most discriminative cue (AUC 0.79), and pose
    # rescues soft shots whose strength is low. The old pure-pose gate is kept as a fallback when
    # the composite one is unavailable.
    if hasattr(mod, "composite_gate"):
        mask, g = mod.composite_gate(
            raw, pose,
            threshold=params.pose_gate_threshold,
            window=params.pose_gate_window,
            one_to_one=params.pose_gate_one_to_one,
            force=params.pose_gate_force,
        )
    else:
        mask, g = mod.gate_hits(
            raw, pose,
            threshold=params.pose_gate_threshold,
            window=params.pose_gate_window,
            one_to_one=params.pose_gate_one_to_one,
            force=params.pose_gate_force,
        )
    trace.update(g)
    trace["regate"] = "applied" if mask is not None else "skipped"
    if mask is None:
        return raw, trace
    return mod.filter_hits(raw, mask), trace


def _store_pose_full(sig: dict, pose) -> None:
    """Persist the full-rate pose curves needed by :func:`_rebuild_pose_signal`."""
    if not isinstance(sig, dict) or pose is None:
        return
    sig["pose_fps"] = [round(float(pose.fps), 3)]
    sig["pose_coverage"] = [round(float(pose.coverage), 4)]
    sig["pose_quiet"] = [round(float(pose.quiet), 5)]
    sig["pose_swing_full"] = [round(float(v), 4) for v in pose.swing]


def recover_pose_from_cache(sig: dict):
    """Best-effort recovery of the full-rate pose signal from the on-disk pose cache.

    Used by the "rebuild hits" path for old analyses saved before full-rate pose was persisted.
    Matches a cached npz by sample count + duration + fps; returns ``None`` when nothing matches, in
    which case the caller must tell the user that a full re-analysis is required.
    """
    existing = _rebuild_pose_signal(sig)
    if existing is not None:
        return existing
    fps = float((sig.get("pose_fps") or [0.0])[0]) or 0.0
    dur = float((sig.get("duration") or [0.0])[0]) or 0.0
    if fps <= 0 or dur <= 0:
        return None
    try:
        from pathlib import Path

        from . import pose as POSE

        cache_dir = POSE.default_cache_dir()
    except Exception:
        return None
    if cache_dir is None:
        return None
    n = int(round(dur * fps))
    for f in sorted(Path(cache_dir).glob("pose_*.npz")):
        got = POSE._load_cache(f, n)
        if got is None:
            continue
        if abs(float(got.fps) - fps) > 0.05 or abs(float(got.duration) - dur) > 1.0:
            continue
        return got
    return None


def _dict_from(sig: dict, key: str, fps_key: str) -> dict | None:
    v = sig.get(key)
    if not v:
        return None
    return {"fps": float((sig.get(fps_key) or [12.0])[0]), key: np.asarray(v, dtype=np.float32)}


def _dict_from_multi(sig: dict, keys: tuple[str, ...], fps_key: str) -> dict | None:
    out: dict = {}
    for k in keys:
        v = sig.get(k)
        if v:
            out[k] = np.asarray(v, dtype=np.float32)
    if not out:
        return None
    out["fps"] = float((sig.get(fps_key) or [12.0])[0])
    return out


def _best_overlap(r: Rally, olds: list[Rally]) -> Rally | None:
    best: Rally | None = None
    best_ov = 0.0
    for o in olds:
        ov = max(0.0, min(r.end, o.end) - max(r.start, o.start))
        if ov > best_ov:
            best_ov, best = ov, o
    # consider it "a continuation of the same rally" only when at least half overlaps
    return best if best is not None and best_ov >= 0.5 * max(0.1, r.duration) else None
