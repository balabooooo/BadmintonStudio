"""REST API for rally annotation and "using annotations to optimize segmentation parameters".

Corresponds to the frontend "Annotation" page:
* Read/save the manual annotations for a media clip;
* One-click generation of a draft from the current AI analysis result (semi-automatic annotation);
* Evaluate with the annotations and search for the best segmentation parameters, returning metrics
  for the user to choose from;
* Export CSV.

The routes are mounted on the main service (:mod:`bms.main`); no separate process is needed.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..analysis import annotation as AN
from ..core import store as ST
from ..core.jobs import manager as JOBS
from ..core.models import MediaInfo, Project
from ..i18n import get_lang, tr

router = APIRouter(prefix="/api/projects/{pid}/media/{mid}/annotation", tags=["annotation"])

#: Maximum overlay window length (seconds). The overlay is a polling visualization, not a video
#: stream: bounding the window bounds npz reads and JSON payload size on the localhost API.
OVERLAY_MAX_WINDOW = 20.0
_OVERLAY_COMPONENTS = ("players", "motion", "audio_hits", "shuttle", "roi")


def _load(pid: str) -> Project:
    try:
        proj = ST.load_project(pid)
    except ValueError as e:
        # Illegal ids are rejected by store (to prevent path traversal); this is the corresponding HTTP semantics
        raise HTTPException(400, str(e)) from e
    if proj is None:
        raise HTTPException(404, tr("api.project_not_found"))
    return proj


def _media(proj: Project, mid: str) -> MediaInfo:
    for m in proj.media:
        if m.id == mid:
            return m
    raise HTTPException(404, tr("api.media_not_found"))


def _auto_draft(proj: Project, mid: str) -> tuple[list[dict[str, Any]], float, float]:
    """Use the rallies in the current analysis result as annotation drafts."""
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        return [], 0.0, 0.0
    out = []
    for i, r in enumerate(sorted(res.rallies, key=lambda x: x.start), 1):
        out.append({
            "start": round(r.start, 3),
            "end": round(r.end, 3),
            "index": i,
            "shots": len(r.shots),
            "score": round(r.scores.total, 1),
        })
    sig = res.signals or {}
    duration = float((sig.get("duration") or [0.0])[0] or 0.0)
    fps = float((sig.get("fps") or [0.0])[0] or 0.0)
    return out, duration, fps


@router.get("")
def get_annotation(pid: str, mid: str) -> dict:
    proj = _load(pid)
    m = _media(proj, mid)
    path = AN.annotation_path(m)
    doc = AN.load_annotation(path)
    auto, duration, fps = _auto_draft(proj, mid)
    if duration <= 0:
        duration = float(m.duration or 0.0)
    if fps <= 0:
        fps = float(m.fps or 0.0)
    res = proj.analyses.get(mid)
    sig = (res.signals or {}) if res is not None else {}
    logger.debug(
        "annotation get: pid={} mid={} rallies={} hits={} focus={} auto={} path={}",
        pid, mid, len(doc.get("rallies") or []), len(doc.get("hits") or []),
        bool(doc.get("focus")), len(auto), path,
    )
    return {
        "media_id": mid,
        "media_name": m.name,
        "duration": duration,
        "fps": fps,
        "path": str(path),
        "rallies": AN.normalize_rallies(doc.get("rallies") or []),
        "hits": AN.normalize_hits(doc.get("hits") or []),
        "focus": doc.get("focus"),
        "note": str(doc.get("note") or ""),
        "auto": auto,
        # Signals for hit-level annotation: the audio envelope plus the raw / gated hit times.
        "envelope": sig.get("envelope") or [],
        "envelope_fps": float((sig.get("envelope_fps") or [0.0])[0] or 0.0),
        "hit_times": sig.get("hit_times") or [],
        "hit_times_raw": sig.get("hit_times_raw") or [],
    }


@router.put("")
def put_annotation(pid: str, mid: str, payload: dict = Body(...)) -> dict:
    proj = _load(pid)
    m = _media(proj, mid)
    _, duration, fps = _auto_draft(proj, mid)
    if duration <= 0:
        duration = float(m.duration or 0.0)
    if fps <= 0:
        fps = float(m.fps or 0.0)
    doc = AN.save_annotation(AN.annotation_path(m), m, payload, duration, fps)
    logger.debug(
        "annotation put: pid={} mid={} rallies={} hits={} focus={} keys={} path={}",
        pid, mid, int(doc.get("count") or 0),
        len(payload.get("hits") or []), bool(payload.get("focus")),
        sorted(payload.keys()), AN.annotation_path(m),
    )
    return {"ok": True, "count": doc["count"], "path": str(AN.annotation_path(m)),
            "updated_at": doc["updated_at"]}


def _pool_curve(values: list, target: int = 600) -> list[float]:
    """Block-mean a full-rate binary/numeric curve to at most ``target`` points for display."""
    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float32)
    except Exception:
        return []
    if arr.size == 0:
        return []
    if arr.size <= target:
        return [round(float(v), 3) for v in arr]
    step = int(np.ceil(arr.size / target))
    trimmed = arr[: arr.size - arr.size % step] if arr.size % step else arr
    pooled = trimmed.reshape(-1, step).mean(axis=1)
    return [round(float(v), 3) for v in pooled]


@router.get("/signals")
def get_annotation_signals(pid: str, mid: str) -> dict:
    """Downsampled multi-track signals for the annotation-page signal panel.

    Pure projection of the stored analysis JSON (no AI rerun): fused activity and thresholds,
    the five fusion components and their weights, player/motion curves, pose swing/overhead
    (display-rate copies), stored hit times, and whether box/pose caches exist for the overlay.
    """
    proj = _load(pid)
    _media(proj, mid)
    res = proj.analyses.get(mid)
    sig = (res.signals or {}) if res is not None else {}
    stats = (res.stats or {}) if res is not None else {}

    def _scalar(key: str) -> float:
        v = sig.get(key)
        return float(v[0]) if isinstance(v, list) and v else 0.0

    weights = sig.get("weights") or []
    weight_map = {k: round(float(weights[i]), 3) for i, k in enumerate(_OVERLAY_COMPONENTS)
                  if i < len(weights)}
    pose_available = bool(sig.get("pose_swing_full"))
    return {
        "duration": _scalar("duration"),
        "fps": _scalar("fps"),
        "activity": sig.get("activity") or [],
        "threshold_hi": _scalar("threshold_hi"),
        "threshold_lo": _scalar("threshold_lo"),
        "weights": weight_map,
        "components": {k: sig.get(f"component_{k}") or [] for k in _OVERLAY_COMPONENTS},
        "shuttle_in_flight": sig.get("shuttle_in_flight") or [],
        "motion": sig.get("motion") or [],
        "motion_fps": _scalar("motion_fps"),
        "player_motion": sig.get("player_motion") or [],
        "active_count": sig.get("active_count") or [],
        "player_coverage": _pool_curve(sig.get("player_coverage_full") or []),
        "player_fps": _scalar("player_fps"),
        "pose": {
            "available": pose_available,
            "swing": sig.get("pose_swing") or [],
            "ok": sig.get("pose_ok") or [],
            "overhead": sig.get("pose_overhead") or [],
            "fps": _scalar("pose_fps"),
            "coverage": _scalar("pose_coverage"),
        },
        "hit_times": sig.get("hit_times") or [],
        "hit_times_raw": sig.get("hit_times_raw") or [],
        "has_boxes_cache": bool(stats.get("boxes_cache")),
        "has_pose_cache": bool(stats.get("pose_cache")),
    }


@router.get("/overlay")
def get_annotation_overlay(pid: str, mid: str, t0: float = 0.0, t1: float = 1.0) -> dict:
    """Windowed per-frame player boxes + COCO skeletons around the playhead.

    Data comes from the compact npz caches referenced by the stored analysis
    (``stats.boxes_cache`` / ``stats.pose_cache``; references live alongside the other traces, not
    in the numeric ``signals`` map); coordinates are normalized 0~1 on the proxy timeline. The
    window is capped at :data:`OVERLAY_MAX_WINDOW` seconds. Missing caches degrade to
    ``available=false`` with empty frames — never an error, so the UI simply hides the layer and
    can offer "rebuild visual cache".

    v2 box caches additionally carry per-box ``conf`` (the UI filters display by a configurable
    detection threshold) and per-frame ``dets`` — raw pre-size-filter detections without a track
    id, drawn as a dashed miss-diagnosis layer. v1 caches serve neither field.
    """
    proj = _load(pid)
    _media(proj, mid)
    res = proj.analyses.get(mid)
    sig = (res.signals or {}) if res is not None else {}
    stats = (res.stats or {}) if res is not None else {}
    duration = float((sig.get("duration") or [0.0])[0] or 0.0)

    if not (t0 >= 0.0 and t1 > t0) or t1 - t0 > OVERLAY_MAX_WINDOW + 1e-6 \
            or (duration > 0 and t0 > duration):
        raise HTTPException(400, tr("annotation.overlay_bad_window",
                                    max_window=OVERLAY_MAX_WINDOW))

    from ..analysis import players as PL
    from ..analysis import pose as POSE

    t_start = time.perf_counter()
    boxes_tag = str(stats.get("boxes_cache") or "")
    pose_tag = str(stats.get("pose_cache") or "")
    t_boxes0 = time.perf_counter()
    boxes_doc = PL.load_boxes_cache(boxes_tag) if boxes_tag else None
    t_boxes_ms = (time.perf_counter() - t_boxes0) * 1000.0
    kp_doc = None
    if pose_tag:
        pose_dir = POSE.default_cache_dir()
        if pose_dir is not None:
            t_pose0 = time.perf_counter()
            kp_doc = POSE.load_keypoints(pose_dir / f"{pose_tag}.npz")
            logger.debug("overlay pose npz load: {:.1f}ms (tag={})",
                         (time.perf_counter() - t_pose0) * 1000.0, pose_tag)

    # frame-index -> payload entry (only frames carrying a box, a skeleton or a raw detection are emitted)
    entries: dict[int, dict[str, Any]] = {}

    def _entry(i: int, fps: float) -> dict:
        e = entries.get(i)
        if e is None:
            e = {"t": round(i / fps, 3), "boxes": [], "skeletons": [], "dets": []}
            entries[i] = e
        return e

    fps = 0.0
    if boxes_doc is not None:
        frame_boxes = boxes_doc["frame_boxes"]
        frame_confs = boxes_doc.get("frame_confs")
        frame_interps = boxes_doc.get("frame_interps")
        raw_dets = boxes_doc.get("raw_dets")
        fps = float(boxes_doc["fps"])
        n = len(frame_boxes)
        i0 = max(0, int(t0 * fps))
        i1 = min(n - 1, int(t1 * fps))
        for i in range(i0, i1 + 1):
            for k, item in enumerate(frame_boxes[i]):
                tid, x1, y1, x2, y2 = item
                box_json: dict[str, Any] = {
                    "track": int(tid),
                    "xyxy": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                }
                # v2 caches carry per-row confidence so the UI can threshold the display;
                # v1 boxes have none and are always shown (no filtering possible).
                if frame_confs is not None and k < len(frame_confs[i]):
                    box_json["conf"] = round(float(frame_confs[i][k]), 3)
                # v3 marks boxes interpolated across short occlusion gaps; the UI draws them dimmed.
                if frame_interps is not None and k < len(frame_interps[i]) and int(frame_interps[i][k]):
                    box_json["interp"] = True
                _entry(i, fps)["boxes"].append(box_json)
            # Raw pre-size-filter detections (v2/v3): what the detector saw before tracking /
            # active-player selection. No track id — the frontend draws them as a dashed
            # miss-diagnosis layer behind the colored tracked boxes.
            if raw_dets is not None:
                for x1, y1, x2, y2, c in raw_dets[i]:
                    _entry(i, fps)["dets"].append({
                        "xyxy": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                        "conf": round(float(c), 3),
                    })

    if kp_doc is not None and kp_doc["xy"].shape[0] > 0:
        kfps = float(kp_doc["fps"])
        fps = fps or kfps
        n = int(kp_doc["n"])
        i0 = max(0, int(t0 * kfps))
        i1 = min(n - 1, int(t1 * kfps))
        frames = kp_doc["frame"]
        mask = (frames >= i0) & (frames <= i1)
        xy = kp_doc["xy"][mask]
        conf = kp_doc["conf"][mask]
        fr = frames[mask]
        trk = kp_doc["track"][mask]
        for r in range(fr.shape[0]):
            kp = [[round(float(xy[r, j, 0]), 4), round(float(xy[r, j, 1]), 4),
                   round(float(conf[r, j]), 2)] for j in range(xy.shape[1])]
            _entry(int(fr[r]), kfps)["skeletons"].append(
                {"track": int(trk[r]), "kp": kp})

    n_boxes = sum(len(e["boxes"]) for e in entries.values())
    n_dets = sum(len(e["dets"]) for e in entries.values())
    n_skel = sum(len(e["skeletons"]) for e in entries.values())
    logger.debug(
        "overlay mid={} window=[{:.2f},{:.2f}] entries={} boxes={} dets={} skeletons={} boxes_npz={:.1f}ms total={:.1f}ms",
        mid, t0, t1, len(entries), n_boxes, n_dets, n_skel, t_boxes_ms,
        (time.perf_counter() - t_start) * 1000.0,
    )
    return {
        "t0": round(max(0.0, t0), 3),
        "t1": round(t1, 3),
        "fps": round(float(fps), 3),
        "duration": round(duration, 3),
        "boxes_available": boxes_doc is not None,
        "skeletons_available": kp_doc is not None,
        "frames": [entries[i] for i in sorted(entries)],
    }


@router.get("/export.csv", response_class=PlainTextResponse)
def export_csv(pid: str, mid: str) -> PlainTextResponse:
    proj = _load(pid)
    m = _media(proj, mid)
    _, _, fps = _auto_draft(proj, mid)
    if fps <= 0:
        fps = float(m.fps or 0.0)
    doc = AN.load_annotation(AN.annotation_path(m))
    csv = AN.annotation_to_csv(doc, fps)
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="rally_annotations.csv"'})


@router.get("/quality")
def annotation_quality(pid: str, mid: str) -> dict:
    """Audit the manual labels against the stored signals (no AI rerun); warnings + snap hints only.

    Labels are never modified here. Stable ASCII warning ``code`` values are translated by the
    frontend; missing signals degrade to structural-only checks.
    """
    proj = _load(pid)
    m = _media(proj, mid)
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, tr("annotation.no_analysis"))

    doc = AN.load_annotation(AN.annotation_path(m))
    gt = [(float(r["start"]), float(r["end"])) for r in AN.normalize_rallies(doc.get("rallies") or [])]
    if not gt:
        raise HTTPException(400, tr("annotation.no_labels"))
    focus = doc.get("focus")
    focus_t = (float(focus[0]), float(focus[1])) if isinstance(focus, (list, tuple)) and len(focus) == 2 else None
    quality = AN.label_quality(res, gt, focus=focus_t)
    quality["focus"] = list(focus_t) if focus_t else None
    return quality


def _read_optimize_labels(m: MediaInfo) -> tuple[list[tuple[float, float]],
                                                  list[tuple[float, bool]],
                                                  tuple[float, float] | None]:
    """Load rally/hit labels for optimization from the media's annotation file.

    Hit-level stages are only meaningful when the user has actually marked at least one
    neighboring-court sound; labels that are all "ours" would simply reward keeping every hit, so
    an all-ours set is downgraded to "no hit labels" here.
    """
    doc = AN.load_annotation(AN.annotation_path(m))
    gt = [(float(r["start"]), float(r["end"])) for r in AN.normalize_rallies(doc.get("rallies") or [])]
    hit_labels = [(float(h["t"]), bool(h.get("ours", True)))
                  for h in AN.normalize_hits(doc.get("hits") or [])]
    if hit_labels and not any(not ours for _, ours in hit_labels):
        hit_labels = []
    focus = doc.get("focus")
    focus_t = (float(focus[0]), float(focus[1])) \
        if isinstance(focus, (list, tuple)) and len(focus) == 2 else None
    return gt, hit_labels, focus_t


@router.post("/optimize")
def optimize(pid: str, mid: str, payload: dict = Body(default={})) -> dict:
    """Evaluate the current segmentation against the annotation and search for the parameter set with the highest F1.

    Runs as a background ``optimize`` job with live progress; the response only carries
    ``{"job_id"}`` — the result (``best`` parameters, top grid rows, ``suggest``) is delivered in
    the job's ``done`` event / job listing. Input validation (missing analysis / labels, bad base
    params) stays synchronous and keeps its 400 semantics; a second optimize for media already
    being optimized is rejected with 409.
    """
    proj = _load(pid)
    m = _media(proj, mid)
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, tr("annotation.no_analysis"))

    gt, _hit_labels, _focus_t = _read_optimize_labels(m)
    if not gt:
        raise HTTPException(400, tr("annotation.no_labels"))

    # Validate the optional "parameters currently in use" base up front (only known fields are
    # applied, same as the old synchronous endpoint).
    base_params = res.params
    for k, v in (payload.get("params") or {}).items():
        if hasattr(base_params, k):
            try:
                base_params = base_params.model_copy(update={k: v})
            except (TypeError, ValueError):
                raise HTTPException(400, tr("annotation.bad_params", key=k)) from None
    params_patch = {k: getattr(base_params, k) for k in (payload.get("params") or {})
                    if hasattr(base_params, k)}

    for j in JOBS.active():
        if j.kind == "optimize" and j.media_id == mid:
            logger.debug("optimize rejected 409: mid={} existing_job={}", mid, j.id)
            raise HTTPException(409, tr("annotation.optimize_running"))

    def work(job):  # noqa: ANN202 - Job worker closure
        # Re-read everything inside the worker: annotation edits and re-analysis may have landed
        # while the job was queued.
        proj2 = _load(pid)
        m2 = _media(proj2, mid)
        res2 = proj2.analyses.get(mid)
        if res2 is None or res2.status != "done":
            raise RuntimeError(tr("annotation.no_analysis"))
        gt2, hit_labels2, focus_t2 = _read_optimize_labels(m2)
        if not gt2:
            raise RuntimeError(tr("annotation.no_labels"))

        # Hit-level sensitivity needs the cached WAV (cheap STFT done once) so the search can
        # re-threshold over a sensitivity grid without re-running any video AI.
        sensitivity_fn = None
        if hit_labels2:
            try:
                from ..analysis import audio_hits as AH
                from ..core import media as _M

                _M.ensure_audio(m2)
                if m2.audio_path:
                    env = AH.build_hit_envelope(m2.audio_path)
                    if env is not None:
                        sensitivity_fn = lambda s, _e=env: AH.pick_hits(_e, sensitivity=float(s))
            except Exception:  # noqa: BLE001
                sensitivity_fn = None

        res2 = res2.model_copy(update={"params": res2.params.model_copy(update=params_patch)})

        def on_progress(p: float, stage: str) -> None:
            # Stage boundaries double as cancellation checkpoints.
            if job.cancelled():
                raise AN.OptimizeCancelled(tr("job.cancelled"))
            job.progress(p, stage=stage, message=tr(f"job.{stage}"))

        logger.debug("optimize job {} start: mid={} gt={} hit_labels={}",
                     job.id, mid, len(gt2), len(hit_labels2))
        result = AN.optimize(res2, gt2, focus=focus_t2, hit_labels=hit_labels2,
                             sensitivity_fn=sensitivity_fn, on_progress=on_progress)
        result["suggest"] = AN.suggest(gt2)
        logger.debug("optimize job {} done: tried={} best_f1={}",
                     job.id, result.get("tried"),
                     (result.get("best") or {}).get("f1"))
        return result

    job = JOBS.submit("optimize", tr("job.title.optimize", name=m.name), work,
                      media_id=mid, lang=get_lang())
    return {"job_id": job.id}
