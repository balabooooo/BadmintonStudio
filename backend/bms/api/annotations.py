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

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import PlainTextResponse

from .. import config as CFG
from ..analysis import annotation as AN
from ..core import store as ST
from ..core.models import AnalysisParams, MediaInfo, Project
from ..i18n import tr

router = APIRouter(prefix="/api/projects/{pid}/media/{mid}/annotation", tags=["annotation"])


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
    return {"ok": True, "count": doc["count"], "path": str(AN.annotation_path(m)),
            "updated_at": doc["updated_at"]}


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


@router.post("/optimize")
def optimize(pid: str, mid: str, payload: dict = Body(default={})) -> dict:
    """Evaluate the current segmentation against the annotation and search for the parameter set with the highest F1.

    Reads only the stored analysis signals; it does not re-run AI. In the return value, ``best`` is
    the parameters to write back and ``results`` are the top results on the grid, for the frontend
    to display.
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

    hit_labels = [(float(h["t"]), bool(h.get("ours", True)))
                  for h in AN.normalize_hits(doc.get("hits") or [])]
    # Hit-level stages are only meaningful when the user has actually marked at least one
    # neighboring-court sound; labels that are all "ours" would simply reward keeping every hit.
    if hit_labels and not any(not ours for _, ours in hit_labels):
        hit_labels = []

    focus = doc.get("focus")
    if isinstance(focus, (list, tuple)) and len(focus) == 2:
        focus_t = (float(focus[0]), float(focus[1]))
    else:
        focus_t = None

    # Hit-level labels unlock two extra calibration stages: the attribution gate threshold and the
    # audio detection sensitivity. The latter needs the cached WAV (cheap STFT done once) so we can
    # re-threshold over a sensitivity grid without re-running any video AI.
    sensitivity_fn = None
    if hit_labels:
        try:
            from ..analysis import audio_hits as AH
            from ..core import media as _M

            _M.ensure_audio(m)
            if m.audio_path:
                env = AH.build_hit_envelope(m.audio_path)
                if env is not None:
                    sensitivity_fn = lambda s, _e=env: AH.pick_hits(_e, sensitivity=float(s))
        except Exception:  # noqa: BLE001
            sensitivity_fn = None

    # Allow the frontend to pass "the parameters currently in use" as a base, so the optimization result matches what is on screen.
    base = res.params
    for k, v in (payload.get("params") or {}).items():
        if hasattr(base, k):
            base = base.model_copy(update={k: v})
    res = res.model_copy(update={"params": base})

    try:
        result = AN.optimize(res, gt, focus=focus_t, hit_labels=hit_labels,
                             sensitivity_fn=sensitivity_fn)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from None
    result["suggest"] = AN.suggest(gt)
    return result
