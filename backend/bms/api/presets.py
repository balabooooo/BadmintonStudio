"""REST API for scene presets (global, cross-project).

Corresponds to the preset thumbnail strip in the frontend "AI Analysis" dialog and "Save as preset"
on the "Annotation" page. A preset = segmentation params + court calibration + a screenshot taken
at save time; the user picks by looking at the screenshot, with no automatic matching.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from ..core import presets as PRE
from ..core import store as ST
from ..i18n import tr

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("")
def list_presets() -> list[dict]:
    return PRE.list_presets()


@router.post("")
def create_preset(payload: dict = Body(...)) -> dict:
    pid = str(payload.get("project_id") or "")
    mid = str(payload.get("media_id") or "")
    if not pid or not mid:
        raise HTTPException(400, tr("api.preset_missing_ids"))
    try:
        proj = ST.load_project(pid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if proj is None:
        raise HTTPException(404, tr("api.project_not_found"))
    media = next((m for m in proj.media if m.id == mid), None)
    if media is None:
        raise HTTPException(404, tr("api.media_not_found"))
    try:
        frame_time = float(payload.get("frame_time") or 0.0)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, tr("api.bad_request", error=str(e))) from e
    try:
        return PRE.save_preset(
            name=str(payload.get("name") or ""),
            note=str(payload.get("note") or ""),
            media=media,
            frame_time=frame_time,
            params=payload.get("params") or {},
            court_poly=payload.get("court_poly"),
            project_id=pid,
            project_name=proj.name,
            media_name=media.name,
            fit=payload.get("fit"),
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, tr("api.preset_save_failed", error=f"{type(e).__name__}: {e}")) from e


@router.patch("/{preset_id}")
def patch_preset(preset_id: str, payload: dict = Body(default={})) -> dict:
    try:
        doc = PRE.rename_preset(
            preset_id,
            name=payload.get("name") if "name" in payload else None,
            note=payload.get("note") if "note" in payload else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if doc is None:
        raise HTTPException(404, tr("api.preset_not_found"))
    return doc


@router.delete("/{preset_id}")
def delete_preset(preset_id: str) -> dict:
    try:
        ok = PRE.delete_preset(preset_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not ok:
        raise HTTPException(404, tr("api.preset_not_found"))
    return {"ok": True}
