"""回合标注与「用标注优化切分参数」的 REST 接口。

对应前端「标注」页：
* 读取/保存某段素材的人工标注；
* 一键从当前 AI 分析结果生成草稿（半自动标注）；
* 用标注评估并搜索最优切分参数，返回指标供用户选择；
* 导出 CSV。

路由挂在主服务上（:mod:`bms.main`），不需要再起一个独立进程。
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

router = APIRouter(prefix="/api/projects/{pid}/media/{mid}/annotation", tags=["annotation"])


def _load(pid: str) -> Project:
    proj = ST.load_project(pid)
    if proj is None:
        raise HTTPException(404, "工程不存在")
    return proj


def _media(proj: Project, mid: str) -> MediaInfo:
    for m in proj.media:
        if m.id == mid:
            return m
    raise HTTPException(404, "素材不存在")


def _auto_draft(proj: Project, mid: str) -> tuple[list[dict[str, Any]], float, float]:
    """把当前分析结果里的回合作为标注草稿。"""
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
    return {
        "media_id": mid,
        "media_name": m.name,
        "duration": duration,
        "fps": fps,
        "path": str(path),
        "rallies": AN.normalize_rallies(doc.get("rallies") or []),
        "focus": doc.get("focus"),
        "note": str(doc.get("note") or ""),
        "auto": auto,
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
    """用标注评估当前切分，并搜索 F1 最高的一组参数。

    只读已存好的分析信号，不重跑 AI。返回里 ``best`` 是建议写回的参数，
    ``results`` 是网格上靠前的结果，供前端展示。
    """
    proj = _load(pid)
    m = _media(proj, mid)
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, "还没有分析结果，请先运行一次 AI 分析")

    doc = AN.load_annotation(AN.annotation_path(m))
    gt = [(float(r["start"]), float(r["end"])) for r in AN.normalize_rallies(doc.get("rallies") or [])]
    if not gt:
        raise HTTPException(400, "还没有人工标注，请先标注几个回合再优化")

    focus = doc.get("focus")
    if isinstance(focus, (list, tuple)) and len(focus) == 2:
        focus_t = (float(focus[0]), float(focus[1]))
    else:
        focus_t = None

    # 允许前端带上「当前正在用的参数」当基座，这样优化结果能与屏幕上看到的一致。
    base = res.params
    for k, v in (payload.get("params") or {}).items():
        if hasattr(base, k):
            base = base.model_copy(update={k: v})
    res = res.model_copy(update={"params": base})

    try:
        result = AN.optimize(res, gt, focus=focus_t)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from None
    result["suggest"] = AN.suggest(gt)
    return result
