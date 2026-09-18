"""BadmintonStudio 本地服务：REST API + WebSocket 进度推送 + 静态前端。"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import (
    APP_DISPLAY_NAME,
    APP_VERSION,
    CACHE_DIR,
    DATA_DIR,
    EXPORT_DIR,
    FRONTEND_DIST,
    MODELS_DIR,
    PROXIES_DIR,
    THUMBS_DIR,
    ensure_dirs,
)
from .core import ffmpeg as ff
from .core import media as M
from .core import store as ST
from .core.jobs import manager as JOBS
from .core.models import AnalysisParams, ExportPreset, MediaInfo, Project, Rally, Timeline, Track, now_ms
from .core.streaming import range_response
from .api.annotations import router as annotations_router

ensure_dirs()

app = FastAPI(title=APP_DISPLAY_NAME, version=APP_VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(annotations_router)


# ------------------------------------------------------------------ WebSocket 广播


class Hub:
    """线程安全的广播中心：后台任务线程 -> 事件循环 -> 所有浏览器连接。"""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def publish(self, payload: dict) -> None:
        """可以被任意线程调用。"""
        if self._loop is None or not self.clients:
            return
        msg = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            asyncio.run_coroutine_threadsafe(self._send_all(msg), self._loop)
        except RuntimeError:
            pass

    async def _send_all(self, msg: str) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


HUB = Hub()
JOBS.subscribe(lambda info: HUB.publish({"type": "job", "job": info.model_dump(mode="json")}))


@app.on_event("startup")
async def _startup() -> None:
    HUB.bind_loop(asyncio.get_running_loop())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await HUB.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "version": APP_VERSION}))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": time.time()}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        HUB.disconnect(ws)


# ------------------------------------------------------------------ 基础


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": APP_DISPLAY_NAME, "version": APP_VERSION, "time": now_ms()}


@app.get("/api/env")
def env_info() -> dict:
    caps: dict[str, Any] = {}
    ffmpeg_path = ""
    err = None
    try:
        ffmpeg_path = ff.find_ffmpeg()
        caps = ff.caps()
    except Exception as e:  # noqa: BLE001
        err = str(e)
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        avail = bool(torch.cuda.is_available())
        gpu = {
            "available": avail,
            "name": torch.cuda.get_device_name(0) if avail else None,
            "torch": torch.__version__,
            "capability": list(torch.cuda.get_device_capability(0)) if avail else None,
        }
    except Exception as e:  # noqa: BLE001
        gpu["error"] = str(e)
    from .core.media import hardware_available

    return {
        "app": APP_DISPLAY_NAME,
        "version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "ffmpeg": ffmpeg_path,
        "ffmpeg_error": err,
        "caps": caps,
        "gpu": gpu,
        "hardware_pipeline": bool(caps.get("cuda_decode") and hardware_available()),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "cache_dir": str(CACHE_DIR),
    }


# ------------------------------------------------------------------ 素材文件访问


def _allowed_roots() -> list[Path]:
    roots = [DATA_DIR, MODELS_DIR]
    try:
        for f in (DATA_DIR / "projects").glob("p_*.json"):
            if f.name.count(".") > 1:
                continue
            try:
                p = Project.model_validate(json.loads(f.read_text(encoding="utf-8")))
                roots.extend(Path(m.path).parent for m in p.media)
                roots.extend(Path(m.proxy_path).parent for m in p.media if m.proxy_path)
            except Exception:
                continue
    except Exception:
        pass
    return roots


def _resolve_asset(raw: str) -> Path:
    p = Path(raw)
    try:
        p = p.resolve()
    except OSError:
        pass
    if not p.is_file():
        raise HTTPException(404, "资源不存在")
    for root in _allowed_roots():
        try:
            rp = root.resolve()
        except OSError:
            continue
        try:
            p.relative_to(rp)
            return p
        except ValueError:
            continue
    raise HTTPException(403, "该路径不在允许访问的范围内")


@app.get("/api/asset")
def asset(request: Request, p: str = Query(...), cache: int = 3600):
    path = _resolve_asset(p)
    return range_response(request, path, cache_seconds=cache)


# ------------------------------------------------------------------ 工程


def _must_project(pid: str) -> Project:
    proj = ST.load_project(pid)
    if proj is None:
        raise HTTPException(404, "工程不存在")
    return proj


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return [s.model_dump(mode="json") for s in ST.list_projects()]


@app.post("/api/projects")
def create_project(payload: dict = Body(default={})) -> dict:
    proj = ST.create_project(payload.get("name") or "未命名工程")
    return proj.model_dump(mode="json")


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    return _must_project(pid).model_dump(mode="json")


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    if "name" in payload and payload["name"]:
        proj.name = str(payload["name"])
    if "ui" in payload and isinstance(payload["ui"], dict):
        proj.ui.update(payload["ui"])
    if "timeline" in payload and isinstance(payload["timeline"], dict):
        proj.timeline = Timeline.model_validate(payload["timeline"])
    ST.save_project(proj, write_analyses=False)
    return proj.model_dump(mode="json")


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict:
    return {"ok": ST.delete_project(pid)}


@app.post("/api/projects/{pid}/duplicate")
def duplicate_project(pid: str, payload: dict = Body(default={})) -> dict:
    p = ST.duplicate_project(pid, payload.get("name"))
    if p is None:
        raise HTTPException(404, "工程不存在")
    return p.model_dump(mode="json")


# ------------------------------------------------------------------ 素材


@app.post("/api/projects/{pid}/media")
def add_media(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    paths = payload.get("paths") or ([payload["path"]] if payload.get("path") else [])
    if not paths:
        raise HTTPException(400, "缺少 path / paths")
    added, failed = [], []
    for raw in paths:
        p = Path(str(raw))
        if not p.is_file():
            failed.append({"path": str(raw), "error": "文件不存在"})
            continue
        try:
            info = M.probe_media(p)
            proj = ST.touch_media(proj, info)
            added.append(info.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            failed.append({"path": str(raw), "error": f"{type(e).__name__}: {e}"})
    return {"project": proj.model_dump(mode="json"), "added": added, "failed": failed}


@app.post("/api/projects/{pid}/media/upload")
async def upload_media(pid: str, request: Request) -> dict:
    """接收浏览器拖进来的文件（原始字节流 + 文件名头）。"""
    proj = _must_project(pid)
    import urllib.parse

    name = request.headers.get("x-filename") or "upload.mp4"
    name = urllib.parse.unquote(name)
    name = Path(name).name or "upload.mp4"
    dest_dir = DATA_DIR / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        dest = dest_dir / f"{dest.stem}_{int(time.time())}{dest.suffix}"
    size = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    info = M.probe_media(dest)
    proj = ST.touch_media(proj, info)
    return {"project": proj.model_dump(mode="json"), "added": info.model_dump(mode="json"), "bytes": size}


@app.delete("/api/projects/{pid}/media/{mid}")
def remove_media(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    proj.media = [m for m in proj.media if m.id != mid]
    proj.analyses.pop(mid, None)
    ST.delete_analysis(pid, mid)
    for t in proj.timeline.tracks:
        t.clips = [c for c in t.clips if c.media_id != mid]
    ST.save_project(proj, write_analyses=False)
    return proj.model_dump(mode="json")


def _media(proj: Project, mid: str) -> MediaInfo:
    for m in proj.media:
        if m.id == mid:
            return m
    raise HTTPException(404, "素材不存在")


@app.get("/api/projects/{pid}/media/{mid}/stream")
def stream_source(pid: str, mid: str, request: Request):
    return range_response(request, Path(_media(_must_project(pid), mid).path), 3600)


@app.get("/api/projects/{pid}/media/{mid}/proxy")
def stream_proxy(pid: str, mid: str, request: Request):
    proj = _must_project(pid)
    m = _media(proj, mid)
    p = Path(m.proxy_path) if m.proxy_path and Path(m.proxy_path).is_file() else Path(m.path)
    return range_response(request, p, 3600)


@app.get("/api/projects/{pid}/media/{mid}/audio")
def stream_audio(pid: str, mid: str, request: Request):
    proj = _must_project(pid)
    m = _media(proj, mid)
    if not m.audio_path or not Path(m.audio_path).is_file():
        raise HTTPException(404, "尚未提取音轨")
    return range_response(request, Path(m.audio_path), 3600)


@app.get("/api/projects/{pid}/media/{mid}/poster")
def poster(pid: str, mid: str):
    proj = _must_project(pid)
    m = _media(proj, mid)
    if not m.poster or not Path(m.poster).is_file():
        M.ensure_poster(m)
        _persist_media(pid, m)
    if m.poster and Path(m.poster).is_file():
        return FileResponse(m.poster, media_type="image/jpeg")
    raise HTTPException(404, "无法生成封面")


@app.get("/api/projects/{pid}/media/{mid}/sprite")
def sprite(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    m = _media(proj, mid)
    info = M.make_sprite(m, count=120, cols=12, tile_w=160)
    if not info:
        raise HTTPException(500, "缩略图生成失败")
    return info


def _persist_media(pid: str, m: MediaInfo) -> None:
    """把素材上的派生路径写回工程并广播，避免每次都要重算代理。"""
    proj = ST.load_project(pid)
    if proj is None:
        return
    for i, x in enumerate(proj.media):
        if x.id == m.id:
            proj.media[i] = m
            ST.save_project(proj, write_analyses=False)
            HUB.publish({"type": "media", "project_id": pid, "media": m.model_dump(mode="json")})
            return


@app.post("/api/projects/{pid}/media/{mid}/prepare")
def prepare_media(pid: str, mid: str, payload: dict = Body(default={})) -> dict:
    """生成代理视频 / 音轨 / 封面（可单独触发，让用户先把播放准备好）。"""
    proj = _must_project(pid)
    m = _media(proj, mid)

    def work(job):
        try:
            job.progress(0.02, "proxy", "生成代理视频")
            M.ensure_proxy(m, on=lambda p, msg: job.progress(0.02 + 0.7 * p, "proxy", msg), cancel=job.cancelled)
            job.progress(0.75, "audio", "提取音轨")
            M.ensure_audio(m, on=lambda p, msg: job.progress(0.75 + 0.2 * p, "audio", msg), cancel=job.cancelled)
            job.progress(0.96, "poster", "生成封面")
            M.ensure_poster(m)
            return {
                "media": m.model_dump(mode="json"),
                "sprite": M.make_sprite(m, count=120, cols=12, tile_w=160),
            }
        finally:
            _persist_media(pid, m)

    job = JOBS.submit("prepare", f"准备素材 {m.name}", work)
    return {"job_id": job.id}


@app.get("/api/media/probe")
def probe_path(path: str = Query(...)) -> dict:
    """探测任意本地文件，返回时长/分辨率等（拖拽导入前的预检）。"""
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    info = M.probe_media(p)
    return info.model_dump(mode="json")


# ------------------------------------------------------------------ 分析


@app.post("/api/projects/{pid}/analyze")
def analyze(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, "缺少 media_id")
    m = _media(proj, mid)
    params = AnalysisParams.model_validate(payload.get("params") or {})
    weights_key = str(payload.get("weights") or "balanced")
    roi = payload.get("roi")
    roi_t = tuple(float(v) for v in roi) if roi and len(roi) == 4 else None

    from .analysis.pipeline import run_analysis

    holder: dict[str, Any] = {}

    def wrapped(job):
        try:
            res = run_analysis(
                m,
                params,
                on_progress=lambda p, s, msg: job.progress(p, s, msg),
                cancel=job.cancelled,
                weights_key=weights_key,
                roi=roi_t,  # type: ignore[arg-type]
            )
            holder["result"] = res
            return res.model_dump(mode="json")
        finally:
            res = holder.get("result")
            if res is not None:
                ST.save_analysis(pid, mid, res)
                p2 = ST.load_project(pid)
                if p2 is not None:
                    p2.analyses[mid] = res
                    ST.save_project(p2, write_analyses=False)
                HUB.publish({"type": "analysis", "project_id": pid, "media_id": mid,
                             "result": res.model_dump(mode="json")})

    job = JOBS.submit("analyze", f"分析 {m.name}", wrapped)
    return {"job_id": job.id}


@app.post("/api/projects/{pid}/player-probe")
def player_probe(pid: str, payload: dict = Body(default={})) -> dict:
    """快速试测人物框尺寸：取若干帧只做检测（不跟踪），秒级返回。

    为什么需要它：调「人物框尺寸筛选」时，用户真正想知道的只有
    「球员的框多大、观众和其他场地的人多大」。等一次完整分析要几分钟，
    而调参往往要来回试好几次。这里在整条视频上均匀抽 6~12 帧批量推理一次，
    直接给出框高分布，界面可以立刻画出直方图并在本地模拟阈值效果。

    ``times``（或 ``at_time``）可以指定**具体时刻**：界面用它做「手动选帧」——
    抓到的那一帧会连**画面本身**一起返回（``frame.image``，缓存里的 JPEG 路径，
    前端用 ``/api/asset`` 取），于是「哪些框被选中、哪些被筛掉」可以直接画在
    画面上核对，而不是只看一个数字分布。

    只读素材，不写分析结果，也不进任务队列 —— 它足够快，直接同步返回。
    """
    from .analysis.pipeline import _manual_poly
    from .analysis import players as PL

    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, "缺少 media_id")
    m = _media(proj, mid)
    params = AnalysisParams.model_validate(payload.get("params") or {})
    count = int(payload.get("count") or 10)
    count = max(1, min(40, count))
    viewpoint = params.viewpoint if params.viewpoint != "auto" else "unknown"
    # 手动选帧：times 明确指定时刻；at_time 是「就抓这一帧」的简写
    times: list[float] | None = None
    raw_times = payload.get("times")
    if isinstance(raw_times, list) and raw_times:
        try:
            times = [float(v) for v in raw_times[:40]]
        except (TypeError, ValueError):
            times = None
    if times is None and payload.get("at_time") is not None:
        try:
            times = [float(payload["at_time"])]
        except (TypeError, ValueError):
            times = None
    # 默认把帧图存下来：界面要把它画出来给用户核对
    save_frames = bool(payload.get("save_frames", True))

    # 场地范围：优先用请求里手动标的边界，其次用已存分析结果里标定好的多边形。
    # 统计口径必须和真正分析时一致，否则试测出来的分布对不上。
    poly = None
    raw_poly = payload.get("court_poly") or params.court_poly or params.court_quad
    manual = _manual_poly(raw_poly)
    if manual is not None:
        poly = [[float(p[0]), float(p[1])] for p in manual]
    else:
        prev = proj.analyses.get(mid)
        cal = (prev.calibration if prev is not None else None) or {}
        got = cal.get("polygon") or cal.get("quad")
        if isinstance(got, list) and len(got) >= 3:
            poly = [[float(p[0]), float(p[1])] for p in got]
        if viewpoint == "unknown" and isinstance(cal.get("viewpoint"), str):
            viewpoint = cal["viewpoint"]

    path = str(m.proxy_path or m.path)
    try:
        return PL.probe_boxes(
            path,
            count=count,
            viewpoint=viewpoint,
            roi_poly=poly,
            times=times,
            save_frames=save_frames,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from None


@app.post("/api/projects/{pid}/resegment")
def resegment(pid: str, payload: dict = Body(default={})) -> dict:
    """用已存好的活跃度曲线快速重切回合（毫秒级），用于调切分参数与评分口径。

    不用重跑球员检测和运动分析，所以调参是即时的。
    """
    from .analysis.pipeline import resegment as _reseg

    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, "缺少 media_id")
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, "还没有分析结果，请先运行一次 AI 分析")

    base = res.params.model_copy(deep=True)
    for k, v in (payload.get("params") or {}).items():
        if hasattr(base, k):
            setattr(base, k, v)
    weights_key = str(payload.get("weights") or res.stats.get("weights") or "balanced")
    try:
        updated = _reseg(res, base, weights_key)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}") from None

    proj.analyses[mid] = updated
    ST.save_analysis(pid, mid, updated)
    remapped = _remap_timeline(proj, mid, updated)
    ST.save_project(proj, write_analyses=False)
    HUB.publish({"type": "analysis", "project_id": pid, "media_id": mid,
                 "result": updated.model_dump(mode="json")})
    if remapped:
        HUB.publish({"type": "timeline", "project_id": pid,
                     "timeline": proj.timeline.model_dump(mode="json")})
    return updated.model_dump(mode="json")


def _remap_timeline(proj: Project, media_id: str, res) -> int:
    """重切分之后，把时间线上已有片段重新挂到新的回合上。

    重切分会让所有回合 id 变化（边界也变了），旧片段就会「悬空」——界面上
    表现为片段全部掉成兜底色、分数标签对不上。这里按**原片时间重叠最多**
    重新挂一次，并刷新标签，让用户不必重新自动剪辑一遍。
    """
    if not res.rallies:
        return 0
    ranges = [(r, r.start, r.end) for r in res.rallies]
    n = 0
    for t in proj.timeline.tracks:
        for c in t.clips:
            if c.media_id != media_id:
                continue
            best = None
            best_ov = 0.0
            for r, rs, re_ in ranges:
                ov = max(0.0, min(c.src_out, re_) - max(c.src_in, rs))
                if ov > best_ov:
                    best_ov, best = ov, r
            if best is None or best_ov <= 0.05:
                continue
            c.rally_id = best.id
            c.label = f"#{best.index} {best.scores.total:.0f}分"
            n += 1
    if n:
        proj.timeline.duration = max(
            [c.tl_start + (c.src_out - c.src_in) / max(c.speed, 1e-6)
             for t in proj.timeline.tracks for c in t.clips] or [0.0]
        )
    return n


@app.get("/api/projects/{pid}/analysis/{mid}")
def get_analysis(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    res = proj.analyses.get(mid)
    if res is None:
        return {"status": "none"}
    return res.model_dump(mode="json")


@app.delete("/api/projects/{pid}/analysis/{mid}")
def clear_analysis(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    proj.analyses.pop(mid, None)
    ST.delete_analysis(pid, mid)
    ST.save_project(proj, write_analyses=False)
    return {"ok": True}


# ------------------------------------------------------------------ 回合编辑


@app.patch("/api/projects/{pid}/rallies/{rid}")
def patch_rally(pid: str, rid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    found: Rally | None = None
    for res in proj.analyses.values():
        for r in res.rallies:
            if r.id == rid:
                found = r
                break
        if found:
            break
    if found is None:
        raise HTTPException(404, "回合不存在")
    for k in ("keep", "starred", "note", "clip_start", "clip_end", "start", "end", "tags"):
        if k in payload:
            setattr(found, k, payload[k])
    ST.save_project(proj, write_analyses=True)
    return found.model_dump(mode="json")


@app.post("/api/projects/{pid}/rallies/bulk")
def bulk_rallies(pid: str, payload: dict = Body(...)) -> dict:
    """批量修改回合（按 id 列表或按筛选条件）。"""
    proj = _must_project(pid)
    mid = payload.get("media_id")
    patch = payload.get("patch") or {}
    ids = set(payload.get("ids") or [])
    flt = payload.get("filter") or {}
    n = 0
    for key, res in proj.analyses.items():
        if mid and key != mid:
            continue
        for r in res.rallies:
            if ids and r.id not in ids:
                continue
            if not ids and not _match_filter(r, flt):
                continue
            for k, v in patch.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            n += 1
    ST.save_project(proj, write_analyses=True)
    return {"updated": n}


@app.get("/api/projects/{pid}/rallies")
def list_rallies(pid: str, media_id: str | None = None) -> dict:
    proj = _must_project(pid)
    out: list[dict] = []
    for key, res in proj.analyses.items():
        if media_id and key != media_id:
            continue
        for r in res.rallies:
            out.append(r.model_dump(mode="json"))
    return {"rallies": out, "count": len(out)}


def _match_filter(r: Rally, flt: dict) -> bool:
    if not flt:
        return True
    if "min_score" in flt and r.scores.total < float(flt["min_score"]):
        return False
    if "max_score" in flt and r.scores.total > float(flt["max_score"]):
        return False
    if "min_duration" in flt and r.duration < float(flt["min_duration"]):
        return False
    if "max_duration" in flt and r.duration > float(flt["max_duration"]):
        return False
    if "min_shots" in flt and r.features.shot_count < int(flt["min_shots"]):
        return False
    if flt.get("starred_only") and not r.starred:
        return False
    if "tags" in flt and flt["tags"]:
        want = set(flt["tags"])
        if not want & set(r.tags):
            return False
    if "keep" in flt and bool(r.keep) != bool(flt["keep"]):
        return False
    return True


@app.post("/api/projects/{pid}/rallies/rescore")
def rescore(pid: str, payload: dict = Body(default={})) -> dict:
    """换一套评分权重重新给已分析的回合打分（不重跑 AI 分析）。"""
    from .analysis import scoring as SC

    proj = _must_project(pid)
    key = str(payload.get("weights") or "balanced")
    w = SC.PRESETS.get(key, SC.PRESETS["balanced"])
    mid = payload.get("media_id")
    n = 0
    for k, res in proj.analyses.items():
        if mid and k != mid:
            continue

        # 每个口径算出来的成绩缓存一份：切走再切回来要能回到原来的分数。
        # 分析结果里存的客观特征是被裁剪过的，重算出来的分数和当初分析时并不完全一致，
        # 不缓存的话「来回切一下就再也回不到原来的排名」。
        cache = res.stats.setdefault("score_cache", {})
        cur = str(res.stats.get("weights") or "balanced")
        if cur and cur not in cache and res.rallies:
            cache[cur] = [
                {
                    "total": r.scores.total,
                    "length": r.scores.length,
                    "intensity": r.scores.intensity,
                    "technique": r.scores.technique,
                    "excitement": r.scores.excitement,
                    "production": r.scores.production,
                    "tags": list(r.tags),
                }
                for r in res.rallies
            ]

        cached = cache.get(key)
        if isinstance(cached, list) and len(cached) == len(res.rallies):
            for r, s in zip(res.rallies, cached):
                r.scores.total = float(s["total"])
                r.scores.length = float(s["length"])
                r.scores.intensity = float(s["intensity"])
                r.scores.technique = float(s["technique"])
                r.scores.excitement = float(s["excitement"])
                r.scores.production = float(s["production"])
                r.tags = list(s.get("tags") or [])
                n += 1
            res.stats["weights"] = key
            continue

        feats = []
        quality = []
        for r in res.rallies:
            feats.append({
                "duration": r.features.duration,
                "shot_count": r.features.shot_count,
                "tempo": r.features.tempo,
                "finish_tempo": r.features.finish_intensity,
                "activity_mean": r.features.motion_energy,
                "activity_peak": r.features.motion_peak,
                "motion_mean": r.features.motion_energy,
                "motion_peak": r.features.motion_peak,
                "shuttle_speed_p90": r.features.shuttle_speed_p95,
                "confidence": r.features.confidence,
                # 评分还要用到这几路：以前这里没传，导致换口径算出来的分数
                # 和当初分析时的口径对不上
                "hit_strength_p90": r.features.hit_strength_p90,
                "player_speed_mean": r.features.player_speed_mean,
                "player_speed_max": r.features.player_speed_max,
                "shuttle_presence": r.features.shuttle_presence,
            })
            quality.append({
                "sharpness": r.features.quality_sharpness,
                "shake": r.features.quality_shake,
                "subject_size": r.features.quality_subject_size,
            })
        scores = SC.score_rallies(feats, w, quality)
        for r, s in zip(res.rallies, scores):
            r.scores.total = float(s["total"])
            r.scores.length = float(s["length"])
            r.scores.intensity = float(s["intensity"])
            r.scores.technique = float(s["technique"])
            r.scores.excitement = float(s["excitement"])
            r.scores.production = float(s["production"])
            r.tags = list(s.get("tags", []))
            n += 1
        res.stats["weights"] = key
        cache[key] = [
            {
                "total": r.scores.total,
                "length": r.scores.length,
                "intensity": r.scores.intensity,
                "technique": r.scores.technique,
                "excitement": r.scores.excitement,
                "production": r.scores.production,
                "tags": list(r.tags),
            }
            for r in res.rallies
        ]
    ST.save_project(proj, write_analyses=True)
    return {"rescored": n, "weights": key}


# ------------------------------------------------------------------ 时间线


@app.post("/api/projects/{pid}/timeline/auto-cut")
def auto_cut(pid: str, payload: dict = Body(...)) -> dict:
    """按筛选条件把回合拼成时间线。"""
    proj = _must_project(pid)
    mid = payload.get("media_id")
    flt = payload.get("filter") or {}
    mode = str(payload.get("mode") or "replace")     # replace | append
    gap = float(payload.get("gap") or 0.0)
    rally_ids = set(payload.get("rally_ids") or [])
    pre = payload.get("pre")
    post = payload.get("post")

    picked: list[tuple[Rally, str, float, float]] = []
    for key, res in proj.analyses.items():
        if mid and key != mid:
            continue
        for r in res.rallies:
            if rally_ids:
                if r.id not in rally_ids:
                    continue
            else:
                if r.keep is False:
                    continue
                if not _match_filter(r, flt):
                    continue
            a = float(pre) if pre is not None else r.clip_start
            b = float(post) if post is not None else r.clip_end
            a = max(0.0, min(a, r.end))
            b = max(a + 0.2, b)
            picked.append((r, key, a, b))

    picked.sort(key=lambda c: c[2])
    timeline = proj.timeline if mode == "append" else Timeline(tracks=[Track(name="视频轨 1", kind="video")])
    if not timeline.tracks:
        timeline.tracks = [Track(name="视频轨 1", kind="video")]
    track = timeline.tracks[0]
    duration = 0.0
    if mode != "append":
        track.clips = []
    elif track.clips:
        duration = max(c.tl_start + c.duration for c in track.clips)

    # 追加时跳过成片里已经有的回合：一个回合只进成片一次
    already = {c.rally_id for c in track.clips if c.rally_id}
    before = len(track.clips)

    from .core.models import Clip

    for r, key, a, b in picked:
        if mode == "append" and r.id in already:
            continue
        already.add(r.id)
        clip = Clip(
            media_id=key,
            src_in=round(a, 3),
            src_out=round(b, 3),
            tl_start=round(duration, 3),
            rally_id=r.id,
            label=f"#{r.index} {r.scores.total:.0f}分",
        )
        track.clips.append(clip)
        duration += clip.duration + gap
    timeline.duration = duration
    proj.timeline = timeline
    ST.save_project(proj, write_analyses=False)
    return {
        "timeline": timeline.model_dump(mode="json"),
        "clip_count": len(track.clips),
        "added": len(track.clips) - before,
        "skipped": max(0, len(picked) - (len(track.clips) - before)),
    }


@app.post("/api/projects/{pid}/timeline")
def set_timeline(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    tl = Timeline.model_validate(payload.get("timeline") or payload)
    total = 0.0
    for t in tl.tracks:
        for c in t.clips:
            total = max(total, c.tl_start + c.duration)
    tl.duration = total
    proj.timeline = tl
    ST.save_project(proj, write_analyses=False)
    return tl.model_dump(mode="json")


# ------------------------------------------------------------------ 导出


@app.get("/api/export/presets")
def export_presets() -> list[dict]:
    return [p.model_dump(mode="json") for p in _presets()]


def _presets() -> list[ExportPreset]:
    return [
        ExportPreset(id="yt1080p", name="横屏 1080p · 高画质", width=1920, height=1080, video_bitrate="16M"),
        ExportPreset(id="yt4k", name="横屏 4K · 高画质", width=3840, height=2160, video_bitrate="45M"),
        ExportPreset(id="yt720p", name="横屏 720p · 体积小", width=1280, height=720, video_bitrate="6M"),
        ExportPreset(id="vertical", name="竖屏 1080x1920 · 抖音/小红书", width=1080, height=1920,
                     video_bitrate="12M", auto_reframe=True),
        ExportPreset(id="square", name="方形 1080x1080 · 朋友圈", width=1080, height=1080, video_bitrate="10M"),
        ExportPreset(id="hevc4k", name="横屏 4K · HEVC 省空间", width=3840, height=2160, vcodec="hevc",
                     video_bitrate="25M"),
        ExportPreset(id="draft", name="快速预览 720p", width=1280, height=720, video_bitrate="4M", crf=26),
    ]


@app.post("/api/projects/{pid}/export")
def export(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    raw = dict(payload.get("preset") or {})
    # 只给了 id 也要能选中预设：直接 model_validate({"id": "draft"}) 会把其余字段
    # 补成默认值（1080p / 12M），于是脚本调用者会静默拿到规格不对的成片。
    known = {p.id: p for p in _presets()}
    base = known.get(str(raw.get("id") or ""))
    if base is not None:
        merged = base.model_dump(mode="json")
        for k, v in raw.items():
            if v is not None:
                merged[k] = v
        preset = ExportPreset.model_validate(merged)
    else:
        preset = ExportPreset.model_validate(raw)
    name = str(payload.get("name") or f"{proj.name}_{int(time.time())}")
    out = EXPORT_DIR / f"{_safe_name(name)}.mp4"
    timeline = proj.timeline
    if not any(t.clips for t in timeline.tracks):
        raise HTTPException(400, "时间线为空，先执行自动剪辑")

    from .render.exporter import export_timeline

    def work(job):
        return export_timeline(proj, timeline, preset, out,
                               on_progress=lambda p, msg: job.progress(p, "export", msg),
                               cancel=job.cancelled)

    job = JOBS.submit("export", f"导出 {out.name}", work)
    return {"job_id": job.id, "output": str(out)}


def _safe_name(s: str) -> str:
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s.strip()[:120] or "export"


@app.get("/api/exports")
def list_exports() -> list[dict]:
    out = []
    if EXPORT_DIR.exists():
        for f in sorted(EXPORT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = f.stat()
            out.append({"name": f.name, "path": str(f), "size": st.st_size, "mtime": int(st.st_mtime * 1000)})
    return out


@app.get("/api/exports/{name}")
def download_export(name: str, request: Request):
    p = EXPORT_DIR / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    return range_response(request, p, 3600)


# ------------------------------------------------------------------ 任务


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.model_dump(mode="json") for j in JOBS.list()]


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.info.model_dump(mode="json")


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str) -> dict:
    return {"ok": JOBS.cancel(jid)}


@app.get("/api/cache/stats")
def cache_stats() -> dict:
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    return {
        "cache": du(CACHE_DIR),
        "proxies": du(PROXIES_DIR),
        "thumbs": du(THUMBS_DIR),
        "exports": du(EXPORT_DIR),
    }


@app.post("/api/cache/clear")
def cache_clear(payload: dict = Body(default={})) -> dict:
    import shutil as _sh

    target = payload.get("target") or "all"
    if target == "proxies":
        dirs = {"proxies": PROXIES_DIR}
    elif target == "thumbs":
        dirs = {"thumbs": THUMBS_DIR}
    else:
        dirs = {
            "proxies": PROXIES_DIR,
            "thumbs": THUMBS_DIR,
            "audio": CACHE_DIR / "audio",
            "frames": CACHE_DIR / "frames",
        }
    freed = 0
    for d in dirs.values():
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            _sh.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
    return {"freed": freed}


# ------------------------------------------------------------------ 静态前端


def _mount_frontend() -> None:
    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def _index():
            return FileResponse(FRONTEND_DIST / "index.html")

        @app.get("/{full_path:path}")
        def _spa(full_path: str):
            if full_path.startswith(("api/", "ws")):
                raise HTTPException(404, "Not found")
            cand = FRONTEND_DIST / full_path
            if cand.is_file():
                return FileResponse(cand)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:
        @app.get("/")
        def _placeholder():
            return JSONResponse({
                "app": APP_DISPLAY_NAME,
                "note": "前端尚未构建。开发时请运行 `npm run dev`（Vite，默认 http://127.0.0.1:5273）；"
                        "构建请运行 `npm run build`（输出到 frontend/dist）。",
                "api_docs": "/api/docs",
            })


_mount_frontend()
