"""BadmintonStudio local service: REST API + WebSocket progress push + static frontend."""

from __future__ import annotations

import asyncio
import json
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .config import (
    APP_DISPLAY_NAME,
    APP_VERSION,
    CACHE_DIR,
    DATA_DIR,
    EXPORT_DIR,
    FRONTEND_DIST,
    MODELS_DIR,
    PROXIES_DIR,
    PROJECTS_DIR,
    THUMBS_DIR,
    ensure_dirs,
)
from .core import ffmpeg as ff
from .core import media as M
from .core import store as ST
from .core import exports as EXPORTS
from .core.jobs import manager as JOBS
from .core.models import AnalysisParams, ExportPreset, MediaInfo, Project, Rally, Timeline, Track, now_ms
from .core.streaming import range_response
from .samples import ensure_sample_files
from .i18n import app_name, get_lang, parse_lang, set_lang, tr
from .api.annotations import router as annotations_router
from .api.presets import router as presets_router
from .logging_setup import setup_logging

ensure_dirs()
setup_logging()

#: Hostnames the local desktop service may be addressed by. A DNS-rebinding domain resolves to
#: 127.0.0.1 but still sends its own name in the Host header, so an allow-list of local names blocks it.
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
#: Browser origins allowed to call the API: same-origin local pages plus the Vite dev server.
LOCAL_ORIGIN_REGEX = r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$"


def _host_is_local(host: str | None) -> bool:
    if not host:
        return True  # no Host header (HTTP/1.0, script clients): no rebinding surface
    h = host.strip()
    if h.startswith("["):  # IPv6 literal, e.g. [::1]:8000
        h = h[1:h.find("]")] if "]" in h else h[1:]
    elif ":" in h:
        h = h.rsplit(":", 1)[0]
    return h.lower() in LOCAL_HOSTNAMES


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True  # no Origin: same-origin navigation or a non-browser client (CLI / scripts)
    if origin == "null":
        return True  # file:// or a sandboxed webview
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "").lower() in LOCAL_HOSTNAMES


app = FastAPI(title=APP_DISPLAY_NAME, version=APP_VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    # Do not allow "*" with credentials: restrict to local origins only.
    allow_origins=[],
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _guard_middleware(request: Request, call_next):
    # Reject cross-origin browser requests and DNS-rebinding Host values. Requests without an Origin
    # header (CLI, scripts, e2e) still work.
    if not _host_is_local(request.headers.get("host")) or not _origin_is_local(request.headers.get("origin")):
        return JSONResponse({"detail": tr("api.origin_denied")}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def _lang_middleware(request: Request, call_next):
    # X-BMS-Lang (exact) first, then Accept-Language (may be a list).
    set_lang(parse_lang(request.headers.get("x-bms-lang"), request.headers.get("accept-language")))
    return await call_next(request)


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    """Log unhandled route errors (a 500 otherwise leaves no trace anywhere durable)."""
    logger.opt(exception=exc).error("unhandled {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse({"detail": tr("api.internal_error")}, status_code=500)


app.include_router(annotations_router)
app.include_router(presets_router)


# ------------------------------------------------------------------ WebSocket broadcast


class Hub:
    """Thread-safe broadcast hub: background job threads -> event loop -> all browser connections."""

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
        """Can be called from any thread."""
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
    # Browsers always send Origin on WebSocket handshakes; apply the same local-origin/Host policy.
    if not _host_is_local(ws.headers.get("host")) or not _origin_is_local(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    set_lang(parse_lang(ws.query_params.get("lang")))
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


# ------------------------------------------------------------------ Basics


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": app_name(), "version": APP_VERSION, "time": now_ms()}


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
    # Whether voice commands are available: only needs the faster-whisper dependency (the model is
    # downloaded on first use); the UI uses this to prompt "missing dependency".
    try:
        import importlib.util

        caps["speech"] = bool(importlib.util.find_spec("faster_whisper"))
    except Exception:  # noqa: BLE001
        caps["speech"] = False
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
    from .core.media import hardware_available, probe_backend_status

    try:
        probe = probe_backend_status()
    except Exception as e:  # noqa: BLE001
        probe = {"active": "unknown", "error": str(e)}

    return {
        "app": app_name(),
        "version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "ffmpeg": ffmpeg_path,
        "ffmpeg_error": err,
        "probe": probe,
        "caps": caps,
        "gpu": gpu,
        "hardware_pipeline": bool(caps.get("cuda_decode") and hardware_available()),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "cache_dir": str(CACHE_DIR),
        "export_dir": str(EXPORT_DIR),
    }


# ------------------------------------------------------------------ Media file access


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


def _is_device_or_unc(raw: str) -> bool:
    """Reject UNC (``\\\\server\\share``) and Windows device (``\\\\?\\`` / ``\\\\.\\``) paths.

    Device paths can bypass drive-root checks and read things like ``\\\\?\\C:\\...``; treat them as
    outside the allowed scope everywhere. A single leading backslash is just a rooted local path.
    """
    s = str(raw).strip()
    return s[:2] in ("\\\\", "//")


def _resolve_asset(raw: str) -> Path:
    if _is_device_or_unc(raw):
        raise HTTPException(403, tr("api.path_not_allowed"))
    p = Path(raw)
    try:
        p = p.resolve()
    except OSError:
        pass
    if not p.is_file():
        raise HTTPException(404, tr("api.asset_not_found"))
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
    raise HTTPException(403, tr("api.path_not_allowed"))


@app.get("/api/asset")
def asset(request: Request, p: str = Query(...), cache: int = 3600):
    path = _resolve_asset(p)
    return range_response(request, path, cache_seconds=cache)


# ------------------------------------------------------------------ Project


_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _check_id(value: str, what: str = "id") -> str:
    """Project/media ids are concatenated into file paths, so the character set must be restricted.

    Otherwise ids like ``..\\..\\x`` would let reads/writes/deletes escape the data/projects directory.
    """
    if not _ID_RE.fullmatch(value or ""):
        raise HTTPException(400, tr("api.id_invalid", what=what))
    return value


def _bad_request(error: object) -> HTTPException:
    """Turn a malformed client value into a 400 instead of letting it bubble up as a 500."""
    return HTTPException(400, tr("api.bad_request", error=str(error)[:200]))


#: Fields a client may change on a rally through the patch/bulk endpoints.
_RALLY_PATCH_FIELDS = {"keep", "starred", "note", "clip_start", "clip_end", "start", "end", "tags"}


def _patched_rally(r: Rally, patch: dict) -> Rally:
    """Validate a rally patch against the whitelist by re-validating the whole model.

    Pydantic v2 does not validate attribute assignment, so ``setattr`` let a wrong type (e.g. a
    string ``clip_end``) into the sidecar; the next ``load_project`` then dropped the entire
    analysis. Building the dict and calling ``model_validate`` rejects bad values up front.
    """
    unknown = sorted(k for k in patch if k not in _RALLY_PATCH_FIELDS)
    if unknown:
        raise HTTPException(400, tr("api.bad_request", error=tr("api.rally_patch_unknown", field=unknown[0])))
    data = r.model_dump(mode="json")
    data.update({k: patch[k] for k in patch})
    try:
        return Rally.model_validate(data)
    except ValidationError as e:
        raise _bad_request(e) from e


def _must_project(pid: str) -> Project:
    _check_id(pid, tr("api.project_id"))
    proj = ST.load_project(pid)
    if proj is None:
        raise HTTPException(404, tr("api.project_not_found"))
    return proj


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return [s.model_dump(mode="json") for s in ST.list_projects()]


@app.post("/api/projects")
def create_project(payload: dict = Body(default={})) -> dict:
    proj = ST.create_project(payload.get("name") or tr("project.untitled"))
    return proj.model_dump(mode="json")


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    return _must_project(pid).model_dump(mode="json")


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, payload: dict = Body(...)) -> dict:
    _check_id(pid, tr("api.project_id"))
    raw_timeline = payload.get("timeline")
    parsed_timeline: Timeline | None = None
    if isinstance(raw_timeline, dict):
        try:
            parsed_timeline = Timeline.model_validate(raw_timeline)
        except ValidationError as e:
            raise _bad_request(e) from e
    elif raw_timeline is not None:
        raise _bad_request("timeline")

    def apply(proj: Project) -> None:
        if payload.get("name"):
            proj.name = str(payload["name"])
        if isinstance(payload.get("ui"), dict):
            proj.ui.update(payload["ui"])
        if parsed_timeline is not None:
            proj.timeline = parsed_timeline

    proj = ST.update_project(pid, apply, write_analyses=False)
    if proj is None:
        raise HTTPException(404, tr("api.project_not_found"))
    return proj.model_dump(mode="json")


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict:
    _check_id(pid, tr("api.project_id"))
    return {"ok": ST.delete_project(pid)}


@app.post("/api/projects/{pid}/duplicate")
def duplicate_project(pid: str, payload: dict = Body(default={})) -> dict:
    _check_id(pid, tr("api.project_id"))
    p = ST.duplicate_project(pid, payload.get("name"))
    if p is None:
        raise HTTPException(404, tr("api.project_not_found"))
    return p.model_dump(mode="json")


# ------------------------------------------------------------------ Media

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".m4v", ".webm", ".ts", ".mpg", ".mpeg", ".m2ts", ".3gp"}


def _natural_key(p: Path) -> list:
    """Make clip2 sort before clip10, avoiding the ordering being scrambled by string sorting."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _expand_media_paths(raw_paths: list) -> tuple[list[Path], list[dict]]:
    """Expand directories into the video files inside them (recursive, natural sort); keep files as-is and deduplicate."""
    found: list[Path] = []
    failed: list[dict] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        p = Path(str(raw))
        if p.is_dir():
            vids = [
                f
                for f in sorted(p.rglob("*"), key=_natural_key)
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            ]
            if not vids:
                failed.append({"path": str(raw), "error": tr("media.no_video_in_dir")})
            for f in vids:
                if f not in seen:
                    seen.add(f)
                    found.append(f)
        elif p.is_file():
            if p not in seen:
                seen.add(p)
                found.append(p)
        else:
            failed.append({"path": str(raw), "error": tr("api.file_not_found")})
    return found, failed


@app.post("/api/projects/{pid}/media")
def add_media(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    raw_paths = payload.get("paths") or ([payload["path"]] if payload.get("path") else [])
    if not raw_paths:
        raise HTTPException(400, tr("api.missing_paths"))
    paths, failed = _expand_media_paths(raw_paths)
    added = []
    for p in paths:
        try:
            info = M.probe_media(p)
            proj = ST.touch_media(proj, info)
            added.append(info.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            failed.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
    return {"project": proj.model_dump(mode="json"), "added": added, "failed": failed}


@app.post("/api/projects/{pid}/media/upload")
async def upload_media(pid: str, request: Request) -> dict:
    """Receive a file dragged in by the browser (raw byte stream + filename header)."""
    # Loading the project and probing the uploaded file (an ffprobe subprocess) are blocking; keep
    # them off the event loop so concurrent requests are not stalled.
    proj = await run_in_threadpool(_must_project, pid)
    import urllib.parse

    name = request.headers.get("x-filename") or "upload.mp4"
    name = urllib.parse.unquote(name)
    name = Path(name).name or "upload.mp4"
    dest_dir = DATA_DIR / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        dest = dest_dir / f"{dest.stem}_{int(time.time())}{dest.suffix}"
    # Write .part first, then rename: a mid-transfer disconnect or probe failure never leaves half a file in uploads.
    part = dest_dir / f"{dest.name}.part"
    part.unlink(missing_ok=True)
    size = 0
    try:
        with open(part, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                size += len(chunk)
        part.replace(dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    try:
        info = await run_in_threadpool(M.probe_media, dest)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(400, tr("media.unrecognized", type=type(e).__name__)) from e
    proj = await run_in_threadpool(ST.touch_media, proj, info)
    return {"project": proj.model_dump(mode="json"), "added": info.model_dump(mode="json"), "bytes": size}


# ------------------------------------------------------------------ Native file dialogs

# tkinter is not thread-safe; allow only one dialog at a time to avoid focus/event-loop conflicts.
_dialog_lock = threading.Lock()


def _with_dialog(fn):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        root.update()
        return fn(root)
    finally:
        root.destroy()


def _pick_video_files() -> list[str]:
    """Open the Windows system file picker (multi-select) and return the chosen local paths."""
    from tkinter import filedialog

    return list(
        _with_dialog(
            lambda root: filedialog.askopenfilenames(
                parent=root,
                title=tr("dialog.pick_videos"),
                filetypes=[
                    (tr("dialog.filter_videos"), "*.mp4 *.mov *.mkv *.avi *.flv *.wmv *.m4v *.webm *.ts *.mpg *.mpeg"),
                    (tr("dialog.filter_all"), "*.*"),
                ],
            )
        )
    )


def _pick_video_folder() -> str:
    """Open the Windows system folder picker and return the chosen directory (empty string means cancelled)."""
    from tkinter import filedialog

    return _with_dialog(
        lambda root: filedialog.askdirectory(parent=root, title=tr("dialog.pick_video_folder"), mustexist=True)
    ) or ""


def _pick_export_folder() -> str:
    """Open the folder picker to choose the export directory for the finished video (empty string means cancelled)."""
    from tkinter import filedialog

    return _with_dialog(
        lambda root: filedialog.askdirectory(parent=root, title=tr("dialog.pick_export_folder"))
    ) or ""


def _dialog(kind: str) -> dict:
    if platform.system() != "Windows":
        raise HTTPException(501, tr("dialog.unsupported"))
    with _dialog_lock:
        try:
            if kind == "videos":
                paths = _pick_video_files()
            elif kind == "export_dir":
                folder = _pick_export_folder()
                paths = [folder] if folder else []
            else:
                # For a folder, just hand the path back; the actual expansion (recursively finding videos) is handled uniformly by add_media.
                folder = _pick_video_folder()
                paths = [folder] if folder else []
        except Exception as e:  # noqa: BLE001
            raise HTTPException(501, tr("dialog.open_failed", type=type(e).__name__, error=e)) from e
    return {"paths": paths, "cancelled": not paths}


@app.post("/api/dialog/videos")
def dialog_videos() -> dict:
    """Open the system file picker to choose videos and hand the local paths to the import API (without copying files).

    Browser fallback: returns 501 on non-Windows or when a window cannot be created, and the frontend falls back to <input type=file> upload.
    """
    return _dialog("videos")


@app.post("/api/dialog/folder")
def dialog_folder() -> dict:
    """Open the system folder picker and return the directory path; add_media recursively expands the videos inside."""
    return _dialog("folder")


@app.post("/api/dialog/export-dir")
def dialog_export_dir() -> dict:
    """Open the system folder picker as the export directory for the finished video."""
    return _dialog("export_dir")


def _purge_media(proj: Project, ids: set[str]) -> int:
    """Remove a batch of media from the project: media / analysis results / analysis files / timeline clips referencing it.

    Only touches the project, never deletes the original files on disk; gaps left by removed timeline
    clips are not backfilled (consistent with single deletion).
    """
    before = len(proj.media)
    proj.media = [m for m in proj.media if m.id not in ids]
    removed = before - len(proj.media)
    for mid in ids:
        proj.analyses.pop(mid, None)
        ST.delete_analysis(proj.id, mid)
    for t in proj.timeline.tracks:
        t.clips = [c for c in t.clips if c.media_id not in ids]
    return removed


@app.delete("/api/projects/{pid}/media/{mid}")
def remove_media(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    _check_id(mid, tr("api.media_id"))
    JOBS.cancel_for_media({mid})
    _purge_media(proj, {mid})
    ST.save_project(proj, write_analyses=False)
    return proj.model_dump(mode="json")


@app.post("/api/projects/{pid}/media/bulk-delete")
def bulk_delete_media(pid: str, payload: dict = Body(...)) -> dict:
    """Bulk-remove media from the project (does not touch the original files on disk)."""
    proj = _must_project(pid)
    raw = payload.get("ids") or []
    if not raw:
        raise HTTPException(400, tr("api.missing_ids"))
    ids = {_check_id(str(x), tr("api.media_id")) for x in raw}
    JOBS.cancel_for_media(ids)
    removed = _purge_media(proj, ids)
    ST.save_project(proj, write_analyses=False)
    return {"project": proj.model_dump(mode="json"), "removed": removed}


def _media(proj: Project, mid: str) -> MediaInfo:
    _check_id(mid, tr("api.media_id"))
    for m in proj.media:
        if m.id == mid:
            return m
    raise HTTPException(404, tr("media.not_found"))


@app.get("/api/projects/{pid}/media/{mid}/stream")
def stream_source(pid: str, mid: str, request: Request):
    return range_response(request, Path(_media(_must_project(pid), mid).path), 3600)


@app.get("/api/projects/{pid}/media/{mid}/proxy")
def stream_proxy(pid: str, mid: str, request: Request):
    proj = _must_project(pid)
    m = _media(proj, mid)
    p = M.proxy_source(m)
    return range_response(request, p, 3600)


@app.get("/api/projects/{pid}/media/{mid}/audio")
def stream_audio(pid: str, mid: str, request: Request):
    proj = _must_project(pid)
    m = _media(proj, mid)
    if not m.audio_path or not Path(m.audio_path).is_file():
        raise HTTPException(404, tr("media.no_audio"))
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
    raise HTTPException(404, tr("media.poster_failed"))


@app.get("/api/projects/{pid}/media/{mid}/sprite")
def sprite(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    m = _media(proj, mid)
    info = M.make_sprite(m, count=120, cols=12, tile_w=160)
    if not info:
        raise HTTPException(500, tr("media.sprite_failed"))
    return info


def _persist_media(pid: str, m: MediaInfo) -> None:
    """Write the media's derived paths back to the project and broadcast, to avoid recomputing the proxy every time."""
    changed = False

    def apply(proj: Project) -> None:
        nonlocal changed
        for i, x in enumerate(proj.media):
            if x.id == m.id:
                proj.media[i] = m
                changed = True
                return

    # load->mutate->save under one lock: otherwise two prepare jobs' saves interleave and the later
    # one overwrites the other's derived paths.
    ST.update_project(pid, apply, write_analyses=False)
    if changed:
        HUB.publish({"type": "media", "project_id": pid, "media": m.model_dump(mode="json")})


#: Proxy encoding is very CPU/GPU-heavy (NVENC also limits the number of concurrent sessions). During
#: bulk import the frontend submits one prepare job per media item; a semaphore queues them so dozens
#: of ffmpeg processes do not all start at once.
_PREPARE_GATE = threading.Semaphore(1)


def _acquire_prepare_gate(job) -> bool:
    """Queue for the prepare gate; cancellable while waiting. Returns True when acquired, False when cancelled."""
    if job.cancelled():
        return False
    job.set(stage="queued", message=tr("job.queued.prepare"))
    while not _PREPARE_GATE.acquire(timeout=0.5):
        if job.cancelled():
            return False
    return True


@app.post("/api/projects/{pid}/media/{mid}/prepare")
def prepare_media(pid: str, mid: str, payload: dict = Body(default={})) -> dict:
    """Generate the proxy video / audio track / poster (can be triggered separately so the user can prepare playback first)."""
    proj = _must_project(pid)
    m = _media(proj, mid)

    def work(job):
        if not _acquire_prepare_gate(job):
            return None
        try:
            if job.cancelled():
                return None
            job.progress(0.02, "proxy", tr("prepare.proxy"))
            M.ensure_proxy(m, on=lambda p, msg: job.progress(0.02 + 0.7 * p, "proxy", msg), cancel=job.cancelled)
            job.progress(0.75, "audio", tr("prepare.audio"))
            M.ensure_audio(m, on=lambda p, msg: job.progress(0.75 + 0.2 * p, "audio", msg), cancel=job.cancelled)
            job.progress(0.96, "poster", tr("prepare.poster"))
            M.ensure_poster(m)
            return {
                "media": m.model_dump(mode="json"),
                "sprite": M.make_sprite(m, count=120, cols=12, tile_w=160),
            }
        finally:
            # Persist before releasing the gate: prepare jobs are serialized, and if the lock were
            # released first, the load->modify->save of two jobs would interleave and the later save
            # would overwrite the derived paths saved by the earlier one.
            try:
                _persist_media(pid, m)
            finally:
                _PREPARE_GATE.release()

    job = JOBS.submit("prepare", tr("job.title.prepare", name=m.name), work, media_id=mid, lang=get_lang())
    return {"job_id": job.id}


@app.get("/api/media/probe")
def probe_path(path: str = Query(...)) -> dict:
    """Probe a local file inside the allowed roots and return duration/resolution etc.

    Arbitrary paths are rejected (same allow-list as ``/api/asset``): without it this endpoint
    leaks metadata about any file on disk to a page able to reach the local port.
    """
    p = _resolve_asset(path)
    try:
        info = M.probe_media(p)
    except Exception as e:  # noqa: BLE001
        # Give a clear 4xx for non-media files instead of letting the exception bubble up as 500
        raise HTTPException(400, tr("media.unrecognized", type=type(e).__name__)) from e
    return info.model_dump(mode="json")


# ------------------------------------------------------------------ Analysis


#: A single AI analysis saturates the GPU and may also spawn ffmpeg frame extraction at the same time.
#: During batch analysis the frontend submits one job per media item; a semaphore queues them to run one
#: by one, to avoid blowing up VRAM/RAM.
_ANALYZE_GATE = threading.Semaphore(1)


def _validated_weights(raw: object, *, fallback: object = "balanced") -> str:
    """Resolve a scoring-weight key against ``scoring.PRESETS``.

    An unknown key must be a 400 rather than being passed into scoring (where it either raises or,
    worse, gets stored in ``stats["weights"]`` and shown in the UI as if it were applied). ``fallback``
    is used only when the request omits the key and is itself coerced to a valid preset.
    """
    from .analysis import scoring as SC

    if raw is None or str(raw) == "":
        internal = str(fallback or "balanced")
        key = internal if internal in SC.PRESETS else "balanced"
    else:
        key = str(raw)
    if key not in SC.PRESETS:
        raise HTTPException(400, tr("analysis.unknown_weights", value=key))
    return key


def _param_with_overrides(base: AnalysisParams, raw: object) -> AnalysisParams:
    """Apply a client params dict onto AnalysisParams, validating the merged result.

    Pydantic v2 does not validate attribute assignment, so ``setattr`` allowed wrong types to enter
    the base params; re-validating the merged model rejects them with a 400. Unknown keys are ignored
    (same as the old ``hasattr`` guard).
    """
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise _bad_request("params")
    data = base.model_dump(mode="json")
    for k, v in raw.items():
        if k in AnalysisParams.model_fields:
            data[k] = v
    try:
        return AnalysisParams.model_validate(data)
    except ValidationError as e:
        raise _bad_request(e) from e


def _acquire_analyze_gate(job) -> bool:
    """Queue for the analysis gate; cancellable while waiting. Returns True when acquired, False when cancelled."""
    if job.cancelled():
        return False
    job.set(stage="queued", message=tr("job.queued.analyze"))
    while not _ANALYZE_GATE.acquire(timeout=0.5):
        if job.cancelled():
            return False
    return True


def _stored_court_poly(proj: Project, mid: str) -> list[list[float]] | None:
    """Read a media item's manually calibrated court boundary (new key court_polys, compatible with the old key court_quads).

    During batch analysis each media uses its own calibration; the boundary drawn for the current media
    must not be applied to all of them.
    """
    ui = proj.ui or {}
    raw = ui.get("court_polys") or ui.get("court_quads") or {}
    q = raw.get(mid) if isinstance(raw, dict) else None
    if isinstance(q, list) and len(q) >= 4:
        try:
            return [[float(p[0]), float(p[1])] for p in q]
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _make_analyze_work(pid: str, m: MediaInfo, params: AnalysisParams,
                       weights_key: str, roi_t) -> Any:
    """Build an analysis job body: acquire the serial gate, run the pipeline, then persist and broadcast."""
    from .analysis.pipeline import run_analysis

    def work(job):
        if not _acquire_analyze_gate(job):
            return None  # cancelled while queued; the gate was not acquired so it does not need releasing
        try:
            res = run_analysis(
                m,
                params,
                on_progress=lambda p, s, msg: job.progress(p, s, msg),
                cancel=job.cancelled,
                weights_key=weights_key,
                roi=roi_t,  # type: ignore[arg-type]
            )
            if res.status != "done":
                # A cancelled/failed result has empty rallies and must never be persisted: that would
                # overwrite the last successful analysis, and a single cancel click would lose the
                # entire analysis.
                if res.status == "cancelled":
                    return {}  # let Job._run see job.cancelled() and mark it as cancelled
                raise RuntimeError(res.error or tr("analysis.failed"))
            ST.save_analysis(pid, m.id, res)
            # The analysis process generates the proxy/audio/poster; these derived paths must also be
            # written back to the project, otherwise playback and analysis would regenerate them next time.
            _persist_media(pid, m)

            def apply(p2: Project) -> None:
                p2.analyses[m.id] = res

            ST.update_project(pid, apply, write_analyses=False)
            HUB.publish({"type": "analysis", "project_id": pid, "media_id": m.id,
                         "result": res.model_dump(mode="json")})
            return res.model_dump(mode="json")
        finally:
            _ANALYZE_GATE.release()

    return work


@app.post("/api/projects/{pid}/analyze")
def analyze(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, tr("api.missing_media_id"))
    m = _media(proj, mid)
    roi = payload.get("roi")
    try:
        params = AnalysisParams.model_validate(payload.get("params") or {})
        roi_t = tuple(float(v) for v in roi) if roi and len(roi) == 4 else None
    except (ValidationError, TypeError, ValueError) as e:
        raise _bad_request(e) from e
    weights_key = _validated_weights(payload.get("weights"))

    job = JOBS.submit("analyze", tr("job.title.analyze", name=m.name),
                      _make_analyze_work(pid, m, params, weights_key, roi_t), media_id=mid, lang=get_lang())
    return {"job_id": job.id}


@app.post("/api/projects/{pid}/analyze-batch")
def analyze_batch(pid: str, payload: dict = Body(default={})) -> dict:
    """Bulk analysis: submit several media at once (default = all in the project); jobs queue up and run one by one.

    Parameters use the settings from the request uniformly, but each media uses its own court
    calibration (read from the manual calibration stored under the project's ui), so the boundary
    drawn for the current media is not applied to all of them. Bulk mode ignores roi — the viewport
    region follows a single video, so sharing one makes no sense.
    """
    proj = _must_project(pid)
    raw_ids = payload.get("media_ids")
    if isinstance(raw_ids, list) and raw_ids:
        ids = [str(x) for x in raw_ids]
    else:
        ids = [m.id for m in proj.media]
    # Deduplicate while preserving order
    seen: set[str] = set()
    ids = [x for x in ids if not (x in seen or seen.add(x))]
    if not ids:
        raise HTTPException(400, tr("analysis.no_media"))

    try:
        params = AnalysisParams.model_validate(payload.get("params") or {})
    except ValidationError as e:
        raise _bad_request(e) from e
    weights_key = _validated_weights(payload.get("weights"))

    job_ids: list[str] = []
    for mid in ids:
        m = next((x for x in proj.media if x.id == mid), None)
        if m is None:
            continue  # media was deleted before the request arrived; skip instead of failing the whole batch
        p = params.model_copy(deep=True)
        p.court_poly = _stored_court_poly(proj, mid)
        job = JOBS.submit("analyze", tr("job.title.analyze", name=m.name),
                          _make_analyze_work(pid, m, p, weights_key, None), media_id=mid, lang=get_lang())
        job_ids.append(job.id)
    if not job_ids:
        raise HTTPException(400, tr("analysis.not_found"))
    return {"job_ids": job_ids, "count": len(job_ids)}


@app.post("/api/projects/{pid}/player-probe")
def player_probe(pid: str, payload: dict = Body(default={})) -> dict:
    """Quickly probe person-box size: take a few frames and run detection only (no tracking), returning in seconds.

    Why it is needed: when tuning "person-box size filtering", what the user really wants to know is
    only "how big are the players' boxes, and how big are spectators and people on other courts".
    Waiting for a full analysis takes minutes, while tuning usually needs several back-and-forth
    tries. Here 6~12 frames are sampled evenly across the whole video and inferred in one batch,
    giving the box-height distribution directly; the UI can immediately draw a histogram and
    simulate threshold effects locally.

    ``times`` (or ``at_time``) can specify **specific moments**: the UI uses it for "manual frame
    selection" — the grabbed frame is returned together with **the image itself** (``frame.image``,
    a cached JPEG path the frontend fetches via ``/api/asset``), so "which boxes are selected and
    which are filtered" can be drawn on the image and checked, instead of only a numeric distribution.

    Read-only on the media; it does not write analysis results and does not enter the job queue —
    it is fast enough to return synchronously.
    """
    from .analysis.pipeline import _manual_poly
    from .analysis import players as PL

    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, tr("api.missing_media_id"))
    m = _media(proj, mid)
    try:
        params = AnalysisParams.model_validate(payload.get("params") or {})
        count = int(payload.get("count") or 10)
    except (ValidationError, TypeError, ValueError) as e:
        raise _bad_request(e) from e
    count = max(1, min(40, count))
    viewpoint = params.viewpoint if params.viewpoint != "auto" else "unknown"
    # Manual frame selection: times explicitly specifies time points; at_time is shorthand for "grab just this frame"
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
    # Save the frame image by default: the UI needs to draw it for the user to verify
    save_frames = bool(payload.get("save_frames", True))

    # Court region: prefer the manually marked boundary in the request, then the polygon calibrated in
    # the saved analysis result. The statistics must match real analysis, otherwise the trial distribution
    # will not line up.
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

    path = str(M.proxy_source(m))
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
    """Quickly resegment rallies from the stored activity curve (milliseconds), for tuning segmentation parameters and scoring profiles.

    Player detection and motion analysis are not re-run, so tuning is immediate.
    """
    from .analysis.pipeline import resegment as _reseg

    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, tr("api.missing_media_id"))
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, tr("analysis.missing_result"))

    base = _param_with_overrides(res.params, payload.get("params"))
    weights_key = _validated_weights(payload.get("weights"), fallback=res.stats.get("weights"))
    try:
        updated = _reseg(res, base, weights_key)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}") from None

    ST.save_analysis(pid, mid, updated)

    remapped = 0

    def apply(proj2: Project) -> None:
        nonlocal remapped
        proj2.analyses[mid] = updated
        remapped = _remap_timeline(proj2, mid, updated)

    proj = ST.update_project(pid, apply, write_analyses=False)
    HUB.publish({"type": "analysis", "project_id": pid, "media_id": mid,
                 "result": updated.model_dump(mode="json")})
    if remapped and proj is not None:
        HUB.publish({"type": "timeline", "project_id": pid,
                     "timeline": proj.timeline.model_dump(mode="json")})
    return updated.model_dump(mode="json")


@app.post("/api/projects/{pid}/rebuild-hits")
def rebuild_hits(pid: str, payload: dict = Body(default={})) -> dict:
    """Rebuild the audio hit sequence from the cached WAV (no video AI) and re-apply the gate.

    Needed by projects analyzed before the pre-gate hit sequence was persisted: afterwards the
    cross-court suppression slider works. Re-runs only the audio detection (cheap STFT) and, when the
    full-rate pose signal can be recovered from the pose cache, re-gates; otherwise the sequence is
    stored without gating and the caller is told a full re-analysis is required for attribution.
    """
    from .analysis import audio_hits as AH
    from .analysis.pipeline import (
        recover_pose_from_cache,
        resegment as _reseg,
        _store_pose_full,
    )

    proj = _must_project(pid)
    mid = payload.get("media_id")
    if not mid:
        raise HTTPException(400, tr("api.missing_media_id"))
    m = _media(proj, mid)
    res = proj.analyses.get(mid)
    if res is None or res.status != "done":
        raise HTTPException(400, tr("analysis.missing_result"))

    base = _param_with_overrides(res.params, payload.get("params"))

    sig = res.signals or {}
    raw = None
    pose = None
    if base.use_audio:
        M.ensure_audio(m)
        if not m.audio_path:
            raise HTTPException(400, tr("analysis.rebuild.no_audio"))
        try:
            raw = AH.detect_hits(m.audio_path, sensitivity=base.hit_sensitivity)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"{type(e).__name__}: {e}") from None
        sig["hit_times_raw"] = [round(float(v), 3) for v in raw.times]
        sig["hit_strength_raw"] = [round(float(v), 3) for v in raw.strength]
        sig["hit_confidence_raw"] = [round(float(v), 3) for v in raw.confidence]
        res.signals = sig
        pose = recover_pose_from_cache(sig)
        if pose is not None:
            _store_pose_full(sig, pose)
    else:
        # Audio is switched off: there is nothing to rebuild. Leave the raw candidates untouched and
        # let recompute run the visual-only path (see pipeline.regate_hits), avoiding a pointless STFT.
        res.signals = sig

    weights_key = _validated_weights(payload.get("weights"), fallback=(res.stats or {}).get("weights"))
    try:
        updated = _reseg(res, base, weights_key)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}") from None
    updated.stats["rebuild_hits"] = {
        "raw_count": int(raw.times.size) if raw is not None else 0,
        "pose_recovered": pose is not None,
        "audio_off": not base.use_audio,
    }

    ST.save_analysis(pid, mid, updated)

    remapped = 0

    def apply(proj2: Project) -> None:
        nonlocal remapped
        proj2.analyses[mid] = updated
        remapped = _remap_timeline(proj2, mid, updated)

    proj = ST.update_project(pid, apply, write_analyses=False)
    HUB.publish({"type": "analysis", "project_id": pid, "media_id": mid,
                 "result": updated.model_dump(mode="json")})
    if remapped and proj is not None:
        HUB.publish({"type": "timeline", "project_id": pid,
                     "timeline": proj.timeline.model_dump(mode="json")})
    return updated.model_dump(mode="json")


def _remap_timeline(proj: Project, media_id: str, res) -> int:
    """After re-segmentation, reattach existing timeline clips to the new rallies.

    Re-segmentation changes all rally ids (and boundaries), leaving old clips "dangling" -- in the UI
    every clip drops to the fallback color and the score labels no longer match. This reattaches by
    **maximum source-time overlap** and refreshes the labels, so the user does not have to run auto-cut again.
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
            c.label = tr("timeline.clip_label", index=best.index, score=best.scores.total)
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
    _check_id(mid, tr("api.media_id"))
    res = proj.analyses.get(mid)
    if res is None:
        return {"status": "none"}
    return res.model_dump(mode="json")


@app.delete("/api/projects/{pid}/analysis/{mid}")
def clear_analysis(pid: str, mid: str) -> dict:
    proj = _must_project(pid)
    _check_id(mid, tr("api.media_id"))
    proj.analyses.pop(mid, None)
    ST.delete_analysis(pid, mid)
    ST.save_project(proj, write_analyses=False)
    return {"ok": True}


# ------------------------------------------------------------------ Rally editing


@app.patch("/api/projects/{pid}/rallies/{rid}")
def patch_rally(pid: str, rid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    found: Rally | None = None
    container: list[Rally] | None = None
    index = -1
    for res in proj.analyses.values():
        for i, r in enumerate(res.rallies):
            if r.id == rid:
                found, container, index = r, res.rallies, i
                break
        if found is not None:
            break
    if found is None or container is None:
        raise HTTPException(404, tr("rally.not_found"))
    updated = _patched_rally(found, payload)
    container[index] = updated
    ST.save_project(proj, write_analyses=True)
    return updated.model_dump(mode="json")


@app.post("/api/projects/{pid}/rallies/bulk")
def bulk_rallies(pid: str, payload: dict = Body(...)) -> dict:
    """Bulk-modify rallies (by id list or by filter criteria)."""
    proj = _must_project(pid)
    mid = payload.get("media_id")
    patch = payload.get("patch") or {}
    flt = payload.get("filter") or {}
    if not isinstance(patch, dict):
        raise _bad_request("patch")
    if not isinstance(flt, dict):
        raise _bad_request("filter")
    unknown = sorted(k for k in patch if k not in _RALLY_PATCH_FIELDS)
    if unknown:
        raise HTTPException(400, tr("api.bad_request", error=tr("api.rally_patch_unknown", field=unknown[0])))
    has_ids = "ids" in payload
    ids = {str(x) for x in (payload.get("ids") or [])}
    # When ids is passed explicitly it must take precedence; an empty list means "do nothing".
    # The old code treated both an empty ids and "ids not passed" as matching by filter, and an empty
    # filter matches everything, so clicking "exclude all" with an empty filter result would change
    # every rally in the whole project.
    if has_ids and not ids:
        return {"updated": 0}
    n = 0
    for key, res in proj.analyses.items():
        if mid and key != mid:
            continue
        for i, r in enumerate(res.rallies):
            if has_ids:
                if r.id not in ids:
                    continue
            elif not _match_filter(r, flt):
                continue
            res.rallies[i] = _patched_rally(r, patch)
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


_FILTER_ALIASES = {
    "minScore": "min_score",
    "maxScore": "max_score",
    "minDuration": "min_duration",
    "maxDuration": "max_duration",
    "minShots": "min_shots",
    "minConfidence": "min_confidence",
    "starredOnly": "starred_only",
}


def _match_filter(r: Rally, flt: dict) -> bool:
    if not flt:
        return True
    # The frontend uses camelCase (minScore...); normalize to snake_case here, otherwise filter
    # conditions other than tags are silently ignored and bulk operations hit every rally.
    flt = dict(flt)
    for cam, snake in _FILTER_ALIASES.items():
        if cam in flt and snake not in flt:
            flt[snake] = flt[cam]
    try:
        if flt.get("keepOnly") and not r.keep:
            return False
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
        if "min_confidence" in flt and r.features.confidence < float(flt["min_confidence"]):
            return False
        if flt.get("starred_only") and not r.starred:
            return False
        if "tags" in flt and flt["tags"]:
            want = set(flt["tags"])
            if not want & set(r.tags):
                return False
        if "keep" in flt and bool(r.keep) != bool(flt["keep"]):
            return False
    except (TypeError, ValueError) as e:
        raise _bad_request(e) from e
    return True


def _rescore_features(r: Rally) -> dict:
    return {
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
        # Scoring also uses these signals: they used to be omitted here, so scores recomputed under a
        # different weighting did not match the weighting used at analysis time
        "hit_strength_p90": r.features.hit_strength_p90,
        "player_speed_mean": r.features.player_speed_mean,
        "player_speed_max": r.features.player_speed_max,
        "shuttle_presence": r.features.shuttle_presence,
        # The voice-command bonus is part of the total, so a rescore must carry it over (otherwise
        # switching the weighting silently drops the bonus) and its phrases must survive for tags.
        "speech_bonus": r.features.speech_bonus,
        "speech_phrases": list(r.features.speech_phrases),
    }


def _rescore_quality(r: Rally) -> dict:
    return {
        "sharpness": r.features.quality_sharpness,
        "shake": r.features.quality_shake,
        "subject_size": r.features.quality_subject_size,
    }


def _rescore_assign(r: Rally, s: dict) -> None:
    r.scores.total = float(s["total"])
    r.scores.length = float(s["length"])
    r.scores.intensity = float(s["intensity"])
    r.scores.technique = float(s["technique"])
    r.scores.excitement = float(s["excitement"])
    r.scores.production = float(s["production"])
    r.tags = list(s.get("tags") or [])


def _rescore_cache_entry(r: Rally) -> dict:
    return {
        "total": r.scores.total,
        "length": r.scores.length,
        "intensity": r.scores.intensity,
        "technique": r.scores.technique,
        "excitement": r.scores.excitement,
        "production": r.scores.production,
        "tags": list(r.tags),
    }


def _apply_bonus_points(rallies: list[Rally], points: float) -> None:
    """Rewrite the absolute speech bonus from a new per-hit point value.

    The bonus is ``points × number of distinct phrases`` (see ``pipeline._apply_speech_bonus``), and
    the phrases are already stored per rally, so changing the points only needs this rewrite plus a
    rescore — no re-recognition or re-segmentation.
    """
    for r in rallies:
        r.features.speech_bonus = points * len(r.features.speech_phrases)


def _rescore_cross_media(proj: Project, key: str, weights, bonus_points: float | None = None) -> int:
    """Recompute all media's rallies as one batch and write back, returning the number of rallies recomputed.

    Scoring uses within-batch percentiles, so scores computed per media are not comparable across media;
    this puts them into the same distribution. The cache is keyed on each media's ``cross:<weighting>``
    key and does not overwrite the per-media cache.

    ``bonus_points`` is an optional override for the absolute voice-command bonus; when set, the bonus
    is rewritten for every rally and every cached score (which embeds the old bonus) is discarded.
    """
    from .analysis import scoring as SC

    ordered = [(k, res) for k, res in proj.analyses.items() if res.rallies]
    flat = [r for _, res in ordered for r in res.rallies]
    if not flat:
        return 0
    cache_key = f"cross:{key}"

    if bonus_points is not None:
        _apply_bonus_points(flat, bonus_points)
        for _, res in ordered:
            res.stats["score_cache"] = {}

    # A complete cache means this weighting has already been batch-computed: refill directly, no recompute when toggling back and forth.
    cached_all = all(
        isinstance(res.stats.get("score_cache", {}).get(cache_key), list)
        and len(res.stats["score_cache"][cache_key]) == len(res.rallies)
        for _, res in ordered
    )
    if cached_all:
        for _, res in ordered:
            for r, s in zip(res.rallies, res.stats["score_cache"][cache_key]):
                _rescore_assign(r, s)
            res.stats["weights"] = cache_key
        return len(flat)

    # Before overwriting, store the current per-media scores into each cache so they can be restored
    # unchanged when switching back to "current media". If the current weighting is already cross it
    # cannot serve as a per-media baseline, so skip it. When a bonus override is applied the current
    # scores still reflect the old bonus, so this baseline snapshot must be skipped too.
    if bonus_points is None:
        for _, res in ordered:
            cache = res.stats.setdefault("score_cache", {})
            cur = str(res.stats.get("weights") or "")
            if cur and not cur.startswith("cross:") and cur not in cache and res.rallies:
                cache[cur] = [_rescore_cache_entry(r) for r in res.rallies]

    scores = SC.score_rallies(
        [_rescore_features(r) for r in flat],
        weights,
        [_rescore_quality(r) for r in flat],
    )
    i = 0
    for _, res in ordered:
        for r in res.rallies:
            _rescore_assign(r, scores[i])
            i += 1
        res.stats["weights"] = cache_key
        res.stats.setdefault("score_cache", {})[cache_key] = [
            _rescore_cache_entry(r) for r in res.rallies
        ]
    return len(flat)


def _rescore_per_media(proj: Project, key: str, weights, mid: str | None, bonus_points: float | None = None) -> int:
    """Per-media weighting recompute (only touches the given media_id; all if not passed), returning the number of rallies recomputed.

    ``bonus_points`` is an optional override for the absolute voice-command bonus; when set, the bonus
    is rewritten for every rally and every cached score (which embeds the old bonus) is discarded.
    """
    from .analysis import scoring as SC

    n = 0
    for k, res in proj.analyses.items():
        if mid and k != mid:
            continue

        if bonus_points is not None:
            _apply_bonus_points(res.rallies, bonus_points)
            res.stats["score_cache"] = {}

        # Cache one copy of the scores computed under each weighting: switching away and back must return
        # to the original scores. The objective features stored in the analysis result are truncated, so
        # recomputed scores are not completely identical to those at analysis time; without caching,
        # "switch back and forth once and the original ranking is lost forever".
        cache = res.stats.setdefault("score_cache", {})
        # When a bonus override is applied the current scores still reflect the old bonus, so skip the
        # baseline snapshot (it would otherwise cache the stale score under the current weighting).
        if bonus_points is None:
            cur = str(res.stats.get("weights") or "")
            # cross weighting scores cannot serve as a per-media baseline; only the current non-cross score is worth caching.
            if cur and not cur.startswith("cross:") and cur not in cache and res.rallies:
                cache[cur] = [_rescore_cache_entry(r) for r in res.rallies]

        cached = cache.get(key)
        if isinstance(cached, list) and len(cached) == len(res.rallies):
            for r, s in zip(res.rallies, cached):
                _rescore_assign(r, s)
                n += 1
            res.stats["weights"] = key
            continue

        scores = SC.score_rallies(
            [_rescore_features(r) for r in res.rallies],
            weights,
            [_rescore_quality(r) for r in res.rallies],
        )
        for r, s in zip(res.rallies, scores):
            _rescore_assign(r, s)
            n += 1
        res.stats["weights"] = key
        cache[key] = [_rescore_cache_entry(r) for r in res.rallies]
    return n


@app.post("/api/projects/{pid}/rallies/rescore")
def rescore(pid: str, payload: dict = Body(default={})) -> dict:
    """Re-score already-analyzed rallies with a different scoring weight set (without re-running the AI analysis).

    With ``cross_media=true``, rallies from all media are pooled and recomputed together so scores
    are comparable across media (scoring uses within-batch percentiles, and scores computed
    separately are not comparable).

    ``speech_bonus_points`` optionally overrides the absolute voice-command bonus (clamped to 0~30);
    the per-rally bonus is rewritten from the stored phrase hits and the scores are recomputed, again
    without re-running the AI or re-segmenting.
    """
    from .analysis import scoring as SC

    _check_id(pid, tr("api.project_id"))
    key = _validated_weights(payload.get("weights"))
    if key not in SC.PRESETS:  # defensive: _validated_weights already guarantees this
        raise HTTPException(400, tr("analysis.unknown_weights", value=key))
    w = SC.PRESETS[key]
    mid = payload.get("media_id")
    cross = bool(payload.get("cross_media"))

    bonus_points: float | None = None
    if payload.get("speech_bonus_points") is not None:
        try:
            bonus_points = float(payload.get("speech_bonus_points"))
        except (TypeError, ValueError):
            raise _bad_request(ValueError("speech_bonus_points")) from None
        bonus_points = min(30.0, max(0.0, bonus_points))

    result: dict[str, int] = {}

    def apply(proj: Project) -> None:
        if cross:
            result["n"] = _rescore_cross_media(proj, key, w, bonus_points)
        else:
            result["n"] = _rescore_per_media(proj, key, w, mid, bonus_points)

    if ST.update_project(pid, apply, write_analyses=True) is None:
        raise HTTPException(404, tr("api.project_not_found"))
    return {"rescored": result.get("n", 0), "weights": key, "cross_media": cross}


# ------------------------------------------------------------------ Timeline


@app.post("/api/projects/{pid}/timeline/auto-cut")
def auto_cut(pid: str, payload: dict = Body(...)) -> dict:
    """Assemble rallies into the timeline by filter criteria."""
    proj = _must_project(pid)
    mid = payload.get("media_id")
    flt = payload.get("filter") or {}
    if not isinstance(flt, dict):
        raise _bad_request("filter")
    mode = str(payload.get("mode") or "replace")     # replace | append
    try:
        gap = float(payload.get("gap") or 0.0)
        pre = float(payload["pre"]) if payload.get("pre") is not None else None
        post = float(payload["post"]) if payload.get("post") is not None else None
    except (TypeError, ValueError) as e:
        raise _bad_request(e) from e
    rally_ids = set(payload.get("rally_ids") or [])

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
            # pre/post are **padding** relative to the rally's clip range, not absolute times.
            # Treating them as absolute values would make every clip become the same segment [pre, post].
            a = r.clip_start - pre if pre is not None else r.clip_start
            b = r.clip_end + post if post is not None else r.clip_end
            a = max(0.0, min(a, r.end))
            b = max(a + 0.2, b)
            picked.append((r, key, a, b))

    # With a single media, sort by source time; across media, group first in the project's media order,
    # otherwise source times from different media would interleave and the output would jump back and
    # forth between them.
    media_rank = {m.id: i for i, m in enumerate(proj.media)}
    picked.sort(key=lambda c: (media_rank.get(c[1], 0), c[2]))
    timeline = proj.timeline if mode == "append" else Timeline(tracks=[Track(name=tr("timeline.track_default", n=1), kind="video")])
    if not timeline.tracks:
        timeline.tracks = [Track(name=tr("timeline.track_default", n=1), kind="video")]
    track = timeline.tracks[0]
    duration = 0.0
    if mode != "append":
        track.clips = []
    elif track.clips:
        duration = max(c.tl_start + c.duration for c in track.clips)

    # When appending, skip rallies already in the output: a rally enters the output only once
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
            label=tr("timeline.clip_label", index=r.index, score=r.scores.total),
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
    # Do not accept an empty body: `payload.get("timeline") or payload` yields {} for {}, which
    # model_validate would fill in as an empty timeline and silently clear all clips.
    if isinstance(payload.get("timeline"), dict):
        raw = payload["timeline"]
    elif "tracks" in payload or "fps" in payload:
        raw = payload
    else:
        raise HTTPException(400, tr("api.missing_timeline"))
    try:
        tl = Timeline.model_validate(raw)
    except ValidationError as e:
        raise _bad_request(e) from e
    total = 0.0
    for t in tl.tracks:
        for c in t.clips:
            total = max(total, c.tl_start + c.duration)
    tl.duration = total
    proj.timeline = tl
    ST.save_project(proj, write_analyses=False)
    return tl.model_dump(mode="json")


# ------------------------------------------------------------------ Export


@app.get("/api/export/presets")
def export_presets() -> list[dict]:
    return [p.model_dump(mode="json") for p in _presets()]


def _presets() -> list[ExportPreset]:
    return [
        ExportPreset(id="yt1080p", name=tr("export.preset.yt1080p.name"), width=1920, height=1080, video_bitrate="16M"),
        ExportPreset(id="yt4k", name=tr("export.preset.yt4k.name"), width=3840, height=2160, video_bitrate="45M"),
        ExportPreset(id="yt720p", name=tr("export.preset.yt720p.name"), width=1280, height=720, video_bitrate="6M"),
        ExportPreset(id="vertical", name=tr("export.preset.vertical.name"), width=1080, height=1920,
                     video_bitrate="12M", auto_reframe=True),
        ExportPreset(id="square", name=tr("export.preset.square.name"), width=1080, height=1080, video_bitrate="10M"),
        ExportPreset(id="hevc4k", name=tr("export.preset.hevc4k.name"), width=3840, height=2160, vcodec="hevc",
                     video_bitrate="25M"),
        ExportPreset(id="draft", name=tr("export.preset.draft.name"), width=1280, height=720, video_bitrate="4M", crf=26),
    ]


@app.post("/api/projects/{pid}/export")
def export(pid: str, payload: dict = Body(...)) -> dict:
    proj = _must_project(pid)
    raw_preset = payload.get("preset") or {}
    if not isinstance(raw_preset, dict):
        raise _bad_request("preset")
    raw = dict(raw_preset)
    # A preset must be selectable even when only the id is given: directly doing
    # model_validate({"id": "draft"}) fills the remaining fields with defaults (1080p / 12M), so script
    # callers silently get output with the wrong specs.
    known = {p.id: p for p in _presets()}
    base = known.get(str(raw.get("id") or ""))
    try:
        if base is not None:
            merged = base.model_dump(mode="json")
            for k, v in raw.items():
                if v is not None:
                    merged[k] = v
            preset = ExportPreset.model_validate(merged)
        else:
            preset = ExportPreset.model_validate(raw)
    except ValidationError as e:
        raise _bad_request(e) from e
    mode = str(payload.get("mode") or "merge")
    if mode not in ("merge", "separate"):
        raise HTTPException(400, tr("export.unknown_mode", mode=mode))
    separate = mode == "separate"
    name = str(payload.get("name") or f"{proj.name}_{int(time.time())}")
    out_dir = _resolve_output_dir(payload.get("output_dir"))
    out = out_dir / f"{_safe_name(name)}.mp4"
    timeline = proj.timeline
    if not any(t.clips for t in timeline.tracks):
        raise HTTPException(400, tr("export.empty_timeline"))

    from .render.exporter import export_timeline

    def work(job):
        result = export_timeline(
            proj, timeline, preset, out,
            on_progress=lambda p, msg: job.progress(p, "export", msg),
            cancel=job.cancelled,
            separate=separate,
        )
        paths = [Path(str(x)) for x in (result.get("paths") or [result.get("path")]) if x]
        result["exports"] = EXPORTS.register(paths, mode, group=_safe_name(name))
        return result

    # Match the exporter's track selection (first video track that actually has clips); ``tracks[0]`` may
    # be an empty/audio track, which made the separate-export clip count wrong.
    track = next((t for t in timeline.tracks if t.kind == "video" and t.clips), None)
    clip_count = len(track.clips) if track else 0
    title = tr("job.title.export_n", count=clip_count) if separate else tr("job.title.export", name=out.name)
    job = JOBS.submit("export", title, work, lang=get_lang())
    # In separate mode no merged file is ever created; point the caller at the output directory instead.
    output = str(out.parent) if separate else str(out)
    return {"job_id": job.id, "output": output, "mode": mode}


def _resolve_output_dir(raw: object) -> Path:
    """Export directory: defaults to data/exports; a custom one must be an absolute path and creatable."""
    if raw is None or str(raw).strip() == "":
        return EXPORT_DIR
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        raise HTTPException(400, tr("export.dir_absolute"))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, tr("export.dir_unusable", error=e)) from e
    return p


def _safe_name(s: str) -> str:
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s.strip()[:120] or "export"


@app.get("/api/exports")
def list_exports() -> list[dict]:
    return EXPORTS.list_all()


@app.get("/api/exports/by-id/{export_id}")
def download_export(export_id: str, request: Request):
    p = EXPORTS.find(export_id)
    if p is None:
        raise HTTPException(404, tr("api.file_not_found"))
    return range_response(request, p, 3600)


@app.post("/api/exports/reveal")
def reveal_export(payload: dict = Body(default={})) -> dict:
    """Locate the exported file in the file explorer."""
    p = EXPORTS.find(str(payload.get("id") or ""))
    if p is None:
        raise HTTPException(404, tr("api.file_not_found"))
    if platform.system() != "Windows":
        raise HTTPException(501, tr("export.reveal_unsupported"))
    import subprocess

    try:
        subprocess.Popen(["explorer", "/select,", str(p)])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(501, tr("export.reveal_failed", error=e)) from e
    return {"ok": True}


# ------------------------------------------------------------------ Jobs


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.model_dump(mode="json") for j in JOBS.list()]


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, tr("job.not_found"))
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

    #: Clearable cache areas. ``all`` is the default; anything else must be one of these.
    targets = {
        "proxies": {"proxies": PROXIES_DIR},
        "thumbs": {"thumbs": THUMBS_DIR},
        "audio": {"audio": CACHE_DIR / "audio"},
        "frames": {"frames": CACHE_DIR / "frames"},
    }
    target = str(payload.get("target") or "all")
    if target == "all":
        dirs = {k: v for group in targets.values() for k, v in group.items()}
    elif target in targets:
        dirs = targets[target]
    else:
        raise HTTPException(400, tr("cache.unknown_target", target=target))
    # Clearing the cache rmtrees proxies/audio being read/written by running jobs, causing mid-job
    # failures. Check *all* jobs (an earlier truncated 200-item list could miss the active one).
    if JOBS.active():
        raise HTTPException(409, tr("cache.busy"))
    freed = 0
    for d in dirs.values():
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            _sh.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
    # Drop the now-dead derived paths from every project. The fields stay set but point at deleted
    # files, and consumers (plus the frontend "already prepared" check) treat a non-empty path as
    # ready, then hand a missing file to cv2/ffmpeg. Clearing them makes the fallback-to-source and
    # regeneration paths kick in.
    field_targets = {k for k in dirs if k in ("proxies", "thumbs", "audio")}
    invalidated = 0
    if field_targets:
        for f in sorted(PROJECTS_DIR.glob("p_*.json")):
            if f.name.count(".") > 1:
                continue  # skip sidecar files
            proj = ST.load_project(f.stem)
            if proj is None:
                continue
            touched = [m for m in proj.media if ST.clear_derived_paths(m, field_targets)]
            if not touched:
                continue
            ST.save_project(proj, write_analyses=False)
            invalidated += len(touched)
            for m in touched:
                HUB.publish({"type": "media", "project_id": proj.id,
                             "media": m.model_dump(mode="json")})
    logger.info("cache cleared: target={} freed={}B invalidated_media={}", target, freed, invalidated)
    return {"freed": freed}


@app.post("/api/samples/seed")
def samples_seed(payload: dict = Body(default={})) -> dict:
    """Generate the guided-tour demo clips under ``<data_dir>/samples`` (blocking,
    synchronous) and return their absolute paths; any failure surfaces as HTTP 500."""
    try:
        ffmpeg = ff.find_ffmpeg()
        files = ensure_sample_files(DATA_DIR, ffmpeg, small=bool(payload.get("small", False)))
    except Exception as e:  # noqa: BLE001
        # Log: an HTTPException(500) would otherwise bypass the unhandled-error handler
        # and leave no durable trace of the ffmpeg/encode failure.
        logger.opt(exception=e).error("sample seeding failed")
        raise HTTPException(500, str(e))
    return {"files": [str(f) for f in files]}


# ------------------------------------------------------------------ Static frontend


def _mount_frontend() -> None:
    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        # index.html must never be cached: after a frontend rebuild its asset hashes change, and a
        # stale cached document references a bundle that no longer exists. The hashed files under
        # /assets are content-addressed, so they may be cached normally. Without this, the pywebview
        # WebView2 cache keeps serving the previous UI on the next launch.
        no_store = {"Cache-Control": "no-store, must-revalidate"}

        @app.get("/")
        def _index():
            return FileResponse(FRONTEND_DIST / "index.html", headers=no_store)

        @app.get("/{full_path:path}")
        def _spa(full_path: str):
            if full_path.startswith(("api/", "ws")):
                raise HTTPException(404, tr("api.not_found"))
            base = FRONTEND_DIST.resolve()
            # With a direct base / full_path, on Windows an absolute path (e.g. C:/Windows/x) or a path
            # containing .. escapes dist, effectively exposing arbitrary local files.
            cand = (base / full_path.lstrip("/\\")).resolve()
            if cand.is_file() and cand.is_relative_to(base):
                return FileResponse(cand)
            return FileResponse(base / "index.html", headers=no_store)
    else:
        @app.get("/")
        def _placeholder():
            return JSONResponse({
                "app": app_name(),
                "note": tr("frontend.not_built"),
                "api_docs": "/api/docs",
            })


_mount_frontend()
