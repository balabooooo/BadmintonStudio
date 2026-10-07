"""Project persistence: one JSON file per project, stored under data/projects.

JSON was chosen over a database so project files stay readable, hand-editable, and easy to
back up together with the media.

Analysis results are stored separately in a ``<project>.<media>.analysis.json`` sidecar file:
it holds signal curves with thousands of points and per-shot information for hundreds of
rallies, and packing it into the main file would make every "tweak a scoring range" write tens
of MB. The main file keeps only media, timeline, and UI state.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from pydantic import ValidationError

from ..config import PROJECTS_DIR, ensure_dirs
from ..i18n import tr
from ..analysis.scoring import migrate_tags
from .models import AnalysisResult, MediaInfo, Project, ProjectSummary, Timeline, Track, now_ms

_lock = threading.RLock()

_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_id(value: str) -> str:
    """The id is concatenated into file paths; restrict the character set so reads/writes/deletes
    cannot escape PROJECTS_DIR.

    The routing layer also validates and returns 400; this is the fallback (scripts and other
    callers go through these functions too).
    """
    if not _ID_RE.fullmatch(value or ""):
        raise ValueError(tr("store.invalid_id", value=value))
    return value


def _path(project_id: str) -> Path:
    return PROJECTS_DIR / f"{_safe_id(project_id)}.json"


def _analysis_path(project_id: str, media_id: str) -> Path:
    return PROJECTS_DIR / f"{_safe_id(project_id)}.{_safe_id(media_id)}.analysis.json"


def _write_json(path: Path, data: dict) -> None:
    # Unique temp name + ``os.replace``: atomic, and two concurrent writers of the same path cannot
    # clobber each other's half-written file (a fixed ``.tmp`` name could).
    tmp = path.with_name(f"{path.name}.{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def list_projects() -> list[ProjectSummary]:
    ensure_dirs()
    out: list[ProjectSummary] = []
    for f in PROJECTS_DIR.glob("p_*.json"):
        if f.name.count(".") > 1:
            continue  # skip sidecar files
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            p = Project.model_validate(data)
            total = sum(m.duration for m in p.media)
            poster = next((m.poster for m in p.media if m.poster), None)
            rally_count = 0
            analyzed = False
            for m in p.media:
                af = _analysis_path(p.id, m.id)
                if af.is_file():
                    try:
                        a = json.loads(af.read_text(encoding="utf-8"))
                        rally_count += len(a.get("rallies", []))
                        analyzed = analyzed or a.get("status") == "done"
                        if not poster:
                            poster = m.poster
                    except Exception:
                        pass
            out.append(ProjectSummary(
                id=p.id, name=p.name, created_at=p.created_at, updated_at=p.updated_at,
                media_count=len(p.media), duration=total, poster=poster,
                rally_count=rally_count, analyzed=analyzed,
            ))
        except Exception:
            continue
    out.sort(key=lambda s: s.updated_at, reverse=True)
    return out


def create_project(name: str | None = None) -> Project:
    ensure_dirs()
    p = Project(name=name or tr("project.untitled"))
    p.timeline = Timeline(tracks=[Track(name=tr("timeline.track_default", n=1), kind="video")])
    save_project(p)
    return p


def load_project(project_id: str) -> Project | None:
    f = _path(project_id)
    if not f.is_file():
        return None
    with _lock:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            # A corrupt main file used to bubble up as a 500 on every route that touches the project;
            # treat it as missing (routes already turn None into a 404) and leave a trace in the log.
            logger.warning("project {} is corrupt ({}); treating as missing", project_id, e)
            return None
    try:
        p = Project.model_validate(data)
    except ValidationError as e:
        logger.warning("project {} has an invalid schema ({}); treating as missing", project_id, e)
        return None
    # Load sidecar analysis results
    for m in p.media:
        try:
            af = _analysis_path(project_id, m.id)
            if not af.is_file():
                continue
            p.analyses[m.id] = AnalysisResult.model_validate(
                json.loads(af.read_text(encoding="utf-8"))
            )
        except Exception as e:  # noqa: BLE001
            # Never drop the whole project because one sidecar is damaged; log it so the loss is visible.
            logger.warning("dropping unreadable analysis sidecar for media {}: {}", m.id, e)
            continue
    # Older projects stored Chinese tags; map them to canonical codes on load.
    for a in p.analyses.values():
        for r in a.rallies:
            r.tags = migrate_tags(r.tags)
    logger.debug(
        "project loaded: id={} media={} analyses={} rallies={}",
        p.id, len(p.media), len(p.analyses),
        sum(len(a.rallies) for a in p.analyses.values()),
    )
    return p


def update_project(project_id: str, mutate: Callable[[Project], Any], *,
                   write_analyses: bool = False) -> Project | None:
    """Atomic read-modify-write: load, mutate, and save while holding the store lock.

    Separate ``load_project`` + ``save_project`` calls take the lock individually, so two concurrent
    requests can interleave and the later save silently overwrites the earlier one's changes. This
    keeps the (reentrant) lock across the whole sequence. ``mutate`` must be fast and non-blocking.
    """
    with _lock:
        p = load_project(project_id)
        if p is None:
            return None
        mutate(p)
        save_project(p, write_analyses=write_analyses)
        return p


def save_project(p: Project, write_analyses: bool = True) -> Project:
    ensure_dirs()
    p.updated_at = now_ms()
    with _lock:
        if write_analyses:
            for mid, res in list(p.analyses.items()):
                try:
                    _write_json(_analysis_path(p.id, mid), res.model_dump(mode="json"))
                except Exception as e:  # noqa: BLE001
                    # A failed sidecar write used to vanish silently, so the analysis was lost with no
                    # trace while the caller believed the save succeeded.
                    logger.warning("failed to save analysis sidecar {}.{}: {}", p.id, mid, e)
                    continue
        payload = p.model_dump(mode="json")
        payload["analyses"] = {}          # analysis results go to the sidecar file
        _write_json(_path(p.id), payload)
    logger.debug("project saved: id={} media={} sidecars_written={} write_analyses={}",
                 p.id, len(p.media), len(p.analyses) if write_analyses else 0, write_analyses)
    return p


def save_analysis(project_id: str, media_id: str, res: AnalysisResult) -> None:
    ensure_dirs()
    with _lock:
        _write_json(_analysis_path(project_id, media_id), res.model_dump(mode="json"))
    logger.debug("analysis sidecar saved: project={} media={} status={} rallies={}",
                 project_id, media_id, res.status, len(res.rallies))


def delete_analysis(project_id: str, media_id: str) -> None:
    # Share one lock with write operations: on Windows deleting a file while another thread has
    # it open raises PermissionError, and interleaving deletes with writes can lose sidecar files.
    with _lock:
        _analysis_path(project_id, media_id).unlink(missing_ok=True)


def delete_project(project_id: str) -> bool:
    f = _path(project_id)
    if not f.is_file():
        return False
    for af in PROJECTS_DIR.glob(f"{project_id}.*.analysis.json"):
        af.unlink(missing_ok=True)
    backup = f.with_suffix(f".deleted_{int(time.time())}.json")
    try:
        shutil.move(str(f), str(backup))
    except OSError:
        f.unlink(missing_ok=True)
    return True


def duplicate_project(project_id: str, new_name: str | None = None) -> Project | None:
    src = load_project(project_id)
    if src is None:
        return None
    new = src.model_copy(deep=True)
    new.id = Project().id
    new.name = new_name or tr("project.copy_name", name=src.name)
    new.created_at = now_ms()
    new.updated_at = now_ms()
    save_project(new)
    return new


def touch_media(p: Project, media: MediaInfo) -> Project:
    """Register media into the project (deduplicated by path) while keeping existing derived paths."""
    for i, m in enumerate(p.media):
        if m.path == media.path:
            media.id = m.id
            # Reuse existing proxy/audio/poster to avoid regenerating them
            media.proxy_path = media.proxy_path or m.proxy_path
            media.proxy_fps = media.proxy_fps or m.proxy_fps
            media.proxy_width = media.proxy_width or m.proxy_width
            media.proxy_height = media.proxy_height or m.proxy_height
            media.audio_path = media.audio_path or m.audio_path
            media.poster = media.poster or m.poster
            p.media[i] = media
            return save_project(p)
    p.media.append(media)
    return save_project(p)


def clear_derived_paths(media: MediaInfo, targets: set[str]) -> bool:
    """Drop derived-asset paths that no longer exist after a cache cleanup.

    Projects keep ``proxy_path``/``audio_path``/``poster`` pointing into ``data/cache``. Removing
    those files (cache clear) leaves the fields set but stale, so every consumer that only checks
    "field is set" -- and the frontend ``ensurePrepare`` -- treats the asset as ready and hands a
    missing path to cv2/ffmpeg. Clearing the fields makes regeneration and the fallback-to-source
    paths kick in. Returns whether anything changed.
    """
    changed = False
    if "proxies" in targets:
        for attr in ("proxy_path", "proxy_fps", "proxy_width", "proxy_height"):
            if getattr(media, attr) is not None:
                setattr(media, attr, None)
                changed = True
    if "audio" in targets and media.audio_path is not None:
        media.audio_path = None
        changed = True
    if "thumbs" in targets and media.poster is not None:
        media.poster = None
        changed = True
    return changed

