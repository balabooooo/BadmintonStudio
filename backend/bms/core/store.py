"""工程持久化：每个工程一个 JSON 文件，放在 data/projects 下。

选择 JSON 而不是数据库，是为了让工程文件可读、可手工修、可随素材一起备份。

分析结果单独存 ``<project>.<media>.analysis.json`` 边车文件：里面有几千点的
信号曲线和上百个回合的逐拍信息，塞进主文件会让每次「改个评分区间」都写几十 MB。
主文件只保留素材、时间线和界面状态。
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from ..config import PROJECTS_DIR, ensure_dirs
from .models import AnalysisResult, MediaInfo, Project, ProjectSummary, Timeline, Track, now_ms

_lock = threading.RLock()


def _path(project_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}.json"


def _analysis_path(project_id: str, media_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}.{media_id}.analysis.json"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def list_projects() -> list[ProjectSummary]:
    ensure_dirs()
    out: list[ProjectSummary] = []
    for f in PROJECTS_DIR.glob("p_*.json"):
        if f.name.count(".") > 1:
            continue  # 跳过边车文件
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


def create_project(name: str = "未命名工程") -> Project:
    ensure_dirs()
    p = Project(name=name)
    p.timeline = Timeline(tracks=[Track(name="视频轨 1", kind="video")])
    save_project(p)
    return p


def load_project(project_id: str) -> Project | None:
    f = _path(project_id)
    if not f.is_file():
        return None
    with _lock:
        data = json.loads(f.read_text(encoding="utf-8"))
    p = Project.model_validate(data)
    # 载入边车分析结果
    for m in p.media:
        af = _analysis_path(project_id, m.id)
        if af.is_file():
            try:
                p.analyses[m.id] = AnalysisResult.model_validate(
                    json.loads(af.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
    return p


def save_project(p: Project, write_analyses: bool = True) -> Project:
    ensure_dirs()
    p.updated_at = now_ms()
    with _lock:
        if write_analyses:
            for mid, res in list(p.analyses.items()):
                try:
                    _write_json(_analysis_path(p.id, mid), res.model_dump(mode="json"))
                except Exception:
                    continue
        payload = p.model_dump(mode="json")
        payload["analyses"] = {}          # 分析结果走边车文件
        _write_json(_path(p.id), payload)
    return p


def save_analysis(project_id: str, media_id: str, res: AnalysisResult) -> None:
    ensure_dirs()
    with _lock:
        _write_json(_analysis_path(project_id, media_id), res.model_dump(mode="json"))


def delete_analysis(project_id: str, media_id: str) -> None:
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
    new.name = new_name or f"{src.name} 副本"
    new.created_at = now_ms()
    new.updated_at = now_ms()
    save_project(new)
    return new


def touch_media(p: Project, media: MediaInfo) -> Project:
    """把素材登记进工程（按路径去重），同时保留已有的派生路径。"""
    for i, m in enumerate(p.media):
        if m.path == media.path:
            media.id = m.id
            # 已有代理/音轨/封面就沿用，避免重复生成
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

