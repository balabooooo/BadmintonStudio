"""Export file registry.

Finished videos can be exported to any directory (no longer fixed to data/exports), and export
records must be listable, playable, and downloadable, so ``data/exports/index.json`` is maintained
here to register every output file. Externally only real paths are provided by id, to avoid
reading arbitrary files through the API.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from ..config import EXPORT_DIR

_INDEX = EXPORT_DIR / "index.json"
_lock = threading.RLock()


def _read() -> list[dict]:
    if not _INDEX.is_file():
        return []
    try:
        data = json.loads(_INDEX.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write(entries: list[dict]) -> None:
    _INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_INDEX)


def _entry(path: Path, mode: str, group: str) -> dict:
    st = path.stat()
    return {
        "id": f"e_{uuid.uuid4().hex[:12]}",
        "name": path.name,
        "path": str(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime * 1000),
        "group": group,
        "mode": mode,
    }


def register(paths: list[Path], mode: str, group: str) -> list[dict]:
    """Register freshly exported files; re-exporting to the same path overwrites the old entry."""
    fresh = [p for p in paths if p and p.is_file()]
    if not fresh:
        return []
    with _lock:
        entries = _read()
        by_path = {str(Path(e.get("path", "")).resolve()): e for e in entries if e.get("path")}
        out: list[dict] = []
        for p in fresh:
            key = str(p.resolve())
            old = by_path.get(key)
            e = _entry(p, mode, group)
            if old:
                e["id"] = old.get("id") or e["id"]
                entries = [x for x in entries if x is not old]
            entries.append(e)
            out.append(e)
        _write(entries)
        return out


def list_all() -> list[dict]:
    """List files from the registry that still exist (size/time taken from disk), newest first."""
    with _lock:
        entries = _read()
    out: list[dict] = []
    for e in entries:
        raw = e.get("path")
        if not raw:
            continue
        p = Path(raw)
        if not p.is_file():
            continue
        st = p.stat()
        out.append({
            "id": str(e.get("id") or ""),
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "mtime": int(st.st_mtime * 1000),
            "group": e.get("group") or "",
            "mode": e.get("mode") or "merge",
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def find(export_id: str) -> Path | None:
    with _lock:
        entries = _read()
    for e in entries:
        if str(e.get("id")) == export_id:
            p = Path(str(e.get("path") or ""))
            return p if p.is_file() else None
    return None
