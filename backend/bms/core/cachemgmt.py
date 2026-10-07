"""Selective cache management: target registry, size scans, safe file-level cleanup.

The settings page lets users pick which cache areas to clear. Every area is an explicit
registry entry backed by a file collector, so:

* scans and deletions always agree on what an area contains;
* user data (uploads, annotations, projects, exports, samples) is unreachable from here;
* deletions happen file-by-file (never a blind ``rmtree`` of ``data/``), so locked files
  are isolated, retried and reported instead of aborting the whole run.

Project-level garbage collection (uploads/annotations orphaned by deleting a project) also
lives here, :func:`purge_orphan_assets`, but is driven from the project-delete route, not the
cache dialog.
"""

from __future__ import annotations

import datetime
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from loguru import logger

from ..config import CACHE_DIR, DATA_DIR, MODELS_DIR

#: Soft time budget for a clear run; the route-level requirement is "respond within 30s".
DEFAULT_DEADLINE_S = 30.0
#: Per-file attempts when Windows keeps a transient lock on the file (1 initial + 2 retries).
DELETE_ATTEMPTS = 3
#: Backoff before retry n (seconds).
RETRY_BACKOFF = (0.05, 0.15)

#: WebView2 profile subdirectories that only hold regenerable caches. Cookies, preferences,
#: localStorage and other profile state live in sibling directories and must stay listed out.
_WEBVIEW_CACHE_DIRS = (
    ("Default", "Cache"),
    ("Default", "Code Cache"),
    ("Default", "GPUCache"),
    ("Default", "DawnGraphiteCache"),
    ("Default", "DawnWebGPUCache"),
    ("Crashpad",),
)

#: Loose model weights in MODELS_DIR ...
_MODEL_FILE_GLOBS = ("*.pt", "*.onnx")
#: ... and fetched model trees (faster-whisper / vosk), removed wholesale once their files go.
_MODEL_DIR_PREFIXES = ("faster-whisper-", "vosk-model-")


@dataclass(frozen=True)
class Target:
    id: str
    #: "safe" (auto-rebuilt derivative), "normal" (regenerable work product), "danger" (user-confirmed).
    level: str
    #: Pre-selected when the dialog opens.
    default: bool


#: Clearable areas, in dialog display order. Anything user-created must NOT be added here.
REGISTRY: tuple[Target, ...] = (
    Target("proxies", "safe", True),
    Target("thumbs", "safe", True),
    Target("audio", "safe", True),
    Target("frames", "safe", True),
    Target("ai", "normal", False),
    Target("eval", "normal", False),
    Target("logs", "normal", False),
    Target("webview", "normal", False),
    Target("models", "danger", False),
)
TARGET_IDS: frozenset[str] = frozenset(t.id for t in REGISTRY)
DANGER_IDS: frozenset[str] = frozenset(t.id for t in REGISTRY if t.level == "danger")
DEFAULT_IDS: tuple[str, ...] = tuple(t.id for t in REGISTRY if t.default)


# -------------------------------------------------------------------------- file collection


def _files(root: Path) -> list[Path]:
    """All files under ``root`` (lexicographic order for stable progress), [] if it is missing."""
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower())


def _is_today(path: Path, today: str) -> bool:
    """Whether a log filename belongs to the current day (the sinks may still be writing to it)."""
    return today in path.name


def _collect(target_id: str, data_dir: Path, models_dir: Path, *, include_active: bool = True) -> list[Path]:
    cache = data_dir / "cache"
    if target_id in ("proxies", "thumbs", "audio", "frames"):
        return _files(cache / target_id)
    if target_id == "ai":
        out: list[Path] = []
        for sub in ("boxes", "pose", "calib"):
            out.extend(_files(cache / sub))
        return sorted(out, key=lambda p: str(p).lower())
    if target_id == "eval":
        # Tuning/eval runs land in cache/eval; loose debug/transcript JSONs sit at the cache root.
        out = _files(cache / "eval")
        if cache.is_dir():
            out.extend(sorted((p for p in cache.glob("*.json") if p.is_file()),
                              key=lambda p: str(p).lower()))
        return out
    if target_id == "logs":
        files = _files(data_dir / "logs")
        if include_active:
            # The size scan shows every log file, including the sinks of the current day.
            return files
        # Deletion must never touch today's log files — the running app may still be writing them.
        today = datetime.date.today().isoformat()
        return [p for p in files if not _is_today(p, today)]
    if target_id == "webview":
        root = data_dir / "webview" / "EBWebView"
        out = []
        for parts in _WEBVIEW_CACHE_DIRS:
            out.extend(_files(root.joinpath(*parts)))
        return out
    if target_id == "models":
        out = []
        if models_dir.is_dir():
            for pattern in _MODEL_FILE_GLOBS:
                out.extend(sorted((p for p in models_dir.glob(pattern) if p.is_file()),
                                  key=lambda p: str(p).lower()))
            for d in sorted((p for p in models_dir.iterdir()
                             if p.is_dir() and p.name.startswith(_MODEL_DIR_PREFIXES)),
                            key=lambda p: str(p).lower()):
                out.extend(_files(d))
        return out
    raise ValueError(f"unknown cache target: {target_id}")


def _model_dirs(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        return []
    return [p for p in models_dir.iterdir()
            if p.is_dir() and p.name.startswith(_MODEL_DIR_PREFIXES)]


# -------------------------------------------------------------------------- scan


def scan_targets(data_dir: Path | str = DATA_DIR,
                 models_dir: Path | str = MODELS_DIR) -> list[dict]:
    """Size/file-count of every registry target, in display order."""
    data_dir, models_dir = Path(data_dir), Path(models_dir)
    out: list[dict] = []
    for t in REGISTRY:
        files = _collect(t.id, data_dir, models_dir)
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                # A file deleted between collection and stat simply counts as zero.
                pass
        out.append({"id": t.id, "level": t.level, "default": t.default,
                    "files": len(files), "bytes": total})
    return out


# -------------------------------------------------------------------------- deletion


def _delete_with_retry(path: Path) -> tuple[bool, str]:
    """Delete one file with a short retry budget; returns (ok, error label). Never raises."""
    for attempt in range(DELETE_ATTEMPTS):
        try:
            path.unlink()
            return True, ""
        except FileNotFoundError:
            return True, ""  # already gone: the desired state
        except OSError as e:
            if attempt < DELETE_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue
            return False, f"{type(e).__name__}: {path.name}"
    return False, f"OSError: {path.name}"


ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def clear_targets(
    ids: Sequence[str],
    *,
    confirms: dict | None = None,
    data_dir: Path | str = DATA_DIR,
    models_dir: Path | str = MODELS_DIR,
    on_progress: ProgressCb | None = None,
    is_cancelled: CancelCb | None = None,
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> dict:
    """Delete files of the selected targets.

    Returns ``{"items": [{id, removed, freed, failed: [error]}], "freed": total,
    "timed_out": [target ids not finished]}``.  Failures are per-file and never abort the run;
    missing danger confirmation / unknown ids raise ``ValueError`` before touching disk.
    """
    data_dir, models_dir = Path(data_dir), Path(models_dir)
    ids = list(ids)
    unknown = [i for i in ids if i not in TARGET_IDS]
    if unknown:
        raise ValueError(f"unknown cache target: {unknown[0]}")
    danger = {t for t in ids if t in DANGER_IDS}
    if danger and not bool((confirms or {}).get("models")):
        raise ValueError("danger target 'models' requires confirmation")

    plan: dict[str, list[Path]] = {
        tid: _collect(tid, data_dir, models_dir, include_active=False) for tid in ids
    }
    total = sum(len(v) for v in plan.values())
    deadline = time.monotonic() + max(0.0, deadline_s)
    done = 0
    freed_total = 0
    timed_out: list[str] = []
    items: list[dict] = []
    stopped = False
    for tid in ids:
        item = {"id": tid, "removed": 0, "freed": 0, "failed": []}
        items.append(item)
        if stopped:
            timed_out.append(tid)
            continue
        for f in plan[tid]:
            if (is_cancelled and is_cancelled()) or time.monotonic() >= deadline:
                stopped = True
                break
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            ok, err = _delete_with_retry(f)
            done += 1
            if on_progress:
                try:
                    on_progress(done, total, tid)
                except Exception:  # noqa: BLE001
                    pass
            if ok:
                item["removed"] += 1
                item["freed"] += size
                freed_total += size
            else:
                item["failed"].append(err)
        if stopped:
            timed_out.append(tid)
        # Fetched model trees: files are gone, so remove the (typically empty) directory tree.
        if tid == "models" and "models" not in timed_out:
            for d in _model_dirs(models_dir):
                shutil.rmtree(d, ignore_errors=True)

    logger.info("cache cleared: targets={} freed={}B removed/{} failed={} timed_out={} cancelled={}",
                ids, freed_total,
                sum(i["removed"] for i in items),
                {i["id"]: len(i["failed"]) for i in items if i["failed"]},
                timed_out, bool(is_cancelled and is_cancelled()))
    return {"items": items, "freed": freed_total, "timed_out": timed_out}


# ------------------------------------------------------------------ project orphan cleanup


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def purge_orphan_assets(
    media_items: Iterable,
    other_media_items: Iterable,
    *,
    data_dir: Path | str = DATA_DIR,
    anno_path_fn: Callable[[object], Path] | None = None,
) -> dict:
    """Delete uploads/annotations referenced only by the removed project.

    ``media_items`` belong to the deleted project; ``other_media_items`` to every still-active
    project. A source file is deleted only when it lives inside ``data/uploads`` AND has no
    remaining reference; external files (videos picked from elsewhere on disk) are never
    deleted. Annotation files are reference-counted the same way via ``anno_path_fn``.
    """
    data_dir = Path(data_dir)
    uploads = _resolved(data_dir / "uploads")
    annotations = _resolved(data_dir / "annotations")
    media_items = list(media_items)
    other_media_items = list(other_media_items)

    other_sources = {_resolved(Path(m.path)) for m in other_media_items if getattr(m, "path", None)}
    other_annos: set[Path] = set()
    if anno_path_fn is not None:
        for m in other_media_items:
            try:
                other_annos.add(_resolved(Path(anno_path_fn(m))))
            except Exception:  # noqa: BLE001
                continue

    freed = 0
    failed: list[str] = []
    uploads_removed = 0
    annotations_removed = 0

    for m in media_items:
        raw = getattr(m, "path", None)
        if not raw:
            continue
        src = _resolved(Path(raw))
        if _within(src, uploads) and src not in other_sources and src.is_file():
            try:
                size = src.stat().st_size
            except OSError:
                size = 0
            ok, err = _delete_with_retry(src)
            if ok:
                uploads_removed += 1
                freed += size
            else:
                failed.append(err)

    if anno_path_fn is not None:
        for m in media_items:
            try:
                ap = _resolved(Path(anno_path_fn(m)))
            except Exception:  # noqa: BLE001
                continue
            if _within(ap, annotations) and ap not in other_annos and ap.is_file():
                try:
                    size = ap.stat().st_size
                except OSError:
                    size = 0
                ok, err = _delete_with_retry(ap)
                if ok:
                    annotations_removed += 1
                    freed += size
                else:
                    failed.append(err)

    logger.info("project orphan assets purged: uploads_removed={} annotations_removed={} freed={}B failed={}",
                uploads_removed, annotations_removed, freed, len(failed))
    return {"freed": freed, "uploads_removed": uploads_removed,
            "annotations_removed": annotations_removed, "failed": failed}
