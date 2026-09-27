"""Scene presets: store the "segmentation params + court calibration + preview frame" obtained
from an annotation / parameter optimization, and reuse them across projects.

Why "manual selection" rather than automatic matching
-----------------------------------------------------
There is no reliable measure of scene similarity, and forcing a match easily applies unsuitable
parameters silently to new media. This only presents the preset together with the frame saved
at the time, leaving it to the user to judge from the screenshot whether it looks similar.

Storage is decoupled from projects: ``data/presets/<id>.json`` + ``<id>.jpg``, visible to all projects.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from ..config import PRESETS_DIR
from ..i18n import tr
from . import media as M
from .models import MediaInfo, _uid

#: Only ids matching this character set are allowed (consistent with the id rule in store, preventing path traversal)
_ID_RE = re.compile(r"^ps_[A-Za-z0-9]+$")

#: Whitelist of parameters allowed into a preset. Keep only the quantities that actually affect
#: segmentation / boundaries / hit attribution, to avoid pulling in the entire AnalysisParams
#: (including use_* switches, etc.) -- those are choices about "how to run this time", not "what
#: parameters this scene should use".
PARAM_FIELDS = (
    "seg_prominence",
    "seg_min_core",
    "seg_min_rest",
    "seg_min_quiet",
    "min_rally_seconds",
    "max_rally_seconds",
    "pre_roll",
    "post_roll",
    "hit_tail_seconds",
    # Hit attribution / cross-court suppression (calibrated from annotations).
    "pose_gate_threshold",
    "pose_gate_window",
    "hit_sensitivity",
)

#: Boolean parameters allowed into a preset (normalized as bool, not float).
BOOL_FIELDS = (
    "pose_gate_one_to_one",
    "pose_gate_force",
)

MIN_POLY_POINTS = 4
MAX_POLY_POINTS = 24


def _path(pid: str) -> Path:
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        raise ValueError(tr("preset.invalid_id", pid=pid))
    return PRESETS_DIR / f"{pid}.json"


def _preview_path(pid: str) -> Path:
    return PRESETS_DIR / f"{pid}.jpg"


def _norm_params(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k in PARAM_FIELDS:
        if k not in raw:
            continue
        try:
            v = float(raw[k])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out[k] = round(v, 4)
    for k in BOOL_FIELDS:
        if k in raw:
            out[k] = bool(raw[k])  # type: ignore[assignment]
    return out


def _norm_poly(raw: Any) -> list[list[float]] | None:
    """Validate and normalize the court polygon; drop it (store None) if invalid, without affecting the params themselves."""
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[list[float]] = []
    for p in raw:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)) or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        out.append([round(x, 5), round(y, 5)])
    if len(out) < MIN_POLY_POINTS or len(out) > MAX_POLY_POINTS:
        return None
    area = 0.0
    n = len(out)
    for i in range(n):
        x1, y1 = out[i]
        x2, y2 = out[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if abs(area) / 2 <= 1e-4:
        return None
    return out


def _write(pid: str, doc: dict[str, Any]) -> None:
    path = _path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_presets() -> list[dict[str, Any]]:
    if not PRESETS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for f in PRESETS_DIR.glob("ps_*.json"):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict) and doc.get("id"):
            out.append(doc)
    out.sort(key=lambda d: d.get("created_at", 0), reverse=True)
    return out


def load_preset(pid: str) -> dict[str, Any] | None:
    try:
        path = _path(pid)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def save_preset(
    *,
    name: str,
    note: str,
    media: MediaInfo,
    frame_time: float,
    params: Any,
    court_poly: Any,
    project_id: str = "",
    project_name: str = "",
    media_name: str = "",
    fit: Any = None,
) -> dict[str, Any]:
    """Extract one frame as a preview + save params/calibration, returning the document written to disk."""
    pid = _uid("ps_")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    frame_time = max(0.0, float(frame_time or 0.0))
    preview = _preview_path(pid)
    got = M.extract_frame(media, frame_time, preview, max_edge=640)
    doc: dict[str, Any] = {
        "id": pid,
        "name": (str(name or "").strip()) or tr("preset.default_name", date=time.strftime("%Y-%m-%d %H:%M")),
        "note": str(note or ""),
        "created_at": int(time.time() * 1000),
        "source": {
            "project_id": project_id,
            "project_name": project_name,
            "media_id": media.id,
            "media_name": media_name or media.name,
            "frame_time": round(frame_time, 3),
        },
        "params": _norm_params(params),
        "court_poly": _norm_poly(court_poly),
        "aspect": round((media.width / media.height) if media.height else 16 / 9, 4),
        "preview": str(preview) if got else "",
    }
    fit_doc = _norm_fit(fit)
    if fit_doc:
        doc["fit"] = fit_doc
    _write(pid, doc)
    return doc


def _norm_fit(raw: Any) -> dict[str, Any]:
    """Keep only the small, displayable calibration provenance (which stage / how good the fit was)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k in ("source", "scope", "f1", "precision", "recall", "hit_f1",
              "hit_precision", "hit_recall", "hit_label_count", "baseline_f1"):
        v = raw.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = round(float(v), 4)
        elif isinstance(v, str) and v:
            out[k] = v[:64]
    return out


def rename_preset(pid: str, name: str | None = None, note: str | None = None) -> dict[str, Any] | None:
    doc = load_preset(pid)
    if doc is None:
        return None
    if name is not None and str(name).strip():
        doc["name"] = str(name).strip()
    if note is not None:
        doc["note"] = str(note)
    _write(pid, doc)
    return doc


def delete_preset(pid: str) -> bool:
    try:
        path = _path(pid)
    except ValueError:
        return False
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    _preview_path(pid).unlink(missing_ok=True)
    return True
