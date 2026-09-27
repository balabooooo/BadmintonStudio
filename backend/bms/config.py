"""Global configuration and path resolution.

All writable state lives under ``data/`` and can be overridden via environment
variables, so that a packaged build can point at the user directory
(``%LOCALAPPDATA%\\BadmintonStudio``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BadmintonStudio"
#: Neutral (language-independent) name used for the FastAPI/OpenAPI title and non-request contexts;
#: user-facing names go through i18n ``app_name()`` instead.
APP_DISPLAY_NAME = "Badminton Studio"
APP_VERSION = "0.1.0"

# Project root: <root>/backend/bms/config.py -> <root>
ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _default_data_dir() -> Path:
    """When frozen (PyInstaller) write to the user directory, otherwise write to the project directory."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / APP_NAME
    return ROOT / "data"


DATA_DIR = _env_path("BMS_DATA_DIR", _default_data_dir())
PROJECTS_DIR = DATA_DIR / "projects"
CACHE_DIR = DATA_DIR / "cache"
PROXIES_DIR = CACHE_DIR / "proxies"
AUDIO_DIR = CACHE_DIR / "audio"
THUMBS_DIR = CACHE_DIR / "thumbs"
FRAMES_DIR = CACHE_DIR / "frames"
EXPORT_DIR = DATA_DIR / "exports"
LOGS_DIR = DATA_DIR / "logs"
#: Manual rally annotations (one <proxy name>.anno.json per media clip), used to auto-optimize segmentation parameters
ANNOTATIONS_DIR = DATA_DIR / "annotations"
#: Scene presets (segmentation params + court calibration + preview frame), reusable across projects
PRESETS_DIR = DATA_DIR / "presets"

MODELS_DIR = _env_path("BMS_MODELS_DIR", ROOT / "models")
TOOLS_DIR = _env_path("BMS_TOOLS_DIR", ROOT / "tools")

FRONTEND_DIST = _env_path("BMS_FRONTEND_DIST", ROOT / "frontend" / "dist")

# ---------------------------------------------------------------- Analysis parameter defaults

#: Longest edge of the proxy video used for analysis (larger = more accurate, smaller = faster)
PROXY_MAX_EDGE = 960
#: Target frame rate of the proxy video (for per-frame AI detection)
PROXY_FPS = 30.0
#: Audio analysis sample rate
AUDIO_SR = 16000


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        PROJECTS_DIR,
        CACHE_DIR,
        PROXIES_DIR,
        AUDIO_DIR,
        THUMBS_DIR,
        FRAMES_DIR,
        EXPORT_DIR,
        LOGS_DIR,
        ANNOTATIONS_DIR,
        PRESETS_DIR,
        MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
