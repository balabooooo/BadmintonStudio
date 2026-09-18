"""全局配置与路径解析。

所有可写状态集中在 ``data/`` 下，可通过环境变量覆盖，便于打包后指向
用户目录（``%LOCALAPPDATA%\\BadmintonStudio``）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "BadmintonStudio"
APP_DISPLAY_NAME = "羽毛球智能剪辑台"
APP_VERSION = "0.1.0"

# 工程根目录：<root>/backend/bms/config.py -> <root>
ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _default_data_dir() -> Path:
    """冻结（PyInstaller）时写入用户目录，开发时写入工程目录。"""
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

MODELS_DIR = _env_path("BMS_MODELS_DIR", ROOT / "models")
TOOLS_DIR = _env_path("BMS_TOOLS_DIR", ROOT / "tools")

FRONTEND_DIST = _env_path("BMS_FRONTEND_DIST", ROOT / "frontend" / "dist")

# ---------------------------------------------------------------- 分析参数默认值

#: 分析用代理视频的最长边（越大越准，越小越快）
PROXY_MAX_EDGE = 960
#: 代理视频目标帧率（逐帧 AI 检测用）
PROXY_FPS = 30.0
#: 音频分析采样率
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
        MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
