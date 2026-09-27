# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Badminton Studio Windows desktop app.

Build with the isolated CPU environment (see packaging/build_release.ps1):

    packaging\\.venv-cpu\\Scripts\\python.exe -m PyInstaller packaging\\BadmintonStudio.spec

Produces a one-folder bundle at ``dist\\BadmintonStudio`` that is then zipped for
distribution. The app bundles a CPU-only torch so it runs on any Windows machine
(GPU users should use the source install documented in README.md).
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Repo root (the spec lives in <root>/packaging).
PROJ = os.path.abspath(os.path.join(SPECPATH, os.pardir))
ENTRY = os.path.join(PROJ, "desktop", "app.py")
FRONTEND_DIST = os.path.join(PROJ, "frontend", "dist")
MODELS_DIR = os.path.join(PROJ, "models")

datas = []
binaries = []
hiddenimports = []

# Bundled read-only resources. config._resource_root() resolves these under
# sys._MEIPASS when frozen, so the destination names must match config defaults.
datas.append((FRONTEND_DIST, "frontend/dist"))
for weights in ("yolo11n.pt", "yolo11n-pose.pt"):
    p = os.path.join(MODELS_DIR, weights)
    if os.path.isfile(p):
        datas.append((p, "models"))

# Package everything under the bms package (analysis/api/core/render/locales).
hiddenimports += collect_submodules("bms")
# uvicorn imports its loop/protocol/lifespan implementations dynamically.
hiddenimports += collect_submodules("uvicorn")

# Packages with dynamic imports or native/data files that need explicit collection.
for pkg in (
    "webview",            # platform backends + bundled js/css
    "ctranslate2",        # faster-whisper native libs
    "pythonnet",          # pywebview's .NET/WebView2 runtime
    "clr_loader",         # pythonnet runtime loader
    "lap",                # ByteTrack linear assignment solver
    "pypinyin",           # pinyin phrase dictionary data
    "polars",             # ultralytics dependency
    "polars_runtime_32",  # polars native runtime
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:  # pragma: no cover - degrade gracefully if a dep is absent
        pass

a = Analysis(
    [ENTRY],
    pathex=[os.path.join(PROJ, "backend"), os.path.join(PROJ, "desktop")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BadmintonStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BadmintonStudio",
)
