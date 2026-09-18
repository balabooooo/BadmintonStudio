# AGENTS.md

Windows desktop AI video editor for badminton: FastAPI backend + React/Vite frontend + pywebview shell.
Code, comments, UI strings, and commit messages are **Chinese**; preserve that convention.

## Layout
- `backend/bms/` — Python package. `main.py` is one large FastAPI app holding all REST routes + WebSocket; `analysis/` is the AI pipeline; `render/exporter.py` is the FFmpeg export path.
- `frontend/` — React 19 + TS + Vite + Tailwind v4 + Zustand. Animation lib is `motion` (not `framer-motion`).
- `desktop/app.py` — uvicorn thread + pywebview window (browser fallback).
- `scripts/` — CLI diagnostics/analysis (`run_analysis.py`, `e2e_test.py`, `eval_segmentation.py`, `probe_video.py`, ...).
- `data/` — gitignored runtime state (project JSON, caches, exports, annotations). `models/*.pt` also gitignored; YOLO weights live there.
- `tests/test_core.py` — the only test suite. No CI, no pre-commit, no Python linter/formatter/typecheck config.

## Environment
- Always use `.venv\Scripts\python.exe`. The venv is created with `--system-site-packages`; `torch`/`torchvision` are **not** in `requirements.txt` and must come from the host env.
- `bms` is not installed as a package. Set `PYTHONPATH=backend` (scripts and `desktop/app.py` patch `sys.path` themselves).
  Example: `$env:PYTHONPATH="D:\Projects\BadmintonStudio\backend"; .\.venv\Scripts\python.exe -m uvicorn bms.main:app --port 8000`
- Launchers set `PYTHONIOENCODING=utf-8`; keep it when running scripts manually (Chinese output).
- Env overrides: `BMS_DATA_DIR`, `BMS_MODELS_DIR`, `BMS_TOOLS_DIR`, `BMS_FRONTEND_DIST`, `BMS_FFMPEG`/`BMS_FFPROBE`; Vite API target is `BMS_API`.

## Commands
- Tests: `.\.venv\Scripts\python.exe tests\test_core.py` (plain pytest also works). Scope is interface/silent-failure regressions, not algorithm accuracy.
- Offline full analysis: `.\.venv\Scripts\python.exe scripts\run_analysis.py "<video>"` (writes `data/cache/last_analysis.json`).
- End-to-end API smoke (requires a running server): `.\.venv\Scripts\python.exe scripts\e2e_test.py "<video>"`.
- Frontend, inside `frontend/`: `npm run build` (`tsc -b && vite build` → `frontend/dist`), `npm run lint` (oxlint), `npm run dev` on port 5273.
- The backend serves `frontend/dist`; frontend changes need `npm run build` before `启动.cmd`/desktop picks them up. Use `pwsh -File scripts/dev.ps1` for HMR (frontend 5273, backend 8000).

## Gotchas
- AI modules **degrade silently**: `pipeline.py` wraps player/shuttle/audio work in `try/except` and records failures in `stats.*_trace`, so a broken call looks like a successful analysis with bad output. When changing a public analysis signature (`analyze_players`, `_select_active_players`, `probe_boxes`, `run_analysis`), update every call site and `tests/test_core.py`.
- Re-segmentation (`_segment_rallies`) must reuse cached full-frame signals and match a full run's output; do not add AI reruns there. Several params only apply to specific segment modes.
- Analysis is GPU-heavy (CUDA/NVENC when available, CPU fallback); long videos generate a proxy under `data/cache/` first.
