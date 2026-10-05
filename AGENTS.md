# AGENTS.md

Windows desktop AI video editor for badminton: FastAPI backend + React/Vite frontend + pywebview shell.
Language: **the UI is bilingual (Chinese default + English mode)**; **CLI output, log/error messages, code
comments and docstrings are English** — and **all git commit messages (subject and body) are written in
English** with a Conventional Commits prefix (same for `README.md`). Preserve these conventions.

`README.md` is the authoritative deep doc (~1000 lines) explaining the design rationale of every analysis
signal and most gotchas; read the relevant section before changing `analysis/`.

## Layout
- `backend/bms/` — Python package. `main.py` is one large FastAPI app holding all REST routes + WebSocket; `analysis/` is the AI pipeline; `api/` has the annotation + preset routers; `render/exporter.py` is the FFmpeg export path.
- `frontend/` — React 19 + TS + Vite + Tailwind v4 + Zustand. Animation lib is `motion` (not `framer-motion`).
- `desktop/app.py` — uvicorn thread + pywebview window (browser fallback).
- `scripts/` — CLI diagnostics/analysis (`run_analysis.py`, `e2e_test.py`, `eval_segmentation.py`, `probe_video.py`, `fetch_speech_model.py`, `tune_clip25.py`, `eval_scoring_ab.py`, `transcribe_speech.py`, ...).
- `data/` — gitignored runtime state (project JSON, caches, exports, annotations, scene presets). `models/*.pt`, `models/*.onnx`, `models/faster-whisper-*/` and `tools/` are also gitignored; **no binaries are committed** (YOLO auto-downloads or is placed in `models/`, the whisper model is fetched by script or auto-downloaded, a bundled ffmpeg lives at `tools/ffmpeg/bin`).
- `tests/test_core.py` — backend test suite (Python, plain functions, pytest-compatible). Scope is interface/silent-failure regressions, not algorithm accuracy.
- `frontend/src/**/*.test.ts(x)` — frontend tests (vitest + @testing-library/react, jsdom). Setup at `frontend/src/test/setup.ts`. Run with `npm run test` / `npm run test:watch` inside `frontend/`. No CI, no pre-commit, no Python linter/formatter/typecheck config.

## Environment
- Always use `.venv\Scripts\python.exe`. The venv is created with `--system-site-packages`; `torch`/`torchvision` are **not** in `requirements.txt` and must come from the host env.
- `bms` is not installed as a package. Set `PYTHONPATH=backend` (scripts and `desktop/app.py` patch `sys.path` themselves).
  Example: `$env:PYTHONPATH="D:\Projects\BadmintonStudio\backend"; .\.venv\Scripts\python.exe -m uvicorn bms.main:app --port 8000`
- Launchers set `PYTHONIOENCODING=utf-8`; keep it when running scripts manually (Chinese output).
- Env overrides: `BMS_DATA_DIR`, `BMS_MODELS_DIR`, `BMS_TOOLS_DIR`, `BMS_FRONTEND_DIST`, `BMS_FFMPEG`/`BMS_FFPROBE`, `BMS_SPEECH_MODEL` (faster-whisper model dir / size / HF repo id); Vite API target is `BMS_API`.

## Commands
- Tests: `.\.venv\Scripts\python.exe tests\test_core.py` (plain pytest also works). Scope is interface/silent-failure regressions, not algorithm accuracy.
- Single test: `.\.venv\Scripts\python.exe -m pytest tests\test_core.py -k <name>`. Tests are plain functions, run in order by the `main()` at the bottom of the file.
- Offline full analysis: `.\.venv\Scripts\python.exe scripts\run_analysis.py "<video>"` (writes `data/cache/last_analysis.json`).
- End-to-end API smoke (requires a running server): `.\.venv\Scripts\python.exe scripts\e2e_test.py "<video>"`.
- Speech model (optional, voice-command scoring): `.\.venv\Scripts\python.exe scripts\fetch_speech_model.py` (pre-fetches `models/faster-whisper-medium/`; otherwise auto-downloads on first use).
- Frontend, inside `frontend/`: `npm run build` (`tsc -b && vite build` → `frontend/dist`), `npm run lint` (oxlint), `npm run dev` on port 5273, `npm run test` (vitest, single run) / `npm run test:watch` (vitest watch).
- Frontend tests use jsdom; `setup.ts` stubs the APIs jsdom lacks (`IntersectionObserver`, `matchMedia`, `scrollIntoView`, `CSS.escape`). Mock the `api` module (`vi.mock('../lib/api', ...)`) to avoid network calls; drive components by setting Zustand store state directly (`useStore.setState(...)`).
- The backend serves `frontend/dist`; frontend changes need `npm run build` before `start.cmd`/desktop picks them up (index.html is served `no-store` and the desktop URL carries a dist-mtime cache-bust, so a restart reloads the new bundle). Use `pwsh -File scripts/dev.ps1` for HMR (frontend 5273, backend 8000).

## Gotchas
- AI modules **degrade silently**: `pipeline.py` wraps player/shuttle/audio work in `try/except` and records failures in `stats.*_trace`, so a broken call looks like a successful analysis with bad output. When changing a public analysis signature (`analyze_players`, `_select_active_players`, `probe_boxes`, `run_analysis`), update every call site and `tests/test_core.py`.
- Re-segmentation (`_segment_rallies`) must reuse cached full-frame signals and match a full run's output; do not add AI reruns there. Several params only apply to specific segment modes.
- Analysis is GPU-heavy (CUDA/NVENC when available, CPU fallback); long videos generate a proxy under `data/cache/` first.
- Commit messages use Conventional Commits with an **English** subject and body (`fix: …`, `feat: …`, `chore: …`, `docs: …`); never write Chinese in commit messages.

## Bilingual UI (i18n)
- **Frontend** (`frontend/src/i18n/`): `useT()` returns `tr(key, params?)` for components; `t()` from `i18n/index.ts` is the non-reactive global (store actions, non-React helpers). Language is a Zustand `lang` field persisted to `localStorage['bms.lang']`, toggled in `SettingsPage`; default `zh`.
  - Catalog: `i18n/catalog/zh.ts` + `en.ts` hold the base keys; large component namespaces live in `i18n/catalog/fragments/*.ts` as `[key, 中文, English]` tuples merged by `fragments/index.ts`. **Add new strings to the fragments (or base catalogs) with BOTH languages.**
  - Domain helpers `i18n/domain.ts`: `jobStageLabel` / `jobKindLabel` / `tagLabel` / `viewpointLabel` translate stable backend codes.
- **Backend** (`backend/bms/i18n.py`): user-facing strings use message keys + `tr(key, **params)`. Catalogs: `locales/zh.py` + `en.py` (base) merged with `locales/fragments/*.py` (`(key, 中文, English)` tuples). Language comes from `X-BMS-Lang` (then `Accept-Language`) for HTTP, `?lang=` for `/ws`; jobs capture the submitting request's language in `JobManager.submit` and install it in the worker thread so `tr()` works deep inside `analysis/`.
- **Never hardcode user-facing Chinese/English**: always go through `tr()` / `tr(...)`. Comments and docstrings stay English.
- **Stable codes, translated at display**: rally tags are canonical ASCII codes (`scoring.TAG_*`, e.g. `many_shots`), not localized strings. `scoring.migrate_tags()` maps legacy Chinese tags on project load. Export preset `id`s, job `stage`/`status`/`kind`, and viewpoint codes are likewise stable; only their display names are translated. `speech_phrases` tags are user data and pass through untranslated.
- `tests/test_core.py::test_i18n_catalogs` asserts backend zh/en key-set parity, interpolation/fallback, and the tag-code behavior. Keep catalog key sets equal. The frontend equivalent is `frontend/src/i18n/catalog.test.ts`, which asserts fragment tuple structure, non-empty translations, and zh/en key-set parity across all `fragments/*.ts`.
