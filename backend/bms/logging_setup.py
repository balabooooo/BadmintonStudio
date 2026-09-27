"""Centralized logging for the backend, built on loguru.

Analysis failures are otherwise swallowed: ``run_analysis`` stores them in ``AnalysisResult.error``
and the job in ``Job.error``, so after the fact there is nothing durable to inspect. Every backend
entry point calls :func:`setup_logging` once, which installs:

* a colorized stderr sink, for interactive / development runs;
* a rotating file sink under ``data/logs/``, for post-mortem debugging;
* a stdlib ``logging`` intercept, so uvicorn / ultralytics / faster-whisper / PIL records land in the
  same sinks instead of a second, unsynchronized logging pipeline.

The function is idempotent: repeated calls (uvicorn import + desktop shell) only reconfigure when
``force=True``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from loguru import logger

from .config import LOGS_DIR

_CONFIGURED = False
_LOG_DIR: Path | None = None

#: Third-party loggers that are extremely chatty at INFO; keep them at WARNING so the analysis log
#: stays readable (ultralytics otherwise prints one line per tracked frame).
_NOISY_LOGGERS = ("ultralytics", "PIL", "matplotlib", "asyncio", "urllib3", "faster_whisper")

_STDERR_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}:{line}</cyan> - <level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)


class _InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records into loguru (standard loguru recipe)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk up out of the logging internals so the reported source is the real caller.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str | None = None, *, log_dir: Path | None = None,
                  force: bool = False) -> Path:
    """Install the loguru sinks and intercept stdlib logging; return the active log directory.

    ``level`` falls back to ``BMS_LOG_LEVEL`` (default ``INFO``). ``log_dir`` is exposed mainly so
    tests can point the file sink at a temporary directory without touching global config.
    """
    global _CONFIGURED, _LOG_DIR
    if _CONFIGURED and not force and _LOG_DIR is not None:
        return _LOG_DIR

    level_name = (level or os.environ.get("BMS_LOG_LEVEL") or "INFO").upper()
    directory = Path(log_dir) if log_dir else LOGS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr, level=level_name, format=_STDERR_FORMAT, colorize=True,
        backtrace=True, diagnose=False, enqueue=False,
    )
    logger.add(
        str(directory / "bms_{time:YYYY-MM-DD}.log"), level=level_name, format=_FILE_FORMAT,
        encoding="utf-8", rotation="10 MB", retention="14 days", enqueue=True,
        backtrace=True, diagnose=False, catch=True,
    )

    # Bridge stdlib logging into loguru and replace any pre-existing handlers so records are not
    # emitted twice.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    _LOG_DIR = directory
    logger.debug("logging configured: level={} dir={}", level_name, directory)
    return directory


__all__ = ["setup_logging"]
