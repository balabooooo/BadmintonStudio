"""Which inference backend / execution provider may serve which task.

The priority lists mirror the Windows table in ``docs/mobile-and-device-ai-plan.md``
(§4.2): TensorRT is only useful on Windows-NVIDIA and its engines are not
portable, DirectML is the only practical GPU path for AMD/Intel on Windows, and
CPU is always last.

Nothing here loads a model. Ranking is a pure function of "what this machine
reports", which keeps it testable and makes the fallback order explicit instead
of being scattered across ``try/except`` blocks.

``BMS_INFER_BACKEND`` overrides the *runtime* choice (``torch`` / ``onnx`` /
``auto``). The execution-provider ranking only applies once the ONNX path is
selected; the Torch path still owns its own device choice in
:mod:`bms.inference.scheduler`.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from loguru import logger

from .capabilities import probe

BACKEND_ENV = "BMS_INFER_BACKEND"

TASK_PLAYERS = "players"
TASK_POSE = "pose"
TASK_SPEECH = "speech"

_CPU = "CPUExecutionProvider"
#: Public alias: the ladder's floor, always tried last.
CPU_PROVIDER = _CPU

#: Windows: TensorRT engines are built per GPU architecture, so they are first only
#: because a working engine is the fastest path; CUDA is the portable NVIDIA path.
_WINDOWS: dict[str, tuple[str, ...]] = {
    TASK_PLAYERS: ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", _CPU),
    TASK_POSE: ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", _CPU),
    TASK_SPEECH: ("CUDAExecutionProvider", _CPU),
}

_PROVIDER_LOCK = threading.Lock()
_PROVIDER_CACHE: list[str] | None = None


def priority(task: str) -> tuple[str, ...]:
    """Ordered provider names Windows prefers for ``task``."""
    return _WINDOWS.get(task, _WINDOWS[TASK_PLAYERS])


def rank_providers(task: str, available: list[str] | tuple[str, ...]) -> list[str]:
    """Intersect the task's priority list with what the runtime actually exposes.

    ``CPUExecutionProvider`` is always appended when available: ORT needs it as
    the fallback partition for any node the accelerator cannot take.
    """
    have = set(available)
    ranked = [p for p in priority(task) if p in have]
    if _CPU in have and _CPU not in ranked:
        ranked.append(_CPU)
    return ranked


def available_providers(refresh: bool = False) -> list[str]:
    """Execution providers reported by the installed ONNX Runtime (empty if absent)."""
    global _PROVIDER_CACHE
    if _PROVIDER_CACHE is not None and not refresh:
        return list(_PROVIDER_CACHE)
    with _PROVIDER_LOCK:
        if _PROVIDER_CACHE is not None and not refresh:
            return list(_PROVIDER_CACHE)
        try:
            import onnxruntime as ort

            _PROVIDER_CACHE = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001 - onnxruntime is optional
            _PROVIDER_CACHE = []
        return list(_PROVIDER_CACHE)


def candidate_providers(task: str) -> list[str]:
    """Ranked, available providers for ``task``; ``[]`` when ORT is not installed."""
    return rank_providers(task, available_providers())


def resolve_detect_backend(requested: str | None = None) -> str:
    """Return ``torch`` or ``onnx`` for the detection / pose path.

    Default is ``torch``: the ONNX path is opt-in until the golden-sample parity
    check passes. Asking for ``onnx`` on a machine without onnxruntime logs and
    falls back to ``torch`` rather than failing the analysis.
    """
    want = str(requested if requested is not None else os.environ.get(BACKEND_ENV, "torch"))
    want = want.strip().lower()
    if want in ("", "torch", "auto"):
        if want == "auto" and available_providers():
            return "onnx"
        return "torch"
    if want == "onnx":
        if available_providers():
            return "onnx"
        logger.warning("BMS_INFER_BACKEND=onnx but onnxruntime is unavailable; using torch")
        return "torch"
    logger.warning("unknown inference backend {!r}; using torch", want)
    return "torch"


def snapshot() -> dict[str, Any]:
    """Structured capability block for ``/api/env``."""
    rep = probe()
    out: dict[str, Any] = {
        "capabilities": rep.as_dict(),
        "ort_providers": available_providers(),
        "detect_backend": resolve_detect_backend(),
        "candidates": {
            TASK_PLAYERS: candidate_providers(TASK_PLAYERS),
            TASK_POSE: candidate_providers(TASK_POSE),
        },
    }
    try:  # imported lazily: benchmark imports this module, so a module-level import would cycle
        from .benchmark import summary as bench_summary

        out["bench"] = bench_summary()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        out["bench"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
