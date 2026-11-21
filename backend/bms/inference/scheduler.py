"""Minimal backend selection built on :mod:`bms.inference.capabilities`.

Only the decisions that already exist in the analysis modules are implemented
here. Keeping them in one place makes the policy explicit and testable; future
runtimes (ONNX Runtime execution providers) extend this module instead of adding
new ``torch.cuda`` checks to analysis code.
"""

from __future__ import annotations

import os

from .capabilities import probe

SHUTTLE_BACKEND_ENV = "BMS_SHUTTLE_BACKEND"


def resolve_torch_device(requested: str | None = "auto") -> str:
    """Resolve a Torch device string, falling back to CPU when CUDA is unavailable.

    Any value starting with ``cuda`` is passed through unchanged when CUDA is
    usable, otherwise it becomes ``cpu``. Non-CUDA values are returned as-is.
    """
    want = str(requested or "cpu").strip()
    if want.lower().startswith("cuda") and not probe().cuda_available:
        return "cpu"
    return want


def resolve_shuttle_backend(force: str | None = None, env_value: str | None = None) -> str:
    """Resolve the shuttle candidate-scan backend (``cpu`` or ``gpu``).

    ``auto`` picks GPU when CUDA is usable. An environment-level ``gpu`` request
    on a CUDA-less machine silently falls back to CPU; an explicit
    ``force="gpu"`` raises, so callers opting into "GPU, no fallback" get a hard
    failure instead of silent CPU work.
    """
    fallback = os.environ.get(SHUTTLE_BACKEND_ENV, "auto") if env_value is None else env_value
    want = str(force if force else fallback).strip().lower()
    if want == "cpu":
        return "cpu"
    if want == "gpu":
        if probe().cuda_available:
            return "gpu"
        if force:
            raise RuntimeError("shuttle backend 'gpu' requested but torch.cuda is unavailable")
        return "cpu"
    return "gpu" if probe().cuda_available else "cpu"


def resolve_speech_runtime() -> tuple[str, str]:
    """Return ``(device, compute_type)`` for faster-whisper / CTranslate2."""
    if probe().ctranslate2_cuda_devices > 0:
        return "cuda", "float16"
    return "cpu", "int8"
