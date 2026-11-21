"""Device capability probing.

The probe is cached for the process lifetime: CUDA availability and device
count do not change without restarting the app. Tests call :func:`reset` after
monkeypatching the vendor probes.

Imports are lazy on purpose. Importing ``torch`` costs seconds on a cold disk
and ``ctranslate2`` may be absent in a CPU-only install; both must degrade to
"unavailable" instead of failing the analysis.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityReport:
    """What the current machine can execute, as reported by the vendor runtimes."""

    torch_available: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str = ""
    ctranslate2_cuda_devices: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_LOCK = threading.Lock()
_REPORT: CapabilityReport | None = None


def _probe_torch() -> tuple[bool, bool, int, str]:
    """Return ``(torch_importable, cuda_available, cuda_device_count, device_name)``."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a missing/broken torch means CPU-only
        return False, False, 0, ""
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        name = str(torch.cuda.get_device_name(0)) if count else ""
        return True, available, count, name
    except Exception:  # noqa: BLE001 - a broken driver must not break analysis
        return True, False, 0, ""


def _probe_ctranslate2() -> int:
    """Return the number of CUDA devices CTranslate2 can see (0 when unavailable)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return 0


def probe(refresh: bool = False) -> CapabilityReport:
    """Return the cached capability report, probing the runtimes on first use."""
    global _REPORT
    if _REPORT is not None and not refresh:
        return _REPORT
    with _LOCK:
        if _REPORT is not None and not refresh:
            return _REPORT
        torch_ok, cuda_ok, count, name = _probe_torch()
        _REPORT = CapabilityReport(
            torch_available=torch_ok,
            cuda_available=cuda_ok,
            cuda_device_count=count,
            cuda_device_name=name,
            ctranslate2_cuda_devices=_probe_ctranslate2(),
        )
        return _REPORT


def reset() -> None:
    """Drop the cached report. Intended for tests and diagnostics."""
    global _REPORT
    with _LOCK:
        _REPORT = None
