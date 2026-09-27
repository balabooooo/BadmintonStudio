"""CUDA runtime discovery for CTranslate2 (used by faster-whisper).

CTranslate2 does its own lazy ``LoadLibrary("cublas64_12.dll")`` and **does not** go through
torch's DLL loader. On a machine where ``torch`` + CUDA already works, CTranslate2 can still
fail with ``Library cublas64_12.dll is not found or cannot be loaded`` because the DLLs are not
on its search path.

torch already ships the exact CUDA 12 runtime CTranslate2 needs (``cublas64_12``,
``cublasLt64_12``, ``cudnn64_9``, ``cudart64_12``). :func:`ensure_cuda_dlls` puts torch's
``lib`` directory (plus any pip ``nvidia-*`` package ``bin`` directories) on ``PATH`` so the
lazy load succeeds. It is a no-op when none of those exist, and never raises.
"""

from __future__ import annotations

import os
from pathlib import Path

_registered = False
_dirs: list[str] = []


def ensure_cuda_dlls() -> list[str]:
    """Register CUDA DLL directories; return the directories added on the first call.

    Safe to call repeatedly: the scan runs once and later calls return the cached list.
    """
    global _registered, _dirs
    if _registered:
        return _dirs
    _registered = True
    dirs: list[str] = []
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir():
            dirs.append(str(lib))
    except Exception:  # noqa: BLE001 - torch is optional
        pass
    try:
        import importlib.util

        for mod in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
            spec = importlib.util.find_spec(mod)
            for base in (spec.submodule_search_locations or []) if spec else []:
                bin_dir = Path(base) / "bin"
                if bin_dir.is_dir():
                    dirs.append(str(bin_dir))
    except Exception:  # noqa: BLE001
        pass
    for d in dirs:
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass
    _dirs = dirs
    return dirs


__all__ = ["ensure_cuda_dlls"]
