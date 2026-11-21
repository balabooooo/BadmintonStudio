"""Measured latency cache for execution providers, and the order they are tried in.

The static priority list in :mod:`bms.inference.registry` encodes "which accelerator
is usually best". That is not the same as "which is best on *this* machine": a
DirectML EP on an old iGPU loses to a fast CPU, TensorRT engines are rebuilt per
driver/GPU, and an NPU may win only for a specific input size. So provider choice
is ranked by measured numbers when a benchmark for the same model/driver exists,
and falls back to the static order when it does not.

The cache is a JSON file under the data directory (never in the repo) keyed by a
fingerprint of everything that can change the number: OS build, GPU name/driver,
ONNX Runtime version, model file hash, imgsz and batch. A stale entry is therefore
ignored instead of silently ranking a different machine's hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import DATA_DIR, MODELS_DIR
from . import registry

CACHE_LOCK = threading.Lock()

#: Provider benchmark rows older than this are ignored (driver/runtime drift).
MAX_AGE_DAYS = 90

_FP_LOCK = threading.Lock()
_FP_CACHE: dict[str, str] | None = None


def cache_path() -> Path:
    """``<data>/cache/inference_bench.json`` (gitignored runtime state)."""
    return Path(DATA_DIR) / "cache" / "inference_bench.json"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - hardware probe must not fail the caller
        pass
    return "unknown"


def _driver_version() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.version.cuda or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _ort_version() -> str:
    try:
        import onnxruntime as ort

        return ort.__version__
    except Exception:  # noqa: BLE001
        return ""


def fingerprint() -> dict[str, str]:
    """Everything that can change an execution provider's latency (cached per process)."""
    global _FP_CACHE
    if _FP_CACHE is not None:
        return dict(_FP_CACHE)
    with _FP_LOCK:
        if _FP_CACHE is None:
            _FP_CACHE = {
                "os": f"{platform.system()} {platform.release()} {platform.version()}",
                "machine": platform.machine(),
                "gpu": _gpu_name(),
                "cuda": _driver_version(),
                "ort": _ort_version(),
            }
        return dict(_FP_CACHE)


def _model_key(model: str) -> str:
    path = Path(model)
    if not path.is_absolute():
        path = Path(MODELS_DIR) / path
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return f"{path.name}:{digest.hexdigest()[:12]}"
    except OSError:
        return f"{model}:missing"


def _read() -> dict[str, Any]:
    path = cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict[str, Any]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record(model: str, provider: str, p50_ms: float, p90_ms: float = 0.0,
           imgsz: int = 640, batch: int = 8, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Store one measurement (upsert by model+provider+shape)."""
    key = _model_key(model)
    entry = {
        "model": key,
        "provider": provider,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "p50_ms": round(float(p50_ms), 3),
        "p90_ms": round(float(p90_ms), 3),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint(),
        "meta": meta or {},
    }
    with CACHE_LOCK:
        data = _read()
        rows = [r for r in data.get("rows", [])
                if not (r.get("model") == key and r.get("provider") == provider
                        and r.get("imgsz") == entry["imgsz"] and r.get("batch") == entry["batch"])]
        rows.append(entry)
        data["rows"] = rows
        _write(data)
    logger.debug("inference bench: {} on {} = {} ms", key, provider, entry["p50_ms"])
    return entry


def table(model: str = "", imgsz: int = 0, batch: int = 0) -> list[dict[str, Any]]:
    """Rows matching the current machine and (optionally) model / shape."""
    fp = fingerprint()
    key = _model_key(model) if model else ""
    out = []
    for row in _read().get("rows", []):
        if row.get("fingerprint") != fp:
            continue
        if key and row.get("model") != key:
            continue
        if imgsz and int(row.get("imgsz", 0)) != int(imgsz):
            continue
        if batch and int(row.get("batch", 0)) != int(batch):
            continue
        out.append(row)
    return out


def preferred_order(task: str, model: str, imgsz: int = 0, batch: int = 0) -> list[str]:
    """Providers to try, fastest measured first, then the static priority order.

    CPU is always last: it is the ladder's floor, and letting a noisy measurement
    put CPU above a real accelerator would waste the whole point of the ranking.
    """
    candidates = registry.candidate_providers(task)
    if not candidates:
        return []
    rows = {r["provider"]: float(r.get("p50_ms") or 0.0)
            for r in table(model, imgsz, batch) if r.get("provider") in candidates}
    measured = sorted((p for p in candidates if p in rows and p != registry.CPU_PROVIDER),
                      key=lambda p: rows[p])
    rest = [p for p in candidates if p not in measured and p != registry.CPU_PROVIDER]
    order = measured + rest
    if registry.CPU_PROVIDER in candidates:
        order.append(registry.CPU_PROVIDER)
    return order


def summary() -> dict[str, Any]:
    """Compact block for ``/api/env``: rows for this machine plus the current order."""
    rows = table()
    return {
        "cache": str(cache_path()),
        "fingerprint": fingerprint(),
        "rows": [
            {"model": r.get("model"), "provider": r.get("provider"),
             "imgsz": r.get("imgsz"), "batch": r.get("batch"),
             "p50_ms": r.get("p50_ms"), "created_at": r.get("created_at")}
            for r in rows
        ],
        "order_players": preferred_order(registry.TASK_PLAYERS, os.environ.get(
            "BMS_ORT_PLAYERS_MODEL", "yolo26n.dynamic.onnx")),
        "order_pose": preferred_order(registry.TASK_POSE, os.environ.get(
            "BMS_ORT_POSE_MODEL", "yolo26n-pose.static.onnx")),
    }
