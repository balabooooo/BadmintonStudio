"""Inference runtime selection.

This package is the single entry point for "which accelerator do we use?".
Analysis modules must not call ``torch.cuda.is_available()`` themselves; they
ask the functions exported here, so capability probing and backend selection
can evolve (ONNX Runtime EPs, new GPU runtimes) without touching call sites.

Layers:

* :mod:`capabilities` - one cached probe of the vendor runtimes.
* :mod:`scheduler` - Torch device / shuttle / speech decisions used today.
* :mod:`registry` - per-task ONNX execution-provider priority and the
  ``torch`` vs ``onnx`` switch (``BMS_INFER_BACKEND``).
* :mod:`parity` - output-level comparison against a reference backend, used by
  the golden-sample check before a new backend may be enabled by default.
"""

from .capabilities import CapabilityReport, probe, reset
from .parity import iou, keypoint_l2, match_boxes, match_frames
from .registry import (
    BACKEND_ENV,
    TASK_PLAYERS,
    TASK_POSE,
    TASK_SPEECH,
    available_providers,
    candidate_providers,
    priority,
    rank_providers,
    resolve_detect_backend,
    snapshot,
)
from .scheduler import resolve_shuttle_backend, resolve_speech_runtime, resolve_torch_device

__all__ = [
    "BACKEND_ENV",
    "TASK_PLAYERS",
    "TASK_POSE",
    "TASK_SPEECH",
    "CapabilityReport",
    "available_providers",
    "candidate_providers",
    "iou",
    "keypoint_l2",
    "match_boxes",
    "match_frames",
    "priority",
    "probe",
    "rank_providers",
    "reset",
    "resolve_detect_backend",
    "resolve_shuttle_backend",
    "resolve_speech_runtime",
    "resolve_torch_device",
    "snapshot",
]
