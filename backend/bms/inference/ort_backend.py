"""ONNX Runtime backend for the YOLO detector and the pose estimator.

Why a second backend: Torch covers Windows-NVIDIA well and nothing else. ONNX
Runtime exposes CUDA / TensorRT / DirectML / CPU execution providers on Windows,
so it is the portable path described in ``docs/mobile-and-device-ai-plan.md`` (§4).

The adapter is written against the **exported, pre-NMS** graphs
(``yolo26n.dynamic.onnx`` output ``(N, 84, 8400)``, pose ``(N, 56, 756)``), which
is what ``ultralytics`` produces for ``simplify=False``. Preprocessing and
postprocessing replicate ultralytics' own letterbox / score filter / NMS /
box-rescale steps, because parity has to be measured against what the Torch path
actually does -- not against an idealised pipeline.

Everything returns *duck-typed* results that mirror ``ultralytics.engine.results``:
``.boxes.xyxy`` / ``.boxes.conf`` expose ``.cpu().numpy()``, and pose adds
``.keypoints.xy`` / ``.keypoints.conf``. That keeps ``players._dets_of`` and
``pose._flush`` unchanged between backends -- the same code reads both.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from loguru import logger

from ..config import MODELS_DIR
from . import registry

#: Dynamic-batch export, used on desktop where the sampled batch length varies.
DEFAULT_DETECT_MODEL = "yolo26n.dynamic.onnx"
#: Static batch exports: NPUs and Core ML delegate much better with fixed shapes.
DEFAULT_DETECT_STATIC = "yolo26n.static.onnx"
DEFAULT_POSE_MODEL = "yolo26n-pose.static.onnx"

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[tuple[str, tuple[str, ...]], Any] = {}


# ------------------------------------------------------------------ tensor shims
# Minimal stand-ins for ``torch.Tensor`` so call sites can keep ``.cpu().numpy()``.


class _Array:
    """A numpy array with the subset of the torch tensor API the call sites use."""

    __slots__ = ("_a",)

    def __init__(self, a: np.ndarray) -> None:
        self._a = a

    def cpu(self) -> "_Array":
        return self

    def numpy(self) -> np.ndarray:
        return self._a

    def __len__(self) -> int:
        return len(self._a)


class _Boxes:
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = _Array(xyxy.astype(np.float32))
        self.conf = _Array(conf.astype(np.float32))
        self.cls = _Array(cls.astype(np.float32))

    def __len__(self) -> int:
        return len(self.xyxy)


class _Keypoints:
    __slots__ = ("xy", "conf")

    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = _Array(xy.astype(np.float32))
        self.conf = _Array(conf.astype(np.float32))

    def __len__(self) -> int:
        return len(self.xy)


class Result:
    """Duck-typed ``ultralytics.engine.results.Results`` (only the fields we read)."""

    __slots__ = ("boxes", "keypoints")

    def __init__(self, boxes: _Boxes, keypoints: _Keypoints | None = None) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


# ------------------------------------------------------------------ preprocessing


def letterbox(
    img: np.ndarray, new_shape: int | tuple[int, int], stride: int = 32, auto: bool = False,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize + pad to a multiple of ``stride`` exactly like ultralytics' ``LetterBox``.

    ``auto=True`` keeps the minimum rectangle (only the missing side is padded to a
    stride multiple), which is what the Torch predictor does for same-sized batches.
    Returns the padded image, the scale ratio and ``(pad_x, pad_y)`` so boxes can be
    mapped back to the original frame.
    """
    import cv2

    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - new_w, new_shape[0] - new_h
    if auto:
        dw, dh = int(np.mod(dw, stride)), int(np.mod(dh, stride))
    dw /= 2.0
    dh /= 2.0
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
    return img, float(r), (float(left), float(top))


def preprocess(
    frames: Sequence[np.ndarray], imgsz: int, auto: bool,
) -> tuple[np.ndarray, list[tuple[float, tuple[float, float], tuple[int, int]]]]:
    """BGR frames -> NCHW float32 RGB in 0..1 plus the geometry needed to invert it."""
    batch = []
    metas: list[tuple[float, tuple[float, float], tuple[int, int]]] = []
    for frame in frames:
        padded, ratio, pad = letterbox(frame, imgsz, auto=auto)
        batch.append(padded[..., ::-1].transpose(2, 0, 1))
        metas.append((ratio, pad, frame.shape[:2]))
    arr = np.ascontiguousarray(np.stack(batch), dtype=np.float32) / 255.0
    return arr, metas


# ------------------------------------------------------------------ postprocessing


def _nms(boxes_xywh: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    """Greedy NMS via OpenCV; indices of the kept boxes in descending score order."""
    import cv2

    if len(boxes_xywh) == 0:
        return np.empty((0,), dtype=np.int64)
    idx = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), float(scores.min()), float(iou_thr),
        top_k=int(max_det),
    )
    if idx is None or len(idx) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(idx).reshape(-1).astype(np.int64)


def _scale_boxes(
    xyxy: np.ndarray, ratio: float, pad: tuple[float, float], orig_hw: tuple[int, int],
) -> np.ndarray:
    """Map letterboxed pixel boxes back to the original frame and clip to it."""
    h, w = orig_hw
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / max(1e-9, ratio)
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / max(1e-9, ratio)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


# ------------------------------------------------------------------ sessions


def model_path(name: str) -> Path:
    """Resolve a model file under ``MODELS_DIR`` (supports an absolute override)."""
    p = Path(name)
    return p if p.is_absolute() else Path(MODELS_DIR) / p


def get_session(
    path: str | Path, providers: Sequence[str] | None = None, task: str = registry.TASK_PLAYERS,
) -> Any:
    """Create (and cache) an ``InferenceSession``, walking down the provider ladder.

    Ordering comes from :func:`bms.inference.benchmark.preferred_order` (measured
    latency first, static priority after). If a provider is registered but fails to
    initialise -- a broken driver, a missing DLL, an unsupported opset -- the next
    rung is tried instead of failing the whole analysis; the CPU rung is always
    last because ORT needs it as the fallback partition anyway.
    """
    import onnxruntime as ort

    resolved = str(model_path(str(path)))
    if providers:
        candidates = [str(p) for p in providers]
    else:
        try:
            from .benchmark import preferred_order

            candidates = preferred_order(task, str(path))
        except Exception:  # noqa: BLE001 - benchmark cache must never block inference
            candidates = registry.candidate_providers(task)
    if not candidates:
        candidates = [registry.CPU_PROVIDER]

    with _SESSION_LOCK:
        sess = _SESSIONS.get((resolved, tuple(candidates)))
        if sess is not None:
            return sess
        last_exc: Exception | None = None
        for i in range(len(candidates)):
            attempt = candidates[i:]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            try:
                sess = ort.InferenceSession(resolved, sess_options=opts, providers=attempt)
            except Exception as exc:  # noqa: BLE001 - demote to the next provider
                last_exc = exc
                logger.warning("execution provider {} unavailable ({}: {}); trying next",
                               attempt[0], type(exc).__name__, exc)
                continue
            _SESSIONS[(resolved, tuple(candidates))] = sess
            _SESSIONS[(resolved, tuple(attempt))] = sess
            logger.debug("onnx session {} providers={}", Path(resolved).name, attempt)
            return sess
        raise last_exc if last_exc else RuntimeError("no execution provider available")


def clear_sessions() -> None:
    """Drop cached sessions (tests / after replacing a model file)."""
    with _SESSION_LOCK:
        _SESSIONS.clear()


def _input_is_dynamic(sess: Any) -> bool:
    """Whether the graph accepts a variable batch (exported with ``dynamic=True``)."""
    try:
        shape = sess.get_inputs()[0].shape
        return any(isinstance(d, str) for d in shape)
    except Exception:  # noqa: BLE001 - treat an unreadable shape as static
        return False


# ------------------------------------------------------------------ detector


class OrtDetector:
    """Person detector on ONNX Runtime with the Torch path's decoding rules."""

    def __init__(
        self,
        model: str = DEFAULT_DETECT_MODEL,
        imgsz: int = 640,
        conf: float = 0.10,
        iou: float = 0.70,
        max_det: int = 300,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model = str(model)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.session = get_session(self.model, providers)
        self.dynamic = _input_is_dynamic(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.batch_limit = 0 if self.dynamic else int(self.session.get_inputs()[0].shape[0])

    def predict(
        self, frames: Sequence[np.ndarray], conf: float | None = None, classes: Iterable[int] = (0,),
    ) -> list[Result]:
        if not frames:
            return []
        conf_thr = self.conf if conf is None else float(conf)
        cls_wanted = np.asarray(list(classes), dtype=np.int64)
        out: list[Result] = []
        step = self.batch_limit or len(frames)
        for start in range(0, len(frames), step):
            chunk = frames[start:start + step]
            real = len(chunk)
            if step and real < step:                      # static graph needs a full batch
                chunk = list(chunk) + [chunk[-1]] * (step - real)
            out.extend(self._run(chunk, real, conf_thr, cls_wanted))
        return out

    def _run(
        self, chunk: Sequence[np.ndarray], real: int, conf_thr: float, cls_wanted: np.ndarray,
    ) -> list[Result]:
        arr, metas = preprocess(chunk, self.imgsz, auto=self.dynamic)
        preds = self.session.run(None, {self.input_name: arr})[0]
        results: list[Result] = []
        for b in range(real):
            raw = np.asarray(preds[b]).T                     # (anchors, 4 + n_cls)
            scores_all = raw[:, 4:]
            cls_ids = scores_all.argmax(1)
            scores = scores_all.max(1)
            keep = scores >= conf_thr
            if cls_wanted.size:
                keep &= np.isin(cls_ids, cls_wanted)
            xywh = raw[keep, :4]
            scores = scores[keep]
            cls_keep = cls_ids[keep]
            if len(xywh) == 0:
                results.append(Result(_Boxes(np.empty((0, 4)), np.empty(0), np.empty(0))))
                continue
            idx = _nms(xywh, scores, self.iou, self.max_det)
            xyxy = _scale_boxes(_xywh_to_xyxy(xywh[idx]), metas[b][0], metas[b][1], metas[b][2])
            results.append(Result(_Boxes(xyxy, scores[idx], cls_keep[idx].astype(np.float32))))
        return results


# ------------------------------------------------------------------ pose


class OrtPoseEstimator:
    """Pose estimator on player crops; returns the same fields ``pose._flush`` reads."""

    def __init__(
        self,
        model: str = DEFAULT_POSE_MODEL,
        imgsz: int = 192,
        conf: float = 0.25,
        iou: float = 0.70,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model = str(model)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.session = get_session(self.model, providers, task=registry.TASK_POSE)
        self.dynamic = _input_is_dynamic(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.batch_limit = 0 if self.dynamic else int(self.session.get_inputs()[0].shape[0])

    def predict(self, patches: Sequence[np.ndarray], conf: float | None = None) -> list[Result]:
        if not patches:
            return []
        conf_thr = self.conf if conf is None else float(conf)
        out: list[Result] = []
        step = self.batch_limit or len(patches)
        for start in range(0, len(patches), step):
            chunk = patches[start:start + step]
            real = len(chunk)
            if step and real < step:
                chunk = list(chunk) + [chunk[-1]] * (step - real)
            out.extend(self._run(chunk, real, conf_thr))
        return out

    def _run(self, chunk: Sequence[np.ndarray], real: int, conf_thr: float) -> list[Result]:
        arr, _metas = preprocess(chunk, self.imgsz, auto=self.dynamic)
        preds = self.session.run(None, {self.input_name: arr})[0]
        results: list[Result] = []
        for b in range(real):
            raw = np.asarray(preds[b]).T                     # (anchors, 4 + 1 + 51)
            scores = raw[:, 4]
            keep = scores >= conf_thr
            xywh, scores = raw[keep, :4], scores[keep]
            kps = raw[keep, 5:]
            if len(xywh) == 0:
                results.append(Result(_Boxes(np.empty((0, 4)), np.empty(0), np.empty(0))))
                continue
            idx = _nms(xywh, scores, self.iou, 300)
            kp = kps[idx].reshape(-1, 17, 3)                 # x, y, confidence per joint
            results.append(Result(
                _Boxes(_xywh_to_xyxy(xywh[idx]), scores[idx], np.zeros(len(idx), np.float32)),
                _Keypoints(kp[..., :2], kp[..., 2]),
            ))
        return results


def available(models: Sequence[str] | None = None) -> dict[str, bool]:
    """Whether the ONNX runtime and the expected model files are present."""
    names = list(models or (DEFAULT_DETECT_MODEL, DEFAULT_POSE_MODEL))
    try:
        import onnxruntime  # noqa: F401
        runtime = True
    except Exception:  # noqa: BLE001
        runtime = False
    found = {name: model_path(name).exists() for name in names}
    if not runtime:
        logger.debug("onnxruntime not installed; ONNX backend unavailable")
    return {"runtime": runtime, **found}
