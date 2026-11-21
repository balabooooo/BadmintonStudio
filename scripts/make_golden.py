"""Freeze a short clip into a reproducible "golden sample" for backend parity checks.

The artifact is what P2/P3 compare against: the exact frames handed to the model
plus the Torch (CPU, FP32) reference outputs. It deliberately stores raw frames
instead of a trimmed video, because seeking a re-encoded clip does not guarantee
the same decoded pixels across codecs.

Output layout (under ``data/golden/<slug>/``, gitignored like all runtime state):

    meta.json                 source, fps, size, sampling times
    frames/f_0000.jpg ...     frames fed to the model
    detections.json           Torch reference: normalized xyxy + confidence
    pose.json                 optional: keypoints on the same frames

Usage:
    python scripts/make_golden.py "D:\\videos\\clip4.mp4" --seconds 30 --frames 40
    python scripts/make_golden.py "D:\\videos\\clip4.mp4" --with-pose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.inference.scheduler import resolve_torch_device  # noqa: E402


def _read_at(video: Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """Decode one frame per timestamp.

    Sequential decoding (``grab`` until the target index, then ``read``) is used
    instead of ``cap.set(CAP_PROP_POS_MSEC)``: seeking a short H.264 clip with
    OpenCV's default backend frequently lands past the last frame, which silently
    collapsed a 24-frame sample down to a single frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out: list[tuple[float, np.ndarray]] = []
    idx = 0
    for t in times:
        target = int(round(t * fps))
        if target < idx:
            continue
        while idx < target and cap.grab():
            idx += 1
        ok, frame = cap.read()
        idx += 1
        if ok and frame is not None:
            out.append((t, frame))
    cap.release()
    return out


def _detect(model, frames: list[np.ndarray], imgsz: int, device: str) -> list[list[dict]]:
    """Run the detector the same way ``analyze_players`` does (conf=LOW_CONF, batch predict)."""
    out: list[list[dict]] = []
    for i in range(0, len(frames), 8):
        chunk = frames[i:i + 8]
        results = model.predict(chunk, classes=[0], conf=0.10, imgsz=imgsz,
                                device=device, verbose=False)
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            out.append([
                {"xyxy": [round(float(v), 6) for v in xyxy[k]], "conf": round(float(conf[k]), 6)}
                for k in range(len(boxes))
            ])
    return out


def _pose(
    model, frames: list[np.ndarray], det_frames: list[list[dict]], crop_size: int,
    device: str, max_people: int = 4, margin: float = 0.35, up_shift: float = 0.05,
) -> list[list[dict]]:
    """Run the pose model on player crops, the way ``analyze_pose`` does.

    Feeding whole frames finds nothing at this imgsz: the two players are a small
    part of a wide low camera shot. Production builds a square crop window from
    each box (:func:`bms.analysis.pose._crop_spec`), resizes it to ``crop_size``
    and infers on that patch, so the reference does exactly the same and stores
    the crop window next to the patch-space keypoints.
    """
    import cv2

    from bms.analysis import pose as POSE

    patches: list[np.ndarray] = []
    slots: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for fi, frame in enumerate(frames):
        h, w = frame.shape[:2]
        dets = sorted(
            det_frames[fi],
            key=lambda d: -((d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1])),
        )[:max_people]
        for k, det in enumerate(dets):
            spec = POSE._crop_spec(tuple(det["xyxy"]), w, h, margin, up_shift, normalized=False)
            x0, y0, x1, y1 = spec
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            patch = cv2.resize(frame[y0:y1, x0:x1], (crop_size, crop_size),
                               interpolation=cv2.INTER_AREA)
            patches.append(patch)
            slots.append((fi, k, spec))

    out: list[list[dict]] = [[] for _ in frames]
    for i in range(0, len(patches), 16):
        chunk = patches[i:i + 16]
        results = model.predict(chunk, imgsz=crop_size, conf=0.10, device=device, verbose=False)
        for (fi, k, spec), res in zip(slots[i:i + 16], results):
            kps = getattr(res, "keypoints", None)
            if kps is None or len(kps) == 0:
                continue
            kp_xy = kps.xy.cpu().numpy()[0]
            kp_cf = kps.conf.cpu().numpy()[0]
            out[fi].append({
                "slot": k,
                "crop": list(spec),
                "kps": [[round(float(x), 4), round(float(y), 4), round(float(c), 4)]
                        for (x, y), c in zip(kp_xy, kp_cf)],
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--seconds", type=float, default=30.0, help="window length from --start")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo26n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu keeps the reference stable across machines")
    ap.add_argument("--with-pose", action="store_true")
    ap.add_argument("--pose-model", default=str(ROOT / "models" / "yolo26n-pose.pt"))
    ap.add_argument("--pose-imgsz", type=int, default=192, help="crop patch size, matches analyze_pose")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    src = Path(args.video)
    if not src.exists():
        raise SystemExit(f"video not found: {src}")
    slug = args.out or f"golden_{src.stem}_{_short_hash(src)}"
    out_dir = Path(slug)
    if not out_dir.is_absolute():
        out_dir = ROOT / "data" / "golden" / slug
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    times = [args.start + args.seconds * i / max(1, args.frames - 1) for i in range(args.frames)]
    sampled = _read_at(src, times)
    if not sampled:
        raise SystemExit("no frame decoded; check --start/--seconds")

    frames: list[np.ndarray] = []
    for i, (t, frame) in enumerate(sampled):
        path = out_dir / "frames" / f"f_{i:04d}.jpg"
        # imencode + tofile instead of imwrite: OpenCV's imwrite rejects non-ASCII paths on Windows.
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise SystemExit(f"cannot encode frame {i}")
        buf.tofile(str(path))
        frames.append(frame)
    print(f"frames: {len(frames)} -> {out_dir / 'frames'}")

    device = resolve_torch_device(args.device)
    det_model = YOLO(args.model)
    detections = _detect(det_model, frames, args.imgsz, device)
    (out_dir / "detections.json").write_text(
        json.dumps({"model": Path(args.model).name, "imgsz": args.imgsz,
                    "device": device, "frames": detections},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pose_path = None
    if args.with_pose:
        pose_model = YOLO(args.pose_model)
        pose = _pose(pose_model, frames, detections, args.pose_imgsz, device)
        pose_path = out_dir / "pose.json"
        pose_path.write_text(
            json.dumps({"model": Path(args.pose_model).name, "imgsz": args.pose_imgsz,
                        "device": device, "frames": pose},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(src),
        "source_size": src.stat().st_size,
        "start": args.start,
        "seconds": args.seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "times": [round(t, 3) for t, _f in sampled],
        "frame_count": len(frames),
        "detections": "detections.json",
        "pose": str(pose_path.name) if pose_path else None,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"golden sample: {out_dir}")
    return 0


def _short_hash(path: Path) -> str:
    key = f"{path.resolve()}|{path.stat().st_size}".encode("utf-8", "replace")
    return hashlib.sha1(key).hexdigest()[:8]


if __name__ == "__main__":
    raise SystemExit(main())
