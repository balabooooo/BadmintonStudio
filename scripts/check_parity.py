"""Compare an inference backend against the golden sample and gate on parity.

The golden sample (``scripts/make_golden.py``) stores the frames, the Torch FP32
reference boxes and, optionally, reference pose keypoints. This script replays the
same frames through the ONNX Runtime backend on a chosen provider and reports:

* detection parity - IoU@0.5 recall / precision, many-to-one coverage, mean IoU,
  confidence delta;
* pose parity - mean / p90 keypoint error in patch pixels;
* latency - p50 batch latency for the same batch size the analysis uses.

Exit code is 0 only when many-to-one coverage clears ``--min-recall`` /
``--min-precision`` (default 0.99): every object the reference saw must have a
counterpart and vice versa. The 1:1 numbers are printed for context -- they are
depressed by near-duplicate boxes that NMS tie-breaking keeps on either side,
which the tracker removes anyway.

Usage:
    python scripts/check_parity.py data/golden/golden_clip4_xxx
    python scripts/check_parity.py data/golden/... --providers CUDAExecutionProvider
    python scripts/check_parity.py data/golden/... --batch 8 --json-out data/cache/parity_ort.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _imread(path: Path):
    """Read an image through ``imdecode``: ``imread`` fails on non-ASCII Windows paths."""
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _load(golden: Path) -> tuple[list, list[list[dict]], list[list[dict]]]:
    meta = json.loads((golden / "meta.json").read_text(encoding="utf-8"))
    frames = []
    for i in range(int(meta["frame_count"])):
        path = golden / "frames" / f"f_{i:04d}.jpg"
        img = _imread(path)
        if img is None:
            raise SystemExit(f"cannot read frame: {path}")
        frames.append(img)
    det = json.loads((golden / "detections.json").read_text(encoding="utf-8"))
    ref_boxes = [[{"xyxy": b["xyxy"], "conf": b["conf"]} for b in f] for f in det["frames"]]
    ref_pose: list[list[dict]] = [[] for _ in frames]
    pose_name = meta.get("pose")
    if pose_name and (golden / pose_name).exists():
        pose = json.loads((golden / pose_name).read_text(encoding="utf-8"))
        ref_pose = pose["frames"]
    return frames, ref_boxes, ref_pose


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden", help="directory produced by scripts/make_golden.py")
    ap.add_argument("--detect-model", default="yolo26n.dynamic.onnx")
    ap.add_argument("--pose-model", default="yolo26n-pose.static.onnx")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--pose-imgsz", type=int, default=192)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--providers", default="", help="comma separated ORT provider list")
    ap.add_argument("--min-recall", type=float, default=0.99)
    ap.add_argument("--min-precision", type=float, default=0.99)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from bms.inference import ort_backend as ORT
    from bms.inference import parity as PAR

    golden = Path(args.golden)
    if not (golden / "meta.json").exists():
        raise SystemExit(f"not a golden sample: {golden}")
    frames, ref_boxes, ref_pose = _load(golden)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()] or None

    det = ORT.OrtDetector(
        model=args.detect_model, imgsz=args.imgsz, conf=args.conf, providers=providers,
    )
    print(f"providers: {det.session.get_providers()}  batch: {args.batch}  frames: {len(frames)}")

    import time

    test_boxes: list[list[dict]] = []
    latencies: list[float] = []
    for start in range(0, len(frames), args.batch):
        chunk = frames[start:start + args.batch]
        t0 = time.perf_counter()
        results = det.predict(chunk)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        for res in results:
            xyxy = res.boxes.xyxy.numpy()
            conf = res.boxes.conf.numpy()
            test_boxes.append([
                {"xyxy": [float(v) for v in xyxy[k]], "conf": float(conf[k])}
                for k in range(len(xyxy))
            ])

    det_parity = PAR.match_frames(ref_boxes, test_boxes, 0.5)
    det_parity["batch_p50_ms"] = round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0
    det_parity["per_frame_ms"] = round(
        det_parity["batch_p50_ms"] / max(1, args.batch if len(frames) >= args.batch else len(frames)), 2)

    out = {"golden": str(golden), "detection": det_parity}

    pose_parity = None
    if any(ref_pose):
        pose = ORT.OrtPoseEstimator(
            model=args.pose_model, imgsz=args.pose_imgsz, conf=0.10, providers=providers,
        )
        crops = []
        for ref_frame, frame in zip(ref_pose, frames):
            import cv2

            for entry in ref_frame:
                x0, y0, x1, y1 = entry["crop"]
                crops.append(cv2.resize(frame[y0:y1, x0:x1], (args.pose_imgsz, args.pose_imgsz),
                                        interpolation=cv2.INTER_AREA))
        results = pose.predict(crops) if crops else []
        # The golden file groups crops per frame; flatten in the same order.
        flat_ref = [entry for f in ref_pose for entry in f]
        test_pose: list[dict] = []
        for res in results:
            kps = res.keypoints
            if kps is None:
                continue
            xy = kps.xy.numpy()[0]
            cf = kps.conf.numpy()[0]
            test_pose.append({"kps": [[float(x), float(y), float(c)] for (x, y), c in zip(xy, cf)]})
        # Re-group back to per-frame slot layout for keypoint_l2.
        grouped: list[list[dict]] = [[] for _ in ref_pose]
        cursor = 0
        for fi, entries in enumerate(ref_pose):
            for entry in entries:
                if cursor < len(test_pose):
                    grouped[fi].append({"slot": entry["slot"], **test_pose[cursor]})
                    cursor += 1
        pose_parity = PAR.keypoint_l2(ref_pose, grouped, crop_size=float(args.pose_imgsz))
        pose_parity["ref_patches"] = len(flat_ref)
        out["pose"] = pose_parity

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = (
        det_parity["coverage_recall"] >= args.min_recall
        and det_parity["coverage_precision"] >= args.min_precision
    )
    print("PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
