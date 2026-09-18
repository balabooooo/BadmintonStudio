"""调查脚本 2：验证「用姿态挥拍 + 球员位移反应 给音频击球做归属」的可行性。

假设：球馆里有多块场地，音频击球声混在一起（实测 audio_reliability≈0.04）。
但**我们的球员**只在真的击球时会挥拍 / 身体突然加速；别的场地的击球声
在我们画面上没有任何对应的动作。

这个脚本对一段窗口逐帧：
1. yolo11n 检测人物 -> 简单跟踪（只求这段窗口内连续）；
2. 把每个框裁出来放大，送 yolo11n-pose 拿关键点（小目标放大后关键点更稳）；
3. 合成两路逐帧证据：
   - ``swing``：手腕速度（相对肩/髋、按身体高度归一）
   - ``react``：框中心速度相对本地基线的突增
4. 对每一个音频击球，看它在 ±0.35s 内有没有 swing / react 证据；
5. 按证据门控后聚类，打印回合结构（间隔、时长、拍数），并把边界帧存图。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PROXY = ROOT / "data" / "cache" / "proxies" / "20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4"
WAV = ROOT / "data" / "cache" / "audio" / "20260913_羽毛球_clip9_m_aa6b31f616ee_16000.wav"
DET_W = ROOT / "models" / "yolo11n.pt"
POSE_W = ROOT / "models" / "yolo11n-pose.pt"
OUT = ROOT / "data" / "cache" / "frames" / "pose_probe2"

L_WRIST, R_WRIST = 9, 10
L_SHLD, R_SHLD = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16


def grab(path: Path, start: float, dur: float, fps: float):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frames, times = [], []
    want = int(dur * fps)
    nxt = start
    idx = int(start * src_fps)
    while len(frames) < want:
        ok, fr = cap.read()
        if not ok:
            break
        t = start + (idx - int(start * src_fps)) / src_fps
        if t >= nxt - 1e-6:
            frames.append(fr)
            times.append(t)
            nxt += 1.0 / fps
        idx += 1
    cap.release()
    return frames, np.asarray(times, dtype=np.float64)


def crops_of(frame: np.ndarray, boxes: np.ndarray, margin: float = 0.18,
             side: int = 192) -> tuple[list[np.ndarray], list[tuple[float, float, float]]]:
    """把每个人物框（带 margin）裁出来并放大到 side×side（保持长宽比，letterbox）。"""
    h, w = frame.shape[:2]
    out, meta = [], []
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        # 正方形窗口，边长 = 高的 (1+2*margin)，保证挥拍时手不出框
        s = bh * (1.0 + 2 * margin)
        x1, y1 = int(round(cx - s / 2)), int(round(cy - s / 2))
        x2, y2 = int(round(cx + s / 2)), int(round(cy + s / 2))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c - x1c < 8 or y2c - y1c < 8:
            out.append(None)
            meta.append((0.0, 0.0, 0.0))
            continue
        patch = frame[y1c:y2c, x1c:x2c]
        interp = cv2.INTER_CUBIC if (x2c - x1c) < side else cv2.INTER_AREA
        patch = cv2.resize(patch, (side, side), interpolation=interp)
        out.append(patch)
        # 把「裁剪窗口左上角 + 缩放比」记下来，好把关键点映射回原图
        meta.append((float(x1c), float(y1c), side / float(s)))
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--gate", type=float, default=0.35)
    args = ap.parse_args()

    cfg = ROOT / "data" / "yolo"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))
    OUT.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    frames, times = grab(PROXY, args.start, args.dur, args.fps)
    n = len(frames)
    H, W = frames[0].shape[:2]
    print(f"frames={n} span={times[0]:.2f}~{times[-1]:.2f} fps={1/np.median(np.diff(times)):.2f}")

    det = YOLO(str(DET_W))
    pose = YOLO(str(POSE_W))

    # ---------- 1) 检测 + 跟踪 ----------
    per_frame: list[list[dict]] = []
    for i in range(0, n, 16):
        res = det.predict(frames[i:i + 16], imgsz=960, conf=0.3, classes=[0],
                          verbose=False, device=0)
        for r in res:
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
            cur = []
            for box in b:
                h = box[3] - box[1]
                # 过滤明显不合理的框：太小（背景小人）、太大（整帧误检）
                if h < 0.10 * H or h > 0.55 * H or (box[2] - box[0]) > 0.5 * W:
                    continue
                cur.append({"box": box, "tid": -1})
            per_frame.append(cur)

    tracks: dict[int, dict] = {}
    next_id = 0
    for fi, cur in enumerate(per_frame):
        for c in cur:
            b = c["box"]
            cx, cy, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]
            best, bd = -1, 1e9
            for tid, tr in tracks.items():
                if tr["last"] < fi - 4:
                    continue
                d = np.hypot(cx - tr["cx"], cy - tr["cy"])
                if d < 1.2 * hh and d < bd:
                    best, bd = tid, d
            if best < 0:
                best = next_id
                next_id += 1
                tracks[best] = {"frames": [], "cx": cx, "cy": cy, "last": fi, "hs": []}
            tr = tracks[best]
            tr["frames"].append(fi)
            tr["hs"].append(hh)
            tr["cx"], tr["cy"], tr["last"] = cx, cy, fi
            c["tid"] = best
    ntrack = len(tracks)
    keep = [t for t, v in tracks.items() if len(v["frames"]) >= int(3.0 * args.fps)]
    print(f"tracks={ntrack} 保留(>=3s)={len(keep)}  det/frame={np.mean([len(x) for x in per_frame]):.2f}")

    # ---------- 2) 逐帧 pose（裁框放大） ----------
    pose_kp: list[dict[int, np.ndarray]] = [dict() for _ in range(n)]
    pose_kc: list[dict[int, np.ndarray]] = [dict() for _ in range(n)]
    for i in range(n):
        if not per_frame[i]:
            continue
        boxes = np.asarray([c["box"] for c in per_frame[i]])
        patches, meta = crops_of(frames[i], boxes)
        idxs = [k for k, p in enumerate(patches) if p is not None]
        if not idxs:
            continue
        res = pose.predict([patches[k] for k in idxs], imgsz=192, conf=0.25,
                           verbose=False, device=0)
        for k, r in zip(idxs, res):
            if r.keypoints is None or len(r.keypoints) == 0:
                continue
            # 取面积最大的那个人
            b = r.boxes.xyxy.cpu().numpy()
            pick = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            kp = r.keypoints.xy.cpu().numpy()[pick]
            kc = r.keypoints.conf.cpu().numpy()[pick]
            ox, oy, sc = meta[k]
            kp = np.stack([ox + kp[:, 0] / sc, oy + kp[:, 1] / sc], axis=1)
            tid = per_frame[i][k]["tid"]
            pose_kp[i][tid] = kp
            pose_kc[i][tid] = kc
        if i % 120 == 0:
            print(f"  pose {i}/{n}")

    hit_rate = np.mean([len(pose_kp[i]) > 0 for i in range(n)])
    print(f"有姿态的帧比例 = {hit_rate:.2f}")

    # ---------- 3) 两路逐帧证据 ----------
    # (a) swing：手腕速度 / 身体高度，去掉整体位移
    swing = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        common = set(pose_kp[i]) & set(pose_kp[i - 1])
        for tid in common:
            kp0, kc0 = pose_kp[i - 1][tid], pose_kc[i - 1][tid]
            kp1, kc1 = pose_kp[i][tid], pose_kc[i][tid]
            # 身体高度：肩->踝
            def _body(kp, kc):
                ys = [kp[j, 1] for j in (L_SHLD, R_SHLD, L_ANK, R_ANK) if kc[j] > 0.3]
                return (max(ys) - min(ys)) if len(ys) >= 2 else np.nan
            b0, b1 = _body(kp0, kc0), _body(kp1, kc1)
            body = np.nanmean([b0, b1])
            if not np.isfinite(body) or body < 0.03 * H:
                continue
            # 参考点：双肩中点（去掉整体位移的影响）
            def _mid(kp, kc, a, b):
                if kc[a] > 0.3 and kc[b] > 0.3:
                    return (kp[a] + kp[b]) / 2
                return None
            sh0 = _mid(kp0, kc0, L_SHLD, R_SHLD)
            sh1 = _mid(kp1, kc1, L_SHLD, R_SHLD)
            if sh0 is None or sh1 is None:
                continue
            for w in (L_WRIST, R_WRIST):
                if kc0[w] < 0.3 or kc1[w] < 0.3:
                    continue
                d = np.hypot((kp1[w, 0] - sh1[0]) - (kp0[w, 0] - sh0[0]),
                             (kp1[w, 1] - sh1[1]) - (kp0[w, 1] - sh0[1]))
                v = float(d) / body * args.fps          # 单位：身体高度/秒
                if v > swing[i]:
                    swing[i] = v
    if n > 4:
        k = np.ones(3, dtype=np.float32) / 3.0
        swing = np.convolve(swing, k, mode="same").astype(np.float32)

    # (b) react：球员位移速度的突增（相对本地中位数）
    spd = np.zeros(n, dtype=np.float32)
    cx = {t: np.full(n, np.nan, dtype=np.float32) for t in keep}
    cy = {t: np.full(n, np.nan, dtype=np.float32) for t in keep}
    hh = {t: np.full(n, np.nan, dtype=np.float32) for t in keep}
    for fi, cur in enumerate(per_frame):
        for c in cur:
            if c["tid"] not in cx:
                continue
            b = c["box"]
            cx[c["tid"]][fi] = (b[0] + b[2]) / 2
            cy[c["tid"]][fi] = (b[1] + b[3]) / 2
            hh[c["tid"]][fi] = b[3] - b[1]
    for t in keep:
        for arr in (cx[t], cy[t], hh[t]):
            ok = ~np.isnan(arr)
            if ok.sum() > 2:
                arr[:] = np.interp(np.arange(n), np.nonzero(ok)[0], arr[ok])
        v = np.hypot(np.diff(cx[t]) / np.maximum(hh[t][1:], 1), np.diff(cy[t]) / np.maximum(hh[t][1:], 1))
        v = np.concatenate([[0.0], v]) * args.fps
        spd = np.maximum(spd, v.astype(np.float32))
    if n > 6:
        spd = np.convolve(spd, np.ones(5, np.float32) / 5, mode="same").astype(np.float32)

    def local_z(sig: np.ndarray, win: float = 3.0) -> np.ndarray:
        w = max(3, int(win * args.fps))
        base = np.array([np.median(sig[max(0, i - w):i + 1]) for i in range(n)], dtype=np.float32)
        amp = np.percentile(sig, 95) - np.percentile(sig, 20) + 1e-6
        return np.clip((sig - base) / amp, 0.0, 3.0)

    sw_z = local_z(swing, 2.0)
    sp_z = local_z(spd, 3.0)
    print(f"swing  p50={np.percentile(swing,50):.2f} p90={np.percentile(swing,90):.2f} max={swing.max():.2f} (身体高/秒)")
    print(f"spd    p50={np.percentile(spd,50):.3f} p90={np.percentile(spd,90):.3f} max={spd.max():.3f} (身位/秒)")

    # ---------- 4) 逐个音频击球算证据 ----------
    from bms.analysis import audio_hits as AH

    det_h = AH.detect_hits(WAV, sensitivity=0.5)
    sel = (det_h.times >= args.start) & (det_h.times <= times[-1])
    ht, hs = det_h.times[sel], det_h.strength[sel]
    g = int(round(args.gate * args.fps))

    rows = []
    for tt, ss in zip(ht, hs):
        fi = int(round((tt - args.start) * args.fps))
        a, b = max(0, fi - g), min(n, fi + g + 1)
        rows.append({
            "t": float(tt - args.start), "strength": float(ss),
            "swing": float(sw_z[a:b].max()) if b > a else 0.0,
            "spd": float(sp_z[a:b].max()) if b > a else 0.0,
        })
    for r in rows:
        r["score"] = 0.62 * min(1.0, r["swing"]) + 0.38 * min(1.0, r["spd"])
    print(f"\n音频击球 {len(rows)} 个；score 分布 p10={np.percentile([r['score'] for r in rows],10):.2f} "
          f"p50={np.percentile([r['score'] for r in rows],50):.2f} "
          f"p90={np.percentile([r['score'] for r in rows],90):.2f}")

    for thr in (0.15, 0.25, 0.35, 0.45, 0.55):
        k = [r for r in rows if r["score"] >= thr]
        print(f"  thr={thr:.2f}: 保留 {len(k):3d}/{len(rows)} ({len(k)/max(1,len(rows)):.0%})")

    # ---------- 5) 门控后聚类 ----------
    for thr in (0.25, 0.35):
        k = sorted([r for r in rows if r["score"] >= thr], key=lambda r: r["t"])
        if not k:
            continue
        gap_thr = 4.0
        clusters, cur = [], [k[0]]
        for r in k[1:]:
            if r["t"] - cur[-1]["t"] <= gap_thr:
                cur.append(r)
            else:
                clusters.append(cur)
                cur = [r]
        clusters.append(cur)
        print(f"\n=== thr={thr:.2f} 门控后聚类（gap>{gap_thr}s）: {len(clusters)} 段 ===")
        for c in clusters:
            if len(c) < 2:
                continue
            print(f"  {c[0]['t']:7.2f} ~ {c[-1]['t']:7.2f}  时长 {c[-1]['t']-c[0]['t']:6.2f}s  拍数 {len(c):3d} "
                  f" 段内最大间隔 {max(np.diff([x['t'] for x in c])) if len(c)>1 else 0:5.2f}s")

    # ---------- 6) 存图核对 ----------
    pick = sorted([r for r in rows if r["score"] >= 0.35], key=lambda r: r["t"])
    save = pick[::max(1, len(pick) // 12)][:14]
    for k, r in enumerate(save):
        fi = int(round(r["t"] * args.fps))
        fr = frames[fi].copy()
        for c in per_frame[fi]:
            b = c["box"].astype(int)
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        for tid, kp in pose_kp[fi].items():
            kc = pose_kc[fi][tid]
            for j, (x, y) in enumerate(kp):
                if kc[j] > 0.35:
                    cv2.circle(fr, (int(x), int(y)), 2, (0, 255, 255), -1)
        name = f"gated_{k:02d}_t{r['t']:7.2f}_s{r['score']:.2f}.jpg"
        cv2.imwrite(str(OUT / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nsaved {len(save)} frames -> {OUT}")

    # 逐秒表
    print("\n逐秒: swing_z | spd_z | 击球(t,strength,score)")
    for s in range(int(times[-1] - args.start)):
        i0, i1 = int(s * args.fps), int((s + 1) * args.fps)
        hh2 = [f"{r['t']-s:.2f}({r['strength']:.2f}/{r['score']:.2f})"
               for r in rows if s <= r["t"] < s + 1]
        print(f" {s:3d}s sw={sw_z[i0:i1].max():4.2f} sp={sp_z[i0:i1].max():4.2f} hits={hh2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
