"""姿态辅助：从球员框里提关键点，做出「挥拍」这一路信号。

为什么需要它
------------
多球场球馆里音频击球是不可信的（实测 30 分钟素材 ``audio_reliability = 0.04``）：
隔壁场地的击球声把「击球序列」填得很密，于是

* :func:`bms.analysis.rally.split_by_hit_gaps` 找不到空档，切不开该切的；
* :func:`bms.analysis.rally.refine_with_hits` 判不出「最后一拍之后没人打球了」，
  不敢把终点收紧。

实测结果就是 0~48.6 秒仍然被粘成一条「回合」。**音频本身提供不了「这一拍是不是
我们打的」这个信息**，而姿态可以：我们的球员只在真的击球时会挥拍，隔壁场地的
击球声在我们的画面上没有任何对应的动作。

设计上的两个关键选择
--------------------
1. **不重新检测、不重新跟踪。** 直接复用 ``PlayerSignal.frame_boxes``
   （已经跟踪好、也已经挑出比赛球员）。既省一半算力，更重要的是避免了
   「两套跟踪结果对不上」这类最难查的 bug。
2. **把框裁出来放大再送姿态模型。** 实测素材里球员只有约 100 像素高，
   整帧直接跑关键点会飘（腕、肘先丢）；裁成正方形窗放大到 192 之后关键点
   稳定可用，而且顺带把隔壁场地的人挡在窗外。
   窗口要**往上偏**并留出余量 —— 头顶击球时手腕会跑到框外。

产出
----
:class:`PoseSignal` 里最重要的是 ``swing``：手腕**相对双肩中点**的位移速度
除以身体高度。减掉双肩中点是为了去掉整体位移（球员跑动时手腕也会跟着动，
那不是挥拍）；除以身体高度是为了和球员远近无关。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

EPS = 1e-9

#: COCO 关键点序号
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16

#: 关键点置信度门限。低于它的关节视为「没测到」，不参与计算。
KP_CONF = 0.35


# ------------------------------------------------------------------ 数据结构


@dataclass
class PoseSignal:
    """逐帧姿态信号（帧率与 ``PlayerSignal.frame_boxes`` 一致）。"""

    fps: float
    duration: float
    #: 逐帧「挥拍强度」：手腕相对肩的位移速度 / 身体高度（单位：身体高/秒）
    swing: np.ndarray
    #: 逐帧是否有可用姿态（0/1）
    ok: np.ndarray
    #: 逐帧挥拍手是否在肩线以上（0/1）—— 用来区分头顶球与下手球
    overhead: np.ndarray
    #: 有可用姿态的帧占比
    coverage: float = 0.0
    #: 挥拍强度的「静息水平」（用来判断某个峰够不够显著）
    quiet: float = 0.0
    #: 诊断信息，直接进 stats 给用户看
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.swing.size)


# ------------------------------------------------------------------ 裁剪


def _crop_spec(box: tuple[float, float, float, float], w: int, h: int,
               margin: float, up_shift: float,
               normalized: bool = True) -> tuple[int, int, int, int]:
    """由球员框算出正方形裁剪窗 ``(x0, y0, x1, y1)``（像素，已夹到画面内）。

    ``PlayerSignal.frame_boxes`` 是**归一化**坐标（0~1），所以默认要按画面尺寸
    还原成像素 —— 直接当像素用会得到一个 4×4 的窗口，姿态模型什么都检不出来。
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    # 自适应兜底：即使调用方传的是像素坐标（画面内 x2/y2 必然 > 1），也不会算错
    if normalized and max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1.5:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    bh = max(4.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - up_shift * bh          # 窗口整体上移
    side = bh * (1.0 + 2.0 * margin)
    a0 = int(round(cx - side / 2.0))
    b0 = int(round(cy - side / 2.0))
    a1 = int(round(cx + side / 2.0))
    b1 = int(round(cy + side / 2.0))
    return (max(0, a0), max(0, b0), min(w, a1), min(h, b1))


def analyze_pose(
    video_path: str,
    frame_boxes: list,
    boxes_fps: float,
    model_name: str = "yolo11n-pose.pt",
    crop_size: int = 192,
    margin: float = 0.35,
    up_shift: float = 0.05,
    confidence: float = 0.25,
    device: str = "cuda",
    max_players_per_frame: int = 4,
    smoothing: float = 0.2,
    cache_dir: str | Path | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PoseSignal | None:
    """跑一遍姿态，返回 :class:`PoseSignal`；拿不到关键点时返回 ``None``。

    Args:
        video_path: 视频路径（代理视频即可，球员框本来就来自它）。
        frame_boxes: ``PlayerSignal.frame_boxes``，逐帧 ``(track_id, x1,y1,x2,y2)``
            归一化坐标。
        boxes_fps: ``frame_boxes`` 的帧率。
        crop_size: 裁剪窗放大到的边长（正方形）。192 是实测的性价比点：
            再小关键点开始丢，再大收益不明显而算力线性上升。
        margin: 裁剪窗相对框高的外扩比例。
        up_shift: 裁剪窗中心相对框中心上移的比例（头顶击球时手腕在框上方）。
        smoothing: 挥拍信号的滑动平均时间（秒）。
        cache_dir: 给了就把结果缓存成 npz（按视频/参数/权重哈希），
            用户反复调切分参数时不必重跑姿态。
        on_progress: ``callable(进度 0~1, 说明)``。
        cancel: ``callable() -> bool``。

    Returns:
        PoseSignal，或 ``None``（没有球员框 / 没有 GPU / 权重缺失 / 全片没检出姿态）。
    """
    n = len(frame_boxes)
    if n == 0 or boxes_fps <= 0:
        return None

    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, video_path, boxes_fps, crop_size,
                                 margin, up_shift, model_name,
                                 boxes_sig=_boxes_signature(frame_boxes))
        cached = _load_cache(cache_path, n)
        if cached is not None:
            cached.trace["cached"] = True
            if on_progress:
                on_progress(1.0, "姿态（缓存）")
            return cached

    weights = _resolve_weights(model_name)
    if weights is None:
        return None
    dev = _pick_device(device)

    swing = np.zeros(n, dtype=np.float32)
    overhead = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=np.float32)

    try:
        import cv2
        from ultralytics import YOLO
    except Exception:
        return None

    try:
        model = YOLO(weights)
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or boxes_fps) or boxes_fps
    step = max(1e-6, boxes_fps / max(src_fps, 1e-6))    # 源帧 -> 采样帧的步长

    # 逐帧裁出来的小图按「批」送 GPU。单帧只有 2~4 个人，
    # 一帧一送会让 GPU 空转（实测批处理能快 3 倍以上）。
    batch_patches: list[np.ndarray] = []
    batch_slots: list[tuple[int, int]] = []             # (帧号, 该帧第几个人)
    kp_store: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _flush() -> None:
        if not batch_patches:
            return
        try:
            res = model.predict(batch_patches, imgsz=crop_size, conf=confidence,
                                verbose=False, device=dev)
        except Exception:
            batch_patches.clear()
            batch_slots.clear()
            return
        for slot, r in zip(batch_slots, res):
            if r.keypoints is None or len(r.keypoints) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            if b is None or len(b) == 0:
                continue
            pick = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            kp_store[slot] = (r.keypoints.xy.cpu().numpy()[pick],
                              r.keypoints.conf.cpu().numpy()[pick])
        batch_patches.clear()
        batch_slots.clear()

    read_idx = 0
    kept = 0
    total = int(n)
    misses = 0
    pos = np.full((n, 4), np.nan, dtype=np.float32)     # 双肩中点 x,y + 两只手腕 x,y
    body = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if cancel and cancel():
            break
        # 跳到该采样帧对应的源帧
        target = int(round(i / max(step, 1e-6)))
        while read_idx < target:
            if not cap.grab():
                break
            read_idx += 1
        ok_read, frame = cap.read()
        read_idx += 1
        if not ok_read or frame is None:
            # 偶发解码失败不该让整段姿态作废，但连续失败就说明到片尾了。
            # 注意：**不要**把 frame_boxes[i] 清空 —— 那是调用方的列表
            # （PlayerSignal.frame_boxes），改它会污染调用方后续的使用。
            misses += 1
            if misses > 30:
                break
            continue
        misses = 0
        h, w = frame.shape[:2]
        boxes = list(frame_boxes[i] or [])[:max_players_per_frame]
        if not boxes:
            continue
        fi = 0
        for item in boxes:
            if len(item) < 5:
                continue
            spec = _crop_spec(item[1:5], w, h, margin, up_shift)
            x0, y0, x1, y1 = spec
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            patch = frame[y0:y1, x0:x1]
            interp = cv2.INTER_CUBIC if (x1 - x0) < crop_size else cv2.INTER_AREA
            patch = cv2.resize(patch, (crop_size, crop_size), interpolation=interp)
            batch_patches.append(patch)
            batch_slots.append((i, fi))
            # 把裁剪窗的几何记在槽位边上：关键点要映射回画面坐标
            kp_store.setdefault(("spec", i, fi), (np.asarray([x0, y0], dtype=np.float32),
                                                  np.asarray([(x1 - x0) / crop_size], dtype=np.float32)))
            fi += 1
            if len(batch_patches) >= 32:
                _flush()
        if (i % 60) == 0 and on_progress:
            on_progress(min(0.99, i / max(total, 1)), "姿态分析")
    _flush()
    cap.release()

    # ---- 关键点 -> 挥拍信号 ----
    # 每帧把「该帧所有人里最快的那个手腕」作为这一帧的挥拍强度。
    for i in range(n):
        for fi in range(max_players_per_frame):
            key = (i, fi)
            if key not in kp_store:
                continue
            spec = kp_store.get(("spec", i, fi))
            if spec is None:
                continue
            kp_raw, kc = kp_store[key]
            x0, y0 = float(spec[0][0]), float(spec[0][1])
            scale = float(spec[1][0])
            kp = np.stack([x0 + kp_raw[:, 0] / scale, y0 + kp_raw[:, 1] / scale], axis=1)
            sh = [kp[j] for j in (L_SHOULDER, R_SHOULDER) if kc[j] > KP_CONF]
            if len(sh) < 1:
                continue
            mid = np.mean(np.asarray(sh, dtype=np.float64), axis=0)
            # 身体高度：肩到踝。没有踝就退回「肩到画面下方」的粗估
            ys = [kp[j, 1] for j in (L_ANKLE, R_ANKLE) if kc[j] > KP_CONF]
            if ys:
                bh = float(max(ys)) - float(mid[1])
            else:
                bh = float(mid[1]) * 0.9
            if bh < 0.02 * h:
                continue
            # 取置信度更高的那只手腕
            cand = [(kc[j], j) for j in (L_WRIST, R_WRIST) if kc[j] > KP_CONF]
            if not cand:
                continue
            cand.sort(reverse=True)
            kc_w, j_w = cand[0]
            if not np.isfinite(pos[i, 0]):
                pos[i, 0], pos[i, 1] = mid
                body[i] = bh
            else:
                body[i] = max(body[i], bh) if np.isfinite(body[i]) else bh
            px, py = float(kp[j_w, 0]), float(kp[j_w, 1])
            if not np.isfinite(pos[i, 2]):
                pos[i, 2], pos[i, 3] = px, py
            else:
                # 同一帧有多个人：留「离肩更远的那个手腕」= 挥得更开的那个人
                d_old = np.hypot(pos[i, 2] - pos[i, 0], pos[i, 3] - pos[i, 1])
                d_new = np.hypot(px - pos[i, 0], py - pos[i, 1])
                if d_new > d_old:
                    pos[i, 2], pos[i, 3] = px, py
            overhead[i] = 1.0 if py < pos[i, 1] else 0.0
            ok[i] = 1.0
            kept += 1

    coverage = float(np.mean(ok > 0)) if n else 0.0
    if coverage < 0.05:
        return None

    # 手腕相对肩的位移速度（画面高度归一），再除以身体高度
    rel = np.stack([pos[:, 2] - pos[:, 0], pos[:, 3] - pos[:, 1]], axis=1)
    good = np.isfinite(rel).all(axis=1)
    if good.sum() < 4:
        return None
    idx = np.arange(n)
    for j in range(2):
        rel[:, j] = np.interp(idx, idx[good], rel[good, j])
    d = np.hypot(np.diff(rel[:, 0]), np.diff(rel[:, 1]))
    d = np.concatenate([[0.0], d]) * boxes_fps
    # 逐帧位移除以「该帧的身体高度」——近处的人身体高、位移也大，比值才是可比的
    bh = np.where(np.isfinite(body) & (body > 1.0), body, np.nan)
    if np.isfinite(bh).sum() > 2:
        bh = np.interp(idx, idx[np.isfinite(bh)], bh[np.isfinite(bh)])
    else:
        bh = np.full(n, max(1.0, 0.2 * h), dtype=np.float64)
    swing = (d / np.maximum(bh, 1e-6)).astype(np.float32)
    # 单帧跳变过大基本是关键点抖动，丢掉
    swing = np.clip(swing, 0.0, 6.0)
    win = max(1, int(round(smoothing * boxes_fps)))
    if win > 1:
        k = np.ones(win, dtype=np.float32) / win
        swing = np.convolve(swing, k, mode="same").astype(np.float32)
    swing[ok <= 0] = 0.0
    quiet = float(np.percentile(swing[swing > 0], 25)) if np.any(swing > 0) else 0.0

    sig = PoseSignal(
        fps=float(boxes_fps), duration=float(n / max(boxes_fps, EPS)),
        swing=swing, ok=ok, overhead=overhead.astype(np.float32),
        coverage=coverage, quiet=quiet,
        trace={"coverage": round(coverage, 3), "frames_with_pose": int(kept),
               "players": int(kept / max(1, int(np.sum(ok > 0)))) if np.any(ok > 0) else 0,
               "swing_p50": round(float(np.percentile(swing, 50)), 3),
               "swing_p90": round(float(np.percentile(swing, 90)), 3),
               "crop_size": crop_size, "model": Path(weights).name},
    )
    if cache_path is not None:
        _save_cache(cache_path, sig)
    if on_progress:
        on_progress(1.0, "姿态分析")
    return sig


# ------------------------------------------------------------------ 击球归属


def swing_peaks(
    pose: PoseSignal,
    min_distance: float = 0.26,
    prominence_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """在挥拍信号上找「挥拍瞬间」，返回 ``(峰所在帧下标, 显著度 0~1)``。

    为什么要找峰而不是直接取「窗口内的最大值」：对拉时每 1.0~1.5 秒就有一拍，
    而击球时刻本身有 ±0.05 秒的精度。如果在 ±0.35 秒的窗口里取最大值，
    这个窗口已经覆盖了半个拍间隔 —— **几乎每个击球附近都能找到一点手腕运动**，
    证据分就失去了区分度（实测 p50 高达 0.72，门控等于没做）。

    峰则把「挥拍」变成了稀疏事件：90 秒里只有几十个，而击球有一百多个，
    于是「一个峰只能解释一个击球」这件事本身就成了最强的约束。
    """
    if pose is None or pose.n < 8:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    from scipy.signal import find_peaks

    sw = pose.swing
    hi = float(np.percentile(sw, 95))
    span = max(hi - pose.quiet, 1e-6)
    dist = max(1, int(round(min_distance * pose.fps)))
    idx, props = find_peaks(sw, distance=dist, prominence=prominence_ratio * span)
    if idx.size == 0:
        return idx.astype(np.int64), np.zeros(0, dtype=np.float32)
    prom = props.get("prominences")
    if prom is None:
        prom = np.full(idx.size, span, dtype=np.float64)
    return idx.astype(np.int64), np.clip(prom / span, 0.0, 1.0).astype(np.float32)


def hit_swing_evidence(
    pose: PoseSignal,
    times: np.ndarray,
    strengths: np.ndarray | None = None,
    window: float = 0.30,
    one_to_one: bool = True,
) -> np.ndarray:
    """每个音频击球在 ±``window`` 秒内匹配到的挥拍证据（0~1）。

    ``one_to_one=True`` 时**一个挥拍峰只解释一个击球**：同一个峰附近若有多个
    击球候选（同一声被重复检出、或者隔壁场地的声音恰好撞上），只保留离峰最近
    （并列时取更强的）那个，其余判为 0。这条约束是这套门控有效的主要原因。

    证据分还乘上一个「离峰多近」的权重：贴着峰的那一拍拿满分，
    偏离 0.3 秒的只能拿一半。
    """
    if pose is None or times is None or len(times) == 0:
        return np.zeros(0, dtype=np.float32)
    ev = np.zeros(len(times), dtype=np.float32)
    idx, prom = swing_peaks(pose)
    if idx.size == 0:
        return ev
    peak_t = idx.astype(np.float64) / max(pose.fps, EPS)
    order = np.argsort(peak_t)
    peak_t, prom = peak_t[order], prom[order]

    # 每个击球先找最近的峰
    pos = np.searchsorted(peak_t, np.asarray(times, dtype=np.float64))
    best_j = np.full(len(times), -1, dtype=np.int64)
    best_dt = np.full(len(times), np.inf, dtype=np.float64)
    for k in range(len(times)):
        for j in (pos[k] - 1, pos[k]):
            if 0 <= j < peak_t.size:
                dt = abs(peak_t[j] - float(times[k]))
                if dt < best_dt[k]:
                    best_dt[k], best_j[k] = dt, j
    ok = (best_j >= 0) & (best_dt <= window)
    if one_to_one and np.any(ok):
        # 同一个峰上的多个候选，只留最近的一个（并列时留强度更高的）
        strengths = (np.asarray(strengths, dtype=np.float64)
                     if strengths is not None and len(strengths) == len(times)
                     else np.zeros(len(times)))
        for j in np.unique(best_j[ok]):
            group = np.nonzero(ok & (best_j == j))[0]
            if group.size <= 1:
                continue
            key = best_dt[group] - 1e-3 * strengths[group]     # 近优先，其次强度
            keep = int(group[int(np.argmin(key))])
            for k in group:
                if k != keep:
                    ok[k] = False
    ev[ok] = prom[best_j[ok]] * (1.0 - 0.5 * (best_dt[ok] / max(window, EPS)))
    return np.clip(ev, 0.0, 1.0).astype(np.float32)


def gate_hits(
    hits,
    pose: PoseSignal | None,
    threshold: float = 0.22,
    min_keep_ratio: float = 0.12,
    max_keep_ratio: float = 0.97,
    window: float = 0.35,
):
    """按姿态证据把音频击球筛成「我们这场比赛打的」。

    返回 ``(mask, trace)``：``mask`` 是布尔数组（True = 保留）。

    **它只在证据足够时动手**，任何一条不满足就原样放行 —— 姿态这条路本身也可能
    失效（球员太小、严重遮挡、非比赛素材），而此时「照旧用全部击球」总比
    「用一半击球把回合切碎」好：

    * 姿态覆盖率太低（< 0.35）→ 放行；
    * 保留比例落在 [12%, 97%] 之外 → 说明门限完全没起作用（全留）或者
      把大部分击球都判掉了（多半是姿态信号本身有问题）→ 放行。
    """
    trace: dict[str, Any] = {}
    if hits is None or hits.times.size == 0 or pose is None:
        return None, trace
    if pose.coverage < 0.35:
        trace["gate_skipped"] = f"姿态覆盖率仅 {pose.coverage:.2f}，未启用击球归属"
        return None, trace
    ev = hit_swing_evidence(pose, hits.times, strengths=hits.strength, window=window)
    mask = ev >= threshold
    ratio = float(np.mean(mask)) if mask.size else 0.0
    trace["evidence_p50"] = round(float(np.percentile(ev, 50)), 3)
    trace["peaks"] = int(swing_peaks(pose)[0].size)
    trace["keep_ratio"] = round(ratio, 3)
    if ratio < min_keep_ratio or ratio > max_keep_ratio:
        trace["gate_skipped"] = (
            f"保留比例 {ratio:.2f} 超出 [{min_keep_ratio:.2f}, {max_keep_ratio:.2f}]，"
            "判定姿态证据不可用，本条未生效")
        return None, trace
    trace["kept"] = int(np.count_nonzero(mask))
    trace["dropped"] = int(mask.size - np.count_nonzero(mask))
    return mask, trace


def filter_hits(hits, mask: np.ndarray):
    """按 ``mask`` 过滤 :class:`~bms.analysis.audio_hits.HitDetection`。"""
    if mask is None or hits is None or hits.times.size == 0:
        return hits
    from .audio_hits import HitDetection

    return HitDetection(
        times=hits.times[mask],
        strength=hits.strength[mask],
        confidence=hits.confidence[mask],
        envelope=hits.envelope,
        env_fps=hits.env_fps,
        threshold=hits.threshold,
        noise_floor_db=hits.noise_floor_db,
    )


# ------------------------------------------------------------------ 缓存


def _boxes_signature(frame_boxes: list) -> str:
    """给逐帧球员框算一个便宜的指纹，用来做缓存键。

    必须带上它：姿态结果是从**球员框**裁出来的，换了尺寸筛选 / 机位之后框会变，
    而视频没变。只按视频哈希的话会命中一份过期缓存，用户会看到「改了参数但
    结果一点没变」这种最难查的问题。抽样就够 —— 每 37 帧取一帧。
    """
    h = hashlib.sha1()
    n = len(frame_boxes)
    h.update(str(n).encode())
    for i in range(0, n, 37):
        for item in (frame_boxes[i] or ()):
            try:
                h.update(("%d:%.4f,%.4f,%.4f,%.4f;" % (int(item[0]), float(item[1]),
                                                       float(item[2]), float(item[3]),
                                                       float(item[4]))).encode())
            except (TypeError, ValueError, IndexError):
                continue
    return h.hexdigest()[:10]


def _cache_path(cache_dir, video_path: str, fps: float, crop: int,
                margin: float, up_shift: float, model_name: str,
                boxes_sig: str = "") -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        st = Path(video_path).stat()
        stamp = f"{video_path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        stamp = str(video_path)
    key = (f"{stamp}|{fps:.3f}|{crop}|{margin:.3f}|{up_shift:.3f}|"
           f"{model_name}|{boxes_sig}")
    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return d / f"pose_{tag}.npz"


def _load_cache(path: Path, n: int) -> PoseSignal | None:
    if path is None or not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if int(z["swing"].size) != n:
            return None
        return PoseSignal(
            fps=float(z["fps"]), duration=float(z["duration"]),
            swing=z["swing"].astype(np.float32),
            ok=z["ok"].astype(np.float32),
            overhead=z["overhead"].astype(np.float32),
            coverage=float(z["coverage"]), quiet=float(z["quiet"]),
            trace={"coverage": round(float(z["coverage"]), 3)},
        )
    except Exception:
        return None


def _save_cache(path: Path, sig: PoseSignal) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fps=sig.fps, duration=sig.duration,
                            swing=sig.swing, ok=sig.ok, overhead=sig.overhead,
                            coverage=sig.coverage, quiet=sig.quiet)
    except Exception:
        pass


# ------------------------------------------------------------------ 环境


def _data_paths() -> tuple[Path, Path]:
    try:
        from ..config import MODELS_DIR  # type: ignore

        return MODELS_DIR.parent / "data" / "yolo", MODELS_DIR
    except Exception:
        root = Path(__file__).resolve().parents[3]
        return root / "data" / "yolo", root / "models"


def default_cache_dir() -> Path | None:
    """默认的姿态缓存目录（``data/cache/pose``）。"""
    try:
        from ..config import CACHE_DIR  # type: ignore

        return CACHE_DIR / "pose"
    except Exception:
        return None


def _resolve_weights(model_name: str) -> str | None:
    """优先用仓库 ``models/`` 下的本地权重；没有就交给 ultralytics 自己下载。"""
    _, models_dir = _data_paths()
    p = Path(model_name)
    if p.is_absolute() and p.exists():
        return str(p)
    local = models_dir / p.name
    if local.exists():
        return str(local)
    return model_name


def _pick_device(device: str) -> str:
    if device not in ("cuda", "cpu"):
        return device
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return device


__all__ = [
    "PoseSignal",
    "analyze_pose",
    "filter_hits",
    "gate_hits",
    "hit_swing_evidence",
    "swing_peaks",
]
