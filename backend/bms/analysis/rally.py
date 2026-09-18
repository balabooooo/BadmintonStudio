"""回合分割融合引擎。

把音频击球、画面运动、球员活跃度、羽毛球出现这四路互补信号融合成
「这是一次回合」的逐帧置信度，再用带迟滞的状态机切出回合区间。

设计原则
--------
1. **每路信号先做鲁棒归一化**（分位数拉伸），避免不同量纲互相压制。
2. **权重按可信度自适应**：某路信号如果区分度差（正态分布、没有双峰），
   自动降权；这样在「音轨被 AGC 污染」或「画面没检出球员」时仍能工作。
3. **迟滞 + 最短持续时间 + 间隔合并**：避免一次回合被切碎，也避免
   脚步声/观众走动触发假回合。
4. **边界回贴音频击球**：可用时把回合起点对齐到第一次击球（发球）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio_hits import HitDetection, cluster_rallies

EPS = 1e-9


# ------------------------------------------------------------------ 工具


def robust_norm(a: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    """分位数拉伸到 0~1；对离群值不敏感。"""
    if a is None or a.size == 0:
        return np.zeros(0, dtype=np.float32)
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_q), np.percentile(a, hi_q)
    if hi - lo < EPS:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def smooth(a: np.ndarray, win: int) -> np.ndarray:
    """滑动均值。"""
    if a.size == 0 or win <= 1:
        return a
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(ap, k, mode="valid")[: a.size].astype(np.float32)


def resample(a: np.ndarray, src_fps: float, dst_fps: float, dst_len: int) -> np.ndarray:
    """把信号按时间轴线性重采样到目标长度。"""
    if a is None or a.size == 0:
        return np.zeros(dst_len, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-9 and a.size == dst_len:
        return a.astype(np.float32)
    src_t = np.arange(a.size) / max(src_fps, EPS)
    dst_t = np.arange(dst_len) / max(dst_fps, EPS)
    return np.interp(dst_t, src_t, a.astype(np.float32)).astype(np.float32)


def spikes_to_signal(times: np.ndarray, values: np.ndarray, fps: float, length: int,
                     decay: float = 1.0) -> np.ndarray:
    """把离散事件（击球）转成逐帧信号：命中处赋值，然后指数衰减。"""
    out = np.zeros(length, dtype=np.float32)
    if times is None or times.size == 0:
        return out
    idx = np.clip(np.round(np.asarray(times) * fps).astype(np.int64), 0, max(0, length - 1))
    v = values if values is not None and len(values) == len(times) else np.ones(len(times), dtype=np.float32)
    np.maximum.at(out, idx, np.asarray(v, dtype=np.float32))
    if decay <= 0:
        return out
    # 指数衰减冲击响应：y[i] = max(out[i], decay * y[i-1])
    for i in range(1, length):
        p = out[i - 1] * decay
        if p > out[i]:
            out[i] = p
    return out


def discriminative_power(a: np.ndarray) -> float:
    """估计一路信号「有没有双峰结构」，作为自适应权重。

    用 (p90 - p50) / (p90 - p10) 衡量：数值越集中在上部，说明越像
    「少数时刻明显高」的稀疏事件信号，值得给高权重。
    """
    if a is None or a.size < 32:
        return 0.35
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    if p90 - p10 < EPS:
        return 0.05
    sep = float((p90 - p50) / (p90 - p10))
    # sep≈0.5 表示均匀/对称（无信息），越接近 0.8~0.95 越像脉冲信号
    return float(np.clip((sep - 0.42) / 0.45, 0.05, 1.0))


# ------------------------------------------------------------------ 融合


@dataclass
class FusedSignal:
    fps: float
    duration: float
    activity: np.ndarray                 # 融合后的活跃度 0~1
    components: dict[str, np.ndarray] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    threshold_hi: float = 0.5
    threshold_lo: float = 0.3
    #: 音频击球信号的可信度 0~1（多球场/AGC 污染时接近 0）
    audio_reliability: float = 1.0


@dataclass
class RallyInterval:
    start: float
    end: float
    confidence: float = 0.5
    #: 区间内命中的音频击球索引
    hit_indices: list[int] = field(default_factory=list)
    serve_time: float | None = None
    serve_side: str = "unknown"
    receive_time: float | None = None
    receive_side: str = "unknown"
    features: dict[str, float] = field(default_factory=dict)


def hit_reliability(hits: HitDetection | None, duration: float, fps: float,
                    n: int) -> tuple[float, float]:
    """评估音频击球信号到底有多可信。

    多球场球馆 + 相机 AGC 会让「击球声」在整个时间轴上均匀出现，这种信号
    几乎不含回合起止信息，强行加权重会把单独一个回合粘成几分钟。

    返回 ``(reliability 0~1, gate 强度分位)``：

    - ``coverage``：被「2 秒内有击球」覆盖的时间比例。真实比赛通常在
      0.25~0.55；若 > 0.8 说明几乎一直在响，基本是噪声或其他场地。
    - ``rate``：平均每秒触球数。羽毛球单场地现实上限约 2.5 次/秒。
    - ``burst``：相邻间隔 < 0.35s 的击球占比，真实回合里很高。
    """
    if hits is None or hits.times.size < 8 or duration <= 0:
        return 0.0, 0.0
    t = hits.times
    rate = t.size / max(duration, 1e-6)
    # 覆盖率：把时间轴切成 2 秒格，统计含击球的格子比例
    nb = max(1, int(np.ceil(duration / 2.0)))
    idx = np.clip((t / 2.0).astype(np.int64), 0, nb - 1)
    coverage = float(np.unique(idx).size / nb)
    gaps = np.diff(t)
    burst = float(np.mean(gaps < 0.35)) if gaps.size else 0.0

    rate_ok = float(np.clip((3.2 - rate) / 2.2, 0.05, 1.0))       # >3.2/s 判定为噪声
    cover_ok = float(np.clip((0.86 - coverage) / 0.36, 0.05, 1.0))  # >0.86 判定为噪声
    burst_ok = float(np.clip(burst / 0.35, 0.2, 1.0))
    rel = float(np.clip(rate_ok * cover_ok * (0.55 + 0.45 * burst_ok), 0.0, 1.0))

    # 强度门限：只保留相对较强的那部分击球
    gate = float(np.percentile(hits.strength, 35)) if hits.strength.size else 0.0
    return rel, gate


def fuse(
    fps: float,
    duration: float,
    hits: HitDetection | None = None,
    motion: dict[str, np.ndarray] | None = None,
    players: dict[str, np.ndarray] | None = None,
    shuttle: dict[str, np.ndarray] | None = None,
    roi_activity: np.ndarray | None = None,
) -> FusedSignal:
    """把各路基线信号融合成单一活跃度曲线。"""
    n = max(1, int(round(duration * fps)))
    comp: dict[str, np.ndarray] = {}

    # --- 音频：击球密度 + 强度（先做可信度门控与强度过滤）
    audio_rel = 0.0
    if hits is not None and hits.times.size:
        audio_rel, gate = hit_reliability(hits, duration, fps, n)
        keep = (hits.strength >= gate) & (hits.confidence >= 0.15)
        times = hits.times[keep]
        strength = hits.strength[keep]
        if times.size:
            density = spikes_to_signal(times, strength, fps, n, decay=0.93)
            cnt = np.zeros(n, dtype=np.float32)
            idx = np.clip(np.round(times * fps).astype(np.int64), 0, n - 1)
            np.add.at(cnt, idx, 1.0)
            dens = smooth(cnt, max(1, int(fps * 2.0)))
            comp["audio_hits"] = 0.6 * robust_norm(dens) + 0.4 * robust_norm(density)
        else:
            comp["audio_hits"] = np.zeros(n, dtype=np.float32)
    else:
        comp["audio_hits"] = np.zeros(n, dtype=np.float32)

    # --- 画面运动
    if motion:
        mfps = float(motion.get("fps", fps))
        court = motion.get("court_motion")
        if court is None or not np.any(court):
            court = motion.get("motion", np.zeros(0))
        m = resample(court, mfps, fps, n)
        comp["motion"] = robust_norm(smooth(m, max(1, int(fps * 0.8))))
    else:
        comp["motion"] = np.zeros(n, dtype=np.float32)

    # --- 球员活跃度
    if players:
        pfps = float(players.get("fps", fps))
        cnt = resample(players.get("active_count", np.zeros(0)), pfps, fps, n)
        spd = resample(players.get("active_speed", np.zeros(0)), pfps, fps, n)
        mspd = resample(players.get("max_speed", np.zeros(0)), pfps, fps, n)
        on_court = np.clip(cnt / 2.0, 0.0, 1.0)
        # 球员速度是脉冲式的：跑一步停下来、再跑一步。所以「最近几秒里
        # 有多少时间在快速移动」比瞬时平均速度更能区分「正在对拉」和「在走动」。
        thr = float(np.percentile(spd, 72)) if spd.size else 0.0
        burst = smooth((spd > thr).astype(np.float32), max(1, int(fps * 3.0)))
        comp["players"] = (
            0.34 * robust_norm(smooth(spd, max(1, int(fps * 1.2))), 10, 92)
            + 0.34 * robust_norm(burst)
            + 0.18 * robust_norm(smooth(mspd, max(1, int(fps * 0.8))), 10, 92)
            + 0.14 * robust_norm(smooth(on_court * spd, max(1, int(fps * 2.0))), 10, 92)
        ).astype(np.float32)
    else:
        comp["players"] = np.zeros(n, dtype=np.float32)

    # --- 羽毛球出现
    if shuttle:
        sfps = float(shuttle.get("fps", fps))
        pres = resample(shuttle.get("presence", np.zeros(0)), sfps, fps, n)
        spd = resample(shuttle.get("max_candidate_speed", np.zeros(0)), sfps, fps, n)
        comp["shuttle"] = (0.6 * robust_norm(smooth(pres, max(1, int(fps * 0.5)))) +
                           0.4 * robust_norm(smooth(spd, max(1, int(fps * 0.5))))).astype(np.float32)
    else:
        comp["shuttle"] = np.zeros(n, dtype=np.float32)

    # --- 外部 ROI 活动
    if roi_activity is not None and roi_activity.size:
        rfps = float(motion.get("fps", fps)) if motion else fps
        comp["roi"] = robust_norm(resample(roi_activity, rfps, fps, n))
    else:
        comp["roi"] = np.zeros(n, dtype=np.float32)

    # --- 自适应权重（音频额外乘上可信度）
    base = {"players": 1.35, "motion": 1.0, "audio_hits": 0.95, "shuttle": 0.9, "roi": 0.8}
    weights: dict[str, float] = {}
    for k, v in comp.items():
        if not np.any(v):
            weights[k] = 0.0
            continue
        w = base.get(k, 0.5) * (0.35 + 0.65 * discriminative_power(v))
        if k == "audio_hits":
            w *= audio_rel
        weights[k] = w
    total = sum(weights.values())
    if total < EPS:
        # 所有信号都不可用时退化成「整段都是候选」，由后续音频/人工修正
        weights = {"audio_hits": 1.0}
        total = 1.0
        comp["audio_hits"] = np.ones(n, dtype=np.float32) * 0.5

    activity = np.zeros(n, dtype=np.float32)
    for k, v in comp.items():
        activity += (weights[k] / total) * v
    activity = smooth(activity, max(1, int(fps * 1.2)))

    # --- 自适应阈值（双阈值迟滞）
    p_hi = float(np.percentile(activity, 78))
    p_lo = float(np.percentile(activity, 55))
    med = float(np.median(activity))
    hi = max(p_hi, med * 1.25)
    lo = max(p_lo * 0.92, med * 1.05)
    return FusedSignal(fps=fps, duration=duration, activity=activity,
                       components=comp, weights=weights, threshold_hi=hi, threshold_lo=lo,
                       audio_reliability=audio_rel)


def _fps_of(sig: dict | None, key: str, default: float) -> float:
    if not sig:
        return default
    try:
        return float(sig.get("fps", default))
    except Exception:
        return default


# ------------------------------------------------------------------ 状态机


def segment(
    sig: FusedSignal,
    gap_seconds: float = 3.0,
    min_seconds: float = 3.0,
    max_seconds: float = 120.0,
    min_on_seconds: float = 0.7,
    min_off_seconds: float = 1.6,
    pad_start: float = 0.0,
    pad_end: float = 0.0,
    target_seconds: float = 28.0,
    split_sensitivity: float = 0.5,
) -> list[RallyInterval]:
    """带迟滞的状态机 + 按「典型回合时长」二次切分。

    为什么需要二次切分：训练/多球练习里球员在两个回合之间只停顿几秒，
    活跃度曲线不会塌到低阈值以下，迟滞状态机就会把连续几个回合粘成一条
    几十秒甚至上百秒的「回合」。所以对超长区间再按**活跃度最低点**递归切，
    直到每段不超过 ``target_seconds`` 的合理倍数。

    ``split_sensitivity`` 0~1 控制积极程度：越大目标时长越短、退出阈值越高。
    """
    a = sig.activity
    fps = sig.fps
    n = a.size
    if n == 0:
        return []

    s = float(np.clip(split_sensitivity, 0.0, 1.0))
    # 目标时长：保守 48s -> 积极 16s
    target = float(np.clip(target_seconds, 8.0, 180.0)) * (1.35 - 0.7 * s)
    target = float(np.clip(target, 8.0, 180.0))

    hi = sig.threshold_hi * (1.0 - 0.12 * s)
    lo = sig.threshold_lo * (1.0 + 0.28 * s)
    on_need = max(1, int(round(min_on_seconds * fps)))
    # 越积极越早判定「这一段结束了」
    off_need = max(1, int(round(min_off_seconds * fps * (1.35 - 0.7 * s))))

    intervals: list[tuple[int, int]] = []
    state = 0
    on_run = off_run = 0
    start_i = 0
    for i in range(n):
        if state == 0:
            on_run = on_run + 1 if a[i] >= hi else 0
            if on_run >= on_need:
                state = 1
                start_i = i - on_run + 1
                off_run = 0
        else:
            off_run = off_run + 1 if a[i] < lo else 0
            if off_run >= off_need:
                intervals.append((start_i, i - off_run + 1))
                state = 0
                on_run = 0
    if state == 1:
        intervals.append((start_i, n - 1))

    if not intervals:
        return []

    # 距离很近的区间合并（同样受积极程度影响）
    merged: list[list[int]] = [list(intervals[0])]
    gap_frames = int(round(max(0.6, gap_seconds * (1.3 - 0.75 * s)) * fps))
    for s0, e0 in intervals[1:]:
        if s0 - merged[-1][1] <= gap_frames:
            merged[-1][1] = e0
        else:
            merged.append([s0, e0])

    out: list[RallyInterval] = []
    min_f = int(round(min_seconds * fps))
    max_f = int(round(max_seconds * fps)) if max_seconds > 0 else 10**9
    target_f = int(round(target * fps))
    for s0, e0 in merged:
        if e0 - s0 < min_f:
            continue
        for ps, pe in _split_by_valley(sig, s0, e0, target_f, max(min_f, int(target_f * 0.35))):
            if pe - ps < min_f:
                continue
            conf = float(np.clip(np.mean(a[ps : pe + 1]) / max(sig.threshold_hi, EPS), 0, 1.5)) / 1.5
            out.append(RallyInterval(
                start=max(0.0, ps / fps - pad_start),
                end=min(sig.duration, (pe + 1) / fps + pad_end),
                confidence=float(np.clip(conf, 0.05, 1.0)),
            ))
    return out


def _split_by_valley(sig: FusedSignal, s: int, e: int, target_f: int, min_f: int) -> list[tuple[int, int]]:
    """把过长的区间在「局部活跃度最低点」递归切开。

    切点选最低点而不是等分，是为了尽量落在「捡球 / 擦汗」的空档上；
    同时在切点附近留一点边距，避免把一拍的收尾切掉。
    """
    if e - s <= target_f * 1.35 or target_f <= 0:
        return [(s, e)]
    a = sig.activity
    rel = int(np.argmin(a[s + min_f : e - min_f])) + s + min_f if (e - min_f) - (s + min_f) > 2 else (s + e) // 2
    if rel <= s + min_f or rel >= e - min_f:
        rel = (s + e) // 2
    left = _split_by_valley(sig, s, rel, target_f, min_f)
    right = _split_by_valley(sig, rel, e, target_f, min_f)
    return left + right


# ------------------------------------------------------------------ 边界回贴与发球识别


def thin_shots(times: np.ndarray, strength: np.ndarray, confidence: np.ndarray,
               min_gap: float = 0.28) -> np.ndarray:
    """在候选击球里挑出物理上合理的一串。

    羽毛球里同一方两次击球间隔不可能小于约 0.3 秒（业余更慢），
    一段回合里出现「每秒 4 拍」基本一定是检测噪声。这里按强度从高到低
    贪心挑选，保证任意两次入选击球的间隔 ≥ ``min_gap``，再按时间排序。
    """
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    score = strength * (0.5 + 0.5 * np.clip(confidence, 0, 1))
    order = np.argsort(-score)
    chosen: list[int] = []
    for i in order:
        t = times[i]
        if all(abs(t - times[j]) >= min_gap for j in chosen):
            chosen.append(int(i))
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.array(sorted(chosen), dtype=np.int64)


#: 同一回合内两次击球的**最大**间隔（秒）的保守下限。
#:
#: 依据是物理而不是调参：一回合内的拍间隔由球的飞行时间决定 —— 业余素材实测
#: 段内最大 2.4~3.6 秒，绝大多数落在 0.4~1.5 秒；而两个回合之间必然隔着
#: 捡球 / 换发球 / 走回接发球位置，**没有任何人击球**的时间普遍 ≥4 秒。
#: 所以「击球序列里的大空档」几乎就是回合边界 —— 这比「球员有没有停下来」
#: 可靠得多，因为球员在捡球时也一直在走动，活跃度根本不塌。
MAX_INTRA_HIT_GAP = 3.0

#: 取击球窗口时向两侧多看的秒数。切分给出的边界本身是粗的（可能落在发球
#: 之后、或最后一拍之前），所以要往外看一点才能把发球/收尾那一拍捞回来。
#: 但**不能太大**，否则会把邻居回合的拍吃进来（见 :func:`refine_with_hits`）。
_HIT_TOL_BEFORE = 1.0
_HIT_TOL_AFTER = 1.2


def hit_gap_limit(
    times: np.ndarray,
    floor: float = MAX_INTRA_HIT_GAP,
    mult: float = 1.8,
    quantile: float = 70.0,
    cap: float = 4.5,
) -> float:
    """估计「同一回合内允许的最大拍间隔」。

    取一个**典型**拍间隔（p70）再乘以余量，而不是取高分位数：分位数取得越高，
    越容易被「回合之间那些几秒到几十秒的大空档」自己抬上去 ——
    实测门控之后（只剩我们自己的击球）p80 就已经到 2.42 秒，
    乘 1.8 得到 4.36 秒的门限，于是 3.7 秒和 3.1 秒的真实停顿全都切不开，
    20 多秒的回合就这么留在那里。

    ``floor`` 是物理下限（一回合内的拍间隔由球的飞行时间决定），
    ``cap`` 防止序列本身很稀疏时门限被放到失效。
    """
    if times is None or len(times) < 3:
        return float(floor)
    g = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    g = g[g > 1e-6]
    if g.size < 2:
        return float(floor)
    return float(min(cap, max(floor, mult * float(np.percentile(g, quantile)))))


def _hit_idx_in_window(hits: HitDetection, t0: float, t1: float,
                       min_confidence: float = 0.18) -> np.ndarray:
    """``[t0, t1]`` 区间里「像我们这场比赛」的击球下标。

    两道过滤：置信度门限 + :func:`thin_shots` 的物理合理性筛选
    （同一方两次击球不可能小于约 0.3 秒，出现「每秒 4 拍」一定是噪声）。
    """
    t = hits.times
    idx = np.nonzero((t >= t0) & (t <= t1))[0]
    if idx.size == 0:
        return idx
    idx = idx[hits.confidence[idx] >= min_confidence]
    if idx.size == 0:
        return idx
    sub = thin_shots(t[idx], hits.strength[idx], hits.confidence[idx])
    return idx[sub]


def split_by_hit_gaps(
    intervals: list[RallyInterval],
    hits: HitDetection | None,
    limit: float | None = None,
    min_side_hits: int = 2,
    min_side_seconds: float = 2.0,
) -> list[RallyInterval]:
    """把「内部含有过大拍间隔」的区间在空档的正中切开。

    这是「一个回合里混进了下一个回合」的正解。旧实现依赖「球员静默段」和
    「活跃度谷值」，两者在多球场球馆里都失效：球员捡球时在走动（活跃度不塌）、
    整帧运动被隔壁场地持续点亮（谷底消失）。于是几个回合被粘成一条几十秒的
    区间。而**球有没有在被击打**是这件事的直接观测：一旦击球序列里出现一个
    远超正常拍间隔的空档，那两段一定是两个回合。

    ``min_side_hits`` / ``min_side_seconds`` 是防误切的护栏：两侧都必须
    真的有足够的拍数、并且各自撑得起一个回合，否则宁可留着不切。
    """
    if hits is None or hits.times.size == 0 or not intervals:
        return intervals
    lim = float(limit) if limit is not None else hit_gap_limit(hits.times)
    out: list[RallyInterval] = []

    def _recurse(iv: RallyInterval, depth: int = 0) -> None:
        if depth > 6 or iv.end - iv.start < 2.0 * min_side_seconds:
            out.append(iv)
            return
        idx = _hit_idx_in_window(hits, iv.start, iv.end)
        if idx.size < 2 * min_side_hits:
            out.append(iv)
            return
        ts = hits.times[idx]
        gaps = np.diff(ts)
        k = int(np.argmax(gaps))
        if gaps[k] <= lim:
            out.append(iv)
            return
        left, right = ts[: k + 1], ts[k + 1:]
        # 护栏：两侧都要够料，否则这不是「两个回合」而是「漏检了几拍」
        if (left.size < min_side_hits or right.size < min_side_hits
                or (left[-1] - left[0]) < min_side_seconds
                or (right[-1] - right[0]) < min_side_seconds):
            out.append(iv)
            return
        cut = float(ts[k] + gaps[k] / 2.0)
        a = RallyInterval(start=iv.start, end=cut, confidence=iv.confidence)
        b = RallyInterval(start=cut, end=iv.end, confidence=iv.confidence)
        for nb in (a, b):
            nb.serve_side = iv.serve_side
            nb.receive_side = iv.receive_side
        _recurse(a, depth + 1)
        _recurse(b, depth + 1)

    for iv in intervals:
        _recurse(iv)
    return sorted(out, key=lambda v: v.start)


def refine_with_hits(
    intervals: list[RallyInterval],
    hits: HitDetection,
    pre_roll: float = 1.0,
    post_roll: float = 1.6,
    tail_seconds: float = 0.9,
    min_hits: int = 1,
    split: bool = True,
    trim_start: bool = True,
    gap_limit: float | None = None,
) -> list[RallyInterval]:
    """用音频击球修正回合边界，并**允许把终点收紧**。

    旧实现在这里写的是 ``iv.end = max(iv.end, last + tail)`` —— 终点只能往后
    推、永远不能往前收。于是无论切分给出的区间有多长（实测有 69 秒、72 拍的
    「一回合」），击球信息都**无法**把球落地之后那一段砍掉。这正是
    「一回合内包含球落地后很长时间」的直接原因。

    现在的规则：

    1. 先按击球序列的大空档把区间切开（:func:`split_by_hit_gaps`），
       消掉「一回合里混进下一个回合」；
    2. 终点**锚到最后一拍**：``iv.end = last + tail_seconds``。
       但只在有证据时才敢收紧 —— 判据是「全局下一次击球离 ``last`` 超过
       ``gap_limit``」：连隔壁场地的声音都没有，说明这一段确实没人打球了。
       若紧接着还有击球，说明球还在飞，就保持原来的更晚终点。
    3. 起点仍然对齐到第一次击球（发球）之前 ``pre_roll``；
       ``trim_start=False`` 用于球员运动切分（它的起点是「球员开始动」，
       本身已经包含了发球准备，不该被推迟）。
    """
    if hits is None or hits.times.size == 0:
        for iv in intervals:
            iv.start = max(0.0, iv.start - pre_roll)
            iv.end = iv.end + post_roll
        return intervals

    t = hits.times
    lim = float(gap_limit) if gap_limit is not None else hit_gap_limit(t)
    if split:
        intervals = split_by_hit_gaps(intervals, hits, limit=lim)

    # 取击球窗口时要**不要越过邻居**：一次击球只能属于一个回合。
    # 不设这个夹逼的话，切点两侧会把同一拍都算进来（左边当成「最后一拍」、
    # 右边当成「第一次击球」），于是左区间往后留 0.9s、右区间往前留 1.2s，
    # 两段直接重叠 2.1 秒 —— 下游 dedupe_overlaps 会在中间切一刀，切完又首尾
    # 相接，最后被 _join_abutting 合并回去，等于白切。
    ordered = sorted(intervals, key=lambda v: v.start)
    bounds: list[tuple[float, float]] = []
    for k, iv in enumerate(ordered):
        prev_end = ordered[k - 1].end if k > 0 else -np.inf
        next_start = ordered[k + 1].start if k + 1 < len(ordered) else np.inf
        lo = max(iv.start - _HIT_TOL_BEFORE, prev_end)
        hi = min(iv.end + _HIT_TOL_AFTER, next_start)
        if hi < lo:
            lo = hi = (iv.start + iv.end) / 2.0
        bounds.append((lo, hi))

    out: list[RallyInterval] = []
    for iv, (lo, hi) in zip(ordered, bounds):
        idx = _hit_idx_in_window(hits, lo, hi)
        iv.hit_indices = idx.tolist()
        if idx.size >= min_hits:
            first = float(t[idx[0]])
            # 「最后一拍」只在区间内部找：窗口右侧多出的 1.6 秒是给
            # 「活动区间结束得比最后一拍早」留的补救余地，不能拿它当终点锚点。
            inside = idx[t[idx] <= iv.end]
            last = float(t[inside[-1]]) if inside.size else first
            anchor = last + max(0.0, tail_seconds)

            # 全局下一次击球离最后一拍多远？超过 gap_limit 就说明这一段真的结束了。
            nxt = int(np.searchsorted(t, last + 1e-6, side="right"))
            gap_after = float(t[nxt] - last) if nxt < t.size else float("inf")
            if gap_after > lim:
                iv.end = min(iv.end, anchor)          # 收紧
            else:
                iv.end = max(iv.end, anchor)          # 后面还有拍，别切掉
            iv.end = max(iv.end, last + 0.05)
            if trim_start:
                iv.start = max(0.0, first - pre_roll)
            else:
                iv.start = min(iv.start, max(0.0, first - pre_roll))
            iv.features["hit_anchored"] = 1.0
            iv.features["tail_gap"] = round(min(gap_after, 999.0), 2)
        else:
            iv.start = max(0.0, iv.start - pre_roll)
            iv.end = iv.end + post_roll
        if idx.size >= 1:
            iv.serve_time = float(t[idx[0]])
        if idx.size >= 2:
            iv.receive_time = float(t[idx[1]])
        out.append(iv)
    return [iv for iv in out if iv.end > iv.start]


def attach_features(
    intervals: list[RallyInterval],
    sig: FusedSignal,
    hits: HitDetection | None = None,
    motion: dict[str, np.ndarray] | None = None,
    players: dict[str, np.ndarray] | None = None,
    shuttle: dict[str, np.ndarray] | None = None,
) -> list[RallyInterval]:
    """给每个回合算客观特征，用于后续评分。"""
    fps = sig.fps
    for iv in intervals:
        a, b = int(iv.start * fps), int(iv.end * fps)
        a, b = max(0, a), min(sig.activity.size, max(b, a + 1))
        seg = sig.activity[a:b]
        # 从已有的 features 出发而不是新建一个字典：边界锚定时写进去的
        # hit_anchored / tail_gap 是诊断信息，界面要靠它解释「终点为什么在这儿」。
        # 旧代码在这里直接换了一个新字典，把它们全丢了。
        f: dict[str, float] = dict(iv.features)
        f.update({
            "duration": float(iv.end - iv.start),
            "activity_mean": float(seg.mean()) if seg.size else 0.0,
            "activity_peak": float(seg.max()) if seg.size else 0.0,
            "confidence": float(iv.confidence),
        })
        if hits is not None and iv.hit_indices:
            hs = hits.strength[iv.hit_indices]
            hc = hits.confidence[iv.hit_indices]
            ts = hits.times[iv.hit_indices]
            f["shot_count"] = float(len(iv.hit_indices))
            f["hit_strength_mean"] = float(hs.mean())
            f["hit_strength_p90"] = float(np.percentile(hs, 90))
            f["hit_conf_mean"] = float(hc.mean())
            if len(ts) > 1:
                d = np.diff(ts)
                f["tempo"] = float(1.0 / max(np.median(d), 1e-3))
                f["rally_span"] = float(ts[-1] - ts[0])
                # 回合末段节奏（越密集说明越激烈）
                tail = d[-max(2, len(d) // 3):]
                f["finish_tempo"] = float(1.0 / max(np.median(tail), 1e-3))
        if players:
            pfps = float(players.get("fps", fps))
            f["player_speed_mean"] = _seg_mean(players.get("active_speed"), pfps, iv.start, iv.end)
            f["player_speed_max"] = _seg_max(players.get("max_speed"), pfps, iv.start, iv.end)
            f["active_count_mean"] = _seg_mean(players.get("active_count"), pfps, iv.start, iv.end)
        if motion:
            mfps = float(motion.get("fps", fps))
            f["motion_mean"] = _seg_mean(motion.get("court_motion", motion.get("motion")), mfps, iv.start, iv.end)
            f["motion_peak"] = _seg_max(motion.get("motion"), mfps, iv.start, iv.end)
        if shuttle:
            sfps = float(shuttle.get("fps", fps))
            f["shuttle_presence"] = _seg_mean(shuttle.get("presence"), sfps, iv.start, iv.end)
            f["shuttle_speed_p90"] = _seg_pct(shuttle.get("max_candidate_speed"), sfps, iv.start, iv.end, 90)
            f["shuttle_speed_max"] = _seg_max(shuttle.get("max_candidate_speed"), sfps, iv.start, iv.end)
        iv.features = f
    return intervals


def _slice(arr, fps, t0, t1):
    if arr is None or len(arr) == 0:
        return None
    a = max(0, int(t0 * fps))
    b = min(len(arr), max(a + 1, int(t1 * fps)))
    if b <= a:
        return None
    return np.asarray(arr[a:b], dtype=np.float32)


def _seg_mean(arr, fps, t0, t1) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(s.mean()) if s is not None and s.size else 0.0


def _seg_max(arr, fps, t0, t1) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(s.max()) if s is not None and s.size else 0.0


def _seg_pct(arr, fps, t0, t1, q) -> float:
    s = _slice(arr, fps, t0, t1)
    return float(np.percentile(s, q)) if s is not None and s.size else 0.0


def dedupe_overlaps(intervals: list[RallyInterval], hits: HitDetection | None = None,
                    fps: float = 15.0, activity: np.ndarray | None = None) -> list[RallyInterval]:
    """消除相邻回合之间的重叠。

    边界回贴时每个回合都会向前留 ``pre_roll``、向后留 ``tail``，两个挨得近的
    回合就会互相盖住几秒。直接拿去拼时间线会让同一段画面出现两次。
    这里把重叠区切在「击球间隔最大处」（或活跃度最低点），比简单取中点更自然。
    """
    if len(intervals) < 2:
        return intervals
    intervals = sorted(intervals, key=lambda v: v.start)
    out = [intervals[0]]
    for cur in intervals[1:]:
        prev = out[-1]
        if cur.start >= prev.end - 0.05:
            out.append(cur)
            continue
        a, b = cur.start, min(prev.end, cur.end)
        cut = _best_cut(a, b, hits, fps, activity)
        prev.end = min(prev.end, cut)
        cur.start = max(cur.start, cut)
        if cur.end - cur.start >= 0.5:
            out.append(cur)
    return [iv for iv in out if iv.end - iv.start >= 0.5]


def _best_cut(a: float, b: float, hits: HitDetection | None, fps: float,
              activity: np.ndarray | None) -> float:
    """在 [a, b] 里挑一个最像「两次回合之间」的时刻。"""
    if b - a < 0.25:
        return (a + b) / 2
    if hits is not None and hits.times.size:
        ht = hits.times
        i0, i1 = int(np.searchsorted(ht, a)), int(np.searchsorted(ht, b))
        seg = ht[i0:i1]
        if seg.size >= 2:
            gaps = np.diff(seg)
            k = int(np.argmax(gaps))
            return float(seg[k] + gaps[k] / 2.0)
        if seg.size == 1:
            # 只有一次击球落在重叠区，切在它前面一点
            return float(max(a, min(seg[0] - 0.15, b)))
    if activity is not None and activity.size and fps > 0:
        i0, i1 = max(0, int(a * fps)), min(activity.size, int(b * fps))
        if i1 - i0 >= 2:
            return float((np.argmin(activity[i0:i1]) + i0) / fps)
    return (a + b) / 2


# ------------------------------------------------------------------ 只靠音频的兜底


def fallback_from_audio(hits: HitDetection, params: Any) -> list[RallyInterval]:
    """没有任何视觉信号时，退回纯音频聚类（老式但可用）。"""
    clusters = cluster_rallies(
        hits,
        gap_seconds=getattr(params, "gap_seconds", 3.2),
        min_seconds=getattr(params, "min_rally_seconds", 2.0),
        max_seconds=getattr(params, "max_rally_seconds", 120.0),
    )
    out = []
    for c in clusters:
        out.append(RallyInterval(start=c.start, end=c.end,
                                 confidence=float(np.clip(len(c.hits) / 12.0, 0.15, 1.0)),
                                 hit_indices=list(c.hits)))
    return out
