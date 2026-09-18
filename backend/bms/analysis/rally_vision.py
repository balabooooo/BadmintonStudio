"""球员运动切分引擎：真正找到「这一回合从哪里开始、在哪里结束」。

为什么需要它
------------
旧的切分完全建立在**整帧融合活跃度**上：把音频击球、整帧运动、球员速度、
羽毛球出现加权求和，再做一次 1.2 秒的滑动平均，然后用「双阈值迟滞状态机」
切区间，最后对超长区间按典型时长递归切分。

这条路在实测素材上是失效的，原因有三个，都是原理性的：

1. **整帧运动里没有回合信息。** 球馆里同时有观众走动、隔壁场地的球、灯光
   变化，它们让整帧运动能量在**任何**时刻都不低。于是活跃度曲线在回合之间
   根本塌不下去，迟滞状态机永远不认为「这一段结束了」。
   实测：30 分钟素材只切出 8 个区间，其中一个是 236 秒、一个是 138 秒。
2. **1.2 秒的平滑把「停顿」抹平了。** 回合之间只有几秒的静默，被 1.2 秒均值
   一平滑再和持续的背景运动相加，谷底就消失了。
3. **于是「-切分」变成了「-均分」。** 超长区间落进 ``_split_by_valley``，
   而该函数只在 ``e - s > target * 1.35`` 时才递归；因为 30 秒的区间确实比
   目标 28 秒长一点，它就会在每个区间内部找最低点再切一刀 ——
   结果就是一堆时长几乎一样的片段（实测 32 个回合平均 20.1 秒、中位数
   19.8 秒，且大量片段**首尾相接**：前一个的 end 正好等于后一个的 start）。
   首尾相接的边界在真实比赛里不可能存在，因为得分之后必然有捡球/换发球的停顿。

本模块的做法
------------
回合的语义是：**发球 → 对拉 → 死球 → 捡球/准备 → 下一次发球**。
球员是唯一贯穿全过程的可靠观测对象：

* 对拉时，两名比赛球员都在**持续快速**移动；
* 死球之后他们会停下来（喘气、看球、捡球），这是**唯一**在每个回合之间
  必然出现的长停顿。

所以真正该用的信号不是「整帧有多吵」，而是「**这两名球员有多久没动了**」。
只需要球员的运动轨迹，不需要知道球在哪里。本模块：

1. 从检测框序列里得到「每一帧该球员移动了多少」（``box_motion``）；
2. 在**局部时间尺度**（默认 4 秒）上做「最低点 + 显著度」分析，
   把曲线切成「活跃段 / 静默段」交替 —— 不需要手动设阈值，
   显著度天然要求谷底相对两边的峰足够低；
3. 用**静默段**（而不是活跃段）定义回合之间的边界。静默段是真实存在的
   物理停顿；活跃段则可能被背景运动污染。
4. 边界回贴：起点退到静默段中真正的「站定」时刻，终点取最后一次明显移动；
5. 最后用**羽毛球是否在飞行**做一次校验（可用时）：一个回合的核心区间里
   应该反复看到球；如果完全看不到球，说明这段其实是捡球/换场。

没有任何球员信号时（检测失败 / 非比赛素材），自动退回
:func:`bms.analysis.rally.segment` 的老路径，行为不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------ 工具


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    """滑动均值（边缘 pad）。"""
    if a.size == 0 or win <= 1:
        return a.astype(np.float32)
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    ap = np.pad(a.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(ap, k, mode="valid")[: a.size].astype(np.float32)


def _robust_norm(a: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros(0, dtype=np.float32)
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_q), np.percentile(a, hi_q)
    if hi - lo < EPS:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resample_to(a: np.ndarray, src_fps: float, dst_fps: float, n: int) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros(n, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-9 and a.size == n:
        return a.astype(np.float32)
    src_t = np.arange(a.size) / max(src_fps, EPS)
    dst_t = np.arange(n) / max(dst_fps, EPS)
    return np.interp(dst_t, src_t, a.astype(np.float32)).astype(np.float32)


def box_motion(boxes_per_frame: list, fps: float, aspect: float = 1.78,
               window: float = 1.0) -> np.ndarray:
    """把「逐帧球员框」变成「球员每秒移动了多少」的信号。

    Args:
        boxes_per_frame: 每帧一个列表，元素是 ``(track_id, x1, y1, x2, y2)``
            归一化坐标；``frame_boxes`` 就是这个结构。
        fps: 该序列的帧率。
        aspect: 画面宽高比，用来把 x 位移换算到「画面高度」尺度。
        window: 求「移动」的时间窗（秒）。用滑动窗内的累计位移而不是逐帧
            位移，是因为球员是「跑一步停一下」——逐帧位移会频繁归零，
            而 1 秒窗内是否移动过才是「有没有在打」的判据。

    Returns:
        长度 = 帧数的数组，单位是「窗内累计位移 / 画面高度」。
    """
    n = len(boxes_per_frame)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ids: set[int] = set()
    for fr in boxes_per_frame:
        for item in fr or ():
            ids.add(int(item[0]))
    if not ids:
        return np.zeros(n, dtype=np.float32)
    col = {tid: k for k, tid in enumerate(sorted(ids))}
    pos = np.full((n, len(ids)), np.nan, dtype=np.float32)   # 归一化 x（已按画面高度换算）
    ys = np.full((n, len(ids)), np.nan, dtype=np.float32)    # 归一化 y
    for i, fr in enumerate(boxes_per_frame):
        for item in fr or ():
            k = col[int(item[0])]
            pos[i, k] = 0.5 * (float(item[1]) + float(item[3])) * aspect
            ys[i, k] = 0.5 * (float(item[2]) + float(item[4]))

    # nan 差分会得到 nan -> 视为「没有位移」（球员被遮挡时不该算作移动）
    step = np.hypot(np.diff(pos, axis=0), np.diff(ys, axis=0))     # 单位：画面高度
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    # 单帧位移超过画面高度 25% 基本是身份交换，丢掉
    step = np.clip(step, 0.0, 0.25)
    step = np.vstack([np.zeros((1, step.shape[1]), dtype=np.float32), step])

    win_f = max(1, int(round(window * fps)))
    # 每个球员在窗口内的累计位移，再取「最活跃的那名球员」
    kern = np.ones(win_f, dtype=np.float32)
    per_player = np.empty_like(step)
    for k in range(step.shape[1]):
        per_player[:, k] = np.convolve(step[:, k], kern, mode="same")
    return per_player.max(axis=1).astype(np.float32)


def detection_coverage(boxes_per_frame: list, fps: float) -> np.ndarray:
    """逐帧「有没有检到比赛球员」，再在时间上做形态学闭运算。

    单人帧也算有效：两名球员里只有一个人被检到时，那个人的运动仍然说明
    「球在飞」。但**整段都检不到人**的时间必须被单独标出来——那段时间
    我们对「球员动没动」一无所知，不能把它当成「球员没动」。
    这正是旧实现容易丢回合的地方：检测失败的一段静默会被当成一次长停顿，
    夹在它两边的真实回合就都被吃掉了。
    """
    n = len(boxes_per_frame)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ok = np.asarray([1.0 if fr else 0.0 for fr in boxes_per_frame], dtype=np.float32)
    # 闭运算：把不超过 1.5 秒的检测空洞填掉（球员被挡住几帧不影响判断）
    from scipy.ndimage import binary_closing

    gap = max(1, int(round(1.5 * fps)))
    closed = binary_closing(ok > 0.5, structure=np.ones(gap * 2 + 1, dtype=bool))
    return closed.astype(np.float32)


def blend_with_activity(player_motion: np.ndarray, coverage: np.ndarray,
                        activity: np.ndarray) -> np.ndarray:
    """在球员信号缺失的时间上，用画面活跃度把信号补起来。

    融合方式是 `covered * player + (1 - covered) * activity`：
    有球员的时候完全信球员（球员运动是干净的「回合内证据」），
    没有球员的时候退回画面活跃度（它不干净，但比「恒为 0」强得多）。
    两者都先归一化到 0~1，所以混合不会引入量纲问题。
    """
    n = player_motion.size
    if n == 0 or activity is None or activity.size != n:
        return player_motion
    cov = np.clip(coverage.astype(np.float32), 0.0, 1.0)
    return (cov * player_motion + (1.0 - cov) * activity).astype(np.float32)


def audio_visual_evidence(
    player_motion: np.ndarray,
    hit_density: np.ndarray | None,
    weight_hits: float = 0.7,
) -> np.ndarray:
    """把「球员在动」和「这一带在连续击球」乘成一条回合证据曲线。

    单看球员运动区分度不够（球员捡球时也走，隔壁场地也在动，实测 AUC≈0.57）；
    单看击球密度会被「隔壁场地恰好也连打几拍」骗到。两者**相乘**要求两件事
    同时成立，正好对应「我们这场比赛正在对拉」：回合间任意一路塌下去，
    证据就塌下去，谷底因此变清晰。

    之所以用 ``(1-weight) + weight*hits`` 而不是直接相乘：直接相乘时只要击球
    密度有一点波动就会把整段证据压没，短回合尤其容易被吃掉；留一个下限让
    「球员确实在快速移动」本身也能支撑起候选，漏检的击球不至于让回合消失。

    两路都先做鲁棒归一化，避免量纲差异。
    """
    if player_motion is None or player_motion.size == 0:
        return hit_density if hit_density is not None else np.zeros(0, dtype=np.float32)
    p = _robust_norm(player_motion)
    if hit_density is None or hit_density.size != p.size or not np.any(hit_density):
        return p
    h = _robust_norm(hit_density)
    w = float(np.clip(weight_hits, 0.0, 1.0))
    return _robust_norm(p * ((1.0 - w) + w * h))


# ------------------------------------------------------------------ 静默段检测


@dataclass
class QuietSpan:
    """一段「球员基本没动」的静默区间（帧索引）。"""

    start: int
    end: int
    depth: float          # 谷底相对两侧峰的深度（0~1）
    floor: float          # 谷底的绝对水平


def find_quiet_spans(
    m: np.ndarray,
    fps: float,
    min_quiet: float = 1.0,
    max_quiet: float = 0.0,
    prominence_ratio: float = 0.30,
) -> list[QuietSpan]:
    """在球员运动曲线上找「静默段」。

    做法是**最低点 + 显著度**，而不是「低于某个阈值」：

    1. ``scipy.signal.find_peaks`` 找局部最低点（对负曲线找峰）；
    2. 每个最低点算显著度（prominence）——谷底相对两侧较高的那个「鞍部」
       下降了多深。显著度按全片运动强度的某个分位数归一化，
       于是「安静球馆里的停顿」和「嘈杂球馆里的停顿」用同一套参数都能抓到；
    3. 显著度不足的最低点直接丢掉（那不是停顿，只是运动强度的正常起伏）；
    4. 相邻的最低点如果离得比 ``min_quiet`` 还近，只留更深的那个。

    Args:
        m: 球员运动曲线（``box_motion`` 的输出）。
        fps: 帧率。
        min_quiet: 两个候选低谷之间至少隔多久才认为是「两次停顿」。
        max_quiet: 单个静默段的最长时长（秒）；0 = 不限。超长的静默段说明
            这段可能根本没有比赛（休息、换场），交给调用方按最大静默截断。
        prominence_ratio: 显著度门限 = 该比例 × (p95 - p20)。默认 0.30 表示
            谷底至少要比「典型活跃水平」低三成。

    Returns:
        按时间排序的 :class:`QuietSpan` 列表。
    """
    if m is None or m.size < max(8, int(fps)):
        return []
    from scipy.signal import find_peaks, peak_prominences

    base = float(np.percentile(m, 20))
    top = float(np.percentile(m, 95))
    span = max(top - base, EPS)
    min_prom = prominence_ratio * span

    dist = max(1, int(round(min_quiet * fps)))
    idx, props = find_peaks(-m, distance=dist, prominence=min_prom)
    if idx.size == 0:
        return []
    prom = props.get("prominences")
    if prom is None:
        prom = peak_prominences(-m, idx)[0]

    quiet = m <= base + 0.35 * span
    out: list[QuietSpan] = []
    for k, i in enumerate(idx):
        # 低谷向两侧扩展到「不再属于静默」为止，得到静默段的宽度
        a = i
        while a > 0 and quiet[a - 1]:
            a -= 1
        b = i
        while b + 1 < m.size and quiet[b + 1]:
            b += 1
        # 静默段至少要有一点点宽度，否则只是曲线的一次抖动
        if b - a < max(1, int(0.25 * fps)):
            a = max(0, i - int(0.15 * fps))
            b = min(m.size - 1, i + int(0.15 * fps))
        out.append(QuietSpan(start=a, end=b,
                             depth=float(np.clip(prom[k] / span, 0.0, 1.5)),
                             floor=float(m[i])))
    if max_quiet > 0:
        lim = int(round(max_quiet * fps))
        for q in out:
            if q.end - q.start > lim:
                mid = (q.start + q.end) // 2
                q.start, q.end = mid - lim // 2, mid + lim // 2
    return out


# ------------------------------------------------------------------ 主切分


@dataclass
class SegmentSignals:
    """切分用到的全部信号（都在同一帧率上）。"""

    fps: float
    duration: float
    #: 球员运动（0~1 归一化后的同一尺度）
    player_motion: np.ndarray
    #: 融合活跃度（旧路径的产物，用作后备证据）
    activity: np.ndarray | None = None
    #: 每帧羽毛球是否在飞行（0~1），可为空
    shuttle: np.ndarray | None = None
    #: 音频击球时刻（秒），可为空
    hit_times: np.ndarray | None = None
    #: 是否真的拿到了球员信号
    has_players: bool = False
    #: 逐帧「球员检测是否有效」（``detection_coverage``）
    coverage: np.ndarray | None = None
    #: 球员检测的有效时间占比
    coverage_ratio: float = 0.0


@dataclass
class SegmentOptions:
    min_rally: float = 2.0
    max_rally: float = 120.0
    #: 两个静默谷之间至少要隔多久才当作「两次停顿」。旧默认 1.0s 配上海量
    #: 噪声会在一次对拉内部切出一堆假边界，所以这里不能取太小。
    min_quiet: float = 0.7
    #: 静默谷的显著度门限（相对 p95-p20）。旧默认 0.30 太高：多球场素材里
    #: 球员捡球时也在走动，谷本来就浅，于是大半回合之间的停顿被判成「不是谷」，
    #: 相邻回合被粘在一起。
    prominence_ratio: float = 0.18
    #: 静默段里「站定」之后还要往前留多久（接发球准备动作）
    pre_roll: float = 1.0
    #: 回合结束后往后留多久（球落地后的收势）
    post_roll: float = 1.6
    #: 静默段短于这个长度就不算「回合结束」（避免把一次长停顿当成回合边界）
    min_rest: float = 0.8
    #: 允许的最短回合，比它短的候选丢掉。旧默认 2.5s 在业余素材上有「刀刃
    #: 效应」：实测有整段因为最长连续移动段 2.42s（差 0.08s）被否掉。
    #: 业余回合里球员「站着看球」的瞬间很多，连续移动段本来就短。
    min_core: float = 1.0


@dataclass
class RawSegment:
    start: float
    end: float
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def segment_by_player_motion(
    sig: SegmentSignals,
    opt: SegmentOptions | None = None,
) -> list[RawSegment]:
    """用「球员运动 + 静默段」切出回合。

    区间定义：从**一段静默结束之后的第一次明显移动**开始，到**下一次静默
    开始之前的最后一次明显移动**结束。这样得到的就是「球在飞」的时间窗，
    不含捡球的走动，也不会跨过两次得分之间的停顿。
    """
    opt = opt or SegmentOptions()
    m = sig.player_motion
    fps = sig.fps
    n = m.size
    if n == 0 or not sig.has_players:
        return []

    quiet = find_quiet_spans(m, fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio)
    if not quiet:
        return []

    base = float(np.percentile(m, 20))
    top = float(np.percentile(m, 95))
    span = max(top - base, EPS)
    # 「明显移动」的门限：静默地板 + 25% 动态范围
    move_thr = base + 0.25 * span
    moving = m >= move_thr

    # 球员检测长时间失效的时间段：那里的「静默」是假的，不能拿它当边界
    unknown = None
    if sig.coverage is not None and sig.coverage.size == n:
        unknown = sig.coverage < 0.5
    first_known = 0
    if unknown is not None:
        known_idx = np.nonzero(~unknown)[0]
        if known_idx.size == 0:
            return []
        first_known = int(known_idx[0])

    def _boundary_ok(i: int) -> bool:
        if unknown is None:
            return True
        a = max(0, i - int(round(0.75 * fps)))
        b = min(n, i + int(round(0.75 * fps)) + 1)
        return bool(np.any(~unknown[a:b]))

    def _add(segs_out: list[RawSegment], lo: int, hi: int, meta: dict) -> None:
        if hi <= lo or not _boundary_ok(lo) or not _boundary_ok(hi - 1):
            return
        core = _trim_to_motion(moving, lo, hi, m, fps, opt)
        if core is None:
            return
        s_i, e_i, score = core
        if not _boundary_ok(s_i) or not _boundary_ok(e_i):
            return
        start = max(0.0, s_i / fps - opt.pre_roll)
        end = min(sig.duration, (e_i + 1) / fps + opt.post_roll)
        if end - start < opt.min_rally:
            return
        if opt.max_rally > 0 and end - start > opt.max_rally:
            segs_out.extend(_split_long(m, fps, start, end, opt))
            return
        segs_out.append(RawSegment(start=start, end=end, score=score, meta=dict(meta)))

    # ---- 静默段 → 回合：相邻两个静默段之间就是候选回合
    segs: list[RawSegment] = []
    for k in range(len(quiet) - 1):
        q0, q1 = quiet[k], quiet[k + 1]
        if q1.start <= q0.end:
            continue
        rest = (q0.end - q0.start) / fps
        if rest < opt.min_rest:
            continue
        if q1.start - q0.end < int(opt.min_core * fps):
            continue
        _add(segs, q0.end, q1.start,
             {"rest_before": round(rest, 2), "quiet_depth": round(q1.depth, 2)})

    # ---- 片头/片尾：如果视频一开始就在对拉，第一个静默段之前也是一个回合
    first = quiet[0]
    if first.start / fps > opt.min_core and first.start > first_known:
        _add(segs, first_known, first.start, {"head": True})
    last = quiet[-1]
    if sig.duration - last.end / fps > opt.min_core:
        _add(segs, last.end, n, {"tail": True})

    segs = _dedupe(segs)
    return sorted(segs, key=lambda s: s.start)


def _trim_to_motion(moving: np.ndarray, lo: int, hi: int, m: np.ndarray,
                    fps: float, opt: SegmentOptions) -> tuple[int, int, float] | None:
    """在候选区间 ``[lo, hi)`` 里收紧到「真的有移动」的核心区段。

    收紧的作用：静默段的边界是按「低于静默地板」定的，而球员在还没完全站定
    的时候就已经不算静默了；反过来运动段的头尾也常常是缓慢起动 / 惯性收势。

    做法是找「最大的连通团」而不是「第一段运动」：一场多拍对拉里球员会有
    短暂站着看球的一瞬（比如球飞过顶时两个人都抬头不动），如果按「第一段」
    截断，一个回合就会被切成两半。
    """
    if hi <= lo:
        return None
    idx = np.nonzero(moving[lo:hi])[0]
    if idx.size == 0:
        return None
    gap = max(1, int(round(0.35 * fps)))
    # 按「间隔 > gap」切成若干连通团
    splits = np.nonzero(np.diff(idx) > gap)[0]
    bounds = np.concatenate([[0], splits + 1, [idx.size]])
    best: tuple[int, int] | None = None
    best_len = -1
    for k in range(len(bounds) - 1):
        a, b = int(idx[bounds[k]]), int(idx[bounds[k + 1] - 1])
        if b - a > best_len:
            best_len, best = b - a, (a, b)
    if best is None:
        return None
    s_i, e_i = best
    seg = m[lo + s_i: lo + e_i + 1]
    base = float(np.percentile(m, 20))
    span = max(float(np.percentile(m, 95)) - base, EPS)
    score = float(np.clip((float(seg.mean()) - base) / span, 0.0, 1.0))
    # 移动占空比太低说明这段主要是少量走动，不是回合
    duty = float(np.count_nonzero(moving[lo + s_i: lo + e_i + 1])) / max(1, e_i - s_i + 1)
    if duty < 0.35 or (e_i - s_i) / fps < opt.min_core:
        return None
    return lo + s_i, lo + e_i, score


def _split_long(m: np.ndarray, fps: float, start: float, end: float,
                opt: SegmentOptions) -> list[RawSegment]:
    """超长回合按内部最深的静默谷切开。"""
    a, b = int(start * fps), int(end * fps)
    if b - a < int(2 * opt.min_core * fps):
        return [RawSegment(start=start, end=end)]
    inner = find_quiet_spans(m[a:b], fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio * 0.7)
    if not inner:
        return [RawSegment(start=start, end=end)]
    best = max(inner, key=lambda q: q.depth)
    cut = (best.start + best.end) // 2
    if cut < int(opt.min_core * fps) or (b - a) - cut < int(opt.min_core * fps):
        return [RawSegment(start=start, end=end)]
    left = _split_long(m, fps, start, (a + cut) / fps, opt)
    right = _split_long(m, fps, (a + cut) / fps, end, opt)
    return left + right


def _dedupe(segs: list[RawSegment]) -> list[RawSegment]:
    """去掉互相重叠的候选，保留更「有料」的那个；并消除首尾相接。"""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[RawSegment] = [segs[0]]
    for cur in segs[1:]:
        prev = out[-1]
        if cur.start >= prev.end - 0.05:
            out.append(cur)
            continue
        # 重叠：谁更长留谁
        if (cur.end - cur.start) > (prev.end - prev.start):
            out[-1] = cur
    return out


# ------------------------------------------------------------------ 主入口


def segment_visual(
    fps: float,
    duration: float,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    activity: np.ndarray | None = None,
    shuttle_presence: np.ndarray | None = None,
    shuttle_fps: float = 0.0,
    hit_times: np.ndarray | None = None,
    hit_density: np.ndarray | None = None,
    opt: SegmentOptions | None = None,
) -> tuple[list[RawSegment], float]:
    """面向流水线的入口：用球员运动切分，并返回球员检测的有效覆盖率。

    做法上有一个容易忽略但很关键的点：**球员检测失效的时间不能被当成
    「球员没动」**。真实素材里比赛球员常常有一半以上的时间检不到
    （在 960×540 代理上人只有 80~120 像素高），如果直接用球员运动曲线，
    那些空洞会被读成一次长停顿，夹在它两边的真实回合就都被吃掉了。
    实测 5 分钟素材上新旧两种写法分别是 5 个回合和 1 个回合。

    所以这里：
      1. 先算逐帧的「检测是否有效」（``detection_coverage``，带 1.5 秒闭运算）；
      2. 在检测失效的时间上**用画面活跃度补位**（``blend_with_activity``）；
      3. 边界只允许落在检测有效的地方（否则那段「静默」不可信）。

    Returns:
        ``(回合列表, 球员检测覆盖率 0~1)``。覆盖率很低时调用方应当改用
        活跃度切分。
    """
    opt = opt or SegmentOptions()
    n = max(1, int(round(duration * fps)))
    boxes = player_boxes or []
    has_players = bool(boxes) and player_fps > 0 and len(boxes) >= int(fps * 2)
    if not has_players:
        return [], 0.0

    m = box_motion(boxes, player_fps, window=1.0)
    m = _resample_to(m, player_fps, fps, n)
    cov = _resample_to(detection_coverage(boxes, player_fps), player_fps, fps, n)
    coverage_ratio = float(np.mean(cov > 0.5))
    m = _robust_norm(_smooth(m, max(1, int(fps * 0.5))))
    act = _resample_to(activity, fps, fps, n) if activity is not None else None
    if act is not None and np.any(act):
        act = _robust_norm(_smooth(act, max(1, int(fps * 1.2))))
        m = blend_with_activity(m, cov, act)
    # 击球密度是这条素材上区分度最高的一路（见 `audio_visual_evidence`）。
    # 把它乘进球员运动里，让「球员在动」和「这一带在连续击球」同时成立才算回合。
    if hit_density is not None and hit_density.size == n:
        m = audio_visual_evidence(m, hit_density)
    if not np.any(m):
        return [], coverage_ratio

    sig = SegmentSignals(
        fps=fps, duration=duration, player_motion=m,
        activity=act,
        shuttle=(_resample_to(shuttle_presence, shuttle_fps, fps, n)
                 if shuttle_presence is not None and shuttle_fps > 0 else None),
        hit_times=hit_times, has_players=True,
        coverage=cov, coverage_ratio=coverage_ratio,
    )
    segs = segment_by_player_motion(sig, opt)
    if segs:
        segs = verify_with_shuttle(segs, sig, opt)
    return segs, coverage_ratio


def segment_activity(
    activity: np.ndarray,
    fps: float,
    duration: float,
    opt: SegmentOptions | None = None,
    shuttle_presence: np.ndarray | None = None,
    shuttle_fps: float = 0.0,
) -> list[RawSegment]:
    """只用融合活跃度曲线切分（球员信号不可用时的主力方案）。

    与旧路径（``rally.segment``）的区别，也正是旧路径切不准的原因：

    1. **不做无条件的递归等分。** 旧路径只要区间比目标时长长一点点，
       就在区间内部找最低点切一刀；这正是「32 个回合平均 20 秒、
       大量片段首尾相接」的来源。这里只在**谷的显著度足够**时才切，
       而且切点用的是谷中心（真正的停顿）而不是任意最低点。
    2. **显著性自适应。** 显著度门限按 (p95 - p20) 的比例定，不写死绝对值，
       所以安静球馆和嘈杂球馆用同一套参数。
    3. **相邻回合之间必须留出停顿。** 输出不会首尾相接。
    """
    opt = opt or SegmentOptions()
    n = activity.size
    if n < max(8, int(fps * 2)):
        return []
    base = float(np.percentile(activity, 20))
    top = float(np.percentile(activity, 95))
    span = max(top - base, EPS)
    quiet = find_quiet_spans(activity, fps, min_quiet=opt.min_quiet,
                             prominence_ratio=opt.prominence_ratio)

    # 活跃门限：静默地板 + 30% 动态范围
    live_thr = base + 0.30 * span
    live = activity >= live_thr

    def trim(lo: int, hi: int) -> tuple[int, int] | None:
        if hi - lo < int(opt.min_core * fps):
            return None
        idx = np.nonzero(live[lo:hi])[0]
        if idx.size == 0:
            return None
        gap = max(1, int(round(0.8 * fps)))
        splits = np.nonzero(np.diff(idx) > gap)[0]
        bounds = np.concatenate([[0], splits + 1, [idx.size]])
        best: tuple[int, int] | None = None
        best_len = -1
        for k in range(len(bounds) - 1):
            a, b = int(idx[bounds[k]]), int(idx[bounds[k + 1] - 1])
            if b - a > best_len:
                best_len, best = b - a, (a, b)
        if best is None or (best[1] - best[0]) / fps < opt.min_core:
            return None
        return lo + best[0], lo + best[1]

    segs: list[RawSegment] = []
    span_bounds: list[tuple[int, int]] = []
    if quiet:
        if quiet[0].start > int(opt.min_core * fps):
            span_bounds.append((0, quiet[0].start))
        for k in range(len(quiet) - 1):
            span_bounds.append((quiet[k].end, quiet[k + 1].start))
        if n - quiet[-1].end > int(opt.min_core * fps):
            span_bounds.append((quiet[-1].end, n))
    else:
        span_bounds.append((0, n))

    for lo, hi in span_bounds:
        core = trim(lo, hi)
        if core is None:
            continue
        s_i, e_i = core
        start = max(0.0, s_i / fps - opt.pre_roll)
        end = min(duration, (e_i + 1) / fps + opt.post_roll)
        if end - start < opt.min_rally:
            continue
        core_act = activity[s_i:e_i + 1]
        score = float(np.clip((float(core_act.mean()) - base) / span, 0.0, 1.0))
        if opt.max_rally > 0 and end - start > opt.max_rally:
            segs.extend(_split_long(activity, fps, start, end, opt))
            continue
        segs.append(RawSegment(start=start, end=end, score=score))

    segs = _dedupe(segs)
    if shuttle_presence is not None and shuttle_fps > 0:
        sig = SegmentSignals(
            fps=fps, duration=duration, player_motion=np.zeros(0), has_players=False,
            shuttle=_resample_to(shuttle_presence, shuttle_fps, fps, n),
        )
        segs = verify_with_shuttle(segs, sig, opt)
    return sorted(segs, key=lambda s: s.start)


def verify_with_shuttle(segs: list[RawSegment], sig: SegmentSignals,
                        opt: SegmentOptions) -> list[RawSegment]:
    """用「羽毛球在不在飞」校验回合。

    羽毛球一旦可见，就说明这一拍确实在打；反过来，如果一个候选回合的核心
    区间里几乎从没看到球，那它更可能是捡球/换场。只在羽毛球信号**足够可靠**
    （全片有一定出现率）时才动手，避免把这个模块变成新的不准确来源。
    """
    sh = sig.shuttle
    if sh is None or sh.size == 0:
        return segs
    cover = float(np.mean(sh > 0.15))
    if cover < 0.05 or cover > 0.9:      # 太少 = 检不到；太多 = 一直是噪声
        return segs
    fps = sig.fps
    out: list[RawSegment] = []
    for s in segs:
        a = max(0, int(s.start * fps))
        b = min(sh.size, max(a + 1, int(s.end * fps)))
        core = sh[a:b]
        if core.size == 0:
            out.append(s)
            continue
        hit_ratio = float(np.mean(core > 0.15))
        s.meta["shuttle_ratio"] = round(hit_ratio, 3)
        if hit_ratio < 0.02:
            # 整段看不到球：不是回合，丢掉
            continue
        # 用「球存在的第一/最后一帧」收紧边界，但保留 pre/post roll 的呼吸
        nz = np.nonzero(core > 0.15)[0]
        if nz.size >= 2:
            s.start = max(s.start, (a + int(nz[0])) / fps - opt.pre_roll * 0.5)
            s.end = min(s.end, (a + int(nz[-1]) + 1) / fps + opt.post_roll * 0.5)
        if s.end - s.start >= opt.min_rally:
            out.append(s)
    return out


__all__ = [
    "QuietSpan",
    "RawSegment",
    "SegmentOptions",
    "SegmentSignals",
    "audio_visual_evidence",
    "blend_with_activity",
    "box_motion",
    "detection_coverage",
    "find_quiet_spans",
    "segment_activity",
    "segment_by_player_motion",
    "segment_visual",
    "verify_with_shuttle",
]
