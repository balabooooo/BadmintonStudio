"""羽毛球（shuttlecock）候选检测与轨迹跟踪 —— 纯 CV，不用神经网络。

设计出发点
----------
机位**完全静止**（本工程素材实测帧间全局位移中位数 0.04 px、最大 < 1 px），
因此不做光流/配准，直接用**时序统计**做背景建模。羽毛球在画面里是一个
**很小的亮白色快速移动点**：4K 原始上约 8~20 px，480p 代理上只有 1~2.5 px。
所以本模块的核心不是"找亮的东西"（球馆里有大量常亮的白线、灯、白墙、白鞋），
而是"找**刚刚才变亮**的**孤立**小白点，并且要求它**连成一条物理上合理的抛物线轨迹**"。

四道串联的筛子
--------------
1. **白色度通道**：``W = min(R, G, B)``。羽毛球是消色差的（三通道同时高），
   而场地是饱和绿（R、B 很低）。实测绿场地 ``W ≈ 33``、白线/白鞋 ``W ≈ 200~255``，
   所以在 ``W`` 上找"亮白点"比在灰度上干净得多（灰度里白线与绿场地的对比只有 ~120，
   ``W`` 上是 ~170，且绿场地的纹理被整体压平）。
2. **时序新颖度**：以该像素在前后各 K 帧窗口里的 85 分位作为参考，
   ``NOV = W - p85``。常亮物体（白线、灯）的 ``NOV ≈ 0``；快速掠过的球 ``NOV`` 很大。
   参考值会先做**全局亮度对齐**，抵消相机自动曝光的漂移
   （本素材开头实测有 +22 灰阶的爬升，不做对齐会在前 15 帧爆出上百个假候选）。
3. **场地先验 + 孤立性**（三条互补的"像不像一个孤立的球"判据）：
   * **周围是绿场地**：候选点自身是**白**的（不能用"绿"去要求它！），
     但它周围一圈的背景必须是绿场地。球馆里的球员皮肤、木墙、地板反光会被
     场地的绿光染色而带上绿偏量，只看候选点本身的颜色是分不开的 ——
     实测只看候选点颜色会有 1144 个候选（1.3 个/帧），改成看周围背景后只剩 68 个
     （0.08 个/帧），是本模块最强的一条降噪判据。
   * **亮度孤立性**：邻居里"与候选点亮度相当"的比例要低。白鞋、白衣服会形成大片白区，
     孤立性差，被丢掉。
   * **novelty 孤立性**：邻居里 novelty 同样高的比例要低。整块白色物体移动时
     novelty 到处都是，孤立小白点只有它自己高。
4. **速度门控 + 抛物线拟合**：羽毛球是全场最快的物体；候选点按速度外推做帧间关联，
   再对轨迹做二次曲线拟合，残差大的丢掉。随机噪声几乎不可能连续 5 帧同时满足
   "位置、速度、曲率"三重要求，这是最后也是最强的过滤器。

已知局限（重要，请如实对待）
----------------------------
* **不是 100% 准确的跟踪**。目标只是提供辅助信号（估计击球时刻、球速、回合激烈程度），
  用于给音频击球检测做交叉验证，不要当作真值。
* **白色球场线是原理性盲区**：球飞到白线上方时 ``W`` 背景本身就是亮的，
  ``NOV`` 接近 0，必然漏检。模块用"静态白区掩码"直接把这些区域排除（宁漏不误）。
* 球贴着球员身体、球拍或白鞋时，与"移动白色大物体"无法区分，会漏检或误检。
* **低码率代理上本模块基本失效 —— 这一点已实测确认，请不要误以为它总能工作。**
  本工程唯一可用的测试素材是 480x270 / 15fps / **441 字节/帧** 的代理：
  8 倍下采样把任何 1~2 px 的小白点都跟绿场地"浆"在一起，导致
  * 候选点的白色度 ``W`` 全部挤在 151~174 的窄带里（球与远场白鞋完全重叠，
    没有任何一个候选 ``W >= 190``，见模块顶部注释的迭代记录）；
  * 平坦场地的时序噪声标准差 p90 就有 11 灰阶、p99 达 37 灰阶，与目标幅度同量级；
  * 该 60 秒里音频检测到 **91 次真实击球**（球确实在飞），
    但灵敏度调到 0.8/1.0 时检出的 presence 帧与真实击球时刻
    **在 0.2 s 内对齐的比例只有 12%/31%，低于随机猜的 ~61%** —— 即
    检出的都是噪声而不是球。
  结论：**这份代理不足以支撑羽毛球的视觉检测**，需要更高分辨率/码率的素材
  （描述里的 4K 原始素材：球有 8~20 px，本模块的设计目标就是它）。
* 仅适用于**静止机位**。手持/摇镜素材必须先做全局配准，否则本模块无意义。
* 速度门控默认上限 0.6 画面高/秒是按规格给的保守值；真实杀球的画面速度远超它，
  在 4K 素材上应把 ``max_speed`` 调到 3~6（并相应放宽 ``max_gap``），否则杀球会被门控丢掉。

迭代记录（在 480x270 测试代理上实测，用于说明这些阈值是怎么来的）
------------------------------------------------------------------
* 第 1 轮：时序中值背景 + 亮度阈值 + 小连通域。**失败**：候选几乎全打在白场线、
  天花板灯和白墙上（一帧 130~180 个候选）。
* 第 2 轮：加入"时序 85 分位新颖度"与全局亮度对齐。开头因自动曝光爬升爆出的
  上百个假候选被压掉，候选降到 p50=0 / p90=2；但 top 事件全部落在**移动的白鞋**上。
* 第 3 轮：加入孤立性判据。第一次写错了 —— 用"候选点自身要偏绿"做场地先验，
  而球是白的、绿偏量≈0，等于把真目标先排除掉了，结果 0 候选；改成
  **"候选点周围一圈背景要是绿场地"** 后，候选从 1144 个（1.3/帧）降到 72 个（0.08/帧）。
* 第 4 轮：加入轨迹级物理约束（最小跨度 + 拟合加速度下限）。走动球员的白鞋
  只能拟合出匀速直线（加速度≈0），被正确拒绝 → 轨迹数 0。
  这正是本素材的诚实结论：剩下的候选里没有球。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------- 公开数据结构


@dataclass
class ShuttlePoint:
    frame: int
    time: float
    x: float          # 归一化 0~1
    y: float          # 归一化 0~1
    score: float      # 0~1 置信度


@dataclass
class ShuttleTrack:
    """一段连续的羽毛球飞行轨迹。"""

    points: list[ShuttlePoint] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    #: 平均/最大像素速度（按画面高度归一化，单位 画面高/秒）
    mean_speed: float = 0.0
    max_speed: float = 0.0
    #: 轨迹起点到终点的总位移（归一化）
    span: float = 0.0
    confidence: float = 0.0


@dataclass
class ShuttleSignal:
    fps: float
    duration: float
    tracks: list[ShuttleTrack] = field(default_factory=list)
    #: 每帧「画面中存在羽毛球」的置信度 0~1
    presence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 每帧检出的候选点数量
    candidate_count: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 每帧所有候选点的最大速度（归一化/秒）
    max_candidate_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------- 常量 / 默认值

#: 阈值标定参考高度。尺度相关阈值都以该高度为基准按工作分辨率等比缩放。
REF_HEIGHT = 480.0
#: 默认时序窗口半径 K（前后各 K 帧参与统计）
DEFAULT_WINDOW = 15
#: 轨迹至少需要多少个点
DEFAULT_MIN_POINTS = 5
#: 关联时允许丢失的最大帧数
DEFAULT_MAX_GAP = 2
#: 速度门控（画面高/秒）。羽毛球是全场最快的物体：低于下限的是静止白点，
#: 高于上限的在物理上不可能是球。高速杀球场景把这个值调大。
DEFAULT_MIN_SPEED = 0.02
DEFAULT_MAX_SPEED = 0.60
#: 抛物线拟合允许的最大 RMS 残差（画面高）
DEFAULT_MAX_RESID = 0.035
#: 轨迹最小跨度（画面高）：短到几乎原地不动的"轨迹"不是球
DEFAULT_MIN_SPAN = 0.04
#: 拟合轨迹的最小加速度（画面高/秒²）。
#: 物理依据：羽毛球在飞行中受重力（~9.8 m/s²）+ 空气阻力，轨迹必然有明显弯曲；
#: 而"在场地上走动的球员的白鞋"只会产生近似匀速的直线。
#: 按典型机位换算（画幅高度约覆盖 8 m），9.8 m/s² ≈ 1.2 画面高/秒²，
#: 一次高远球/平抽的拟合加速度落在 0.3~1.5 之间。默认 0.25 是留了余量的下限。
#: **注意**：这个值依赖机位与画幅覆盖的真实距离，换机位需要标定；
#: 设为 0 可关闭该判据。
DEFAULT_MIN_ACCEL = 0.25
#: 工作分辨率宽度上限。4K 原生逐像素处理太慢且没必要，缩到这个宽度足够
#: （3840 -> 1280 后球仍有 2.5~6.5 px）。
DEFAULT_WORK_WIDTH = 1280
#: 连通域面积的标定范围（@480 行高，单位 px²）
_AREA_MIN_REF = 1.0
_AREA_MAX_REF = 80.0


# ---------------------------------------------------------------- 底层工具


def _white_map(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由 BGR 帧算「白色度」W 与「绿偏量」GEX。"""
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    w = np.minimum(np.minimum(r, g), b)
    gex = g - 0.5 * (r + b)
    return w, gex


def _odd(v: float, lo: int = 3) -> int:
    k = int(round(v))
    if k < lo:
        k = lo
    if k % 2 == 0:
        k += 1
    return k


def _pctl_axis0(stack: np.ndarray, q: float) -> np.ndarray:
    """对 (T, H, W) 沿时间轴求每像素 q 分位。

    用 :func:`numpy.partition`（O(T)）而不是 :func:`numpy.percentile`（要排序），
    在本模块的窗口长度（T≈31）上快 3~4 倍；不做插值，误差远小于阈值余量。
    """
    t = stack.shape[0]
    k = int(round((t - 1) * q / 100.0))
    k = max(0, min(t - 1, k))
    return np.partition(stack, k, axis=0)[k]


def _sensitivity_thresholds(sensitivity: float, scale: float, work_h: int,
                            court_gex: float | None = None) -> dict:
    """把 sensitivity(0~1) 与分辨率换算成一组实际阈值。

    分辨率自适应：面积按 ``scale**2`` 缩放（面积是二维量，所以是平方缩放，
    而不是线性缩放 —— 线性缩放会让高分辨率下的面积上限过小而丢球），
    孤立性窗口与开运算核按 ``scale`` 线性缩放。

    默认值是在本工程的 480x270 测试代理上实测标定的（见模块文档的迭代记录）。
    """
    s = float(np.clip(sensitivity, 0.0, 1.0))
    area_min = max(1, int(round(_AREA_MIN_REF * scale * scale)))
    area_max = max(area_min + 1, int(round(_AREA_MAX_REF * scale * scale)))
    return {
        # 绝对白色度下限。球是白的：实测白线/白鞋 200~255，而球员皮肤、
        # 木墙、地板反光都在 100~160，所以这条把绝大部分杂波挡在外面。
        "w": 170.0 - 40.0 * s,
        # 新颖度阈值（相对时序 85 分位参考）
        "nov": 36.0 - 18.0 * s,
        # 局部对比：候选点必须明显亮于周围一圈的中值
        "contrast": 58.0 - 26.0 * s,
        # 场地先验：候选点**周围一圈**背景的绿偏量中值下限（注意不是候选点自己）
        "bg_gex": 55.0 - 20.0 * s if court_gex is None else float(court_gex),
        # 开运算核尺寸；分辨率太低时目标本身只有 1~2 px，开运算会把它一起吃掉，
        # 所以低分辨率下退化为 0（不做开运算）
        "open": 3 if work_h >= 360 else 0,
        "area_min": area_min,
        "area_max": area_max,
        "iso_win": _odd(11.0 * scale, 5),
        # 亮度孤立性：与候选点亮度相当的邻居占比上限（白鞋/白衣服内部会接近 1）
        "iso_w": 0.20 + 0.30 * s,
        # novelty 孤立性：novelty 同样高的邻居占比上限
        "iso_n": 0.05 + 0.20 * s,
    }


# ---------------------------------------------------------------- 候选检测


def _detect_candidates(
    w: np.ndarray,
    w_ref: np.ndarray,
    gex: np.ndarray,
    static_white: np.ndarray | None,
    mask_roi: np.ndarray | None,
    th: dict,
) -> list[tuple[float, float, float, float]]:
    """单帧候选点提取，返回 [(x, y, score, novelty), ...]（像素坐标）。

    ``gex`` 是**本帧的绿偏量图**，只用来判断候选点**周围**是不是绿场地。
    注意不能用它去要求候选点自身 —— 羽毛球是白色的，绿偏量接近 0，
    早期版本正是栽在这个反了的先验上（详见模块文档的迭代记录）。
    """
    import cv2

    nov = w - w_ref
    m = (nov > th["nov"]) & (w > th["w"])
    if static_white is not None:
        m &= ~static_white
    if mask_roi is not None:
        m &= mask_roi
    if not m.any():
        return []

    m8 = m.astype(np.uint8)
    osz = th["open"]
    if osz >= 3:
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz, osz)))
        if not m8.any():
            return []

    nl, _lab, st, cent = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if nl <= 1:
        return []

    excess = np.maximum(nov, 0.0)
    h, wd = w.shape
    S = th["iso_win"]
    half = S // 2
    out: list[tuple[float, float, float, float]] = []
    for j in range(1, nl):
        area = int(st[j, cv2.CC_STAT_AREA])
        if area < th["area_min"] or area > th["area_max"]:
            continue
        x0, y0 = int(st[j, cv2.CC_STAT_LEFT]), int(st[j, cv2.CC_STAT_TOP])
        x1 = x0 + int(st[j, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(st[j, cv2.CC_STAT_HEIGHT])

        # 亚像素质心：连通域内用新颖度做灰度加权
        sub = excess[y0:y1, x0:x1]
        tot = float(sub.sum())
        if tot <= 1e-6:
            cx, cy = float(cent[j][0]), float(cent[j][1])
        else:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cx = float((sub * xs).sum() / tot)
            cy = float((sub * ys).sum() / tot)

        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < wd and 0 <= iy < h):
            continue
        nov_peak = float(nov[iy, ix])
        lvl = float(w[iy, ix])

        # 以候选点为中心的局部窗口
        ax0, ax1 = max(0, ix - half), min(wd, ix + half + 1)
        ay0, ay1 = max(0, iy - half), min(h, iy + half + 1)
        pw = w[ay0:ay1, ax0:ax1]
        pn = nov[ay0:ay1, ax0:ax1]
        pg = gex[ay0:ay1, ax0:ax1]
        if pw.size < 9:
            continue
        bg = float(np.median(pw))

        # ---- 局部对比：候选点必须明显亮于它周围一圈的中值
        contrast = lvl - bg
        if contrast < th["contrast"]:
            continue

        # 去掉中心 3x3 核心区，避免候选点自己算进"邻居"
        core = np.zeros_like(pw, dtype=bool)
        core[max(0, iy - ay0 - 1):min(pw.shape[0], iy - ay0 + 2),
             max(0, ix - ax0 - 1):min(pw.shape[1], ix - ax0 + 2)] = True
        n_out = float(max(1, int((~core).sum())))

        # ---- 场地先验：周围一圈背景是否为绿场地
        ring = pg[~core]
        if ring.size < 4 or float(np.median(ring)) < th["bg_gex"]:
            continue

        # ---- 亮度孤立性：邻居里"与候选点亮度相当"的比例
        # （阈值取 bg~lvl 的 60% 处，因而与绝对亮度无关，暗背景亮背景都适用）
        iso_w = float(((pw > bg + 0.60 * contrast) & ~core).sum()) / n_out
        if iso_w > th["iso_w"]:
            continue

        # ---- novelty 孤立性：邻居里 novelry 同样高的比例
        # 白鞋/白衣整块移动时 novelty 到处都是，这条把它们整体否掉；
        # 孤立小白点周围 novelty ≈ 0，比值接近 0。
        nbg = float(np.median(pn))
        iso_n = float(((pn > nbg + 0.5 * (nov_peak - nbg)) & ~core).sum()) / n_out
        if iso_n > th["iso_n"]:
            continue

        sn = min(1.0, nov_peak / max(1.0, th["nov"] * 3.0))
        sc = min(1.0, contrast / max(1.0, th["contrast"] * 2.5))
        sw_ = max(0.0, 1.0 - iso_w / max(1e-6, th["iso_w"]))
        sn_ = max(0.0, 1.0 - iso_n / max(1e-6, th["iso_n"]))
        score = float(np.clip(0.35 * sn + 0.30 * sc + 0.15 * sw_ + 0.20 * sn_, 0.0, 1.0))
        out.append((cx, cy, score, nov_peak))
    return out


# ---------------------------------------------------------------- 轨迹关联


def _associate(
    per_frame: dict[int, list[tuple[float, float, float, float]]],
    fps: float,
    height: int,
    min_points: int,
    max_gap: int,
    max_speed: float,
) -> list[list[tuple[int, float, float, float]]]:
    """按「速度外推 + 最近邻」把逐帧候选点串成轨迹。

    返回 ``[[(frame, x, y, score), ...], ...]``，每条已按帧号升序。
    """
    frames = sorted(per_frame)
    active: list[list[tuple[int, float, float, float]]] = []
    done: list[list[tuple[int, float, float, float]]] = []

    for f in frames:
        cands = per_frame[f]
        used = [False] * len(cands)
        # 先处理"停得最久"的轨迹，避免新轨迹把老轨迹的目标抢走
        active.sort(key=lambda tr: tr[-1][0])
        still: list[list[tuple[int, float, float, float]]] = []
        for tr in active:
            gap = f - tr[-1][0]
            if gap > max_gap + 1:          # 断太久，轨迹终结
                if len(tr) >= min_points:
                    done.append(tr)
                continue

            # 用最后两点的速度外推预测位置
            if len(tr) >= 2:
                pf, px, py, _ = tr[-2]
                lf, lx, ly, _ = tr[-1]
                dt = max(1e-6, (lf - pf) / fps)
                step = gap / fps
                pred_x = lx + (lx - px) / dt * step
                pred_y = ly + (ly - py) / dt * step
            else:
                pred_x, pred_y = tr[-1][1], tr[-1][2]

            # 搜索半径由速度上限决定，再给一个 2 px 的地板值防止亚像素抖动被卡死
            radius = max(2.0, max_speed * height * (gap / fps))
            best, best_cost = -1, 1e18
            for k, (cx, cy, sc, _nv) in enumerate(cands):
                if used[k]:
                    continue
                d = float(np.hypot(cx - pred_x, cy - pred_y))
                if d > radius:
                    continue
                cost = d / radius - 0.25 * sc      # 越近越好，候选置信度越高越好
                if cost < best_cost:
                    best, best_cost = k, cost
            if best >= 0:
                used[best] = True
                cx, cy, sc, _nv = cands[best]
                tr.append((f, cx, cy, sc))
            still.append(tr)
        active = still

        # 未被认领的候选点开新轨迹
        for k, (cx, cy, sc, _nv) in enumerate(cands):
            if not used[k]:
                active.append([(f, cx, cy, sc)])

    for tr in active:
        if len(tr) >= min_points:
            done.append(tr)
    return done


def _fit_track(
    pts: list[tuple[int, float, float, float]],
    fps: float,
    height: int,
    width: int,
    min_speed: float,
    max_speed: float,
    max_resid: float,
    min_points: int,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
) -> tuple[ShuttleTrack | None, float]:
    """二次曲线（抛物线）拟合并打分；返回 ``(轨迹, RMS残差/画面高)``。

    物理依据：羽毛球受重力 + 空气阻力，短时段内轨迹在画面里近似抛物线，
    位置对时间做二次拟合即可。除了残差要小，还要求**拟合出的加速度"不能太小"**：
    匀速直线运动（走动的人、缓慢移动的白色物件）虽然残差也小，但它的二次项
    接近 0；真正的球一定带着重力造成的弯曲。这条把"物理上像球"和
    "数学上拟合得好"区分开了。
    """
    n = len(pts)
    if n < min_points:
        return None, 1e9
    t = np.array([p[0] for p in pts], dtype=np.float64) / fps
    x = np.array([p[1] for p in pts], dtype=np.float64)
    y = np.array([p[2] for p in pts], dtype=np.float64)
    sc = np.array([p[3] for p in pts], dtype=np.float64)
    t = t - t[0]

    deg = 2 if n >= 5 else 1
    try:
        cx = np.polyfit(t, x, deg)
        cy = np.polyfit(t, y, deg)
    except Exception:
        return None, 1e9
    rx = x - np.polyval(cx, t)
    ry = y - np.polyval(cy, t)
    resid = float(np.sqrt(np.mean(rx * rx + ry * ry)) / max(1.0, height))
    if resid > max_resid:
        return None, resid

    dt = np.diff(t)
    dist = np.hypot(np.diff(x), np.diff(y))
    speeds = dist / np.maximum(dt, 1e-6) / max(1.0, height)
    mean_speed = float(speeds.mean()) if speeds.size else 0.0
    obs_max_speed = float(speeds.max()) if speeds.size else 0.0
    # 平均速度低于下限 => 静止白点（场地线、灯）而非羽毛球
    if mean_speed < min_speed or mean_speed > max_speed * 1.5:
        return None, resid

    span = float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / max(1.0, height))
    if span < min_span:
        return None, resid

    # 拟合出的加速度（二次项 2a）：球一定被重力"弯"过
    accel = 0.0
    if deg == 2:
        accel = float(np.hypot(2.0 * cx[0], 2.0 * cy[0]) / max(1.0, height))
    if min_accel > 0 and accel < min_accel:
        return None, resid
    # 加速度大得离谱（多半是拟合被噪声带跑）也丢掉
    if accel > 12.0:
        return None, resid

    score_mean = float(sc.mean())
    n_f = min(1.0, n / (2.0 * min_points))
    fit_q = max(0.0, 1.0 - resid / max(1e-6, max_resid))
    plaus = float(np.clip((speeds <= max_speed).mean() if speeds.size else 0.0, 0.0, 1.0))
    acc_q = 1.0 if min_accel <= 0 else float(np.clip(accel / (2.0 * min_accel), 0.0, 1.0))
    conf = float(np.clip(0.28 * score_mean + 0.22 * n_f + 0.28 * fit_q
                         + 0.12 * plaus + 0.10 * acc_q, 0.0, 1.0))

    track = ShuttleTrack(
        points=[
            ShuttlePoint(frame=int(p[0]), time=float(p[0] / fps),
                         x=float(p[1] / width), y=float(p[2] / height), score=float(p[3]))
            for p in pts
        ],
        start=float(pts[0][0] / fps),
        end=float(pts[-1][0] / fps),
        mean_speed=mean_speed,
        max_speed=obs_max_speed,
        span=span,
        confidence=conf,
    )
    return track, resid


# ---------------------------------------------------------------- 主入口


def analyze_shuttle(
    video_path: str,
    sample_fps: float = 30.0,
    roi: tuple[float, float, float, float] | None = None,
    max_seconds: float = 0.0,
    sensitivity: float = 0.5,
    on_progress=None,
    cancel=None,
    *,
    work_width: int = 0,
    window: int = DEFAULT_WINDOW,
    min_points: int = DEFAULT_MIN_POINTS,
    max_gap: int = DEFAULT_MAX_GAP,
    min_speed: float = DEFAULT_MIN_SPEED,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_resid: float = DEFAULT_MAX_RESID,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
    court_gex: float | None = None,
) -> ShuttleSignal:
    """分析视频，返回羽毛球候选轨迹与逐帧辅助信号。

    ``work_width`` 及其后的参数是可选调参项（都有默认值，不影响规格给定的调用方式）。

    Args:
        video_path: 输入视频。**机位必须静止**，否则结果无意义。
        sample_fps: 目标采样帧率；源帧率更低时不会上采样。
        roi: 归一化 ``(x0, y0, x1, y1)``，只在其中找球；None 表示全画面。
        max_seconds: 只分析前 N 秒；0 表示全片。
        sensitivity: 0~1，越大越灵敏（阈值越低、候选越多、噪声也越多）。
        on_progress: ``callable(progress: float, stage: str)``。
        cancel: ``callable() -> bool``，返回 True 时尽快停止并返回已有结果。
        work_width: 工作分辨率宽度；0 表示自动（不超过 :data:`DEFAULT_WORK_WIDTH`）。
        window: 时序窗口半径 K，参考值取前后各 K 帧。
        min_points: 轨迹最少点数，少于该值的轨迹丢弃。
        max_gap: 关联时允许丢失的最大帧数。
        min_speed / max_speed: 速度门控（画面高/秒）。
        max_resid: 抛物线拟合允许的最大 RMS 残差（画面高）。
        min_span: 轨迹最小跨度（画面高）。
        min_accel: 拟合轨迹的最小加速度（画面高/秒²），用于排除匀速直线运动
            （走动的人）；0 表示关闭。
        court_gex: 场地先验阈值（候选点周围一圈背景的绿偏量中值下限）。
            None 表示按 sensitivity 自动取值（s=0.5 时为 45）。设为 -999 可关闭该先验，
            让模块在全画面找球（适合球常常飞在深色天花板/墙体前的机位）。

    Returns:
        :class:`ShuttleSignal`。``fps`` 是**实际采样帧率**（源帧率低时低于 ``sample_fps``）。
    """
    sig, _dbg = _run(
        video_path, sample_fps, roi, max_seconds, sensitivity, on_progress, cancel,
        work_width=work_width, window=window, min_points=min_points, max_gap=max_gap,
        min_speed=min_speed, max_speed=max_speed, max_resid=max_resid,
        min_span=min_span, min_accel=min_accel, court_gex=court_gex, want_debug=False,
    )
    return sig


def analyze_shuttle_debug(
    video_path: str,
    sample_fps: float = 30.0,
    roi: tuple[float, float, float, float] | None = None,
    max_seconds: float = 0.0,
    sensitivity: float = 0.5,
    on_progress=None,
    cancel=None,
    **kw,
) -> tuple[ShuttleSignal, dict]:
    """诊断辅助入口（规格之外的附加函数）：额外返回逐帧候选点与轨迹归属。

    debug 字典含 ``candidates``（每帧归一化 ``(x, y, score)``）、
    ``tracked``（每帧属于有效轨迹的归一化 ``(x, y)``）、``width``/``height``/``step``/``frames``。
    ``analyze_shuttle`` 的行为与返回值不受影响。
    """
    return _run(
        video_path, sample_fps, roi, max_seconds, sensitivity, on_progress, cancel,
        want_debug=True, **kw,
    )


def _run(
    video_path: str,
    sample_fps: float,
    roi: tuple[float, float, float, float] | None,
    max_seconds: float,
    sensitivity: float,
    on_progress,
    cancel,
    *,
    work_width: int = 0,
    window: int = DEFAULT_WINDOW,
    min_points: int = DEFAULT_MIN_POINTS,
    max_gap: int = DEFAULT_MAX_GAP,
    min_speed: float = DEFAULT_MIN_SPEED,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_resid: float = DEFAULT_MAX_RESID,
    min_span: float = DEFAULT_MIN_SPAN,
    min_accel: float = DEFAULT_MIN_ACCEL,
    court_gex: float | None = None,
    want_debug: bool = False,
) -> tuple[ShuttleSignal, dict]:
    import cv2

    def emit(p: float, s: str) -> None:
        if on_progress is not None:
            try:
                on_progress(float(np.clip(p, 0.0, 1.0)), s)
            except Exception:
                pass

    empty = ShuttleSignal(fps=0.0, duration=0.0)
    if not video_path:
        return empty, {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return empty, {}

    # ---- 工作分辨率
    ww = int(work_width) if work_width and work_width > 0 else min(src_w, DEFAULT_WORK_WIDTH)
    ww = max(64, min(ww, src_w))
    scale = ww / float(src_w)
    wh = max(2, int(round(src_h * scale)))
    wf = float(wh) / REF_HEIGHT            # 相对参考高度的缩放
    thr = _sensitivity_thresholds(sensitivity, wf, wh, court_gex)

    step = max(1, int(round(src_fps / max(1e-3, float(sample_fps)))))
    eff_fps = src_fps / step
    limit_frames = int(max_seconds * src_fps) if max_seconds and max_seconds > 0 else (total or 10 ** 9)

    # ---- ROI 掩码
    mask_roi = None
    if roi is not None:
        x0, y0, x1, y1 = roi
        ax0 = int(np.clip(round(min(x0, x1) * ww), 0, ww))
        ax1 = int(np.clip(round(max(x0, x1) * ww), 0, ww))
        ay0 = int(np.clip(round(min(y0, y1) * wh), 0, wh))
        ay1 = int(np.clip(round(max(y0, y1) * wh), 0, wh))
        if ax1 <= ax0 or ay1 <= ay0:
            cap.release()
            return empty, {}
        mask_roi = np.zeros((wh, ww), bool)
        mask_roi[ay0:ay1, ax0:ax1] = True

    K = max(1, int(window))
    WIN = 2 * K + 1
    emit(0.02, "分析羽毛球候选点")

    # 环形缓冲只存 uint8：W = min(R,G,B) 天然在 0..255，GEX 加 128 偏移后也够用
    # （场地先验只关心 gex 是否大于 ~35，裁剪到 [-128,127] 不影响判断）。
    wq: collections.deque = collections.deque(maxlen=WIN)
    gq: collections.deque = collections.deque(maxlen=WIN)
    bright_acc = np.zeros((wh, ww), np.float32)             # 静态白区累计
    n_read = 0          # 已读取的源帧数
    n_samp = 0          # 已采样的帧数（= 输出时间轴长度）
    per_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    top_frames: list[tuple[int, int]] = []

    def process(target_local: int, frame_no: int) -> None:
        """对窗口内第 target_local 帧做候选提取，结果归到时间轴第 frame_no 帧。"""
        stack = np.stack(wq).astype(np.float32)
        gimg = gq[target_local].astype(np.float32) - 128.0
        # 全局亮度对齐，抵消自动曝光漂移
        gm = np.median(stack.reshape(stack.shape[0], -1), axis=1)
        off = gm[target_local]
        if np.any(np.abs(gm - off) > 0.5):
            stack -= (gm - off)[:, None, None]
        w_now = stack[target_local]
        w_ref = _pctl_axis0(stack, 85.0)

        frac = n_samp / max(1.0, float(n_read))
        if frac > 0.15:
            sw = bright_acc > (0.35 * n_read)
            static_white = cv2.dilate(sw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        else:
            static_white = None

        pts = _detect_candidates(w_now, w_ref, gimg, static_white, mask_roi, thr)
        if pts:
            per_frame[frame_no] = pts
            top_frames.append((frame_no, len(pts)))

    idx = 0
    while True:
        if cancel is not None and cancel():
            break
        if not cap.grab():
            break
        if idx % step != 0 and idx != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue
        if n_read >= limit_frames:
            break
        n_read += 1

        if frame.shape[1] != ww or frame.shape[0] != wh:
            frame_w = cv2.resize(frame, (ww, wh), interpolation=cv2.INTER_AREA)
        else:
            frame_w = frame
        w, gex = _white_map(frame_w)
        # GEX 加 128 后存 uint8（省内存）；场地先验只关心它是否超过 ~35
        gq.append(np.clip(gex + 128.0, 0, 255).astype(np.uint8))
        bright_acc += (w > 150.0)
        wq.append(np.clip(w, 0, 255).astype(np.uint8))

        if len(wq) < WIN:
            # 预热：时序参考还没建立，此时任何"新颖度"都不可信。
            # 素材开头实测有自动曝光爬升（+22 灰阶/8 帧），预热期强行出候选
            # 会一次性爆出上百个假点，所以干脆等窗口填满再开始。
            n_samp += 1
            continue

        cur = n_samp                      # 当前帧在时间轴上的序号
        n_samp += 1
        # 窗口满了：目标取正中帧，即 K 帧之前的那一帧
        process(K, cur - K)

        if n_read % 16 == 0:
            p = n_read / float(min(total, limit_frames)) if total else 0.5
            emit(0.02 + 0.73 * min(1.0, p), "分析羽毛球候选点")

    # 收尾：窗口满了以后，最后 K 帧还没被处理过，在最终窗口里从中心往后补完
    done_all = cancel is None or not cancel()
    if done_all and len(wq) == WIN and n_samp > K:
        for r in range(K + 1, WIN):
            process(r, n_samp - (WIN - 1 - r))
    cap.release()
    emit(0.78, "关联轨迹")

    n_frames = n_samp
    duration = n_frames / eff_fps if eff_fps > 0 else 0.0
    presence = np.zeros(n_frames, np.float32)
    cand_count = np.zeros(n_frames, np.float32)
    max_speed_arr = np.zeros(n_frames, np.float32)
    for f, pts in per_frame.items():
        if 0 <= f < n_frames:
            cand_count[f] = len(pts)

    # ---- 逐帧「最大候选速度」：候选点与前一采样帧候选点的最近邻位移
    # 只在物理上合理的搜索半径内才算数（超出的视为「无法关联」，记 0 而不是无穷大），
    # 因此它是一个**下界意义上的**上界估计：真实球速超过门控时这里会被截断。
    gate = max(float(max_speed), 1e-3)
    dt = 1.0 / max(eff_fps, 1e-6)
    radius = gate * float(wh) * dt
    prev_pts: list[tuple[float, float, float, float]] | None = None
    for f in range(n_frames):
        pts = per_frame.get(f)
        if pts and prev_pts:
            pa = np.array([(p[0], p[1]) for p in pts], np.float32)
            pb = np.array([(p[0], p[1]) for p in prev_pts], np.float32)
            d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
            dmin = d.min(axis=1)
            ok_d = dmin[dmin <= radius]
            if ok_d.size:
                max_speed_arr[f] = float(ok_d.max()) / float(wh) / dt
        if pts:
            prev_pts = pts

    # ---- 关联 + 抛物线拟合
    wf_h = float(wh)
    chains = _associate(per_frame, eff_fps, wh, min_points, max_gap, max_speed)
    tracks: list[ShuttleTrack] = []
    tracked_by_frame: dict[int, list[tuple[float, float]]] = {}
    for chain in chains:
        tr, _resid = _fit_track(chain, eff_fps, wh, ww, min_speed, max_speed,
                                max_resid, min_points, min_span, min_accel)
        if tr is None:
            continue
        tracks.append(tr)
        for p in chain:
            tracked_by_frame.setdefault(p[0], []).append((p[1] / ww, p[2] / wf_h))

    tracks.sort(key=lambda t: (-t.confidence, t.start))
    for tr in tracks:
        for p in tr.points:
            if 0 <= p.frame < n_frames:
                presence[p.frame] = max(presence[p.frame], float(np.clip(tr.confidence, 0.0, 1.0)))

    emit(1.0, "完成")
    sig = ShuttleSignal(
        fps=float(eff_fps),
        duration=float(duration),
        tracks=tracks,
        presence=presence,
        candidate_count=cand_count,
        max_candidate_speed=max_speed_arr,
    )
    if not want_debug:
        return sig, {}

    dbg = {
        "candidates": {f: [(p[0] / ww, p[1] / wf_h, p[2]) for p in pts]
                       for f, pts in per_frame.items()},
        "tracked": tracked_by_frame,
        "width": ww,
        "height": wh,
        "step": step,
    }
    return sig, dbg


__all__ = [
    "ShuttlePoint",
    "ShuttleTrack",
    "ShuttleSignal",
    "analyze_shuttle",
    "analyze_shuttle_debug",
]
