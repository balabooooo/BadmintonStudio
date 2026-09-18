"""场地标定与机位识别：让分析不再假设「摄像机在某一块场地的后方」。

问题
----
原来的分析里，机位是被**写死在代码里的**：球员是画面里最大的两个人框、
位置在画面高度 0.40~0.65 的横带里、框越大就越靠近相机、场地是绿色的。
这些先验在「超广角鱼眼、低机位、架在场地后方」这一种拍法上成立，换一个
角度就会静默地给出错误结果（不会报错，只是切得不对、跟错了人）。

本模块提供一块**中立的几何基座**，让下游模块不必再猜：

1. :func:`court_mask`：从颜色先验里找出「场地」区域（绿/蓝/灰/木地板都支持），
   并把它做成一个 mask；
2. :func:`find_court_poly`：从 mask 里拟合出场地边界（**多边形**，点数 4~16）；
3. :func:`estimate_viewpoint`：判断这是哪一种拍法
   （``rear`` 场地后方 / ``side`` 边线 / ``elevated`` 高机位 / ``overhead`` 俯拍）；
4. :func:`CourtCalibration`：给出**单应变换**，把画面坐标投影到「场地平面」，
   于是「谁离相机近」「两人相距多远」「球速多少」都可以在一个与机位无关的
   坐标系里回答。

为什么是**多边形**而不是四边形
------------------------------
2025 版之前标定只有「四个角」这一种表示。这条约定隐含了一个假设：
**场地边界在画面里是直线**。全景相机 / 鱼眼镜头拍出来的场地边界是弯的
（桶形畸变，越靠画面边缘弯得越厉害），于是：

* 用四个角去框一块弯边的场地，要么把边角切掉一块（近端两个角落在画面外），
  要么把场地外的观众区一起圈进来；
* 判定「球员在不在场地里」用的是**矩形**外接框，弯边造成的误差在画面边角
  会被放大到整块看台 —— 而这恰恰是全景素材里背景人员最多的地方。

所以本模块的核心表示改成 **N 点多边形**（``polygon``，归一化坐标，4~24 点）：

* 多边形直接用于「在不在场地里」（:func:`point_in_poly`，逐点射线法）
  和 ROI 外接框（:func:`CourtCalibration.court_mask_roi`）；
* 需要单应变换时（机位判定、距离比较），从多边形里**拟合出面积最大的那个
  四边形**（:func:`fit_quad_from_poly`）再算单应 —— 单应要求四点对应，
  而多点边界只能这样近似；
* 「多边形比四边形多出来的面积」被当作**畸变程度**（
  ``CourtCalibration.distortion``）记录下来，界面可以据此提示「这块场地的
  边缘弯得厉害，建议多加几个点」。

四边形是点数 = 4 的特例，所以旧的手动标定（``court_quad``）与自动标定
依然完全兼容：拟合出来的四边形如果已经够准，多边形就是那个四边形。

坐标系约定
----------
标定后的场地坐标用**羽毛球场真实米制**：``u`` 沿场地短边（双打边线之间
6.10 m），``v`` 沿场地长边（端线之间 13.40 m），原点在场地中心。
四角按「离相机最近的角」为起点顺时针编号（``c0..c3``），于是 ``near``
一侧就是 ``v < 0``（靠近原点的那半边）。即使没有真实场地（比如训练馆里
画线不清楚），只要拟合出了四边形，下游也至少能拿到「两人在画面的什么位置、
沿着哪个方向分开」这类与机位无关的信息。

全部计算只依赖 numpy / opencv，不需要 GPU，也不改变任何既有信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

EPS = 1e-9

#: 标准羽毛球场尺寸（米）——双打边线宽 6.10，两端线间距 13.40
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40

#: 场地多边形允许的点数区间。下限 4 是因为单应变换需要四点对应
#: （三点只能给出仿射，无法表达透视）；上限 24 是「够用且不会让
#: 拟合四边形退化成噪声」的折中——弯边再多也不会多出这么多拐点。
MIN_POLY_POINTS = 4
MAX_POLY_POINTS = 24

Viewpoint = Literal["rear", "side", "elevated", "overhead", "unknown"]

VIEWPOINT_LABEL: dict[str, str] = {
    "rear": "场地后方（底线后）",
    "side": "边线侧方",
    "elevated": "高机位斜俯",
    "overhead": "正俯拍",
    "unknown": "未识别（通用模式）",
}


# ------------------------------------------------------------------ 颜色 / 场地


@dataclass
class CourtColor:
    """场地的颜色统计（用于建 mask）。"""

    #: 主色在 HSV 里的中心
    hue: float = 60.0
    sat: float = 120.0
    val: float = 140.0
    #: 主色像素占比
    ratio: float = 0.0
    #: 主色的人类可读名称
    name: str = "unknown"


def _hsv_of(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def estimate_court_color(frames: list[np.ndarray], max_side: int = 240) -> CourtColor:
    """从若干抽样帧里估计**地面**的主色。

    关键在于「取样位置」，而不是直方图本身：球馆的墙和顶棚常常和地面同色系
    （实测素材里墙是青色木饰面、地面是青绿色地胶，色相只差几个 bin），
    对全画幅统计会把墙选成主色，拟合出的「场地」变成一整条横带。

    所以这里在每帧的**下方 45%、画面水平中间 60%** 的区域内取样：
    无论机位是在底线后、边线侧还是高角度俯拍，那一块都必然落在场地上。
    由此得到的色相/饱和度/明度作为 :func:`court_mask` 的中心值。
    """
    import cv2

    if not frames:
        return CourtColor()
    hist = np.zeros(180, dtype=np.float64)
    sat_acc = np.zeros(180, dtype=np.float64)
    val_acc = np.zeros(180, dtype=np.float64)
    total = 0
    for fr in frames:
        h, w = fr.shape[:2]
        scale = max_side / max(1, max(h, w))
        if scale < 1.0:
            fr = cv2.resize(fr, (max(2, int(w * scale)), max(2, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
        h, w = fr.shape[:2]
        patch = fr[int(h * 0.55):, int(w * 0.20):int(w * 0.80)]
        if patch.size == 0:
            patch = fr
        hsv = _hsv_of(patch)
        hh = hsv[:, :, 0].astype(np.int32).ravel()
        ss = hsv[:, :, 1].astype(np.float64).ravel()
        vv = hsv[:, :, 2].astype(np.float64).ravel()
        # 只看有颜色、亮度正常的像素（白色球场线、阴影会被排除）。
        # 阈值故意放得很低：地面远端会被灯光洗淡，饱和度掉到几十，
        # 门槛高了就只剩近端那一小块，色相估计反而更偏。
        sel = (ss > 30) & (vv > 32)
        if not np.any(sel):
            continue
        idx = hh[sel]
        wgt = ss[sel] / 255.0
        np.add.at(hist, idx, wgt)
        np.add.at(sat_acc, idx, ss[sel])
        np.add.at(val_acc, idx, vv[sel])
        total += int(sel.sum())
    if hist.sum() <= EPS or total == 0:
        return CourtColor()
    # 在 12 bin 宽的窗上做滑动求和，窗内加权平均 -> 对色相的抖动不敏感
    win = 12
    kernel = np.ones(win, dtype=np.float64)
    ext = np.concatenate([hist, hist[: win - 1]])
    scores = np.convolve(ext, kernel[::-1], mode="valid")
    best = int(np.argmax(scores))
    idxs = np.arange(best, best + win) % 180
    wsum = float(hist[idxs].sum())
    if wsum <= EPS:
        return CourtColor()
    hue = float((idxs * hist[idxs]).sum() / wsum)
    sat = float(sat_acc[idxs].sum() / wsum)
    val = float(val_acc[idxs].sum() / wsum)
    ratio = float(wsum / max(hist.sum(), EPS))
    return CourtColor(hue=hue % 180.0, sat=sat, val=val, ratio=ratio,
                      name=_hue_name(hue % 180.0))


def _hue_name(hue: float) -> str:
    if 35 <= hue < 46:
        return "yellow"
    if 46 <= hue < 78:
        return "green"
    if 78 <= hue < 100:
        return "cyan"
    if 100 <= hue < 130:
        return "blue"
    if hue < 12 or hue >= 168:
        return "red"
    if 12 <= hue < 35:
        return "orange"
    return "unknown"


def court_mask(frame: np.ndarray, color: CourtColor, tol: float = 12.0,
               sat_min: float = 0.45, val_min: float = 0.62,
               sat_abs: float = 60.0) -> np.ndarray:
    """按估计出的地面主色生成球场区域的二值 mask（uint8 0/255）。

    ``sat_min`` / ``val_min`` 是**相对**于主色的比例而不是绝对值：地胶在
    不同灯光和距离下有明有暗（实测近端饱和度 215、远端只有 81），但
    「饱和度不低于主色的某个比例、亮度不低于主色的某个比例」这个相对
    关系比较稳定，同时又能把更暗更灰的墙面挡在外面。
    另外再加一条绝对下限 ``sat_abs``：白色球场线、灰色地板的饱和度很低，
    不设绝对下限时会把它们也算进来。
    """
    import cv2

    hsv = _hsv_of(frame)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    dh = np.abs(h - float(color.hue))
    dh = np.minimum(dh, 180.0 - dh)          # 色相是环形的
    m = ((dh <= tol)
         & (s >= max(sat_abs, float(color.sat) * sat_min))
         & (v >= max(25.0, float(color.val) * val_min)))
    mask = (m.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def _bottom_anchor(mask: np.ndarray) -> float:
    """mask 在画面底部中央区域里的覆盖率（用来判断「地面找到了没有」）。"""
    h, w = mask.shape[:2]
    band = mask[int(h * 0.90):, int(w * 0.25):int(w * 0.75)]
    return float(np.mean(band > 0)) if band.size else 0.0


def _union_mask(frames: list[np.ndarray], color: CourtColor, tol: float,
                sat_min: float, val_min: float, sat_abs: float,
                keep_ratio: float = 0.45) -> np.ndarray | None:
    """在若干抽样帧上建 mask，取「每一帧都被选中」的交叠区域。

    交叠的意义：真正的地面在每一帧里都会被选中，而球员、观众、其他场地
    只会偶尔命中，取交叠等于让这些干扰自己消失。
    """
    import cv2

    acc: np.ndarray | None = None
    for fr in frames:
        m = court_mask(fr, color, tol=tol, sat_min=sat_min, val_min=val_min, sat_abs=sat_abs)
        # 只保留够大的连通块，滤掉零散噪点
        con, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(m)
        for c in con:
            if cv2.contourArea(c) > 0.01 * m.size:
                cv2.drawContours(keep, [c], -1, 255, -1)
        m = keep
        if acc is None:
            acc = m.astype(np.float32)
        else:
            if m.shape != acc.shape:
                m = cv2.resize(m, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_NEAREST)
            acc += m.astype(np.float32)
    if acc is None:
        return None
    acc /= float(len(frames))
    mask = (acc > keep_ratio).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return mask


def _seed_component(mask: np.ndarray, seed_y: float = 0.93, seed_x: float = 0.5,
                    min_area_ratio: float = 0.02) -> np.ndarray | None:
    """在 mask 里取「包含画面底部中央那一点」的连通块。

    这一点一定是场地，所以用它当种子比「取最大连通块」可靠：
    球馆里最大的色块未必是场地（实测顶棚和木饰面墙比地面更亮）。
    为了避免这一行正好落在球场线的白条上，在底部横向扫一条带取多数票。
    """
    import cv2

    h, w = mask.shape[:2]
    band_y0 = int(h * max(0.0, seed_y - 0.03))
    band_x0, band_x1 = int(w * 0.35), int(w * 0.65)
    band = mask[band_y0:, band_x0:band_x1]
    seed = None
    if band.size:
        ys, xs = np.nonzero(band)
        if xs.size:
            k = int(np.argmax(np.bincount(xs)))
            rows = ys[xs == k]
            seed = (band_x0 + int(k), band_y0 + int(np.median(rows)))
    if seed is None:
        return None
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    lbl = labels[seed[1], seed[0]]
    if lbl == 0:
        return None
    comp = (labels == lbl).astype(np.uint8) * 255
    if float(np.mean(comp > 0)) < min_area_ratio:
        return None
    return comp


def measure_court_band(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """逐行判断「这一行有多少是场地」，得到场地的上下边界。

    这是整套标定里最关键的一步，做法来自实测：从画面**最底部**往上逐行分析，
    一行被算作场地当且仅当

    * 该行有足够比例的像素落在「底部区域测出来的地面颜色范围」内；
    * 且这些像素**连成一片**（覆盖了该行横向跨度的很大一部分），
      零散命中不算（那是墙上的杂色，不是地面）。

    实测素材的逐行剖面（480 宽、270 高，色相接近地面主色的像素）：

    | 行位置 | 饱和度 | 亮度 | 判定 |
    | --- | --- | --- | --- |
    | 0.00–0.44 | 60–95 | 45–107 | 顶棚 / 墙面：饱和度低 |
    | 0.50 | 82 | 142 | 过渡带：远端场地 + 墙裙 |
    | 0.56–0.94 | 194–220 | 160–180 | 近端场地：饱和度极高 |

    可以看到**饱和度**把地面和墙分得非常干净（差 2 倍以上），而色相只差十几个
    bin（墙面木饰面与地胶同为青绿色调）。所以判据用饱和度而不是色相。

    Returns:
        ``(逐行是否为场地的布尔数组, 统计信息)``；数组长度等于画面高度。
    """
    if not frames:
        return np.zeros(0, dtype=bool), {}
    fr = frames[0]
    h, w = fr.shape[:2]
    hsv = _hsv_of(fr)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # 从底部 12% 的区域里取地面色相的众数（那里必定是场地）
    band = slice(int(h * 0.88), h)
    hs = H[band].ravel()
    ss = S[band].ravel()
    vs = V[band].ravel()
    sel0 = ss > 60
    if int(sel0.sum()) < 50:
        return np.zeros(h, dtype=bool), {"reason": "底部取样不足"}
    hue_hist = np.bincount(hs[sel0].astype(np.int32), minlength=180)
    # 12 bin 滑窗取峰，对色相抖动不敏感
    win = 12
    ext = np.concatenate([hue_hist, hue_hist[: win - 1]])
    scores = np.convolve(ext, np.ones(win), mode="valid")
    best = int(np.argmax(scores))
    idxs = np.arange(best, best + win) % 180
    wsum = float(hue_hist[idxs].sum())
    hue = float((idxs * hue_hist[idxs]).sum() / max(wsum, EPS)) if wsum > 0 else float(best)
    sat_ref = float(np.median(ss[sel0]))

    dh = np.abs(H - hue)
    dh = np.minimum(dh, 180.0 - dh)
    sat_thr = max(60.0, sat_ref * 0.62)
    cand = (dh <= 20.0) & (S >= sat_thr) & (V >= 40.0)

    # 逐行：命中比例 + 最长连续段
    row_hit = np.zeros(h, dtype=np.float32)
    row_span = np.zeros(h, dtype=np.float32)
    for y in range(h):
        row = cand[y]
        hit = float(row.mean())
        row_hit[y] = hit
        if hit < 0.05:
            continue
        xs = np.nonzero(row)[0]
        best_run = 1
        run = 1
        for k in range(1, xs.size):
            run = run + 1 if xs[k] == xs[k - 1] + 1 else 1
            if run > best_run:
                best_run = run
        row_span[y] = best_run / float(w)

    # 判据要允许两种「断口」：球场线把一行切成好几段，以及球网柱、球员把
    # 局部遮住。所以既看命中比例，也看**最长的连续段**，两个门限都放得比较松
    # （实测近端地胶因为球场线的分布，最长连续段只有整行的 40% 左右）。
    is_court = (row_hit >= 0.60) & (row_span >= 0.30)
    # 从最底部往上找：一旦连续多行不是场地就停下（避免被上方的零散行干扰）
    court_rows = np.zeros(h, dtype=bool)
    miss = 0
    for y in range(h - 1, -1, -1):
        if is_court[y]:
            court_rows[y] = True
            miss = 0
        else:
            miss += 1
            if miss > max(2, int(0.02 * h)):
                break
    info = {
        "hue": round(hue, 1), "sat_ref": round(sat_ref, 1), "sat_thr": round(sat_thr, 1),
        "top_ratio": round(float(np.nonzero(court_rows)[0].min()) / h, 3)
        if court_rows.any() else 1.0,
        "cover": round(float(np.mean(court_rows)), 3),
    }
    return court_rows, info


def court_mask_from_rows(frame: np.ndarray, court_rows: np.ndarray,
                         hue: float, sat_thr: float) -> np.ndarray:
    """按「逐行场地判定」+ 颜色条件生成该帧的球场 mask。"""
    import cv2

    h, w = frame.shape[:2]
    if court_rows.size != h:
        court_rows = np.resize(court_rows, h)
    hsv = _hsv_of(frame)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    dh = np.abs(H - hue)
    dh = np.minimum(dh, 180.0 - dh)
    cand = ((dh <= 26.0) & (S >= max(45.0, sat_thr * 0.55))).astype(np.uint8)
    cand[~court_rows, :] = 0
    mask = cand * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    # 只保留包含底部中央的连通块
    comp = _seed_component(mask, min_area_ratio=0.03)
    return comp if comp is not None else mask


def refine_court_color(frames: list[np.ndarray],
                       base: CourtColor) -> tuple[CourtColor, np.ndarray | None]:
    """在基准色附近小范围搜索，找出「最像一整块地面」的那组参数。

    打分标准：必须能圈出一个**包含画面底部中央**的连通块，面积越大越好，
    但超过 0.8 说明把墙 / 顶棚也算进来了，要重罚。
    """
    grid = [
        (dh, sm)
        for dh in (-6.0, -3.0, 0.0, 3.0, 6.0, 10.0)
        for sm in (0.08, 0.20, 0.35)
    ]
    best_score = -1.0
    best_color = base
    best_mask: np.ndarray | None = None
    for dh, sm in grid:
        cand = CourtColor(hue=(base.hue + dh) % 180.0, sat=base.sat, val=base.val,
                          ratio=base.ratio, name=_hue_name((base.hue + dh) % 180.0))
        union = _union_mask(frames, cand, tol=12.0, sat_min=sm, val_min=0.35, sat_abs=60.0)
        if union is None:
            continue
        comp = _seed_component(union)
        if comp is None:
            continue
        cover = float(np.mean(comp > 0))
        if cover < 0.05 or cover > 0.98:
            continue
        score = min(cover, 0.80) - max(0.0, cover - 0.80) * 3.0
        if score > best_score:
            best_score, best_color, best_mask = score, cand, comp
    return best_color, best_mask


def _contour_area(pts: np.ndarray) -> float:
    """多边形的鞋带面积（像素²，取绝对值）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def find_court_quad(mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray | None:
    """从球场 mask 里拟合四边形，返回 4×2 的像素坐标（顺序不定）。

    优先用 ``approxPolyDP`` 找真正的四边形；退化时用 ``minAreaRect``；
    再不行就用凸包的四个极值点。面积太小（不足画面 5%）视为没找到。

    .. note::
       这个函数保留给「必须恰好四个点」的旧调用方（以及回归测试）。
       新的代码请用 :func:`find_court_poly` —— 全景 / 鱼眼素材的场地边界
       是弯的，硬压成四边形会白白丢掉边界精度。
    """
    import cv2

    if mask is None or mask.size == 0:
        return None
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < min_area_ratio * w * h:
        return None
    peri = cv2.arcLength(big, True)
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        ap = cv2.approxPolyDP(big, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(big)
    return cv2.boxPoints(rect).astype(np.float32)


def _resample_ring(pts: np.ndarray, n: int, phase: float = 0.0) -> np.ndarray:
    """把闭合轮廓按**弧长**均匀重采样成 n 个点（``phase`` 偏移起点）。

    为什么要按弧长重采样，而不是继续用 ``approxPolyDP``：``approxPolyDP`` 的
    容差是「相对周长」的，同一个容差在弯边形状上会直接跳过中间状态 ——
    容差小一点就保留几十个点，大一点就把整条弧压成一条弦（弯边被抹平）。
    于是「用几个点描述这条弯边」这个问题它没法回答。按弧长重采样则可以
    精确地问：「用 n 个点，能把这圈边界描述到多准？」

    ``phase`` 是必要的：均匀采样的起点落在哪里决定了顶点能不能正好压在
    「角」上（起点落在边中间时会切掉一个角，面积凭空少一块）。调用方会对
    同一个 n 试若干相位，取最好的那个。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return p[:n]
    # 去掉相邻重复点：轮廓上极常见，会让弧长出现 0 长度段，插值就不单调了
    keep = np.ones(p.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
    p = p[keep]
    if p.shape[0] < 3:
        return p[:n]
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return p[:n]
    t = ((np.arange(n) / float(n) + float(phase) % 1.0) % 1.0) * total
    xs = np.interp(t, s, closed[:, 0])
    ys = np.interp(t, s, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def find_court_poly(mask: np.ndarray, min_area_ratio: float = 0.05,
                    max_points: int = 16, iou_min: float = 0.985,
                    phases: int = 12) -> np.ndarray | None:
    """从球场 mask 里拟合**多边形**边界，返回 N×2 像素坐标（4 ≤ N ≤ max_points）。

    判据是「**用最少的点，把这块区域描述到多准**」：从 4 个点开始逐点增加，
    取第一个能把「与原轮廓的交并比」做到 ``iou_min`` 以上的点数。

    * 边界笔直（普通机位）时 4 个点就够了 —— 交并比几乎为 1，于是返回的就是
      传统四边形，行为与旧版本一致；
    * 边界是弯的（全景 / 鱼眼）时 4 个点会切掉边角，交并比掉下来，
      于是自动加到 6、8……个点，直到弯边被描述清楚为止。

    每个点数都会比较两族候选（取交并比高的那个）：

    * **RDP**（``approxPolyDP``，容差二分到「刚好剩下 n 个点」）。它擅长保住
      「角」：四条边长短不一的矩形用 4 个点就能精确描述，这是弧长均匀采样
      做不到的（均匀采样的顶点按周长等分，长短边不等时压不到角上）；
    * **弧长均匀采样**（试 ``phases`` 个起点相位）。它擅长描述**没有角**的
      平滑弯边：RDP 的起点是轮廓给的、固定不动，弯边上可能被这个固定起点
      占掉一个顶点。

    两族都算一遍的代价很小（轮廓通常几百个点，十几毫秒），换来的是
    「直的用四边形、弯的自动加点」不用用户告诉我们是哪种素材。

    为什么用交并比而不是「面积差」：面积差看不出「形状对了但整体偏了」。
    交并比同时约束位置和形状，而我们最终关心的正是「多边形圈出来的这块区域
    对不对」—— 球场 ROI 的唯一用途就是这个。栅格化在 256 宽上做，
    所以判据本身有约半个百分点的量化误差。
    """
    import cv2

    if mask is None or mask.size == 0:
        return None
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    raw_area = float(cv2.contourArea(big))
    if raw_area < min_area_ratio * w * h:
        return None
    contour = big.reshape(-1, 2).astype(np.float64)
    if contour.shape[0] < 4:
        return None

    # 低分辨率栅格上算交并比：判据的精度需求远低于像素级，256 宽足够，
    # 而且能把「每个点数 × 每族候选」都算一遍（十几毫秒）。
    scale = 256.0 / max(1, w)
    sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    ref = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(ref, [np.round(contour * scale).astype(np.int32).reshape(-1, 1, 2)], 1)
    if int(np.count_nonzero(ref)) == 0:
        return None

    def _iou(cand: np.ndarray) -> float:
        m = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(cand * scale).astype(np.int32).reshape(-1, 1, 2)], 1)
        inter = int(np.count_nonzero(np.logical_and(m, ref)))
        union = int(np.count_nonzero(np.logical_or(m, ref)))
        return float(inter / union) if union else 0.0

    diag = float(np.hypot(w, h))
    cnt32 = contour.astype(np.float32).reshape(-1, 1, 2)

    def _rdp(n: int) -> np.ndarray | None:
        """容差二分：找到「刚好只剩 n 个点」的最小容差，返回那个多边形。

        RDP 的点数是容差的阶梯函数（容差大一点就少一个点），所以直接二分
        容差比给一串固定容差靠谱：固定容差网格很容易整段跳过想要的点数。
        """
        lo, hi = 0.0, diag
        ap = cv2.approxPolyDP(cnt32, hi, True)
        if len(ap) > n:
            return None                     # 再大的容差也降不到 n 个点
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            cur = cv2.approxPolyDP(cnt32, mid, True)
            if len(cur) > n:
                lo = mid
            else:
                hi = mid
        ap = cv2.approxPolyDP(cnt32, hi, True).reshape(-1, 2).astype(np.float64)
        return ap if 3 <= ap.shape[0] <= n else None

    lo_n = max(MIN_POLY_POINTS, 3)
    hi_n = int(max(lo_n, min(max_points, MAX_POLY_POINTS)))
    best_any: tuple[float, np.ndarray] | None = None
    for n in range(lo_n, hi_n + 1):
        cands: list[np.ndarray] = []
        rdp = _rdp(n)
        if rdp is not None:
            cands.append(rdp)
        for k in range(max(1, int(phases))):
            cands.append(_resample_ring(contour, n, phase=k / float(max(1, int(phases)))))
        best_iou = -1.0
        best_pts = cands[0]
        for cand in cands:
            if cand.shape[0] < 3:
                continue
            v = _iou(cand)
            if v > best_iou:
                best_iou, best_pts = v, cand
        if best_any is None or best_iou > best_any[0]:
            best_any = (best_iou, best_pts)
        if best_iou >= iou_min:
            return best_pts.astype(np.float32)
    # 到上限仍不达标（边界非常弯、或者是块不规则的场地）：退回「最好的那个」，
    # 至少保证下游有标定可用 —— 「它不够准」由调用方记进 notes。
    if best_any is not None and best_any[0] >= 0.90:
        return best_any[1].astype(np.float32)
    return find_court_quad(mask, min_area_ratio=min_area_ratio)


def poly_area_norm(poly: np.ndarray) -> float:
    """归一化多边形的面积（0~1，画面总面积 = 1）。"""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def order_poly(poly: np.ndarray, aspect: float = 1.7778) -> np.ndarray:
    """把任意点序的多边形整理成「近左起点、沿画面向右」的环序。

    三种输入都要能吃下：

    * 用户随手点的（可能是乱的，甚至来回打结）；
    * ``approxPolyDP`` 出来的（本来就有序，但起点不定）；
    * 程序拼出来的。

    做法：按「相对质心的极角」排序得到环序，再把起点转到**最近的那个顶点**
    （y 最大，也就是画面最下方；并列时取更靠左的，与 :func:`order_quad`
    的约定一致）。极角用 ``atan2(dy, dx * aspect)``：x 乘画面宽高比之后，
    「离得近」在不同画面比例下才是可比的，否则竖屏素材里极角会被横向压扁，
    环序容易在近端两个点上互换。

    排序后顺手做一次「凹凸体检」：鞋带面积明显小于凸包面积说明点序自交
    （用户点出了蝴蝶结），这时改用凸包点序 —— 自交多边形会让
    :func:`point_in_poly` 的判定完全反直觉，而且没法做单应拟合。
    """
    import cv2

    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    n = p.shape[0]
    if n < 3:
        return p
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    ang = np.arctan2(p[:, 1] - cy, (p[:, 0] - cx) * max(0.2, float(aspect)))
    order = np.argsort(-ang)                    # 降序 = 从近端向右绕
    ring = p[order]
    # 自交体检（只有点数 > 4 才有意义，四点一定是简单多边形变形）
    if n > 4:
        try:
            hull = cv2.convexHull(ring.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) >= 3 and _contour_area(ring) < 0.90 * _contour_area(hull):
                hcx, hcy = float(hull[:, 0].mean()), float(hull[:, 1].mean())
                hang = np.arctan2(hull[:, 1] - hcy, (hull[:, 0] - hcx) * max(0.2, float(aspect)))
                ring = hull[np.argsort(-hang)]
                n = ring.shape[0]
        except Exception:
            pass
    # 起点：y 最大（最近）里最靠左的那个
    y_max = float(ring[:, 1].max())
    tol = max(1e-6, 0.06 * float(ring[:, 1].max() - ring[:, 1].min()))
    near = np.nonzero(ring[:, 1] >= y_max - tol)[0]
    start = int(near[np.argmin(ring[near, 0])]) if near.size else 0
    return np.roll(ring, -start, axis=0).astype(np.float32)


def fit_quad_from_poly(poly: np.ndarray) -> np.ndarray | None:
    """从多边形里挑出**面积最大的那个四边形**（4×2，像素坐标）。

    单应变换必须是四点对应，而多边形给的是 N 个边界点，所以这里要「选四个」。
    选法用了场地边界的一个普适性质：**真实场地的四个角一定是边界上最外凸的
    那四个点**，因此「顶点构成的内接四边形面积最大」与「选到四个角」等价，
    而且不需要知道相机参数。

    点数 ≤ 4 时直接返回（不足 4 点返回 ``None``）。点数多的时候穷举
    C(N,4)：N 被限制在 :data:`MAX_POLY_POINTS`（24）以内，最坏 10626 种组合，
    每种只算一次鞋带面积，微秒级，不值得为它写更聪明的启发式。
    """
    from itertools import combinations

    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    n = p.shape[0]
    if n < 4:
        return None
    if n == 4:
        return order_quad(p)
    if n > MAX_POLY_POINTS:
        # 均匀抽稀到上限，保持环序（抽稀会丢一点边界精度，但不会丢角点：
        # 角点两侧的采样点仍在，穷举仍能挑到它们附近）
        idx = np.linspace(0, n - 1, MAX_POLY_POINTS).round().astype(int)
        p = p[np.unique(idx)]
        n = p.shape[0]
    best_area = -1.0
    best: np.ndarray | None = None
    for combo in combinations(range(n), 4):
        q = p[list(combo)]
        a = _contour_area(q)
        if a > best_area:
            best_area = a
            best = q
    if best is None or best_area <= EPS:
        return None
    return order_quad(best)


def offset_poly(poly: np.ndarray, dist: float) -> np.ndarray:
    """把多边形**向外平移** ``dist``（同一坐标系单位，通常是归一化坐标）。

    做法是几何上的标准「多边形偏移」：把每条边沿外法向平移 ``dist``，
    再把相邻两条平移后的边求交点当作新顶点。对凸多边形（球场边界就是凸的）
    这给出的是精确的等距外扩 —— 每一条边都往外让出同样的距离。

    为什么不直接用「按质心等比放大」：那会让离质心远的边让得多、近的边让得少，
    而球场多边形在画面里往往是一个扁横条（近半场），质心到远边的距离远小于
    到左右两边的距离，等比放大会把远端边界几乎不动、左右边界推得很远 ——
    正好和「远端真球员站在边界上」这个最需要放宽的方向相反。

    自交或退化（点太少、面积没变大）时返回原多边形：宁可少放宽一点，
    也不要产出一个形状诡异的 ROI 把所有人都判到外面。
    """
    q = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = q.shape[0]
    if n < 3 or dist <= 0:
        return q.astype(np.float32)
    # 环绕方向决定外法向的符号：图坐标系（y 向下）里鞋带面积为正时，
    # 法向取 (e_y, -e_x) 指向外侧
    x, y = q[:, 0], q[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    sign = 1.0 if signed > 0 else -1.0

    pts: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for i in range(n):
        a = q[i]
        b = q[(i + 1) % n]
        e = b - a
        L = float(np.linalg.norm(e))
        if L < EPS:
            continue
        d = e / L
        nvec = sign * np.array([d[1], -d[0]], dtype=np.float64)
        dirs.append(d)
        pts.append(a + nvec * float(dist))
    m = len(pts)
    if m < 3:
        return q.astype(np.float32)

    out: list[np.ndarray] = []
    for i in range(m):
        p1, d1 = pts[i], dirs[i]
        p2, d2 = pts[(i + 1) % m], dirs[(i + 1) % m]
        # 两条直线求交：p1 + t*d1 = p2 + s*d2
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:            # 平行（共线）边：直接取平移后的端点
            out.append(p1)
            continue
        rhs = p2 - p1
        t = (rhs[0] * d2[1] - rhs[1] * d2[0]) / denom
        out.append(p1 + d1 * t)
    # out[j] 是「边 j 与边 j+1 的交点」，也就是原顶点 j+1 的新位置；
    # 往回挪一格，让输出的第 i 个点仍然对应输入的第 i 个点（起点不变）
    cand = np.roll(np.asarray(out, dtype=np.float64), 1, axis=0) if m == n else np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(cand)):
        return q.astype(np.float32)
    # 外扩后面积必须**变大**（只看绝对值：环序可能是顺时针也可能是逆时针，
    # 用带符号面积判断方向会在顺时针的环上把「外扩成功」误判成失败），
    # 否则说明偏移算错了，宁可原样返回也不要用一个形状诡异的多边形当 ROI。
    a_cand = _poly_area_signed(cand)
    if not np.isfinite(a_cand) or abs(a_cand) <= abs(signed):
        return q.astype(np.float32)
    return cand.astype(np.float32)


def _poly_area_signed(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_poly(pts: np.ndarray, poly: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """逐点判断「是否落在多边形内」（射线交叉法，向量化）。

    Args:
        pts: ``(N, 2)`` 待判定点（与 ``poly`` 同一坐标系，通常是归一化坐标）。
        poly: ``(M, 2)`` 多边形顶点（环序，M ≥ 3）。
        margin: 向外放宽的距离（与坐标同单位，0 = 精确判定）。

    Returns:
        长度 N 的 bool 数组。

    为什么不用「外接矩形」近似：全场素材里弯边造成的误差集中出现在画面边角，
    而那正是「其他场地的人 / 观众」最多的地方 —— 用矩形判定等于把这些人
    全部放进候选池，后续只能靠尺寸门限硬筛。

    ``margin`` 是为**筛选球员**准备的。两个原因：

    * 射线法对「正好压在边线上」的点给出的是半开区间的结果（上边界算外、
      下边界算内），而球员的脚正好落在画出来的边线上完全可能发生；
    * 场地标定圈出来的「场地」本来就是保守的近半场，远端的真球员常常正好
      站在边界那一行上。

    判错方向的代价不对称：多留一个人只是多一点噪声（尺寸筛选和活跃度评分
    还会再筛一遍），漏掉一个人会让下游的活跃度曲线和裁切跟随直接断档。
    精确判定（画图、统计）用默认的 ``margin=0``。
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if margin > 0 and q.shape[0] >= 3:
        q = offset_poly(q, float(margin)).astype(np.float64)
    m = q.shape[0]
    if p.shape[0] == 0 or m < 3:
        return np.ones(p.shape[0], dtype=bool)
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(p.shape[0], dtype=bool)
    x1, y1 = q[:, 0], q[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for i in range(m):
        ax, ay, bx, by = x1[i], y1[i], x2[i], y2[i]
        if abs(by - ay) < EPS and abs(bx - ax) < EPS:
            continue
        # 该边是否跨越这条水平射线（半开区间，避免顶点被算两次）
        crosses = ((ay > y) != (by > y))
        if not np.any(crosses):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (y - ay) / np.where(by == ay, EPS, by - ay)
        xint = ax + t * (bx - ax)
        inside ^= crosses & (x <= xint)
    return inside


def order_quad(quad: np.ndarray) -> np.ndarray:
    """把四个角点排成 [近左, 近右, 远右, 远左]。

    「近」= 在画面里更靠下（y 更大），这是任何机位下都成立的判据：
    地面上的点越靠近画面下方就越靠近相机。
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    order = np.argsort(-q[:, 1])          # y 从大到小 = 从近到远
    near = q[order[:2]]
    far = q[order[2:]]
    near = near[np.argsort(near[:, 0])]   # 近边按 x 排序 -> 近左, 近右
    far = far[np.argsort(far[:, 0])]      # 远边同理
    return np.array([near[0], near[1], far[1], far[0]], dtype=np.float32)


# ------------------------------------------------------------------ 标定


@dataclass
class CourtCalibration:
    """一次分析的场地标定结果。

    核心表示是**多边形**（``polygon`` / ``polygon_norm``）：4 个点时它就是
    传统的四角四边形，点更多时它描述的是全景 / 鱼眼素材里弯掉的场地边界。
    ``quad`` / ``quad_norm`` 是从多边形拟合出来的四边形，只为单应变换服务。
    """

    #: 是否成功标定出场地多边形（能建出单应变换才算成功）
    ok: bool = False
    #: 拟合出来的四边形（像素坐标，[近左, 近右, 远右, 远左]）——单应用
    quad: np.ndarray | None = None
    #: 归一化（0~1）的四边形，便于前端画框（旧字段，保留兼容）
    quad_norm: list[list[float]] = field(default_factory=list)
    #: 场地边界多边形（像素坐标，环序：近左起点、沿画面向右绕）
    polygon: np.ndarray | None = None
    #: 归一化（0~1）的场地边界多边形 —— 前端画线、场内判定都用它
    polygon_norm: list[list[float]] = field(default_factory=list)
    #: 多边形来自哪里：``auto`` 自动识别 / ``manual`` 用户手动标定 / ``none``
    source: str = "none"
    #: 「多边形比拟合四边形多出来的面积」比例 0~1，弯边（畸变）越厉害越大
    distortion: float = 0.0
    #: 画面坐标 -> 场地坐标（单位方框 [0,1]×[0,1]，u 横向、v 由近及远）
    homography: np.ndarray | None = None
    #: 逆变换
    homography_inv: np.ndarray | None = None
    #: 识别出的机位
    viewpoint: Viewpoint = "unknown"
    #: 机位识别的置信度 0~1
    confidence: float = 0.0
    #: 场地颜色
    color: CourtColor = field(default_factory=CourtColor)
    #: 场地在画面里的面积占比
    area_ratio: float = 0.0
    #: 画面宽高比
    aspect: float = 1.7778
    #: 「远边比近边短多少」——衡量近大远小的透视压缩程度（1 = 没有压缩）
    foreshortening: float = 1.0
    #: 球员（归一化框底边中心）落在场地内/外的比例
    in_court_ratio: float = 0.0
    #: 说明性信息
    notes: list[str] = field(default_factory=list)

    # ---- 坐标变换 ----
    def to_court(self, pts: np.ndarray) -> np.ndarray:
        """画面像素坐标 -> 场地坐标（单位方框，``ok`` 为真时有效）。"""
        if not self.ok or self.homography is None:
            return np.zeros((0, 2), dtype=np.float32)
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        import cv2

        return cv2.perspectiveTransform(p, self.homography).reshape(-1, 2)

    def to_court_norm(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        """归一化画面坐标 -> 场地坐标。"""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        return self.to_court(np.stack([p[:, 0] * w, p[:, 1] * h], axis=1))

    def contains(self, pts_norm: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """归一化画面坐标里，哪些点落在**场地多边形内**（bool 数组）。

        有多边形就按多边形判；只有四边形（老数据）就按四边形判 —— 两者
        是同一条路径，不需要调用方分支。
        """
        poly = self.polygon_norm or self.quad_norm
        p = np.asarray(pts_norm, dtype=np.float32).reshape(-1, 2)
        if not poly:
            return np.ones(p.shape[0], dtype=bool)
        return point_in_poly(p, np.asarray(poly, dtype=np.float32), margin=margin)

    def contains_box_bottom(self, boxes: "list | np.ndarray", margin: float = 0.02) -> np.ndarray:
        """一批归一化框（xyxy）的**底边中心**是否落在场地里。

        底边中心是「人在场上的落点」的近似，球员检测的 ROI 过滤用的就是它。
        默认带 2% 的外扩：球员站在边线上时不能判成场外（漏掉真球员的代价
        远大于多留一个背景人员）。
        """
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if arr.size == 0:
            return np.zeros(0, dtype=bool)
        pts = np.stack([0.5 * (arr[:, 0] + arr[:, 2]), arr[:, 3]], axis=1)
        return self.contains(pts, margin=margin)

    def court_mask_roi(self, margin: float = 0.06) -> tuple[float, float, float, float] | None:
        """场地范围的外接矩形（归一化，带外扩），可直接当作分析 ROI。

        外接矩形只是「粗筛」用的（把解码范围缩小、把明显在画面另一端的人排掉），
        精细的场内判定请用 :meth:`contains`。
        """
        src = self.polygon_norm or self.quad_norm
        if not src:
            return None
        arr = np.asarray(src, dtype=np.float32)
        x0, y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
        x1, y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
        return (max(0.0, x0 - margin), max(0.0, y0 - margin),
                min(1.0, x1 + margin), min(1.0, y1 + margin))

    def poly_norm(self) -> list[list[float]]:
        """多边形（归一化坐标）供前端画线。"""
        return self.polygon_norm or self.quad_norm

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def as_payload(self) -> dict[str, Any]:
        """给前端/统计用的可序列化摘要。"""
        poly = self.polygon_norm or self.quad_norm
        return {
            "ok": self.ok,
            "viewpoint": self.viewpoint,
            "viewpoint_label": VIEWPOINT_LABEL.get(self.viewpoint, self.viewpoint),
            "confidence": round(float(self.confidence), 3),
            # polygon 是主表示；quad 保留给「只认识四个角」的老前端 / 老工程
            "polygon": poly,
            "quad": self.quad_norm,
            "point_count": len(poly),
            "source": self.source,
            "distortion": round(float(self.distortion), 4),
            "court_color": self.color.name,
            "court_area_ratio": round(float(self.area_ratio), 3),
            "foreshortening": round(float(self.foreshortening), 3),
            "in_court_ratio": round(float(self.in_court_ratio), 3),
            "roi": list(self.court_mask_roi()) if poly else None,
            "notes": list(self.notes),
        }


#: 场地平面的目标坐标系：单位方框，u 横向 0~1（左→右），v 纵向 0~1（近→远）
_COURT_TARGET = np.array([
    [0.0, 0.0],   # 近左
    [1.0, 0.0],   # 近右
    [1.0, 1.0],   # 远右
    [0.0, 1.0],   # 远左
], dtype=np.float32)


def build_calibration(poly: np.ndarray, frame_w: int, frame_h: int,
                      color: CourtColor | None = None,
                      source: str = "auto") -> CourtCalibration:
    """用场地边界（**N 点多边形**，N ≥ 4；4 个点即传统四边形）建立标定。

    单应变换是四点对应，所以这里先 :func:`fit_quad_from_poly` 从多边形里挑出
    面积最大的内接四边形，再用它算单应。多边形本身则用于场内判定与 ROI ——
    两者配合起来，弯边素材不会被硬压成直线，同时下游拿到的
    ``to_court()`` / ``in_court_ratio`` / 机位判定完全不变。

    Args:
        poly: ``(N, 2)`` 像素坐标，点序随意（会按 :func:`order_poly` 整理）。
        frame_w / frame_h: 原始画面尺寸（多边形坐标所在的空间）。
        color: 场地颜色估计（可为空）。
        source: ``auto`` / ``manual``，会随 payload 一起下发。

    **注意**：目标坐标系是单位方框，不是真实米制。原因很实在：
    画面里往往只能看到场地的一部分（实测素材只看到近半场），把它硬映射到
    13.40 m × 6.10 m 会把所有纵向距离放大两倍以上，比不标定更糟。
    单位方框下「两名球员相距 0.4 个场宽」这类相对量是可靠的，
    判断机位、判断谁在近端、裁剪跟随都不需要真实米制。
    """
    import cv2

    cal = CourtCalibration(ok=False, color=color or CourtColor(),
                           aspect=float(frame_w) / max(1, frame_h), source=source)
    if poly is None:
        cal.note("未找到场地边界")
        return cal

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < MIN_POLY_POINTS:
        cal.note(f"场地边界只有 {pts.shape[0]} 个点，至少需要 {MIN_POLY_POINTS} 个"
                 f"（单应变换要求四点对应）")
        return cal

    ring = order_poly(pts, aspect=cal.aspect)
    cal.polygon = ring
    cal.polygon_norm = [[round(float(x) / frame_w, 5), round(float(y) / frame_h, 5)]
                        for x, y in ring]
    poly_area = poly_area_norm(np.asarray(cal.polygon_norm, dtype=np.float64))
    if poly_area < 0.02:
        cal.note("场地多边形太小（不足画面 2%），标定没有意义")
        return cal

    quad = fit_quad_from_poly(ring)
    if quad is None:
        cal.note("无法从场地边界里拟合出四边形")
        return cal
    q = order_quad(quad)
    cal.quad = q
    cal.quad_norm = [[round(float(x) / frame_w, 5), round(float(y) / frame_h, 5)] for x, y in q]

    quad_area = float(abs(cv2.contourArea(q.reshape(-1, 1, 2))) / max(1, frame_w * frame_h))
    cal.area_ratio = poly_area
    # 「多边形比四边形多出来的面积比例」= 边界弯曲（镜头畸变 / 球场地胶起拱）的程度
    cal.distortion = float(np.clip(1.0 - quad_area / max(poly_area, EPS), 0.0, 1.0))
    if ring.shape[0] > 4:
        cal.note(f"场地边界 {ring.shape[0]} 点多边形（弯边占 {cal.distortion:.1%}），"
                 f"场内判定按多边形、单应按拟合四边形")
    if cal.distortion > 0.06:
        cal.note(f"边界明显不是直线（拟合四边形丢了 {cal.distortion:.0%} 的面积）："
                 f"超广角/鱼眼畸变的典型特征，ROI 已按多边形处理")

    near_w = float(np.linalg.norm(q[1] - q[0]))
    far_w = float(np.linalg.norm(q[2] - q[3]))
    cal.foreshortening = float(min(near_w, far_w) / max(near_w, far_w, EPS))
    H = cv2.getPerspectiveTransform(q, _COURT_TARGET)
    if not np.all(np.isfinite(H)) or abs(float(H[2, 2])) < EPS:
        cal.note("单应矩阵退化")
        return cal
    HH = H / float(H[2, 2])
    cal.homography = HH.astype(np.float32)
    cal.homography_inv = np.linalg.inv(cal.homography).astype(np.float32)
    cal.ok = True
    return cal


# ------------------------------------------------------------------ 机位判定


def _as_xyxy(item: Any) -> tuple[float, float, float, float] | None:
    """把「逐帧框」统一成 ``(x1, y1, x2, y2)``。

    工程里两种表示都有：``PlayerSignal.frame_boxes`` 是
    ``(track_id, x1, y1, x2, y2)``，检测阶段的逐帧框是 ``(x1, y1, x2, y2)``。
    机位判定只关心几何量，所以两种都要能吃下 —— 早期只认五元组，
    四元组会被整条跳过（表现为「球员框太少，无法判定机位」）。
    """
    try:
        arr = np.asarray(item, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size >= 5:
        return (float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]))
    if arr.size >= 4:
        return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    return None


def estimate_viewpoint(
    cal: CourtCalibration,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    frame_w: int = 1000,
    frame_h: int = 1000,
    hint: Viewpoint | None = None,
) -> CourtCalibration:
    """判断机位类型，并把判别依据写进 ``notes``。

    用到的都是**与画面朝向无关**的量：

    * ``foreshortening``：场地远端宽度 / 近端宽度。斜视时远端明显更窄
      （近大远小），俯拍时接近 1。它是「相机离地面多高、离场地多远」的直接反映；
    * 球员的**画面坐标**分布：横向铺开 vs 纵向铺开；
    * ``size_ratio``：同屏两名球员的框面积比（深度差的代理量）。

    于是：

    * 球员主要沿画面横向分开、框大小接近 -> **side**（边线侧方看过去的典型特征）；
    * 球员主要沿画面纵向分开、一近一远大小差明显 -> **rear**（底线后方机位）；
    * 场地四边形几乎不变形（``foreshortening`` 接近 1）-> **overhead**；
    * 介于两者之间、球员分布靠近画面中心 -> **elevated**（高机位斜俯）。
    """
    if player_boxes is None or player_fps <= 0 or not player_boxes:
        cal.viewpoint = hint or "unknown"
        cal.confidence = 0.9 if hint else 0.0
        cal.note("没有球员框，无法判定机位" + ("（采用指定机位）" if hint else "，使用通用模式"))
        return cal

    # ---- 球员框统计（无论标定成功与否都能算）
    xs: list[float] = []
    ys: list[float] = []
    heights: list[float] = []
    ratios: list[float] = []
    bottoms: list[float] = []          # 框底边的 y（「人站在场上的位置」）
    parsed: list[tuple[float, float, float, float]] = []
    for fr in player_boxes:
        fr_boxes: list[tuple[float, float, float, float]] = []
        for item in fr or ():
            b = _as_xyxy(item)
            if b is None:
                continue
            x1, y1, x2, y2 = b
            fr_boxes.append(b)
            xs.append(0.5 * (x1 + x2))
            ys.append(0.5 * (y1 + y2))
            bottoms.append(y2)
            heights.append(abs(y2 - y1))
        if fr_boxes:
            parsed.extend(fr_boxes)
        if len(fr_boxes) >= 2:
            h = sorted((abs(b[3] - b[1]) for b in fr_boxes), reverse=True)
            if len(h) >= 2 and h[1] > EPS:
                ratios.append(float(h[0] / h[1]))
    if len(xs) < 8:
        cal.viewpoint = "unknown"
        cal.note("球员框太少，无法判定机位")
        return cal

    asp = float(frame_w) / max(1, frame_h)
    spread_x = float(np.percentile(xs, 92) - np.percentile(xs, 8)) * asp
    spread_y = float(np.percentile(ys, 92) - np.percentile(ys, 8))
    med_h = float(np.median(heights)) if heights else 0.0
    size_ratio = float(np.median(ratios)) if ratios else 1.0
    area_ratio = cal.area_ratio
    foreshort = float(cal.foreshortening)

    # ---- 标定成功时再看球员投影到场地坐标后的分布
    spread_u = spread_v = -1.0
    if cal.ok:
        pts = np.asarray([(x, y) for x, y in zip(xs, ys)], dtype=np.float32)
        court = cal.to_court_norm(pts, frame_w, frame_h)
        good = (np.isfinite(court).all(axis=1)
                & (np.abs(court[:, 0]) < 3.0) & (np.abs(court[:, 1]) < 3.0))
        cal.in_court_ratio = float(np.mean(good))
        if good.sum() >= 8:
            spread_u = float(np.percentile(court[good, 0], 92) - np.percentile(court[good, 0], 8))
            spread_v = float(np.percentile(court[good, 1], 92) - np.percentile(court[good, 1], 8))
        # 「场内」的另一个口径：直接拿**框底边中心**去判多边形（不经过单应）。
        # 底边中心＝球员在场上的落点，也就是球员检测的 ROI 真正用的那个判据；
        # 用框中心判会偏低（人框的上半截常常在场地之外：举手、跳起、或者框把
        # 记分牌一起框进去了），得到的是一个没人用的口径。
        if (cal.polygon_norm or cal.quad_norm) and parsed:
            arr = np.asarray(parsed, dtype=np.float32)
            ratio = float(np.mean(cal.contains_box_bottom(arr)))
            if abs(ratio - float(np.mean(good))) > 0.05:
                cal.note(f"按场地多边形判定，球员底边有 {ratio:.0%} 落在场内"
                         f"（按单应坐标算 {float(np.mean(good)):.0%}；"
                         f"前者才是球员筛选用到的口径）")
            cal.in_court_ratio = ratio

    cal.note(f"画面横向分布 {spread_x:.2f} / 纵向分布 {spread_y:.2f}（都是画面高度为单位），"
             f"球员框高中位数 {med_h:.3f}，同屏高度比 {size_ratio:.2f}")
    if cal.ok and spread_u >= 0:
        cal.note(f"场地坐标下的分布：横向 {spread_u:.2f} / 纵向 {spread_v:.2f} 个场宽，"
                 f"近大远小系数 {foreshort:.2f}，场地区域占画面 {area_ratio:.0%}")
    elif not cal.ok:
        cal.note("没有场地标定，只能按球员在画面里的分布做粗略判断")

    ratios_uv = spread_u / max(spread_v, 1e-3) if spread_u >= 0 else 0.0
    horiz = spread_x / max(spread_y, 1e-3)
    ratio_hint = hint if hint in ("rear", "side", "elevated", "overhead") else None

    # 球员框在画面里的高度是判定机位高度最直接的量：机位越高，球员越小。
    # 阈值按「人在画面里占多高」划：俯拍/高机位通常 < 0.12，正常斜视机位
    # 0.12~0.5，特写 > 0.5。实测素材是 0.18~0.22（低机位超广角）。
    tall = med_h >= 0.16
    short = med_h < 0.11

    conf = 0.45
    if not cal.ok:
        # 没有标定：只能用「球员往哪个方向铺开」和「大小差 / 框高」
        if short and horiz < 1.6 and size_ratio < 2.0:
            cal.viewpoint, conf = "elevated", 0.45
        elif horiz > 1.8 and size_ratio < 2.2:
            cal.viewpoint, conf = "side", 0.5
        elif size_ratio > 1.8 and horiz < 1.6:
            cal.viewpoint, conf = "rear", 0.55
        else:
            cal.viewpoint, conf = "unknown", 0.3
    elif ratio_hint:
        # 用户/上层明确指定了机位：信它，但把实测到的两个量记下来供核对
        cal.viewpoint = ratio_hint
        conf = 0.9
        cal.note(f"采用指定的机位：{VIEWPOINT_LABEL.get(ratio_hint, ratio_hint)}")
    elif short and foreshort > 0.80 and area_ratio > 0.30:
        cal.viewpoint = "overhead"
        conf = min(0.9, 0.5 + area_ratio * 0.5)
    elif short:
        cal.viewpoint = "elevated"
        conf = 0.6
    elif ratios_uv > 1.5 and size_ratio < 2.2:
        cal.viewpoint = "side"
        conf = min(0.9, 0.5 + 0.12 * min(ratios_uv - 1.5, 3.0))
    elif ratios_uv < 1.0 and size_ratio > 1.35:
        cal.viewpoint = "rear"
        conf = min(0.9, 0.5 + 0.15 * min(size_ratio - 1.35, 3.0))
    elif horiz > 1.8 and size_ratio < 2.2:
        cal.viewpoint = "side"
        conf = 0.5
    elif size_ratio > 1.6 and tall:
        cal.viewpoint = "rear"
        conf = 0.55
    elif tall:
        # 球员够大说明机位不高、离场地不远；分不清侧方还是后方时按后方处理
        # （后方是最常见的拍法，先验参数也按它调）
        cal.viewpoint = "rear"
        conf = 0.4
    else:
        cal.viewpoint = "unknown"
        conf = 0.3
    cal.confidence = float(np.clip(conf, 0.0, 1.0))
    cal.note(f"判定机位：{VIEWPOINT_LABEL.get(cal.viewpoint, cal.viewpoint)}"
             f"（置信度 {cal.confidence:.2f}）")
    return cal


# ------------------------------------------------------------------ 主入口


def _sample_frames(video_path: str, count: int = 24, max_side: int = 320) -> list[np.ndarray]:
    """在整条视频上均匀抽若干帧（用于颜色/场地估计），结果落盘缓存。"""
    cached = _load_sampled(video_path, count, max_side)
    if cached is not None:
        return cached
    frames = _read_sampled(video_path, count, max_side)
    _save_sampled(video_path, count, max_side, frames)
    return frames


def _sample_key(video_path: str, count: int, max_side: int) -> str:
    import hashlib
    from pathlib import Path

    p = Path(video_path)
    #: 抽样逻辑的版本号。改了抽帧 / 标定算法就要改它，否则会读到旧缓存，
    #: 表现为「代码改了但结果一点没变」，非常难查。
    version = "2"
    try:
        st = p.stat()
        sig = f"{version}|{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{count}|{max_side}"
    except OSError:
        sig = f"{version}|{video_path}|{count}|{max_side}"
    return hashlib.sha1(sig.encode("utf-8", "replace")).hexdigest()[:16]


def _sample_cache_path(video_path: str, count: int, max_side: int):
    from pathlib import Path

    try:
        from ..config import CACHE_DIR

        root = Path(CACHE_DIR) / "calib"
    except Exception:  # pragma: no cover
        root = Path(__file__).resolve().parents[3] / "data" / "cache" / "calib"
    return root / f"{_sample_key(video_path, count, max_side)}.npz"


def _load_sampled(video_path: str, count: int, max_side: int) -> list[np.ndarray] | None:
    """读缓存。抽帧要顺序解码 16 次，在 4K 长素材上要一两秒，值得缓存。"""
    p = _sample_cache_path(video_path, count, max_side)
    if not p.is_file():
        return None
    try:
        z = np.load(p)
        arr = z["frames"]
        return [arr[i] for i in range(arr.shape[0])]
    except Exception:
        return None


def _save_sampled(video_path: str, count: int, max_side: int,
                  frames: list[np.ndarray]) -> None:
    if not frames:
        return
    p = _sample_cache_path(video_path, count, max_side)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            return
        np.savez_compressed(p, frames=np.stack(frames, axis=0))
    except Exception:
        pass


def _read_sampled(video_path: str, count: int, max_side: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        for i in range(count):
            pos = int(total * (i + 0.5) / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fr = cap.read()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                s = max_side / max(1, max(h, w))
                if s < 1.0:
                    fr = cv2.resize(fr, (max(2, int(w * s)), max(2, int(h * s))),
                                    interpolation=cv2.INTER_AREA)
                frames.append(fr)
    else:
        while len(frames) < count:
            ok, fr = cap.read()
            if not ok:
                break
            if len(frames) % 10 == 0:
                frames.append(fr)
    cap.release()
    return frames


def calibrate(
    video_path: str,
    player_boxes: list | None = None,
    player_fps: float = 0.0,
    frame_size: tuple[int, int] | None = None,
    samples: int = 24,
    hint: Viewpoint | None = None,
) -> CourtCalibration:
    """从视频里标定场地与机位。

    Args:
        video_path: 视频路径（用代理即可，抽样帧会被缩到 320 像素内）。
        player_boxes: ``PlayerSignal.frame_boxes``（归一化框）；用于判定机位。
        player_fps: 上面那个序列的帧率。
        frame_size: 原始画面 (w, h)；不给则从抽样帧推断。
        hint: 用户在界面上指定的机位；给了就以它为准（仍然会算并记录实测指标）。

    Returns:
        :class:`CourtCalibration`；失败时 ``ok=False``，调用方应退回通用模式。
    """
    frames = _sample_frames(video_path, count=samples)
    if not frames:
        cal = CourtCalibration()
        cal.note("无法读取视频抽样帧")
        return cal
    h0, w0 = frames[0].shape[:2]
    if frame_size is None:
        frame_size = (w0, h0)
    fw, fh = int(frame_size[0]), int(frame_size[1])

    base = estimate_court_color(frames)
    if base.ratio < 0.35:
        cal = CourtCalibration(color=base, aspect=float(fw) / max(1, fh))
        cal.note(f"地面颜色不明确（主色 {base.name}，占比 {base.ratio:.0%}），跳过几何标定")
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    color, mask = refine_court_color(frames, base)
    cal = CourtCalibration(color=color, aspect=float(fw) / max(1, fh))
    if mask is None:
        cal.note(f"地面主色 {color.name}，但没能圈出稳定的场地区域，跳过几何标定")
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    # 逐行判定场地的上下边界（这一步用饱和度把地面和墙分开，见 measure_court_band）
    rows, band_info = measure_court_band(frames)
    if not rows.any():
        cal.note(f"未能从逐行分析里定出场地范围（{band_info.get('reason', '行匹配不足')}），跳过几何标定")
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est

    # 在多帧上用「行判定 + 颜色」建 mask 并取交叠，再按底部种子取连通块
    acc: np.ndarray | None = None
    for fr in frames:
        m = court_mask_from_rows(fr, rows, float(band_info["hue"]), float(band_info["sat_thr"]))
        acc = m.astype(np.float32) if acc is None else acc + m.astype(np.float32)
    assert acc is not None
    acc /= float(len(frames))
    union = (acc > 0.5).astype(np.uint8) * 255
    seed_comp = _seed_component(union, min_area_ratio=0.03)
    comp = seed_comp if seed_comp is not None else union
    cover = float(np.mean(comp > 0))
    cal.note(f"场地区域占画面 {cover:.0%}")

    quad_small = find_court_poly(comp, min_area_ratio=0.03)
    if quad_small is None:
        cal.note("未能从颜色 mask 里拟合出场地边界")
        est = estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)
        est.ok = False
        return est
    sx = float(fw) / comp.shape[1]
    sy = float(fh) / comp.shape[0]
    poly = quad_small.astype(np.float32) * np.array([sx, sy], dtype=np.float32)
    cal = build_calibration(poly, fw, fh, color, source="auto")
    cal.note(f"地面主色 {color.name}（色相 {band_info.get('hue')}），场地区域占画面 {cover:.0%}")
    return estimate_viewpoint(cal, player_boxes, player_fps, fw, fh, hint=hint)


def refine_viewpoint(cal: CourtCalibration, player_boxes: list | None,
                     player_fps: float, frame_size: tuple[int, int],
                     hint: Viewpoint | None = None) -> CourtCalibration:
    """球员检测跑完之后再判一次机位（这次有真实球员框可用）。"""
    if player_boxes is None or player_fps <= 0 or not player_boxes:
        return cal
    cal.notes = [n for n in cal.notes
                 if not n.startswith(("没有球员框", "球员框太少", "画面横向分布",
                                      "场地坐标下的分布", "判定机位"))]
    return estimate_viewpoint(cal, player_boxes, player_fps,
                              int(frame_size[0]), int(frame_size[1]), hint=hint)


def detect_viewpoint_only(player_boxes: list | None, player_fps: float,
                          frame_size: tuple[int, int]) -> Viewpoint:
    """没有场地标定时，仅凭球员框的相对分布给一个**粗**机位判断。

    只用「球员在画面里的横向/纵向分布」和「同屏框面积比」：

    * 两名球员横向拉得很开、框大小接近 -> 边线侧方；
    * 一近一远、框大小差明显 -> 场地后方；
    * 球员挤在画面中央一小块、框都很小 -> 高机位/俯拍。

    这个判断不如带标定的版本可靠，所以只用于「选先验参数」，
    判断不出来时返回 ``unknown``（下游按通用参数跑）。
    """
    if not player_boxes or player_fps <= 0:
        return "unknown"
    xs: list[float] = []
    ys: list[float] = []
    ratios: list[float] = []
    sizes: list[float] = []
    for fr in player_boxes:
        if not fr:
            continue
        for item in fr:
            try:
                _, x1, y1, x2, y2 = item
            except Exception:
                continue
            xs.append(0.5 * (float(x1) + float(x2)))
            ys.append(0.5 * (float(y1) + float(y2)))
            sizes.append(abs(float(y2) - float(y1)))
        if len(fr) >= 2:
            a = sorted((abs(float(b[3]) - float(b[1])) for b in fr if len(b) >= 5), reverse=True)
            if len(a) >= 2 and a[1] > EPS:
                ratios.append(a[0] / a[1])
    if len(xs) < 8:
        return "unknown"
    asp = float(frame_size[0]) / max(1, float(frame_size[1]))
    spread_x = float(np.percentile(xs, 92) - np.percentile(xs, 8)) * asp
    spread_y = float(np.percentile(ys, 92) - np.percentile(ys, 8))
    size_ratio = float(np.median(ratios)) if ratios else 1.0
    med_h = float(np.median(sizes)) if sizes else 0.0
    if med_h < 0.10 and spread_x < 0.35 and spread_y < 0.25:
        return "elevated"
    if spread_x > 1.5 * max(spread_y, 0.05) and size_ratio < 2.0:
        return "side"
    if size_ratio > 1.6:
        return "rear"
    return "unknown"


__all__ = [
    "COURT_LENGTH_M",
    "COURT_WIDTH_M",
    "MAX_POLY_POINTS",
    "MIN_POLY_POINTS",
    "VIEWPOINT_LABEL",
    "CourtCalibration",
    "CourtColor",
    "build_calibration",
    "calibrate",
    "court_mask",
    "detect_viewpoint_only",
    "estimate_court_color",
    "estimate_viewpoint",
    "find_court_poly",
    "find_court_quad",
    "fit_quad_from_poly",
    "order_poly",
    "order_quad",
    "offset_poly",
    "point_in_poly",
    "poly_area_norm",
    "refine_viewpoint",
]
