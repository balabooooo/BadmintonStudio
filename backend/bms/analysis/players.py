"""人物检测与跟踪：从画面里挑出「正在比赛的球员」。

场景特点（已实测确认）：
    * 机位完全静止的超广角鱼眼低机位，架在一块场地的后方；
    * 正在比赛的球员是画面中**最大**的两个人框，会跑动/跨步/起跳，
      位置落在画面高度约 0.40~0.65 的横带里；
    * 其余是大量背景人员（其他场地的人、观众、工作人员）：框更小、
      基本不动，其中还有「一直在旁边小范围走动的工作人员」这种干扰项。

因此本模块做三件事：
    1. **检测**：ultralytics YOLO（``classes=[0]`` 只要人），按 ``sample_fps``
       抽帧后**批量**送 GPU 推理（绝不逐帧调用）；
    2. **跟踪**：自己实现的简易多目标跟踪器，代价 = 归一化框 IoU + 中心距离，
       用匀速外推预测后再匹配，允许丢失若干帧，尽量避免把一名球员断成两条轨迹；
    3. **判定比赛球员**：按「框大 + 速度快 + 出场久」综合打分，并对
       「长时间低速」「水平位置几乎固定」「长期贴画面边缘」等特征扣分，
       最后取综合分最高的 1~4 条轨迹（通常 2 条，双打可能 4 条）。

只依赖 ultralytics / opencv / numpy / scipy / torch，无 GPU 时自动回退 CPU。
"""

from __future__ import annotations

import bisect
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .court_calib import point_in_poly

# ------------------------------------------------------------------ 常量

#: 低于该置信度的框直接忽略（跟踪用）
LOW_CONF = 0.15
#: 允许轨迹「丢失」的采样帧数，超过则结束该轨迹
MAX_MISSING = 15
#: 归一化框面积上限：超过它基本是「人贴到镜头前」的异常大框
MAX_BOX_AREA = 0.12
#: 归一化框高下限：更小的框必然是噪声
MIN_BOX_HEIGHT = 0.02
#: 框的宽高比（w/h）合理区间，用来丢掉横条/竖条误检
ASPECT_RANGE = (0.12, 1.8)
#: 同一帧内几乎重叠的重复框的 IoU 阈值
DUP_IOU = 0.75
#: 匹配门限：IoU 下限
IOU_GATE = 0.03
#: 匹配门限：中心距离下限（以框高为单位，乘以框高得到实际门限）
DIST_GATE_RATIO = 0.45
#: 匹配门限：相邻两帧之间的框面积比值上限（防止「跳」到旁边另一个人身上）
AREA_RATIO_MAX = 3.5
#: 轨迹合并：断开时间上限（秒）。球员跑出画面/被完全挡住时可能断几秒，
#: 实测片段里有 3.2 秒的断口，这里留到 5 秒；位置与尺度检查仍然很严。
MERGE_MAX_GAP = 5.0
#: 轨迹合并：位置差上限（以画面高度为单位）
MERGE_MAX_DIST = 0.10
#: 轨迹合并：框面积比值上限
MERGE_AREA_RATIO = 3.5
#: 轨迹合并：时间上重叠时，重合帧里「像同一个人」的比例下限
MERGE_OVERLAP_RATIO = 0.5
#: 轨迹越长越可信：观测达到该帧数后，允许丢失的时间翻倍
RELIABLE_FRAMES = 30

#: 尺寸筛选：直方图的分箱数与上限（框高以画面高度为单位，0.5 以上基本是特写）
SIZE_HIST_BINS = 24
SIZE_HIST_MAX = 0.50
#: 尺寸筛选：随统计结果一起下发的样本数上限（界面用它实时拖动阈值预览）
SIZE_SAMPLE_MAX = 600
#: 场地多边形 ROI 的外扩量（归一化单位，每一条边都向外让出这么多）。
#: 与旧的矩形 ROI 外扩 0.04 一致。球员正好站在边线上时必须算「场内」：
#: 判错方向的代价不对称 —— 多留一个人只是噪声（后面还有尺寸筛和活跃度评分），
#: 漏掉真球员会让下游的活跃度曲线和裁切跟随直接断档。
ROI_POLY_MARGIN = 0.04

Progress = Callable[[float, str], None]

#: 相对工程根目录的默认路径（导入 bms.config 失败时的兜底）
_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ 公开数据结构


@dataclass
class PlayerTrack:
    """一个被持续跟踪的人。"""

    track_id: int
    frames: list[int]           # 出现过的帧序号（相对分析起点，从 0 开始）
    times: list[float]          # 对应秒数
    boxes: list[tuple[float, float, float, float]]   # 归一化 xyxy (0~1)
    speeds: list[float]         # 每个出现帧的归一化速度（框中心位移/秒，按画面高度归一）
    confidences: list[float]
    # 汇总统计
    mean_area: float = 0.0      # 归一化框面积均值
    max_area: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0
    active_score: float = 0.0   # 综合「活跃度」0~1，用于挑出比赛球员
    total_travel: float = 0.0   # 归一化累计位移

    @property
    def n(self) -> int:
        """出现过的采样帧数。"""
        return len(self.frames)


@dataclass
class PlayerSignal:
    fps: float
    duration: float
    tracks: list[PlayerTrack] = field(default_factory=list)
    #: 每帧「场上活跃球员」数量（按 active_player_ids 统计）
    active_count: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 每帧活跃球员的平均速度（归一化/秒）
    active_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 每帧最大速度
    max_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 每帧画面中所有人（含背景）的平均速度
    crowd_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: 被判定为「比赛球员」的 track_id 列表（通常 2 个，双打可能 4 个）
    active_player_ids: list[int] = field(default_factory=list)
    #: 每帧每人的框，用于自动裁切跟随：[frame] -> [(track_id, x1,y1,x2,y2), ...]（只含 active）
    frame_boxes: list[list[tuple[int, float, float, float, float]]] = field(default_factory=list)
    #: 人物框尺寸筛选的统计（过滤前的框高分布、筛掉多少、参考尺度），供界面调参
    size_stats: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ 小工具


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个归一化 xyxy 框的 IoU。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_height(box: Sequence[float]) -> float:
    return abs(float(box[3]) - float(box[1]))


@dataclass
class SizeFilter:
    """人物框尺寸筛选：把「明显不是这场比赛球员」的人框在检测阶段就丢掉。

    为什么需要它：原有的尺寸门限是**自适应**的（拿「每帧最大框的 90 分位」
    当参考尺度，再砍掉明显偏小的轨迹），在「球员是画面里最大的人」这种素材上
    很好用。但有两种素材它会失灵：

    * **看台 / 观众离相机比球员更近**（边线侧方机位、看台就在镜头后面）：
      参考尺度会被最大的观众框劫持，真正的球员反而成了「偏小」的那一批；
    * **全景 / 鱼眼**：同一个球员在画面中心与画面边角的框高能差一倍以上，
      「多大才算球员」在画面不同位置本来就不是同一个数。

    这两种情况只能靠用户指认，所以这里给出两个筛选口径：

    * ``absolute``：框高占画面高度的**绝对**比例。适合机位固定、人框大小稳定；
    * ``relative``：框高 ÷ **同帧最大框高**的比值。适合畸变严重 / 观众更近的
      素材 —— 它在每一帧里重新归一，天然免疫「位置不同框大小不同」。

    面积上下限两种口径下都同时生效（面积对「贴到镜头前的巨大误检」更敏感，
    因为畸变会把框拉宽而不是拉高）。

    ``mode == "off"`` 时不做任何筛选，行为与旧版本完全一致。
    """

    mode: str = "off"
    min_height: float = 0.05
    max_height: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0

    @property
    def active(self) -> bool:
        if self.mode not in ("absolute", "relative"):
            return False
        return bool(self.min_height > 0 or self.max_height > 0
                    or self.min_area > 0 or self.max_area > 0)

    def thresholds(self, ref_h: float) -> tuple[float, float]:
        """给定「同帧最大框高」，算出实际生效的框高上下限（0 = 不限）。"""
        if self.mode == "relative":
            if ref_h <= 0.02:            # 这一帧没有可信参考（人都很小）
                return 0.0, 0.0
            return (self.min_height * ref_h,
                    self.max_height * ref_h if self.max_height > 0 else 0.0)
        return float(self.min_height), float(self.max_height)

    def keep(self, box: Sequence[float], ref_h: float = 0.0) -> bool:
        """这个框是否通过筛选。``ref_h`` 只在 ``relative`` 口径下起作用。"""
        if not self.active:
            return True
        lo, hi = self.thresholds(ref_h)
        h = _box_height(box)
        if lo > 0 and h < lo:
            return False
        if hi > 0 and h > hi:
            return False
        area = _box_area(box)
        if self.min_area > 0 and area < self.min_area:
            return False
        if self.max_area > 0 and area > self.max_area:
            return False
        return True

    def as_dict(self) -> dict[str, float | str]:
        return {"mode": self.mode, "min_height": round(float(self.min_height), 4),
                "max_height": round(float(self.max_height), 4),
                "min_area": round(float(self.min_area), 5),
                "max_area": round(float(self.max_area), 5)}


def size_filter_from(raw: "SizeFilter | dict | None") -> SizeFilter:
    """把参数（模型 / 字典 / 本类实例）统一成 :class:`SizeFilter`。

    pipeline 传的是 ``AnalysisParams`` 里的几个字段，界面调试时可能直接传字典，
    所以这里做一次容错归一化，别让类型问题把球员检测整条链路带崩。
    """
    if raw is None:
        return SizeFilter()
    if isinstance(raw, SizeFilter):
        return raw
    if isinstance(raw, dict):
        def _f(key: str, default: float) -> float:
            try:
                v = float(raw.get(key, default))
                return v if np.isfinite(v) else default
            except Exception:
                return default

        mode = str(raw.get("mode") or "off")
        return SizeFilter(
            mode=mode if mode in ("off", "absolute", "relative") else "off",
            min_height=max(0.0, _f("min_height", 0.05)),
            max_height=max(0.0, _f("max_height", 0.0)),
            min_area=max(0.0, _f("min_area", 0.0)),
            max_area=max(0.0, _f("max_area", 0.0)),
        )
    return SizeFilter()


def _sanitize_box(x1: float, y1: float, x2: float, y2: float,
                  viewpoint: str = "unknown") -> tuple[float, float, float, float] | None:
    """把检测框归一化、裁剪到 [0,1]，并丢掉明显不合理的框。

    ``viewpoint`` 会影响几何门限：俯拍/高机位下球员从上方看是「矮而宽」的，
    宽高比会超过 1.8，而远处的人又只占画面高度的百分之几 —— 用低机位那套
    写死的门限会把真实球员直接丢掉。所以这里按机位放宽或者在信息不足时
    干脆不做这些硬性裁剪（置信度由检测器自己给）。
    """
    w = max(1e-9, float(x2) - float(x1))
    h = max(1e-9, float(y2) - float(y1))
    box = (max(0.0, float(x1)), max(0.0, float(y1)), min(1.0, float(x2)), min(1.0, float(y2)))
    if box[2] - box[0] <= 1e-6 or box[3] - box[1] <= 1e-6:
        return None
    top_view = viewpoint in ("overhead", "elevated")
    lo_h = 0.008 if top_view else MIN_BOX_HEIGHT
    lo_r, hi_r = (0.08, 6.0) if top_view else ASPECT_RANGE
    hi_area = 0.30 if top_view else MAX_BOX_AREA
    if box[3] - box[1] < lo_h:
        return None
    ratio = w / h
    if ratio < lo_r or ratio > hi_r:
        return None
    if _box_area(box) > hi_area:      # 贴到镜头前的巨大误检
        return None
    return box


def _same_person(b1: Sequence[float], b2: Sequence[float], aspect: float = 1.78) -> bool:
    """判断两个框是不是「同一个人的重复检测」。

    实测发现：480p 小目标上 YOLO 经常对同一个人输出两个略有差异的框
    （IoU 0.4~0.7；跨步时一个是「身体竖条」、一个是「连拍带人的横条」）。
    如果不清理，同一个人会同时长出两条轨迹、彼此抢检测，
    是轨迹碎片化的主要来源。因此除了 IoU，还允许按
    「中心几乎重合 + 尺度接近」判定重复。

    ``aspect`` 必须是**运行时**的画面宽高比。以前这里写死 1.78，
    竖屏素材（9:16）下会把横向距离放大 3 倍以上：两个站在同一高度、
    水平相隔一段距离的不同球员会被判成「同一个人」而被合并掉。
    """
    iou = _iou(b1, b2)
    if iou > DUP_IOU:
        return True
    h1 = max(1e-6, b1[3] - b1[1])
    h2 = max(1e-6, b2[3] - b2[1])
    if iou <= 0.30 and abs(_box_center(b1)[1] - _box_center(b2)[1]) > 0.5 * min(h1, h2):
        return False
    c1, c2 = _box_center(b1), _box_center(b2)
    dist = math.hypot((c1[0] - c2[0]) * max(0.2, float(aspect)), c1[1] - c2[1])
    if dist > 0.45 * min(h1, h2):
        return False
    a1 = _box_area(b1)
    a2 = _box_area(b2)
    ratio = a1 / max(a2, 1e-9)
    return 0.30 <= ratio <= 3.2


def _dedup_dets(
    dets: list[tuple[tuple[float, float, float, float], float]],
    aspect: float = 1.78,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """同一帧内去掉「同一个人」的重复框，保留置信度最高的那个。"""
    ordered = sorted(dets, key=lambda z: -z[1])
    kept: list[tuple[tuple[float, float, float, float], float]] = []
    for box, conf in ordered:
        if any(_same_person(box, kb, aspect) for kb, _ in kept):
            continue
        kept.append((box, conf))
    return kept


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


# ------------------------------------------------------------------ 环境准备


def _data_paths() -> tuple[Path, Path]:
    """返回 (可写的 ultralytics 配置目录, 权重目录)。"""
    try:  # 优先复用工程配置，保证与后端其它模块写到同一处
        from ..config import DATA_DIR, MODELS_DIR  # type: ignore

        return Path(DATA_DIR) / "yolo", Path(MODELS_DIR)
    except Exception:  # pragma: no cover - 独立运行时的兜底
        return _ROOT / "data" / "yolo", _ROOT / "models"


def _prepare_yolo_env(cfg_dir: Path) -> None:
    """显式设置 YOLO_CONFIG_DIR，避免 ultralytics 去写不可写的用户目录。"""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cfg_dir)


def _resolve_weights(model_name: str, models_dir: Path) -> str:
    """定位权重：显式路径 > models/ > 工程根目录 > 交给 ultralytics 下载到 models/。"""
    p = Path(model_name)
    if p.is_file():
        return str(p)
    models_dir.mkdir(parents=True, exist_ok=True)
    cand = models_dir / model_name
    if cand.is_file():
        return str(cand)
    root_cand = _ROOT / model_name
    if root_cand.is_file():                     # 已经下载过就别重复下载
        return str(root_cand)
    return str(cand)


def _pick_device(device: str) -> str:
    """无 GPU 时自动回退 CPU。"""
    want = (device or "cpu").strip()
    if want.lower().startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return want


# ------------------------------------------------------------------ 跟踪器


class _TrackBuf:
    """内部轨迹缓冲：保存观测序列，并维护匀速预测所需的状态。"""

    __slots__ = (
        "track_id", "aspect", "frames", "times", "boxes", "speeds", "confs",
        "last_box", "last_center", "last_time", "vx", "vy", "missed", "dead",
    )

    @property
    def n(self) -> int:
        """已记录的观测帧数。"""
        return len(self.frames)

    def __init__(self, track_id: int, aspect: float) -> None:
        self.track_id = track_id
        self.aspect = aspect
        self.frames: list[int] = []
        self.times: list[float] = []
        self.boxes: list[tuple[float, float, float, float]] = []
        self.speeds: list[float] = []
        self.confs: list[float] = []
        self.last_box: tuple[float, float, float, float] | None = None
        self.last_center: tuple[float, float] | None = None
        self.last_time: float = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0
        self.dead = False

    # ---- 预测（匀速外推） ----
    def predict(self, time: float) -> tuple[float, float] | None:
        """用最近的速度把中心点外推到 ``time`` 时刻。

        丢失越久，速度衰减得越多（球员被遮挡后往往已经停下/变向），
        避免预测点飞得太远、把旁边的其他人「抢」过来。
        """
        if self.last_center is None or self.last_box is None:
            return None
        gap = max(0.0, time - self.last_time)
        decay = 0.80 ** min(self.missed, 6)
        cx = self.last_center[0] + self.vx * gap * decay
        cy = self.last_center[1] + self.vy * gap * decay
        return (cx, cy)

    def predicted_box(self, time: float) -> tuple[float, float, float, float] | None:
        """外推后的框（尺寸沿用最后一次观测，用来算 IoU）。"""
        c = self.predict(time)
        if c is None or self.last_box is None:
            return None
        w = self.last_box[2] - self.last_box[0]
        h = self.last_box[3] - self.last_box[1]
        return (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)

    # ---- 观测更新 ----
    def observe(self, frame_idx: int, time: float, box: tuple[float, float, float, float], conf: float) -> None:
        cx, cy = _box_center(box)
        speed = 0.0
        if self.last_center is not None:
            dt = time - self.last_time
            if dt > 1e-6:
                # 画面宽高比换算：dx 以画面宽归一，乘 aspect 后与 dy 同尺度（都以画面高度为单位）
                dx = (cx - self.last_center[0]) * self.aspect
                dy = cy - self.last_center[1]
                speed = math.hypot(dx, dy) / dt
                vx = (cx - self.last_center[0]) / dt
                vy = (cy - self.last_center[1]) / dt
                # 指数平滑，抑制单帧抖动；同时限幅避免误检把速度带飞
                self.vx = 0.5 * self.vx + 0.5 * float(np.clip(vx, -4.0, 4.0))
                self.vy = 0.5 * self.vy + 0.5 * float(np.clip(vy, -4.0, 4.0))
        self.frames.append(int(frame_idx))
        self.times.append(float(time))
        self.boxes.append(box)
        self.speeds.append(float(speed))
        self.confs.append(float(conf))
        self.last_box = box
        self.last_center = (cx, cy)
        self.last_time = float(time)
        self.missed = 0

    # ---- 合并另一条轨迹（同一人被跟成两条轨迹时） ----
    def absorb(self, other: "_TrackBuf") -> None:
        """把另一条轨迹的观测并进来（按时间排序，保证时间轴单调）。"""
        self.frames.extend(other.frames)
        self.times.extend(other.times)
        self.boxes.extend(other.boxes)
        self.speeds.extend(other.speeds)
        self.confs.extend(other.confs)
        order = sorted(range(len(self.times)), key=lambda k: self.times[k])
        self.frames = [self.frames[k] for k in order]
        self.times = [self.times[k] for k in order]
        self.boxes = [self.boxes[k] for k in order]
        self.speeds = [self.speeds[k] for k in order]
        self.confs = [self.confs[k] for k in order]
        if other.times and other.times[-1] >= self.times[-1]:
            self.last_box = other.last_box
            self.last_center = other.last_center
            self.last_time = other.last_time
            self.vx, self.vy = other.vx, other.vy
            self.missed = other.missed
        other.dead = True


class _MultiObjectTracker:
    """极简多目标跟踪：IoU + 中心距离代价，匈牙利匹配 + 匀速预测。

    与朴素实现的差别（都是为了「不要把球员断成两条轨迹」）：
        * 用匀速外推后的**预测框**去匹配，快速跑动时依然对得上；
        * 距离门限随框高自适应，并对**框面积突变**设上限——否则球员短暂消失时，
          轨迹会「跳」到旁边那个更小的背景路人身上（身份交换）；
        * 观测越多的轨迹越可信，匹配代价有小幅优惠，丢失容忍度也翻倍；
        * 每帧结束后做一次「同一人只留一条轨迹」的收尾合并。
    """

    def __init__(
        self,
        fps: float,
        aspect: float,
        max_missing: int = MAX_MISSING,
    ) -> None:
        self.fps = max(1e-6, fps)
        self.aspect = aspect
        self.max_missing = int(max_missing)
        self.tracks: list[_TrackBuf] = []
        self._next_id = 1

    # 距离统一换算成「画面高度」为单位，便于和 IoU 一起构成代价
    def _dist(self, c1: tuple[float, float], c2: tuple[float, float]) -> float:
        return math.hypot((c1[0] - c2[0]) * self.aspect, c1[1] - c2[1])

    def update(
        self,
        frame_idx: int,
        time: float,
        dets: list[tuple[tuple[float, float, float, float], float]],
    ) -> None:
        """用当前帧的检测更新所有轨迹。"""
        alive = [t for t in self.tracks if not t.dead]
        n_t, n_d = len(alive), len(dets)
        matches: list[tuple[int, int]] = []

        if n_t and n_d:
            det_area = [_box_area(b) for b, _c in dets]
            cost = np.full((n_t, n_d), 1e6, dtype=np.float64)
            for i, tr in enumerate(alive):
                pb = tr.predicted_box(time)
                pc = tr.predict(time)
                if pb is None or pc is None or tr.last_box is None:
                    continue
                tr_h = max(1e-6, tr.last_box[3] - tr.last_box[1])
                tr_area = max(1e-9, _box_area(tr.last_box))
                # 距离门限：以「框高的比例」为单位，人越大允许的位移越大
                gate = max(0.025, DIST_GATE_RATIO * tr_h)
                for j, (box, _conf) in enumerate(dets):
                    # 门限一：面积突变过大 -> 大概率是旁边的另一个人，宁可不匹配
                    ratio = det_area[j] / tr_area
                    if ratio > AREA_RATIO_MAX or ratio < 1.0 / AREA_RATIO_MAX:
                        continue
                    iou = _iou(pb, box)
                    dist = self._dist(pc, _box_center(box))
                    # 门限二：IoU 够大 或 中心够近
                    if iou < IOU_GATE and dist > gate:
                        continue
                    # 代价：IoU 越小越贵，中心越远越贵，丢失越久越倾向新建轨迹；
                    # 观测充分的轨迹有微小优惠，避免身份被新轨迹抢走
                    stable = min(1.0, tr.n / 30.0)
                    cost[i, j] = (
                        (1.0 - iou)
                        + 1.2 * dist
                        + 0.02 * min(tr.missed, 10)
                        - 0.05 * stable
                    )
            if (cost < 1e5).any():
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows.tolist(), cols.tolist()):
                    if cost[r, c] < 1e5:
                        matches.append((r, c))

        used_t = {m[0] for m in matches}
        used_d = {m[1] for m in matches}

        # 已匹配：更新轨迹
        for ti, di in matches:
            box, conf = dets[di]
            alive[ti].observe(frame_idx, time, box, conf)

        # 未匹配检测：先看看它是不是「已经在跟踪的那个人」换了个框型
        # （跨步时 YOLO 会在「身体竖框」和「连拍横框」之间反复横跳）。
        # 如果是，就不再新建轨迹，否则同一个人会长出两条轨迹互相抢检测。
        for j in range(n_d):
            if j in used_d:
                continue
            box, conf = dets[j]
            claimed = False
            for tr in alive:
                if tr.dead or tr.last_box is None or tr.missed > 2:
                    continue
                if _same_person(box, tr.last_box, self.aspect):
                    claimed = True
                    break
            if claimed:
                continue
            tr = _TrackBuf(self._next_id, self.aspect)
            self._next_id += 1
            tr.observe(frame_idx, time, box, conf)
            self.tracks.append(tr)

        # 未匹配轨迹：累加丢失计数，超限则结束
        for i, tr in enumerate(alive):
            if i in used_t:
                continue
            tr.missed += 1
            # 已经稳定跟踪一段时间的轨迹（多半是真球员）容忍更久的遮挡
            limit = self.max_missing * (2 if tr.n >= RELIABLE_FRAMES else 1)
            if tr.missed > limit:
                tr.dead = True

        # 收尾：同一时刻若两条活跃轨迹都指着同一个人（同一帧里被断成两条），
        # 把年轻的那条并进年长的那条。这是「一名球员被断成两条轨迹」的最后一道保险。
        self._consolidate()

    def _consolidate(self) -> None:
        """把「同一帧里指向同一个人」的多条活跃轨迹合并成一条。"""
        live = [t for t in self.tracks if not t.dead and t.n > 0 and t.last_box is not None]
        if len(live) < 2:
            return
        live.sort(key=lambda t: -t.n)
        for i, strong in enumerate(live):
            if strong.dead:
                continue
            for weak in live[i + 1:]:
                if weak.dead or weak.missed > 2 or strong.missed > 2:
                    continue
                if abs(strong.last_time - weak.last_time) > 2.0 / self.fps:
                    continue
                if _same_person(strong.last_box, weak.last_box, self.aspect):
                    strong.absorb(weak)


# ------------------------------------------------------------------ 轨迹合并


def _junction_ok(prev: _TrackBuf, nxt: _TrackBuf, aspect: float) -> bool:
    """判断两条轨迹的「接缝」是否像同一个人。

    分两种情况：
      * **时间上断开**：用速度把前一段外推到后一段的起始时刻，
        位置必须落在 :data:`MERGE_MAX_DIST` 内，尺度也要接近；
      * **时间上重叠**：重叠时间段里，两条轨迹的框大多数要「像同一个人」
        （同一名球员被同时跟成两条轨迹时就是这种情况）。
    """
    if prev.times[-1] < nxt.times[0]:
        if nxt.times[0] - prev.times[-1] > MERGE_MAX_GAP:
            return False
        c_prev = prev.predict(nxt.times[0]) or _box_center(prev.boxes[-1])
        c_new = _box_center(nxt.boxes[0])
        dist = math.hypot((c_prev[0] - c_new[0]) * aspect, c_prev[1] - c_new[1])
        if dist > MERGE_MAX_DIST:
            return False
        a_prev = _box_area(prev.boxes[-1])
        a_new = _box_area(nxt.boxes[0])
        return min(a_prev, a_new) > 1e-9 and max(a_prev, a_new) / min(a_prev, a_new) <= MERGE_AREA_RATIO

    # 时间重叠：抽查较短那条轨迹落在重叠区间内的观测
    o0 = max(prev.times[0], nxt.times[0])
    o1 = min(prev.times[-1], nxt.times[-1])
    if o1 < o0:
        return False
    short, other = (prev, nxt) if prev.n <= nxt.n else (nxt, prev)
    # 用二分查找定位最近邻。原来的 `min(range(other.n), key=...)` 是 O(n)，
    # 套在「每条轨迹都要和短轨迹的每个观测比对」上就是 O(n²)，
    # 30 分钟素材（几百条轨迹、每条上千个观测）会直接把 CPU 打满几十分钟。
    other_times = other.times
    hit = 0
    total = 0
    for k, t in enumerate(short.times):
        if t < o0 or t > o1:
            continue
        total += 1
        j = bisect.bisect_left(other_times, t)
        best = -1
        best_d = 1e9
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < other.n:
                d = abs(other_times[cand] - t)
                if d < best_d:
                    best_d, best = d, cand
        if best < 0 or best_d > 0.2:
            continue
        box = short.boxes[k]
        o_box = other.boxes[best]
        c1, c2 = _box_center(box), _box_center(o_box)
        dist = math.hypot((c1[0] - c2[0]) * aspect, c1[1] - c2[1])
        a1, a2 = _box_area(box), _box_area(o_box)
        ratio = max(a1, a2) / max(1e-9, min(a1, a2))
        if dist <= MERGE_MAX_DIST and ratio <= MERGE_AREA_RATIO:
            hit += 1
    return total > 0 and hit / total >= MERGE_OVERLAP_RATIO


def _merge_tracks(bufs: list[_TrackBuf], aspect: float) -> list[_TrackBuf]:
    """把「同一人被断开的两段轨迹」合并成一条。

    这是「不让球员被追成两条轨迹」的最后一道保险，处理两种情况：

    1. 球员短暂离开画面/被遮挡后又出现（时间上有间隔）；
    2. 同一名球员在同一时刻被跟成了两条并行轨迹（框型来回变化导致）。

    合并前都要过 :func:`_junction_ok` 的接缝检查：位置和尺度对不上就不合，
    避免把旁边两个不同的人粘成一条「大杂烩」。

    注意：已经「结束」的轨迹（长时间没匹配上而停止更新）同样参与合并——
    它们的数据是有效的，而且正是「球员离开画面一会儿又回来」这种需要重连的情况。

    性能：先用「时间区间 + 空间包围盒」做粗筛，避免对全部轨迹两两做接缝检查
    （几十万次比对在长视频上非常慢）。
    """
    live = [t for t in bufs if t.n > 0]
    live.sort(key=lambda t: (t.times[0], -t.n))

    # 预计算每条轨迹的时空范围，用于粗筛
    meta: list[tuple[float, float, float, float, float, float]] = []
    for t in live:
        xs = [b[0] for b in t.boxes]
        ys = [b[1] for b in t.boxes]
        ws = [b[2] for b in t.boxes]
        hs = [b[3] for b in t.boxes]
        meta.append((t.times[0], t.times[-1], min(xs), min(ys), max(xs), max(ys)))
        del ws, hs

    merged: list[_TrackBuf] = []
    merged_meta: list[tuple[float, float, float, float, float, float]] = []
    for tr, mt in zip(live, meta):
        target: _TrackBuf | None = None
        for idx, prev in enumerate(merged):
            pm = merged_meta[idx]
            # 时间上断得太久 -> 跳过
            if pm[1] < tr.times[0] and tr.times[0] - pm[1] > MERGE_MAX_GAP:
                continue
            # 时间上完全晚于对方 -> 不可能接得上（live 已按起始时间排序）
            if pm[0] > mt[1]:
                continue
            # 空间包围盒相距太远 -> 跳过（留出 3 倍容差）
            tol = MERGE_MAX_DIST * 3.0
            if (mt[2] - pm[4] > tol or pm[2] - mt[4] > tol or
                    mt[3] - pm[5] > tol or pm[3] - mt[5] > tol):
                continue
            if _junction_ok(prev, tr, aspect):
                target = prev
                break
        if target is None:
            merged.append(tr)
            merged_meta.append(mt)
        else:
            target.absorb(tr)
            i = merged.index(target)
            old = merged_meta[i]
            merged_meta[i] = (
                min(old[0], mt[0]), max(old[1], mt[1]),
                min(old[2], mt[2]), min(old[3], mt[3]),
                max(old[4], mt[4]), max(old[5], mt[5]),
            )
    return merged


# ------------------------------------------------------------------ 汇总与判定


class _Feat:
    """判定「比赛球员」用的派生特征（内部使用，不进入公开数据结构）。"""

    __slots__ = ("area_p90", "cx_mean", "x_range", "duration", "static_ratio")

    def __init__(self, area_p90: float, cx_mean: float, x_range: float,
                 duration: float, static_ratio: float) -> None:
        self.area_p90 = area_p90
        self.cx_mean = cx_mean
        self.x_range = x_range
        self.duration = duration
        self.static_ratio = static_ratio


def _track_feat(track: PlayerTrack, fps: float) -> _Feat:
    """从轨迹里算出判定所需的派生特征。

    * ``area_p90``：框面积 90 分位数（比均值/最大值都稳，见 :func:`_select_active_players`）；
    * ``cx_mean`` / ``x_range``：水平位置的均值与跨度（识别「一直待在同一小块区域的人」）；
    * ``static_ratio``：低速帧占比（识别「长时间站着不动的人」）。
    """
    areas = np.asarray([_box_area(b) for b in track.boxes], dtype=np.float64)
    cxs = np.asarray([0.5 * (b[0] + b[2]) for b in track.boxes], dtype=np.float64)
    sp = np.asarray(track.speeds[1:], dtype=np.float64) if len(track.speeds) > 1 else np.zeros(0)
    return _Feat(
        area_p90=float(np.percentile(areas, 90)) if areas.size else 0.0,
        cx_mean=float(cxs.mean()) if cxs.size else 0.5,
        x_range=float(cxs.max() - cxs.min()) if cxs.size else 0.0,
        duration=float(track.times[-1] - track.times[0]) if track.times else 0.0,
        static_ratio=float(np.mean(sp < 0.05)) if sp.size else 1.0,
    )


def _summarize(bufs: list[_TrackBuf], fps: float) -> list[PlayerTrack]:
    """把内部轨迹缓冲转换成公开的 PlayerTrack，并算出各种统计量。"""
    out: list[PlayerTrack] = []
    for tr in bufs:
        if tr.n == 0:
            continue
        # 同一帧可能因为轨迹合并留下多个观测：只保留置信度最高的那个，
        # 保证 frames / times 里每一帧只出现一次。
        best: dict[int, int] = {}
        for k, f in enumerate(tr.frames):
            j = best.get(f)
            if j is None or tr.confs[k] > tr.confs[j]:
                best[f] = k
        order = sorted(best.values(), key=lambda k: tr.times[k])
        frames = [tr.frames[k] for k in order]
        times = [tr.times[k] for k in order]
        boxes = [tr.boxes[k] for k in order]
        speeds = [tr.speeds[k] for k in order]
        confs = [tr.confs[k] for k in order]

        areas = np.asarray([_box_area(b) for b in boxes], dtype=np.float64)
        track = PlayerTrack(
            track_id=tr.track_id,
            frames=frames,
            times=times,
            boxes=boxes,
            speeds=speeds,
            confidences=confs,
        )
        track.mean_area = float(areas.mean())
        track.max_area = float(areas.max())
        # 速度统计忽略第 0 帧（无历史，恒为 0），避免拖低均值
        sp = np.asarray(speeds[1:], dtype=np.float64) if len(speeds) > 1 else np.zeros(0)
        track.mean_speed = float(sp.mean()) if sp.size else 0.0
        track.max_speed = float(sp.max()) if sp.size else 0.0
        # 累计位移：相邻出现帧之间的框中心距离之和（按画面高度归一）
        travel = 0.0
        for k in range(1, len(times)):
            p0, p1 = _box_center(boxes[k - 1]), _box_center(boxes[k])
            dt = times[k] - times[k - 1]
            if dt > 4.0 / fps:      # 中间断过帧，不算连续位移
                continue
            travel += math.hypot((p1[0] - p0[0]) * tr.aspect, p1[1] - p0[1])
        track.total_travel = float(travel)
        out.append(track)
    return out


def _box_xyxy(item: Sequence[float]) -> tuple[float, float, float, float] | None:
    """把「逐帧框」的两种表示统一成 ``(x1, y1, x2, y2)``。

    工程里同时存在两种：

    * ``PlayerSignal.frame_boxes``：``(track_id, x1, y1, x2, y2)``（带轨迹号，
      下游要按人取框）；
    * 检测阶段的逐帧框：``(x1, y1, x2, y2)``。

    这里必须同时认两种。**这是一个真实踩到的坑**：``_box_size_stats`` 原来只认
    5 元组（用 ``b[4] - b[2]`` 算框高），而 ``analyze_players`` 传进去的是 4 元组，
    于是「这场比赛里球员大概多大」永远算不出来（``ref`` 恒为 0），
    自适应尺寸硬门限在真实流水线里**从来没有生效过** —— 单元测试里喂的是
    5 元组，所以一直没被发现。少了这道门限，观众和隔壁场地的人会一直挤进候选。
    """
    n = len(item)
    try:
        if n >= 5:
            return (float(item[1]), float(item[2]), float(item[3]), float(item[4]))
        if n >= 4:
            return (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
    except (TypeError, ValueError):
        return None
    return None


def _box_size_stats(boxes: list, aspect: float = 1.7778) -> dict[str, float]:
    """从逐帧球员框估计「这场比赛里球员大概多大」。

    比赛球员是画面里最大的那批人，所以取**每帧最大框**的 90 分位当参考尺度，
    而不是全体中位数（背景人员数量远多于球员，中位数会被他们拉低）。
    做法对机位不敏感：无论机位多高、球场在画面哪一块，最大的那个框
    总是离相机最近的那个人。

    Returns:
        ``{"ref": 参考框高, "min_abs": 绝对下限, "max_abs": 绝对上限}``
        （都以画面高度为单位）。
    """
    per_frame_max: list[float] = []
    for fr in boxes:
        if not fr:
            continue
        hs = []
        for item in fr:
            b = _box_xyxy(item)
            if b is not None:
                hs.append(abs(b[3] - b[1]))
        if hs:
            per_frame_max.append(max(hs))
    if len(per_frame_max) < 5:
        return {"ref": 0.0, "min_abs": 0.0, "max_abs": 1.0}
    arr = np.asarray(per_frame_max, dtype=np.float64)
    ref = float(np.percentile(arr, 90))
    # 参考尺度至少要有画面高度的 6%，否则说明这一整段都没检到像球员的人，
    # 这时候不该拿它去硬筛（否则把所有人都筛掉）
    if ref < 0.06:
        return {"ref": 0.0, "min_abs": 0.0, "max_abs": 1.0}
    return {
        "ref": ref,
        "min_abs": max(0.03, ref * 0.42),
        # 上限放得很宽（4 倍参考尺度）：特写镜头里球员会占很大一块，
        # 这里只想挡掉「整个人贴到镜头前」的误检
        "max_abs": min(1.0, ref * 4.0),
    }


def _build_size_stats(size_filter: SizeFilter,
                      samples: list[list[float]],
                      dropped: int,
                      frames: int,
                      ref_scale: float) -> dict[str, Any]:
    """把「检测到的框」整理成界面能直接用的统计（直方图 + 样本 + 计数）。

    Args:
        samples: 每个框一项 ``[框高, 框面积, 同帧最大框高]``。
        dropped: 被尺寸筛选砍掉的框数。
        frames: 参与统计的采样帧数。
        ref_scale: 自适应参考尺度（每帧最大框高的 90 分位）。

    统计的是**过滤之前**的所有框（只经过几何门限与 ROI）。这样界面才能画出
    「阈值一拖会砍掉多少」—— 如果只统计过滤后的框，用户就永远看不到被砍掉的
    那部分，调参也就没有依据。

    样本里带上「同帧最大框高」和面积，前端就能在本地**精确**模拟两种口径下
    任意阈值的效果：拖滑杆时不需要每动一下就重跑一遍检测。
    """
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, 3) if samples else np.zeros((0, 3))
    h = arr[:, 0] if arr.size else np.zeros(0)
    total = int(h.size)
    edges = np.linspace(0.0, SIZE_HIST_MAX, SIZE_HIST_BINS + 1)
    if total:
        hist, _ = np.histogram(np.clip(h, 0.0, SIZE_HIST_MAX), bins=edges)
        overflow = int(np.sum(h > SIZE_HIST_MAX))
    else:
        hist = np.zeros(SIZE_HIST_BINS, dtype=np.int64)
        overflow = 0
    step = max(1, total // SIZE_SAMPLE_MAX) if total else 1
    sample = [[round(float(v), 5) for v in arr[i]] for i in range(0, total, step)][:SIZE_SAMPLE_MAX]
    return {
        **size_filter.as_dict(),
        "active": bool(size_filter.active),
        "total": total,
        "kept": int(total - dropped),
        "dropped": int(dropped),
        "frames": int(frames),
        # 自适应参考尺度（每帧最大框的 90 分位）：relative 口径的「1.0」在这里
        "ref": round(float(ref_scale), 5),
        "bins": SIZE_HIST_BINS,
        "hist_max": SIZE_HIST_MAX,
        "hist": [int(v) for v in hist],
        "overflow": overflow,
        "hist_median": round(float(np.median(h)), 5) if total else 0.0,
        "hist_p90": round(float(np.percentile(h, 90)), 5) if total else 0.0,
        #: 每项 [框高, 框面积, 同帧最大框高]
        "sample": sample,
    }


def _median_box(track: PlayerTrack) -> tuple[float, float, float, float]:
    """轨迹的代表框：逐帧框的逐坐标中位数。

    用中位数而不是均值：球员跨步/冲网时框会突然拉长，均值会被这些瞬间带偏，
    而尺寸筛选用的是「这个人平时多大」。逐坐标取中位数也保证结果仍是一个
    合法的矩形（取面积中位数那种做法会得到一个并不存在的框）。
    """
    if not track.boxes:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(track.boxes, dtype=np.float64).reshape(-1, 4)
    med = np.median(arr, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]), float(med[3]))


def _track_median_size(track: PlayerTrack, aspect: float = 1.7778) -> float:
    """轨迹的典型尺寸：框高与框宽里较大的那个（按画面高度为单位）。

    取「高 / 宽」的较大值是为了适配俯拍：从上往下看人是「矮而宽」的，
    只看框高会把这些球员误判成小目标而筛掉。
    """
    if not track.boxes:
        return 0.0
    vals: list[float] = []
    for b in track.boxes:
        h = abs(float(b[3]) - float(b[1]))
        w = abs(float(b[2]) - float(b[0])) * max(0.2, float(aspect))
        vals.append(max(h, w))
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _select_active_players(
    tracks: list[PlayerTrack],
    fps: float,
    total_duration: float,
    max_players: int = 4,
    viewpoint: str = "unknown",
    boxes: list | None = None,
    aspect: float = 1.7778,
    size_filter: SizeFilter | None = None,
) -> list[PlayerTrack]:
    """挑出「正在比赛的球员」并写入各自的 ``active_score``。

    判定依据（每一条都对应实测观察到的现象）：

    1. **框大**（低机位下权重最高）：比赛球员离机位近，是画面里最大的两个人框。
       用「框面积 90 分位数」而不是均值——球员跨步/冲网的一瞬间框会突然变大
       （实测面积差 5~10 倍），均值会被大量「站着的普通帧」拉平，而 max 又太
       容易被单帧误检带偏。最后再除以全部候选轨迹的 90 分位数做鲁棒归一。
    2. **移动快**：跑动/跨步/起跳让他们的 mean_speed、max_speed 远高于
       站着看球的人和观众；同样用 90 分位数做鲁棒归一，并按观测帧数做
       样本量收缩（只出现一两秒的碎片估计不可靠，要打折扣）。
    3. **出场时长**（弱权重）：背景路人可能一闪而过，但观众也会长期存在，
       所以时长只给很小的权重，不单独作为依据。
    4. **扣分项**：
       - 「长时间低速」：一直站着的人（观众、休息球员）即使框不小也扣分；
       - 「水平位置几乎固定」：一直在场边小范围来回走的工作人员扣分；
       - 「长期贴画面左右边缘」：鱼眼边缘会把远处的人拉大，靠边的人
         （大多是坐在场边/靠墙的观众）扣分。

    **「框够大」是硬门限，不是加分项。** 比赛球员必须比其他所有人明显大：
    实测素材里真正球员的框高约 0.16~0.22（画面高度的 16%~22%），而观众、
    隔壁场地的人只有 0.02~0.06。如果只用加权求和，一条「框很小但一直在动」
    的轨迹（观众走动、隔壁场地热身）有机会靠速度分挤进前几名。
    给定 ``boxes``（逐帧框）时会算出「典型球员尺寸」——取所有轨迹里最大的
    那批框高的中位数——然后把明显偏小的轨迹直接排除。

    **机位相关**：「框最大 = 离相机最近」只在后方/侧方低机位成立。高机位和
    俯拍下所有人在画面里大小差不多，此时**运动**才是唯一可靠的判据，
    所以权重会切换成「运动为主、面积只作微调」；这时也取消「贴边扣分」
    （俯拍时球场本来就可能偏向画面一侧）。

    最后按分数从高到低取：通常 2 条；如果第 3、4 名与第 1 名同档且都明显高于
    其余轨迹，则取 4 条（支持双打）。**同一名球员被断成多条轨迹时只算一个人**，
    把名额留给另一名球员；只有当某条碎片能明显延长这名球员的时间覆盖时才会
    一并纳入（否则球员会在某些时段「凭空消失」，下游裁切就没得跟了）。
    """
    if not tracks:
        return []
    top_view = viewpoint in ("overhead", "elevated")
    w_area, w_speed, w_dur = ((0.20, 0.65, 0.15) if top_view else (0.45, 0.40, 0.15))

    # 太短的轨迹（不到约 0.4 秒）直接不算候选
    min_frames = max(3, int(round(0.4 * fps)))
    cand = [t for t in tracks if t.n >= min_frames]
    if not cand:
        cand = sorted(tracks, key=lambda t: t.n, reverse=True)[:2]
    if not cand:
        return []

    feats: dict[int, _Feat] = {t.track_id: _track_feat(t, fps) for t in cand}

    # ---- 尺寸硬门限：把「太小的一定不是比赛球员」提前排除掉。
    # 参考尺寸用「候选里最大的那一批框高」——比赛球员就是画面里最大的人，
    # 所以这个参考量在任何机位下都指向真实球员的尺度。
    size = _box_size_stats(boxes, aspect) if boxes else None
    size_ref = float(size["ref"]) if size else 0.0
    if size_filter is not None and size_filter.active:
        # 用户显式指定了尺寸筛选：它优先于自适应门限（自适应门限会被
        # 「离相机更近的观众」劫持，这正是用户要手动介入的场景）。
        # 轨迹级别的框是逐帧框的中位数，直接按同一套阈值判它即可。
        kept = [t for t in cand if size_filter.keep(_median_box(t), size_ref)]
        if kept:
            cand = kept
            feats = {t.track_id: _track_feat(t, fps) for t in cand}
    elif size is not None and size_ref > 0:
        lo = max(size["min_abs"], size_ref * 0.42)
        sized = [t for t in cand if _track_median_size(t, aspect) >= lo]
        if len(sized) >= 1:
            dropped = len(cand) - len(sized)
            if dropped:
                cand = sized
                feats = {t.track_id: _track_feat(t, fps) for t in cand}

    areas = np.asarray([feats[t.track_id].area_p90 for t in cand], dtype=np.float64)
    speeds = np.asarray([t.mean_speed for t in cand], dtype=np.float64)
    peaks = np.asarray([t.max_speed for t in cand], dtype=np.float64)
    area_ref = max(_pct(areas, 90), 1e-6)
    speed_ref = max(_pct(speeds, 90), 1e-6)
    peak_ref = max(_pct(peaks, 90), 1e-6)
    dur_ref = max(total_duration, 1e-6)

    for t in cand:
        f = feats[t.track_id]
        # 样本量收缩：见得越少的轨迹，速度/面积的估计越不可信
        shrink = t.n / (t.n + 2.0 * fps)
        # --- 正向得分 ---
        area_score = _clip01(f.area_p90 / area_ref)
        speed_score = shrink * _clip01(
            0.65 * t.mean_speed / speed_ref + 0.35 * t.max_speed / peak_ref
        )
        duration_score = _clip01(f.duration / dur_ref)
        # --- 扣分 ---
        penalty_static = 0.35 * _clip01(f.static_ratio)                  # 长时间低速
        penalty_fixed = 0.25 if (f.x_range < 0.06 and f.duration > 0.5 * dur_ref) else 0.0
        # 高机位/俯拍下球场本来就可能偏向画面一侧，贴边不代表是观众
        edge = _clip01((abs(f.cx_mean - 0.5) - 0.42) / 0.08)             # 贴画面边缘
        penalty_edge = 0.0 if top_view else 0.30 * edge

        score = (
            w_area * area_score
            + w_speed * speed_score
            + w_dur * duration_score
            - penalty_static
            - penalty_fixed
            - penalty_edge
        )
        t.active_score = _clip01(score)

    ranked = sorted(cand, key=lambda t: t.active_score, reverse=True)

    # 事后尺寸校验：把「尺寸和第一名差太多」的整条轨迹丢掉。
    # 加权求和总有可能让一条「框很小但一直在动」的轨迹挤进来（观众走动、
    # 隔壁场地热身），这里按尺寸再砍一刀——比赛球员之间尺寸不会差 2 倍以上。
    if size is not None and size["ref"] > 0 and len(ranked) > 1:
        ref_size = _track_median_size(ranked[0], aspect)
        if ref_size > 0:
            keep = [t for t in ranked
                    if _track_median_size(t, aspect) >= max(size["min_abs"], ref_size * 0.45)]
            if keep:
                ranked = keep

    best = ranked[0].active_score
    if best <= 1e-6:
        return ranked[:1]

    def _same_player(a: PlayerTrack, b: PlayerTrack) -> bool:
        """两条轨迹是不是「同一名球员」（时间重叠 + 位置/尺度接近）。"""
        fa, fb = feats[a.track_id], feats[b.track_id]
        overlap = min(a.times[-1], b.times[-1]) - max(a.times[0], b.times[0])
        if overlap <= 0.0:
            return False
        if abs(fa.cx_mean - fb.cx_mean) > 0.08:
            return False
        return max(fa.area_p90, fb.area_p90) / max(1e-9, min(fa.area_p90, fb.area_p90)) <= 2.5

    # 「球员组」：同一名球员可能被断成多条轨迹，但它们仍然只算一个人。
    # 只有当某条碎片主要是「新增的时间覆盖」（该球员此前没被跟到的时段）时才一并
    # 纳入——这样球员在某个时段就不会「凭空消失」，下游裁切跟随也不会中途丢人。
    EXTEND_MIN = 2.0
    groups: list[dict[str, Any]] = []
    picked: list[PlayerTrack] = []
    for t in ranked:
        if len(picked) >= max_players:
            break
        f = feats[t.track_id]
        t0, t1 = t.times[0], t.times[-1]
        grp = next((g for g in groups if _same_player(t, g["rep"])), None)
        if grp is not None:
            gain = max(0.0, grp["t0"] - t0) + max(0.0, t1 - grp["t1"])
            span = max(1e-6, t1 - t0)
            # 既要有可观的新增时间，又要占这条轨迹自身时长的一半以上；
            # 否则它只是同一名球员的重复碎片，不该占用「球员名额」。
            if gain < EXTEND_MIN or gain < 0.5 * span:
                continue
        # 会动 + 框不小，是「比赛球员」的必要条件
        moving = (t.mean_speed / speed_ref) >= 0.20 and (f.area_p90 / area_ref) >= 0.20
        if not moving:
            continue
        # 第 2 名起：必须与第 1 名同档（0.55 倍以上），且绝对分不能太低
        if picked and t.active_score < max(0.55 * best, 0.30):
            break
        picked.append(t)
        if grp is None:
            groups.append({"rep": t, "t0": t0, "t1": t1})
        else:
            grp["t0"] = min(grp["t0"], t0)
            grp["t1"] = max(grp["t1"], t1)
    return picked


#: 逐时间窗重挑比赛球员时，窗口长度与步长（秒）。窗口要比「一次换人 /
#: 一次遮挡」长得多，球员才能在窗内有足够的观测；步长取一半保证交界处
#: 不会出现谁都没被选中的缝。
_ACTIVE_WINDOW_S = 180.0
_ACTIVE_WINDOW_STEP_S = 90.0
#: 逐窗合并后允许保留的最大轨迹条数（安全阀，防止背景误检把 frame_boxes 撑爆）。
_ACTIVE_MAX_TRACKS = 64


def _select_active_players_windowed(
    tracks: list[PlayerTrack],
    fps: float,
    duration: float,
    max_players: int = 4,
    viewpoint: str = "unknown",
    boxes: list | None = None,
    aspect: float = 1.7778,
    size_filter: SizeFilter | None = None,
) -> list[PlayerTrack]:
    """**逐时间窗**挑比赛球员，再合并结果。

    为什么不能全片只挑一次：实测 30 分钟素材有 285 条轨迹 —— 球员中途被遮挡、
    走出画面、或者和背景的人交叠，跟踪器就会给他一个新 id。而
    :func:`_select_active_players` 是按**全片**统计量排名的，取前 4 条很可能
    全部落在视频中段。后果非常严重：``PlayerSignal.frame_boxes`` 在其余时段
    是空的，于是

    * ``active_count`` / ``player_motion`` 整段为 0（实测前 372 秒恒为 0）；
    * 切分时球员覆盖率只有 0.48，「球员运动切分」这条路直接作废，
      退回「整帧活跃度」——而后者在多球场球馆里从来不塌，回合被粘成
      几十秒一条。

    做法：按 ``_ACTIVE_WINDOW_S`` 切窗（步长一半，保证交界处不漏），每个窗
    只拿**与该窗有时间重叠**的轨迹去排名，然后把各窗选中的 id 取并集。
    全片那一遍仍然要跑：它的 ``size_ref`` 是「画面里最大的人有多大」的
    全局参考量，用来做最后一道尺寸闸门，防止某个窗口把远处的观众选进来。

    ``duration`` 短于一个窗口时行为与旧实现完全一致。
    """
    if not tracks:
        return []
    if duration <= _ACTIVE_WINDOW_S:
        return _select_active_players(tracks, fps, duration, max_players=max_players,
                                      viewpoint=viewpoint, boxes=boxes, aspect=aspect,
                                      size_filter=size_filter)

    # 全片那一遍：拿到「参考尺寸」和一组基线 ids
    base = _select_active_players(tracks, fps, duration, max_players=max_players,
                                  viewpoint=viewpoint, boxes=boxes, aspect=aspect,
                                  size_filter=size_filter)
    base_ids = {t.track_id for t in base}

    gsize = _box_size_stats(boxes, aspect) if boxes else None
    g_ref = float(gsize["ref"]) if gsize else 0.0
    g_min = float(gsize["min_abs"]) if gsize else 0.0
    # 逐窗挑出来的轨迹也要过同一道「够大」的闸门：只有当它明显小于全片参考
    # 尺寸（说明是远处观众 / 隔壁场地的人）时才丢。
    size_floor = max(g_min, g_ref * 0.45) if g_ref > 0 else 0.0

    n_win = max(1, int(np.ceil((duration - _ACTIVE_WINDOW_S)
                               / _ACTIVE_WINDOW_STEP_S))) + 1
    chosen: dict[int, PlayerTrack] = {t.track_id: t for t in base}
    for w in range(n_win):
        t0 = w * _ACTIVE_WINDOW_STEP_S
        t1 = min(duration, t0 + _ACTIVE_WINDOW_S)
        if t1 - t0 < 8.0:
            continue
        sub = [t for t in tracks
               if t.n > 0 and t.times[-1] >= t0 and t.times[0] <= t1]
        if not sub:
            continue
        sub_boxes = None
        if boxes:
            i0 = max(0, int(t0 * fps))
            i1 = min(len(boxes), max(i0 + 1, int(t1 * fps)))
            sub_boxes = boxes[i0:i1]
        try:
            picked = _select_active_players(sub, fps, t1 - t0,
                                            max_players=max_players, viewpoint=viewpoint,
                                            boxes=sub_boxes, aspect=aspect,
                                            size_filter=size_filter)
        except Exception:
            continue
        for t in picked:
            if t.track_id in chosen:
                continue
            if size_floor > 0 and _track_median_size(t, aspect) < size_floor:
                continue
            chosen[t.track_id] = t

    out = sorted(chosen.values(), key=lambda t: (t.times[0] if t.n else 0.0))
    if len(out) > _ACTIVE_MAX_TRACKS:
        # 太多说明背景误检混进来了：优先保留全片基线 + 出现时间最长的
        out = sorted(out, key=lambda t: (t.track_id not in base_ids, -t.n))
        out = out[: _ACTIVE_MAX_TRACKS]
    return out


# ------------------------------------------------------------------ 主入口


def analyze_players(
    video_path: str,
    sample_fps: float = 15.0,
    roi: tuple[float, float, float, float] | None = None,   # 归一化 x0,y0,x1,y1；人框底边中心落在其中才算候选
    max_seconds: float = 0.0,      # 0 = 全片
    model_name: str = "yolo11n.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    batch_hint: int = 16,
    viewpoint: str = "unknown",
    roi_poly: list[list[float]] | None = None,   # 归一化场地多边形；比 roi 精确
    size_filter: "SizeFilter | dict | None" = None,
    on_progress: Progress | None = None,
    cancel: Any = None,            # callable() -> bool
) -> PlayerSignal:
    """检测并跟踪视频里的人物，挑出正在比赛的球员。

    Args:
        video_path: 视频路径（可以是低分辨率代理视频，速度更快）。
        sample_fps: 抽帧分析的帧率，输出时间轴也按它。
        roi: 归一化 (x0, y0, x1, y1)。给定后，只有**框底边中心**落在其中的
            框才作为候选（用于先粗筛掉看台/其他场地的人）。场地标定成功时
            由流水线自动传入，等价于「只在这块场地里找人」。
        roi_poly: 归一化多边形（4~24 点）。给了它就**取代** ``roi`` 做场内判定：
            全景 / 鱼眼素材的场地边界是弯的，用外接矩形判会把弯边以外的
            大片区域（往往就是看台）算成「场内」。
        size_filter: :class:`SizeFilter` 或等价字典；在检测阶段按人物框的
            高度/面积筛掉「明显不是这场比赛球员」的框。``None`` / ``mode="off"``
            时行为与旧版本完全一致。
        max_seconds: 只分析前若干秒；0 表示整个视频。
        model_name: YOLO 权重名或路径；优先用本地 ``models/`` 下的权重。
        imgsz: 推理输入尺寸。
        conf: 检测置信度阈值（跟踪时会自动放宽到 0.15 以穿过短暂遮挡）。
        device: ``cuda`` / ``cpu``；不可用时自动回退。
        batch_hint: 每次送进 GPU 的帧数。
        viewpoint: 机位类型（``rear`` / ``side`` / ``elevated`` / ``overhead`` /
            ``unknown``）。影响两件事：几何门限（俯拍下人是「矮而宽」的）
            和「挑比赛球员」的权重（高机位下所有人一样大，只能靠运动区分）。
        on_progress: ``callable(progress: float, stage: str)`` 进度回调。
        cancel: ``callable() -> bool``，返回 True 时尽快中断并返回已分析的部分。

    Returns:
        PlayerSignal（``size_stats`` 里带着过滤前的框高分布，界面据此调参）
    """
    import cv2

    if not video_path:
        raise ValueError("video_path 不能为空")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    sf = size_filter_from(size_filter)

    cfg_dir, models_dir = _data_paths()
    _prepare_yolo_env(cfg_dir)
    weights = _resolve_weights(model_name, models_dir)
    dev = _pick_device(device)

    def _report(p: float, stage: str) -> None:
        if on_progress is not None:
            on_progress(_clip01(p), stage)

    def _cancelled() -> bool:
        try:
            return bool(cancel is not None and cancel())
        except Exception:
            return False

    _report(0.0, "准备模型")

    # ---- 打开视频，计算抽帧步长 ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0 or not np.isfinite(src_fps):
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, probe = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"无法读取视频帧: {video_path}")
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    aspect = float(width) / float(max(1, height))

    step = max(1, int(round(src_fps / max(1e-3, sample_fps))))
    eff_fps = src_fps / step                      # 实际抽帧率
    step_dt = 1.0 / eff_fps
    limit = int(round(max_seconds * src_fps)) if max_seconds > 0 else (total_frames or 10 ** 9)
    batch_size = max(1, int(batch_hint))

    # ---- 加载模型 ----
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        cap.release()
        raise RuntimeError(f"无法导入 ultralytics: {exc}") from exc

    model = YOLO(weights)
    # 跟踪需要更宽松的框（能穿过短暂遮挡），但低于 0.15 的一律不采信
    det_conf = float(min(conf, LOW_CONF))

    tracker = _MultiObjectTracker(fps=eff_fps, aspect=aspect, max_missing=MAX_MISSING)

    def _predict(frames: list[np.ndarray]) -> list[Any]:
        """批量推理；CUDA 出错时回退 CPU 重试一次。"""
        nonlocal dev
        try:
            return model.predict(
                frames, classes=[0], conf=det_conf, imgsz=imgsz,
                device=dev, verbose=False,
            )
        except Exception:
            if dev != "cpu":
                dev = "cpu"
                return model.predict(
                    frames, classes=[0], conf=det_conf, imgsz=imgsz,
                    device="cpu", verbose=False,
                )
            raise

    # ---- 抽帧 + 批量推理 ----
    roi_arr = None
    if roi is not None:
        roi_arr = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
    poly_arr = None
    if roi_poly:
        p = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
        p = p[np.all(np.isfinite(p), axis=1)]
        if p.shape[0] >= 3:
            poly_arr = p
            # 多边形与矩形同时给出时以多边形为准：它才是场地的真实形状
            roi_arr = None

    def _passes_roi(box: tuple[float, float, float, float]) -> bool:
        """框底边中心是否落在场地里。

        有场地多边形就用多边形（弯边素材必须这样），否则退回外接矩形。
        底边中心而不是框中心：人框的上半截常常在场地之外（举手、跳起、
        或者框把记分牌一起框进去了），用中心判会把真正的球员排掉。
        """
        bcx = 0.5 * (box[0] + box[2])
        by = box[3]
        if poly_arr is not None:
            # ROI_POLY_MARGIN：球员正好站在边线上时按「场内」算（见 point_in_poly）
            return bool(point_in_poly(np.asarray([[bcx, by]], dtype=np.float32),
                                      poly_arr, margin=ROI_POLY_MARGIN)[0])
        if roi_arr is None:
            return True
        return (roi_arr[0] <= bcx <= roi_arr[2]) and (roi_arr[1] <= by <= roi_arr[3])

    def _dets_of(result: Any) -> list[tuple[tuple[float, float, float, float], float]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cf = boxes.conf.cpu().numpy()
        out: list[tuple[tuple[float, float, float, float], float]] = []
        for k in range(len(boxes)):
            c = float(cf[k])
            if c < LOW_CONF:             # 置信度太低的框直接忽略
                continue
            b = _sanitize_box(
                float(xyxy[k, 0]) / width, float(xyxy[k, 1]) / height,
                float(xyxy[k, 2]) / width, float(xyxy[k, 3]) / height,
                viewpoint,
            )
            if b is None or not _passes_roi(b):
                continue
            out.append((b, c))
        # 同一帧内几乎重叠的重复框：只保留置信度最高的那个
        return _dedup_dets(out, aspect)

    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    proc = 0                     # 已处理的采样帧数
    read = 0                     # 已 grab 的原始帧数
    idx = 0
    # 逐采样帧的原始检测框（归一化，含背景人员）。用来自适应地估计
    # 「这场比赛里球员大概多大」，见 `_box_size_stats`。
    det_frames: list[list[tuple[float, float, float, float]]] = []
    # ---- 尺寸筛选的统计（统计的是**筛选前**的框，界面才能画出「砍掉了什么」）
    #: 每个框一项 [框高, 框面积, 该帧参考框高]
    size_samples: list[list[float]] = []
    size_dropped = 0
    size_frames = 0

    def _apply_size_filter(dets: list[tuple[tuple[float, float, float, float], float]],
                           ) -> list[tuple[tuple[float, float, float, float], float]]:
        """按尺寸筛掉不合适的人框，并把统计记下来（原地更新上面的累加器）。

        ``relative`` 口径需要「同帧最大框高」当分母：这里用**本帧所有候选框**的
        最大值。它必须是本帧的、而不是全片的：检测框大小随人物远近变化，
        用全片参考会把「镜头扫过看台」那一帧的所有人都判成合格。
        """
        nonlocal size_dropped, size_frames
        if not dets:
            return dets
        size_frames += 1
        heights = [_box_height(b) for b, _c in dets]
        ref_h = max(heights) if heights else 0.0
        size_samples.extend([[_box_height(b), _box_area(b), ref_h] for b, _c in dets])
        if not sf.active:
            return dets
        kept = [(b, c) for (b, c) in dets if sf.keep(b, ref_h)]
        size_dropped += len(dets) - len(kept)
        # 全被筛掉时保留原样：宁可多给跟踪器一点噪声，也不要出现「整段时间
        # 一个框都没有」——那会让下游的活跃度和裁切跟随直接断档。
        return kept or dets

    while True:
        if _cancelled():
            break
        ok = cap.grab()
        if not ok:
            break
        read += 1
        if idx % step != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue
        if read > limit:
            break
        buf.append(frame)
        buf_idx.append(proc)
        proc += 1

        if len(buf) >= batch_size:
            results = _predict(buf)
            for k, res in enumerate(results):
                dets = _apply_size_filter(_dets_of(res))
                det_frames.append([b for b, _c in dets])
                tracker.update(buf_idx[k], buf_idx[k] * step_dt, dets)
            buf.clear()
            buf_idx.clear()
            if total_frames:
                _report(min(0.95, read / min(total_frames, limit)), "检测球员")

    if buf and not _cancelled():
        results = _predict(buf)
        for k, res in enumerate(results):
            dets = _apply_size_filter(_dets_of(res))
            det_frames.append([b for b, _c in dets])
            tracker.update(buf_idx[k], buf_idx[k] * step_dt, dets)
        buf.clear()
        buf_idx.clear()
    cap.release()

    duration = proc * step_dt                 # 实际分析的秒数
    _report(0.97, "跟踪与统计")

    # ---- 轨迹合并 + 汇总 ----
    bufs = _merge_tracks(tracker.tracks, aspect)
    tracks = _summarize(bufs, eff_fps)

    # ---- 判定比赛球员 ----
    # 逐时间窗判定：长视频里球员会被断成很多条轨迹，全片只挑一次会让
    # 「中段之外的时间」完全没有球员信号（实测 frame_boxes 有 58% 的时间是空的）。
    active = _select_active_players_windowed(tracks, eff_fps, duration, viewpoint=viewpoint,
                                             boxes=det_frames, aspect=aspect,
                                             size_filter=sf)
    active_ids = [t.track_id for t in active]

    # ---- 组装时间轴（长度 = ceil(duration * sample_fps)，索引 i 对应 i/sample_fps） ----
    n = max(1, int(math.ceil(duration * sample_fps - 1e-9)))

    def _ti(t: float) -> int:
        return min(n - 1, max(0, int(round(t * sample_fps))))

    active_present: list[set[int]] = [set() for _ in range(n)]
    active_sp: list[list[float]] = [[] for _ in range(n)]
    crowd_sp: list[list[float]] = [[] for _ in range(n)]
    max_sp = np.zeros(n, dtype=np.float32)
    boxes_map: list[dict[int, tuple[int, float, float, float, float]]] = [dict() for _ in range(n)]
    active_set = set(active_ids)

    for tr in tracks:
        for j in range(tr.n):
            i = _ti(tr.times[j])
            s = float(tr.speeds[j])
            crowd_sp[i].append(s)
            if s > max_sp[i]:
                max_sp[i] = s
            if tr.track_id in active_set:
                active_present[i].add(tr.track_id)
                active_sp[i].append(s)
                b = tr.boxes[j]
                boxes_map[i][tr.track_id] = (tr.track_id, float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    active_count = np.asarray([len(s) for s in active_present], dtype=np.float32)
    active_speed = np.asarray(
        [float(np.mean(v)) if v else 0.0 for v in active_sp], dtype=np.float32
    )
    crowd_speed = np.asarray(
        [float(np.mean(v)) if v else 0.0 for v in crowd_sp], dtype=np.float32
    )
    frame_boxes = [list(m.values()) for m in boxes_map]

    size_stats = _build_size_stats(
        sf, size_samples, size_dropped, size_frames,
        float((_box_size_stats(det_frames, aspect) or {}).get("ref", 0.0)) if det_frames else 0.0,
    )

    _report(1.0, "完成")
    return PlayerSignal(
        fps=float(sample_fps),
        duration=float(duration),
        tracks=tracks,
        active_count=active_count,
        active_speed=active_speed,
        max_speed=max_sp,
        crowd_speed=crowd_speed,
        active_player_ids=active_ids,
        frame_boxes=frame_boxes,
        size_stats=size_stats,
    )


# ------------------------------------------------------------------ 框尺寸试测


#: 试测用的模型缓存：``权重路径 -> YOLO 实例``。
#: 为什么要缓存：界面里调尺寸阈值时用户会**反复抓同一段视频的不同帧**，
#: 而 ``YOLO(weights)`` 加载权重本身要 1~2 秒（比推理一帧还慢）。
#: 只给试测用：完整分析一次加载、连跑几千帧，缓存对它没有收益，
#: 而且共用同一个模型对象会让「分析中」和「抓帧」互相干扰。
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
#: ultralytics 的 ``predict`` 会改写模型对象内部状态，多个线程同时调用同一个
#: 缓存实例不安全；抓帧是短调用，串行化代价可以接受。
_PREDICT_LOCK = threading.Lock()


def _load_model_cached(weights: str) -> Any:
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(weights)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(weights)
            _MODEL_CACHE[weights] = model
        return model


def _probe_frame_path(video_path: str, t: float, index: int = 0) -> Path:
    """试测帧图的落盘路径（放在缓存目录下，由 ``/api/asset`` 提供访问）。"""
    import hashlib

    try:
        from ..config import FRAMES_DIR

        root = Path(FRAMES_DIR) / "probe"
    except Exception:  # pragma: no cover - 独立运行时的兜底
        root = Path(__file__).resolve().parents[3] / "data" / "cache" / "frames" / "probe"
    key = hashlib.sha1(
        f"{Path(video_path).resolve()}|{t:.3f}|{index}".encode("utf-8", "replace")
    ).hexdigest()[:16]
    return root / f"{key}.jpg"


def _read_frame_at(cap: Any, src_fps: float, t: float, tries: int = 6) -> np.ndarray | None:
    """定位到第 ``t`` 秒并读出一帧，跳过 seek 之后常见的黑帧。

    长 GOP 素材 seek 之后常常先返回几帧全黑/花屏的帧。试测帧图是给用户
    **看**的，黑帧会让「框位置对不对」完全没法核对，所以这里往后多读几帧，
    取第一个「不是纯色」的帧；实在都是纯色就把最后一帧交出去（至少不是空）。
    """
    import cv2

    idx = int(round(max(0.0, t) * max(src_fps, 1e-6)))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    except Exception:
        pass
    last: np.ndarray | None = None
    for _ in range(max(1, int(tries))):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        last = fr
        try:
            if float(fr.std()) > 2.0:      # 纯色帧的 std≈0
                return fr
        except Exception:
            return fr
    return last


def probe_boxes(
    video_path: str,
    count: int = 10,
    model_name: str = "yolo11n.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cuda",
    viewpoint: str = "unknown",
    roi: tuple[float, float, float, float] | None = None,
    roi_poly: list[list[float]] | None = None,
    max_side: int = 960,
    times: Sequence[float] | None = None,
    save_frames: bool = False,
    on_progress: Progress | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """在若干帧上**只做检测**（不跟踪），返回人物框与逐帧的框尺寸分布。

    用途：界面里调「人物框尺寸筛选」时，如果只能等完整分析跑完再回来看结果，
    调一次参数要等几分钟 —— 实际上用户真正想知道的只有一件事：
    **「球员的框有多大、观众和其他场地的人的框有多大」**。

    两种取帧方式：

    * 默认在整条视频上**均匀抽** ``count`` 帧（一次批量推理，1~3 秒），
      给出这条素材的整体分布；
    * 给了 ``times`` 就**只取这些时刻**（用户手动选帧 / 抓当前播放位置），
      这时候每帧还会带上 ``image``（落盘的 JPEG 路径），
      界面可以把它画出来，把「哪些框被选中、哪些被筛掉」直接摆在画面上。

    Args:
        video_path: 视频路径（优先传代理视频，解码更快）。
        count: 均匀抽帧时的帧数（1~40）；给了 ``times`` 时忽略。
        roi / roi_poly: 与 :func:`analyze_players` 同义；给了就按场地过滤，
            这样统计出来的分布才与真正分析时一致。
        max_side: 推理前把画面缩到这个边长以内（只影响速度，不影响归一化坐标）。
        times: 指定的时刻（秒）；最多 40 个。
        save_frames: 是否把取到的帧写成 JPEG 并返回路径（给界面显示用）。

    Returns:
        ``{"frames": [{"t", "boxes", "confs", "ref", "image"}...],
        "points": [[框高, 框面积, 同帧最大框高]...], ...}``
    """
    import cv2
    import time

    if not video_path:
        raise ValueError("video_path 不能为空")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    t0 = time.time()
    want_times: list[float] = []
    if times:
        for v in list(times)[:40]:
            try:
                want_times.append(max(0.0, float(v)))
            except (TypeError, ValueError):
                continue
    count = len(want_times) if want_times else int(max(1, min(40, count)))

    cfg_dir, models_dir = _data_paths()
    _prepare_yolo_env(cfg_dir)
    weights = _resolve_weights(model_name, models_dir)
    dev = _pick_device(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0 or not np.isfinite(src_fps):
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[np.ndarray] = []
    used_times: list[float] = []
    for i in range(count):
        if cancel is not None and cancel():
            break
        if want_times:
            fr = _read_frame_at(cap, src_fps, want_times[i])
            t_used = want_times[i]
        elif total_frames > 0:
            pos = int(total_frames * (i + 0.5) / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fr = cap.read()
            if not ok:
                fr = None
            t_used = pos / src_fps
        else:
            # 帧数未知（部分容器）时顺序读
            ok, fr = cap.read()
            if not ok:
                fr = None
            t_used = len(frames) * 1.0 / max(src_fps, 1e-3)
        if fr is None:
            continue
        used_times.append(float(t_used))
        h, w = fr.shape[:2]
        s = max_side / max(1, max(h, w))
        if s < 1.0:
            fr = cv2.resize(fr, (max(2, int(w * s)), max(2, int(h * s))),
                            interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        return {"frames": [], "points": [], "count": 0, "duration": 0.0,
                "width": 0, "height": 0, "aspect": 1.7778,
                "viewpoint": viewpoint, "error": "无法读取视频帧"}

    # 归一化坐标必须按**实际解码出来的画面**算（而不是容器上报的尺寸）：
    # 试测的框要能直接画在返回的那张图上，两者必须来自同一个像素空间。
    height, width = frames[0].shape[:2]
    aspect = float(width) / float(max(1, height))

    # 帧图落盘：与推理用的是同一帧，所以框和画面严格对齐
    images: list[str | None] = [None] * len(frames)
    if save_frames:
        for i, fr in enumerate(frames):
            p = _probe_frame_path(video_path, used_times[i] if i < len(used_times) else 0.0, i)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(p), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
                    images[i] = str(p)
            except Exception:
                pass

    if on_progress is not None:
        on_progress(0.3, "加载模型")
    di = float(min(conf, LOW_CONF))
    try:
        model = _load_model_cached(weights)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"无法加载 ultralytics 模型: {exc}") from exc

    poly_arr = None
    if roi_poly:
        p = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
        p = p[np.all(np.isfinite(p), axis=1)]
        if p.shape[0] >= 3:
            poly_arr = p
    roi_arr = tuple(float(v) for v in roi) if (roi is not None and poly_arr is None) else None

    def _predict_all() -> list[Any]:
        nonlocal dev
        with _PREDICT_LOCK:
            try:
                return model.predict(frames, classes=[0], conf=di, imgsz=imgsz,
                                     device=dev, verbose=False)
            except Exception:
                if dev != "cpu":
                    dev = "cpu"
                    return model.predict(frames, classes=[0], conf=di, imgsz=imgsz,
                                         device="cpu", verbose=False)
                raise

    if on_progress is not None:
        on_progress(0.6, "检测人物框")
    results = _predict_all()

    out_frames: list[dict[str, Any]] = []
    points: list[list[float]] = []
    for k, res in enumerate(results):
        fh = int(frames[k].shape[0]) if k < len(frames) else height
        fw = int(frames[k].shape[1]) if k < len(frames) else width
        boxes = getattr(res, "boxes", None)
        dets: list[tuple[tuple[float, float, float, float], float]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cf = boxes.conf.cpu().numpy()
            for j in range(len(boxes)):
                c = float(cf[j])
                if c < LOW_CONF:
                    continue
                b = _sanitize_box(float(xyxy[j, 0]) / fw, float(xyxy[j, 1]) / fh,
                                  float(xyxy[j, 2]) / fw, float(xyxy[j, 3]) / fh,
                                  viewpoint)
                if b is None:
                    continue
                bcx, by = 0.5 * (b[0] + b[2]), b[3]
                if poly_arr is not None:
                    if not bool(point_in_poly(np.asarray([[bcx, by]], dtype=np.float32),
                                              poly_arr, margin=ROI_POLY_MARGIN)[0]):
                        continue
                elif roi_arr is not None:
                    if not (roi_arr[0] <= bcx <= roi_arr[2] and roi_arr[1] <= by <= roi_arr[3]):
                        continue
                dets.append((b, c))
        dets = _dedup_dets(dets, aspect)
        ref = max((_box_height(b) for b, _c in dets), default=0.0)
        out_frames.append({
            "t": round(float(used_times[k]) if k < len(used_times) else 0.0, 3),
            "boxes": [[round(float(v), 5) for v in b] for b, _c in dets],
            "confs": [round(float(c), 3) for _b, c in dets],
            "ref": round(float(ref), 5),
            # 与 boxes 同一帧的画面（JPEG 路径，界面用 /api/asset 取）
            "image": images[k] if k < len(images) else None,
        })
        # [框高, 框面积, 同帧最大框高]：界面据此在本地模拟任意阈值
        points.extend([[_box_height(b), _box_area(b), ref] for b, _c in dets])

    if on_progress is not None:
        on_progress(1.0, "完成")
    return {
        "frames": out_frames,
        "points": points,
        "count": len(out_frames),
        # 取到的最后一帧的时刻（不是视频总长）
        "duration": round(float(max(used_times)) if used_times else 0.0, 3),
        "video_duration": round(float(total_frames / src_fps) if total_frames else 0.0, 3),
        "width": int(width),
        "height": int(height),
        "aspect": round(aspect, 5),
        "viewpoint": viewpoint,
        "elapsed": round(time.time() - t0, 2),
        "model": Path(weights).name,
        "device": dev,
        "frames_saved": bool(save_frames),
    }
