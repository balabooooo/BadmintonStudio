"""领域数据模型（Pydantic v2）。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


# ------------------------------------------------------------------ 媒体


class MediaInfo(BaseModel):
    id: str = Field(default_factory=lambda: _uid("m_"))
    path: str
    name: str = ""
    size: int = 0
    duration: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    rotation: int = 0
    vcodec: str = ""
    acodec: str | None = None
    has_audio: bool = False
    created_at: int = Field(default_factory=now_ms)
    # 派生资源（分析用）
    proxy_path: str | None = None
    proxy_fps: float | None = None
    proxy_width: int | None = None
    proxy_height: int | None = None
    audio_path: str | None = None
    poster: str | None = None

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 16 / 9


# ------------------------------------------------------------------ 回合与击球

PlayerSide = Literal["near", "far", "unknown"]
ShotKind = Literal["serve", "receive", "clear", "drop", "smash", "drive", "net", "lift", "unknown"]


class ShotEvent(BaseModel):
    """一次击球（球拍触球时刻）。"""

    time: float
    player: PlayerSide = "unknown"
    kind: ShotKind = "unknown"
    confidence: float = 0.5
    #: 触球后球速估计（场地坐标，m/s），未知为 None
    speed: float | None = None
    #: 该拍是否伴随明显起跳（杀球/跳杀）
    airborne: bool = False
    #: 触球点相对场地的高度（m），未知为 None
    height: float | None = None


class RallyFeatures(BaseModel):
    """用于评分的客观特征量。"""

    duration: float = 0.0
    shot_count: int = 0
    #: 每秒拍数
    tempo: float = 0.0
    #: 球速分位（像素/秒，已按场地尺度归一）
    shuttle_speed_p50: float = 0.0
    shuttle_speed_p95: float = 0.0
    #: 回合内场地运动能量均值（0~1 归一）
    motion_energy: float = 0.0
    motion_peak: float = 0.0
    #: 双方跑动距离（米）
    travel_near: float = 0.0
    travel_far: float = 0.0
    #: 杀球次数
    smash_count: int = 0
    #: 最大连续多拍（= shot_count）
    longest_exchange: int = 0
    #: 回合结束前最后 2 秒是否激烈
    finish_intensity: float = 0.0
    #: 是否出现极限救球（球员瞬时加速度峰值）
    scramble: float = 0.0
    #: 比分接近度 0~1（有记分牌识别时可用）
    closeness: float | None = None
    #: 是否关键分（局点/赛点/平分）
    clutch: bool = False
    #: 分析置信度
    confidence: float = 0.5

    # ---- 下面这些是评分直接用到的中间量。
    # 只存分项分是算不出「换一套口径」的分数的：换口径时要用同样的原始特征重算，
    # 所以这里把评分用到的量都留下来（老的分析结果里它们是默认值）。
    #: 击球力度 90 分位
    hit_strength_p90: float = 0.0
    #: 球员跑动速度均值 / 峰值（像素每秒，已按场地尺度归一）
    player_speed_mean: float = 0.0
    player_speed_max: float = 0.0
    #: 羽毛球在画面里出现的比例 0~1
    shuttle_presence: float = 0.0
    #: 画面质量：清晰度 / 抖动 / 主体大小
    quality_sharpness: float = 0.6
    quality_shake: float = 0.3
    quality_subject_size: float = 0.25


class RallyScores(BaseModel):
    """分项评分（0~100）。"""

    total: float = 0.0
    length: float = 0.0
    intensity: float = 0.0
    technique: float = 0.0
    excitement: float = 0.0
    production: float = 0.0  # 画面/取景质量（清晰度、抖动、遮挡）


class Rally(BaseModel):
    id: str = Field(default_factory=lambda: _uid("r_"))
    index: int = 0
    #: 回合本体（发球开始 -> 死球）
    start: float = 0.0
    end: float = 0.0
    #: 建议剪辑区间（含准备动作留白）
    clip_start: float = 0.0
    clip_end: float = 0.0
    #: 发球 / 接发球
    serve_time: float | None = None
    serve_player: PlayerSide = "unknown"
    receive_time: float | None = None
    receive_player: PlayerSide = "unknown"
    shots: list[ShotEvent] = Field(default_factory=list)
    features: RallyFeatures = Field(default_factory=RallyFeatures)
    scores: RallyScores = Field(default_factory=RallyScores)
    tags: list[str] = Field(default_factory=list)
    #: 用户干预
    keep: bool = True
    starred: bool = False
    note: str = ""
    #: 自动判定该回合结果（有记分牌时）
    winner: PlayerSide | None = None
    #: 回合本体时长（= end - start），随序列化一起下发，方便前端直接使用
    duration: float = 0.0


class AnalysisParams(BaseModel):
    """分析可调参数。"""

    #: 击球检测灵敏度 0~1（越大越灵敏）
    hit_sensitivity: float = 0.5
    #: 判断回合结束的静音时长（秒）
    gap_seconds: float = 3.2
    #: 最短回合时长（秒），低于此丢弃
    min_rally_seconds: float = 2.0
    #: 最长回合时长（秒），超出则截断
    max_rally_seconds: float = 120.0
    #: 典型回合时长（秒）。连续训练/多球练习时，球员在两个回合之间只停几秒，
    #: 活跃度曲线不会塌陷，光靠阈值会把好几个回合粘成一条。
    #: 超过这个长度的区间会继续在「活跃度最低点」递归切开。
    target_rally_seconds: float = 28.0
    #: 切分积极程度 0~1：越大越倾向切细（映射到更短的目标时长与更早的退出阈值）
    split_sensitivity: float = 0.5
    #: 低于这个置信度的回合在界面上默认标灰（可在界面上再调）
    confidence_min: float = 0.0
    #: 剪辑时在回合前后各保留的留白。
    #: ``pre_roll`` 是发球准备（球员走到位置、抛球前的停顿）；
    #: ``post_roll`` 是终点之后额外留的一点呼吸，加了它才是最终导出的
    #: ``clip_end``。因为 ``hit_tail_seconds`` 已经把「球落地」算进去了，
    #: 这里默认给得很小 —— 旧默认 1.6 会让每回合凭空多出一秒半的死球画面。
    pre_roll: float = 1.2
    post_roll: float = 0.5
    #: **最后一拍之后保留多久**（秒），用于把回合终点锚定到「球落地」。
    #: 一拍打出去之后球还要飞一会儿才落地，所以终点不能直接等于最后一拍。
    #: 实测业余素材这段飞行多数在 0.4~1.2 秒；给 0.9 能覆盖常见情况，
    #: 同时把「球落地后还留很久」和「吃进下一个回合」一起消掉。
    #: 调大 = 更保险但更松，调小 = 更紧但可能吃掉大力高远球的落地瞬间。
    hit_tail_seconds: float = 0.9
    #: 启用各分析模块
    use_audio: bool = True
    use_motion: bool = True
    use_players: bool = True
    #: 姿态辅助（YOLO-pose）：给音频击球做「是不是我们这场比赛打的」归属判定。
    #: 多球场球馆里音频无法区分「谁在击球」，而我们的球员只在真击球时挥拍，
    #: 所以这是解决「回合被粘长 / 吃进下一个回合」的关键证据。
    #: 失败（没有 GPU / 权重缺失 / 球员太小）会自动降级，行为与该开关关闭时一致。
    use_pose: bool = True
    #: 羽毛球轨迹跟踪：计算量与「帧数 × 像素数 × 时间窗」成正比，
    #: 30 分钟 4K 素材能跑到小时级，所以默认关闭，长视频按需开启。
    use_shuttle: bool = False
    use_scoreboard: bool = False
    #: 羽毛球跟踪的采样帧率（比球员检测更高才好抓快速飞行的球）
    shuttle_fps: float = 10.0
    #: 羽毛球跟踪的时间预算（秒，0 = 全片）。覆盖不足 60% 时整路信号会被丢弃，
    #: 避免「只在前半段有数据」把融合结果带偏。
    shuttle_budget_seconds: float = 420.0
    #: 场地朝向：自动 | 横屏 | 竖屏
    court_orientation: Literal["auto", "landscape", "portrait"] = "auto"
    #: 机位/拍法。``auto`` = 自动识别；其余用于用户明确知道自己的拍法时跳过猜测。
    #:   rear   = 场地后方（底线后，最常见）
    #:   side   = 边线侧方
    #:   elevated = 高机位斜俯（看台/二楼）
    #:   overhead = 正俯拍
    #: 不同机位下「谁离相机近」「球员框该多大」完全不同，识别出来才能选对先验。
    viewpoint: Literal["auto", "rear", "side", "elevated", "overhead"] = "auto"
    #: 是否自动标定场地（颜色 + 多边形 + 单应变换）。关掉则全程用全画幅，
    #: 也就等于退回旧行为（多球场/颜色异常时可用）。
    auto_calibrate: bool = True
    #: **手动标定的场地边界**（归一化 0~1，4~24 个点，顺序不限）。
    #: 用户在预览画面上点一圈即可；给了它就不再猜测，直接用它建标定。
    #: 颜色标定在多球场 / 地胶颜色异常 / 场地只占画面一角时会失败，
    #: 这时候手动标一次比继续调算法有效得多。
    #: **全景 / 鱼眼素材请多加几个点**：弯掉的边界用四个角描述会切掉边角，
    #: 而边角正是背景人员最密集的地方。
    court_poly: list[list[float]] | None = None
    #: 旧字段：手动标定的四角（等价于 ``court_poly`` 只给 4 个点）。
    #: 保留是为了让已经存过的旧工程继续生效；新代码请写 ``court_poly``。
    court_quad: list[list[float]] | None = None
    #: ---- 人物框尺寸筛选 ----
    #: 检测到的人框按尺寸过滤的模式：
    #:   ``off``      不筛选（只保留原有的几何门限）
    #:   ``absolute`` 按「框高占画面高度的比例」筛选（min/max 是绝对比例）
    #:   ``relative`` 按「框高 ÷ 同帧最大框高」筛选（比值）
    #: 全景 / 鱼眼素材里同一个人在画面中心与边角的框高能差一倍以上，
    #: 这时 ``absolute`` 很容易把靠边的真球员筛掉，``relative`` 更稳。
    player_size_mode: Literal["off", "absolute", "relative"] = "off"
    #: 框高下限（absolute = 占画面高度比例；relative = 相对同帧最大框的比值）
    player_min_height: float = 0.05
    #: 框高上限（0 = 不限）
    player_max_height: float = 0.0
    #: 框面积下限（归一化面积 0~1，0 = 不限）。畸变下框会变宽，
    #: 想更细地卡「贴到镜头前的人」时用面积比高度准。
    player_min_area: float = 0.0
    #: 框面积上限（0 = 不限）
    player_max_area: float = 0.0
    #: 回合切分方式：``auto`` 优先用球员运动切分（推荐），``activity`` 用旧的
    #: 融合活跃度 + 迟滞状态机，``hybrid`` 两种都跑再择优。
    segment_mode: Literal["auto", "activity", "hybrid"] = "auto"
    #: ---- 静默段切分尺度（人工标注校准的目标参数）----
    #: 这些是「球员运动 × 击球密度」证据曲线上找静默谷的内部尺度。它们不在旧
    #: 参数里是因为以前没有 ground truth 可依据（见 HANDOVER 8.6）。有了标注
    #: 之后它们就是最值得校准的量，所以提升为可持久化、可被优化器写入的参数。
    #: 单个静默谷的最短宽度：太短会把曲线抖动当成停顿。
    seg_min_quiet: float = 0.7
    #: 静默谷的显著度门限（相对 p95-p20）；越大要求谷越深。
    seg_prominence: float = 0.18
    #: 两段静默之间至少隔多久才算「回合结束」。
    seg_min_rest: float = 0.8
    #: 允许的最短连续移动段，比它短的候选丢掉。
    seg_min_core: float = 1.0
    #: 逐帧 AI 分析的最大帧数（用于长视频限速；0 = 不限）
    max_frames: int = 0
    #: 分析帧率
    sample_fps: float = 15.0


class AnalysisResult(BaseModel):
    media_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"] = "pending"
    stage: str = ""
    message: str = ""
    progress: float = 0.0
    error: str | None = None
    params: AnalysisParams = Field(default_factory=AnalysisParams)
    started_at: int | None = None
    finished_at: int | None = None
    #: 时间序列信号（下采样后返回前端画波形）
    signals: dict[str, list[float]] = Field(default_factory=dict)
    signal_fps: float = 0.0
    hits: list[ShotEvent] = Field(default_factory=list)
    rallies: list[Rally] = Field(default_factory=list)
    court: dict[str, Any] | None = None
    #: 场地标定与机位识别结果（``court_calib.CourtCalibration.as_payload()``）
    calibration: dict[str, Any] | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ 工程 / 时间线


class Transform(BaseModel):
    scale: float = 1.0
    x: float = 0.0  # 归一化偏移
    y: float = 0.0
    rotation: float = 0.0


class Clip(BaseModel):
    id: str = Field(default_factory=lambda: _uid("c_"))
    media_id: str
    #: 源素材内的入点 / 出点（秒）
    src_in: float = 0.0
    src_out: float = 0.0
    #: 时间线上的起点；为 None 时按顺序排列
    tl_start: float = 0.0
    speed: float = 1.0
    volume: float = 1.0
    transform: Transform = Field(default_factory=Transform)
    #: 关联的回合（AI 切出来的片段）
    rally_id: str | None = None
    label: str = ""
    #: 竖屏自动裁切
    vertical_crop: bool = False
    #: 保持原速的部分（变速时保护，未使用则空）
    protected: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, (self.src_out - self.src_in) / max(self.speed, 1e-6))


class Track(BaseModel):
    id: str = Field(default_factory=lambda: _uid("t_"))
    name: str = "视频轨 1"
    kind: Literal["video", "audio", "overlay"] = "video"
    muted: bool = False
    locked: bool = False
    clips: list[Clip] = Field(default_factory=list)


class Timeline(BaseModel):
    tracks: list[Track] = Field(default_factory=list)
    duration: float = 0.0
    fps: float = 30.0
    width: int = 1920
    height: int = 1080


class ExportPreset(BaseModel):
    id: str
    name: str
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    vcodec: Literal["h264", "hevc"] = "h264"
    encoder: Literal["auto", "nvenc", "x264", "qsv"] = "auto"
    video_bitrate: str = "12M"
    audio_bitrate: str = "192k"
    crf: int | None = None
    container: str = "mp4"
    #: 竖屏自动跟随裁切
    auto_reframe: bool = False


class Project(BaseModel):
    id: str = Field(default_factory=lambda: _uid("p_"))
    name: str = "未命名工程"
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)
    media: list[MediaInfo] = Field(default_factory=list)
    analyses: dict[str, AnalysisResult] = Field(default_factory=dict)
    timeline: Timeline = Field(default_factory=Timeline)
    #: 播放器/界面状态
    ui: dict[str, Any] = Field(default_factory=dict)
    version: int = 1


class ProjectSummary(BaseModel):
    id: str
    name: str
    created_at: int
    updated_at: int
    media_count: int = 0
    duration: float = 0.0
    poster: str | None = None
    rally_count: int = 0
    analyzed: bool = False


# ------------------------------------------------------------------ 任务


class JobInfo(BaseModel):
    id: str = Field(default_factory=lambda: _uid("j_"))
    kind: str
    title: str = ""
    status: Literal["queued", "running", "done", "error", "cancelled"] = "queued"
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    error: str | None = None
    result: Any = None
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)
