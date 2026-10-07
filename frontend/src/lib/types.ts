/** 与后端 Pydantic 模型一一对应的 TypeScript 类型。 */

export type PlayerSide = 'near' | 'far' | 'unknown'

export interface ShotEvent {
  time: number
  player: PlayerSide
  kind: string
  confidence: number
  speed: number | null
  airborne: boolean
  height: number | null
}

export interface RallyFeatures {
  duration: number
  shot_count: number
  tempo: number
  shuttle_speed_p50: number
  shuttle_speed_p95: number
  motion_energy: number
  motion_peak: number
  travel_near: number
  travel_far: number
  smash_count: number
  longest_exchange: number
  finish_intensity: number
  scramble: number
  closeness: number | null
  clutch: boolean
  confidence: number
  /** 语音口令加分：命中配置短语后加在总分上的固定分（0 表示没命中） */
  speech_bonus: number
  /** 本回合命中的口令短语（最多 2 个） */
  speech_phrases: string[]
}

export interface RallyScores {
  total: number
  length: number
  intensity: number
  technique: number
  excitement: number
  production: number
}

export interface Rally {
  id: string
  index: number
  start: number
  end: number
  clip_start: number
  clip_end: number
  serve_time: number | null
  serve_player: PlayerSide
  receive_time: number | null
  receive_player: PlayerSide
  shots: ShotEvent[]
  features: RallyFeatures
  scores: RallyScores
  tags: string[]
  keep: boolean
  starred: boolean
  note: string
  winner: PlayerSide | null
  duration: number
}

/** 机位 / 拍法。不同机位下「谁离相机近」「球员框该多大」完全不同。 */
export type Viewpoint = 'auto' | 'rear' | 'side' | 'elevated' | 'overhead'

/** 人物框尺寸筛选口径。
 *
 * - `off`：不筛选
 * - `absolute`：按「框高占画面高度的比例」筛（全景 / 鱼眼素材下同一个人的
 *   框高在画面中心和边角能差一倍以上，绝对阈值容易误伤）
 * - `relative`：按「框高 ÷ 同帧最大框高」筛，免疫「位置不同框大小不同」
 */
export type PlayerSizeMode = 'off' | 'absolute' | 'relative'

export interface AnalysisParams {
  hit_sensitivity: number
  gap_seconds: number
  min_rally_seconds: number
  max_rally_seconds: number
  target_rally_seconds: number
  split_sensitivity: number
  confidence_min: number
  pre_roll: number
  post_roll: number
  /** 最后一拍之后保留多久（球落地所需的飞行时间）—— 回合终点的锚点 */
  hit_tail_seconds: number
  use_audio: boolean
  use_motion: boolean
  use_players: boolean
  /** 姿态辅助：给音频击球做「是不是我们这场比赛打的」归属判定 */
  use_pose: boolean
  /** 击球归属门控阈值 0~1：越大越严格（剔除更多邻场击球声），即时可调 */
  pose_gate_threshold: number
  /** 挥拍峰能解释击球的时间窗（秒） */
  pose_gate_window: number
  /** 一次挥拍只能解释一次击球（去除恰好同期的邻场击球） */
  pose_gate_one_to_one: boolean
  /** 保留率超出安全区间时仍强制应用门控 */
  pose_gate_force: boolean
  use_shuttle: boolean
  use_scoreboard: boolean
  shuttle_fps: number
  shuttle_budget_seconds: number
  /** 语音口令加分：识别「好球」这类短口令并给命中回合加分 */
  use_speech: boolean
  /** 要检测的口令短语，最多 2 个、每个最多 3 字 */
  speech_phrases: string[]
  /** 每命中一个短语加多少分（总分封顶 100） */
  speech_bonus_points: number
  /** faster-whisper 模型规格：越大越准、越慢、下载越大 */
  speech_model: 'tiny' | 'base' | 'small' | 'medium' | 'large-v3'
  /** 近音容错：把「到球/倒球」这类听错也算命中「好球」（需要 pypinyin），并对回合区间做短窗复核提高召回 */
  speech_fuzzy: boolean
  court_orientation: 'auto' | 'landscape' | 'portrait'
  /** 机位：auto = 自动识别 */
  viewpoint: Viewpoint
  /** 是否自动标定场地（颜色 + 多边形 + 单应变换） */
  auto_calibrate: boolean
  /** 手动标定的场地边界（归一化 0~1，4~24 点）；给了就不再自动猜测 */
  court_poly?: number[][]
  /** 旧字段：手动四角（等价于 court_poly 只给 4 个点） */
  court_quad?: number[][]
  /** 人物框尺寸筛选口径 */
  player_size_mode: PlayerSizeMode
  /** 框高下限（absolute = 画面高度比例；relative = 相对同帧最大框的比值） */
  player_min_height: number
  /** 框高上限（0 = 不限） */
  player_max_height: number
  /** 框面积下限（归一化面积，0 = 不限） */
  player_min_area: number
  /** 框面积上限（0 = 不限） */
  player_max_area: number
  /** 回合切分方式：auto 优先用球员运动切分 */
  segment_mode: 'auto' | 'activity' | 'hybrid'
  /** 静默段：最短宽度（秒） */
  seg_min_quiet: number
  /** 静默段：显著度门限（相对 p95-p20） */
  seg_prominence: number
  /** 静默段：两段之间至少隔多久才算回合结束（秒） */
  seg_min_rest: number
  /** 允许的最短连续移动段（秒） */
  seg_min_core: number
  max_frames: number
  sample_fps: number
  /** Fusion component base weights (annotation optimizer stage E); defaults match the backend. */
  fuse_weight_players: number
  fuse_weight_motion: number
  fuse_weight_audio: number
  fuse_weight_shuttle: number
  fuse_weight_roi: number
}

/* ------------------------------------------------------------------ 人工标注 */

export interface AnnotationRally {
  start: number
  end: number
  note?: string
  source?: string
}

/** 击球级标注：t = 击球时刻（秒），ours = true 我方 / false 邻场 */
export interface AnnotationHit {
  t: number
  ours: boolean
}

/** 自动切分草稿（当前分析结果的回合，用于半自动标注） */
export interface AnnotationDraft {
  start: number
  end: number
  index: number
  shots: number
  score: number
}

export interface AnnotationResponse {
  media_id: string
  media_name: string
  duration: number
  fps: number
  path: string
  rallies: AnnotationRally[]
  hits: AnnotationHit[]
  focus: number[] | null
  note: string
  auto: AnnotationDraft[]
  /** 音频包络（用于击球标注的波形显示） */
  envelope: number[]
  envelope_fps: number
  /** 门控后的击球时刻（当前结果） */
  hit_times: number[]
  /** 门控前的原始击球时刻（用于重新标注/重门控） */
  hit_times_raw: number[]
}

/** 叠加层：某一帧的一个球员框（归一化坐标） */
export interface OverlayBox {
  track: number
  xyxy: [number, number, number, number]
  /** 检测置信度（v2 可视化缓存才有；v1 缺失时不做阈值过滤，全部显示） */
  conf?: number
  /** v3 caches: box was linearly interpolated across a short occlusion gap rather than detected (drawn dimmed) */
  interp?: boolean
}

/** 叠加层：某一帧的一条原始检测框（未跟踪、无 track id，灰色虚线诊断层；v2 缓存才有） */
export interface OverlayDet {
  xyxy: [number, number, number, number]
  conf: number
}

/** 叠加层：某一帧一个人的 COCO 17 关键点，[x, y, conf]，xy 已归一化 */
export interface OverlaySkeleton {
  track: number
  kp: [number, number, number][]
}

/** 叠加层：单帧数据（仅包含有框或有关键点的帧） */
export interface OverlayFrame {
  t: number
  boxes: OverlayBox[]
  skeletons: OverlaySkeleton[]
  /** 该帧的全部原始检测（v2 缓存；尺寸过滤前、跟踪器实际看到的全集） */
  dets?: OverlayDet[]
}

/** GET annotation/overlay 响应 */
export interface OverlayResponse {
  t0: number
  t1: number
  fps: number
  duration: number
  boxes_available: boolean
  skeletons_available: boolean
  frames: OverlayFrame[]
}

/** GET annotation/signals 响应（降采样多轨信号） */
export interface AnnotationSignals {
  duration: number
  fps: number
  activity: number[]
  threshold_hi: number
  threshold_lo: number
  weights: Record<string, number>
  components: Record<string, number[]>
  /** 羽毛球「在飞」覆盖曲线（仅开启 use_shuttle 的新分析有） */
  shuttle_in_flight?: number[]
  motion: number[]
  motion_fps: number
  player_motion: number[]
  active_count: number[]
  player_coverage: number[]
  player_fps: number
  pose: {
    available: boolean
    swing: number[]
    ok: number[]
    overhead: number[]
    fps: number
    coverage: number
  }
  hit_times: number[]
  hit_times_raw: number[]
  has_boxes_cache: boolean
  has_pose_cache: boolean
}

/** 击球归属门控的判定指标 */
export interface HitMetric {
  tp: number
  fp: number
  fn: number
  precision: number
  recall: number
  f1: number
}

/** 一次参数评估的指标 */
export interface SegmentMetric {
  iou: number
  n: number
  tp: number
  fp: number
  fn: number
  precision: number
  recall: number
  f1: number
  params: Record<string, number>
  /** ±1.0s 边界带 F1（起/止均值，P2 混合目标） */
  boundary_band_f1?: number
  /** 0.7 IoU F1 + 0.3 边界带 F1 的组合分 */
  score?: number
  /** 有击球级标注时，该组参数下的击球归属指标 */
  hit?: HitMetric
}

/** 单条标注的一个质量告警（code 为稳定 ASCII 码，UI 翻译） */
export interface QualityWarning {
  code: 'overlap' | 'too_close' | 'duration_outlier' | 'boundary_off_quiet' |
    'boundary_no_hit' | 'evidence_contradiction' | string
  severity: 'warn' | 'info'
  side?: 'start' | 'end'
  /** 与最近参照（quiet 谷/击球/相邻标注）的距离，秒 */
  distance?: number
  /** 建议吸附时刻（只建议，不自动改写），秒 */
  snap_t?: number
  snap_kind?: 'quiet' | string
  snap_distance?: number
  /** 最近击球距离，秒 */
  nearest_hit?: number
  /** duration_outlier：时长与稳健 z */
  duration?: number
  z?: number
  /** evidence_contradiction：内外活动度对比（归一化） */
  contrast?: number
}

/** GET annotation/quality 响应里的一条标注审计结果 */
export interface QualityItem {
  index: number
  start: number
  end: number
  severity: 'ok' | 'info' | 'warn'
  warnings: QualityWarning[]
}

/** GET annotation/quality 响应 */
export interface QualityReport {
  count: number
  severity_counts: { warn: number; info: number }
  items: QualityItem[]
  focus: [number, number] | null
}

/** 分阶段搜索中单个阶段的结果 */
export interface OptimizeStage {
  search_fields: string[]
  tried: number
  best: SegmentMetric | null
  /** weights 阶段：实际接受的权重移动步数 */
  accepted_moves?: number
  factors?: number[]
}

export interface OptimizeResult {
  gt_count: number
  focus: number[]
  iou_threshold: number
  baseline: SegmentMetric
  best: SegmentMetric | null
  results: SegmentMetric[]
  tried: number
  search_fields: string[]
  /** 分阶段搜索：segment / gate / sensitivity / padding */
  stages?: Record<string, OptimizeStage>
  hit_label_count?: number
  suggest: Record<string, unknown>
  /** 优化目标：边界带宽（秒）与权重 */
  objective?: { band: number; band_weight: number }
}

/** 场地标定与机位识别结果 */
export interface CourtCalibration {
  ok: boolean
  viewpoint: Exclude<Viewpoint, 'auto'> | 'unknown'
  viewpoint_label: string
  confidence: number
  /** 场地边界多边形（归一化 N 点，近左起点、沿画面向右绕）；弯边素材会多于 4 点 */
  polygon: number[][]
  /** 拟合出来的四边形（归一化 4 点：近左 / 近右 / 远右 / 远左），单应变换用 */
  quad: number[][]
  /** 多边形点数 */
  point_count: number
  /** 来源：auto 自动识别 / manual 手动标定 */
  source: 'auto' | 'manual' | 'none' | string
  /** 「多边形比拟合四边形多出来的面积」比例：弯边（畸变）越厉害越大 */
  distortion: number
  court_color: string
  court_area_ratio: number
  foreshortening: number
  in_court_ratio: number
  roi: number[] | null
  notes: string[]
}

/** 人物框尺寸筛选的实测统计（分析结果里 player_trace.size_filter） */
export interface PlayerSizeStats {
  mode: PlayerSizeMode | string
  active: boolean
  min_height: number
  max_height: number
  min_area: number
  max_area: number
  /** 参与统计的框数（**筛选前**，只经过几何门限与场地 ROI） */
  total: number
  kept: number
  dropped: number
  frames: number
  /** 自适应参考尺度：每帧最大框高的 90 分位 */
  ref: number
  bins: number
  hist_max: number
  hist: number[]
  overflow: number
  hist_median: number
  hist_p90: number
  /** 每个框的 [框高, 框面积, 同帧最大框高]，界面据此本地模拟任意阈值 */
  sample: number[][]
}

/** 人物框尺寸试测（/player-probe）：只检测不跟踪，秒级返回 */
export interface PlayerProbeFrame {
  t: number
  boxes: number[][]
  confs: number[]
  /** 同帧最大框高（相对口径的参考尺度） */
  ref: number
  /** 这一帧的画面（缓存里的 JPEG 绝对路径，用 api.assetUrl 取）；没存图时为 null */
  image?: string | null
}

export interface PlayerProbe {
  frames: PlayerProbeFrame[]
  /** 每个框的 [框高, 框面积, 同帧最大框高] */
  points: number[][]
  count: number
  /** 取到的最后一帧的时刻 */
  duration: number
  /** 视频总长（秒）；容器读不到时为 0 */
  video_duration?: number
  width: number
  height: number
  aspect: number
  viewpoint: string
  elapsed: number
  model: string
  device: string
  frames_saved?: boolean
  error?: string
}

export interface AnalysisSignals {
  [key: string]: number[] | undefined
}

export interface AnalysisResult {
  media_id: string
  status: 'none' | 'pending' | 'running' | 'done' | 'error' | 'cancelled'
  stage: string
  message: string
  progress: number
  error: string | null
  params: AnalysisParams
  started_at: number | null
  finished_at: number | null
  signals: Record<string, number[]>
  signal_fps: number
  hits: ShotEvent[]
  rallies: Rally[]
  court: Record<string, unknown> | null
  calibration: CourtCalibration | null
  stats: Record<string, any>
}

export interface MediaInfo {
  id: string
  path: string
  name: string
  size: number
  duration: number
  fps: number
  width: number
  height: number
  rotation: number
  vcodec: string
  acodec: string | null
  has_audio: boolean
  created_at: number
  proxy_path: string | null
  proxy_fps: number | null
  proxy_width: number | null
  proxy_height: number | null
  audio_path: string | null
  poster: string | null
}

export interface Transform {
  scale: number
  x: number
  y: number
  rotation: number
}

export interface Clip {
  id: string
  media_id: string
  src_in: number
  src_out: number
  tl_start: number
  speed: number
  volume: number
  transform: Transform
  rally_id: string | null
  label: string
  vertical_crop: boolean
  protected: boolean
}

export interface Track {
  id: string
  name: string
  kind: 'video' | 'audio' | 'overlay'
  muted: boolean
  locked: boolean
  clips: Clip[]
}

export interface Timeline {
  tracks: Track[]
  duration: number
  fps: number
  width: number
  height: number
}

export interface Project {
  id: string
  name: string
  created_at: number
  updated_at: number
  media: MediaInfo[]
  analyses: Record<string, AnalysisResult>
  timeline: Timeline
  ui: Record<string, unknown>
  version: number
}

export interface ProjectSummary {
  id: string
  name: string
  created_at: number
  updated_at: number
  media_count: number
  duration: number
  poster: string | null
  rally_count: number
  analyzed: boolean
}

export interface JobInfo {
  id: string
  kind: string
  title: string
  /** 该任务服务的素材 id（prepare 任务用），界面据此把进度贴到对应卡片 */
  media_id?: string | null
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  stage: string
  message: string
  error: string | null
  result: any
  created_at: number
  updated_at: number
}

/** 场景预设：一次标注/优化得到的切分参数 + 场地标定 + 保存时的预览帧。 */
export interface ScenePreset {
  id: string
  name: string
  note: string
  created_at: number
  source: {
    project_id: string
    project_name: string
    media_id: string
    media_name: string
    frame_time: number
  }
  params: Partial<AnalysisParams>
  court_poly: [number, number][] | null
  aspect: number
  /** 预览帧的绝对路径；用 api.assetUrl 取图 */
  preview: string
  /** 标定来源与拟合质量（标注优化写入），仅用于展示 */
  fit?: Record<string, number | string>
}

export interface ExportPreset {
  id: string
  name: string
  width: number
  height: number
  fps: number
  vcodec: 'h264' | 'hevc'
  encoder: 'auto' | 'nvenc' | 'x264' | 'qsv'
  video_bitrate: string
  audio_bitrate: string
  crf: number | null
  container: string
  auto_reframe: boolean
}

export interface EnvInfo {
  app: string
  version: string
  python: string
  platform: string
  ffmpeg: string
  ffmpeg_error: string | null
  /** 媒体探测链路：pyav / ffprobe / ffmpeg，以及各自的可用性 */
  probe?: {
    active: string
    pyav?: boolean
    pyav_error?: string | null
    ffprobe?: string | null
    ffmpeg?: string
    error?: string
  }
  caps: Record<string, boolean>
  gpu: { available: boolean; name?: string | null; torch?: string; capability?: number[] }
  data_dir: string
  models_dir: string
  cache_dir: string
  /** 默认导出目录（data/exports），自定义导出时的初始值 */
  export_dir: string
  /** Directory holding the rotating backend debug logs (bms_debug_YYYY-MM-DD.log). */
  logs_dir?: string
}

/** 导出记录条目（来自后端登记表，路径可能在任意自定义目录） */
export interface ExportItem {
  id: string
  name: string
  path: string
  size: number
  mtime: number
  group?: string
  mode?: 'merge' | 'separate'
}
