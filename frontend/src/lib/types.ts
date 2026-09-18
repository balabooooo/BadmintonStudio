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
  use_shuttle: boolean
  use_scoreboard: boolean
  shuttle_fps: number
  shuttle_budget_seconds: number
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
  max_frames: number
  sample_fps: number
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
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  stage: string
  message: string
  error: string | null
  result: any
  created_at: number
  updated_at: number
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
  caps: Record<string, boolean>
  gpu: { available: boolean; name?: string | null; torch?: string; capability?: number[] }
  data_dir: string
  models_dir: string
  cache_dir: string
}
