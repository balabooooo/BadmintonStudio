/** 全局状态：会话、工程、分析结果、时间线编辑、播放、任务、通知。 */

import { create } from 'zustand'
import { api } from '../lib/api'
import { timecode } from '../lib/format'
import { ws } from '../lib/ws'
import { applyDocumentLang, getLang, setLang as setRuntimeLang, t as tr, type Lang } from '../i18n'
import type {
  AnalysisParams,
  AnalysisResult,
  Clip,
  EnvInfo,
  JobInfo,
  MediaInfo,
  Project,
  ProjectSummary,
  Rally,
  ScenePreset,
  Timeline,
} from '../lib/types'

export type View = 'library' | 'studio' | 'annotate' | 'exports' | 'settings'

/**
 * 这个回合在成片里被盖住了多少（0~1），以及盖得最多的那一段。
 *
 * 为什么要按「原片时间范围」而不是只看 ``rally_id``：重新分析或重新切分之后，
 * 成片里的旧片段挂的还是老 id，只看 id 会以为「这个回合还没进成片」，
 * 于是同一个回合被重复加进去——那正是「每个回合只加入一次」要避免的事。
 * 把成片里所有片段与这个回合重叠的长度加起来，够长就算已经加过了。
 */
export function rallyFilmCoverage(
  clips: Clip[],
  rally: Rally,
  mediaId?: string,
): { ratio: number; best: Clip | null } {
  const len = Math.max(0.05, rally.clip_end - rally.clip_start)
  let covered = 0
  let best: Clip | null = null
  let bestOverlap = 0
  for (const c of clips) {
    // 跨素材时只认同一素材的片段：不同素材的源片时间轴会重叠，
    // 不按 media_id 过滤会把别的素材的片段错认成「这个回合已经进成片了」。
    if (mediaId && c.media_id !== mediaId) continue
    const overlap = Math.max(0, Math.min(c.src_out, rally.clip_end) - Math.max(c.src_in, rally.clip_start))
    if (overlap <= 0) continue
    covered += overlap
    if (overlap > bestOverlap) {
      bestOverlap = overlap
      best = c
    }
  }
  return { ratio: Math.min(1, covered / len), best }
}

/** 覆盖到这个比例就算「这个回合已经在成片里了」 */
export const FILM_COVERED_RATIO = 0.75

export interface Toast {
  id: string
  kind: 'info' | 'success' | 'warn' | 'error'
  title: string
  detail?: string
  ttl?: number
}

export interface RallyFilter {
  minScore: number
  maxScore: number
  minDuration: number
  maxDuration: number
  minShots: number
  /** 最低分析置信度 0~1：来回都检不到球员的片段会给低置信度，可以直接滤掉 */
  minConfidence: number
  starredOnly: boolean
  keepOnly: boolean
  tags: string[]
  search: string
}

export const DEFAULT_FILTER: RallyFilter = {
  minScore: 0,
  maxScore: 100,
  minDuration: 0,
  maxDuration: 999,
  minShots: 0,
  minConfidence: 0,
  starredOnly: false,
  keepOnly: false,
  tags: [],
  search: '',
}

/** 回合列表的范围：只看当前素材，还是合并工程内所有已分析素材。 */
export type RallyScope = 'current' | 'all'

/** 带归属信息的回合：Rally 本身没有 media_id，跨素材列表/播放队列需要知道它属于谁。 */
export type RallyRef = Rally & { media_id: string; media_name: string }

/**
 * 与后端 ``scoring.PRESETS`` 的键一一对应。
 * 文案不在这里写死，只存 i18n key：语言切换后列表与讲解会立即跟着变。
 * ``detail`` 是给用户的原理讲解：总分是五个分项按这里的权重加权求和，
 * 再乘一个「分析置信度」折扣。把配方直接摊开，用户才知道该选哪个。
 */
export const WEIGHT_PRESETS_MAP = [
  {
    value: 'balanced',
    labelKey: 'weight.balanced.label',
    hintKey: 'weight.balanced.hint',
    detail: {
      summaryKey: 'weight.balanced.summary',
      weights: [
        ['weight.dim.intensity', 30],
        ['weight.dim.length', 22],
        ['weight.dim.technique', 20],
        ['weight.dim.excitement', 20],
        ['weight.dim.production', 8],
      ] as [string, number][],
    },
  },
  {
    value: 'highlight',
    labelKey: 'weight.highlight.label',
    hintKey: 'weight.highlight.hint',
    detail: {
      summaryKey: 'weight.highlight.summary',
      weights: [
        ['weight.dim.intensity', 34],
        ['weight.dim.technique', 26],
        ['weight.dim.excitement', 20],
        ['weight.dim.length', 16],
        ['weight.dim.production', 4],
      ] as [string, number][],
    },
  },
  {
    value: 'long_rally',
    labelKey: 'weight.long_rally.label',
    hintKey: 'weight.long_rally.hint',
    detail: {
      summaryKey: 'weight.long_rally.summary',
      weights: [
        ['weight.dim.length', 42],
        ['weight.dim.intensity', 20],
        ['weight.dim.excitement', 18],
        ['weight.dim.technique', 14],
        ['weight.dim.production', 6],
      ] as [string, number][],
    },
  },
  {
    value: 'technique',
    labelKey: 'weight.technique.label',
    hintKey: 'weight.technique.hint',
    detail: {
      summaryKey: 'weight.technique.summary',
      weights: [
        ['weight.dim.technique', 42],
        ['weight.dim.intensity', 18],
        ['weight.dim.excitement', 18],
        ['weight.dim.length', 14],
        ['weight.dim.production', 8],
      ] as [string, number][],
    },
  },
  {
    value: 'training',
    labelKey: 'weight.training.label',
    hintKey: 'weight.training.hint',
    detail: {
      summaryKey: 'weight.training.summary',
      weights: [
        ['weight.dim.length', 30],
        ['weight.dim.intensity', 26],
        ['weight.dim.technique', 18],
        ['weight.dim.production', 16],
        ['weight.dim.excitement', 10],
      ] as [string, number][],
    },
  },
  {
    value: 'highlight_pro',
    labelKey: 'weight.highlight_pro.label',
    hintKey: 'weight.highlight_pro.hint',
    detail: {
      summaryKey: 'weight.highlight_pro.summary',
      weights: [
        ['weight.dim.highlight', 20],
        ['weight.dim.intensity', 22],
        ['weight.dim.technique', 22],
        ['weight.dim.excitement', 24],
        ['weight.dim.length', 10],
        ['weight.dim.production', 2],
      ] as [string, number][],
    },
  },
]

export const DEFAULT_PARAMS: AnalysisParams = {
  hit_sensitivity: 0.5,
  gap_seconds: 3.0,
  min_rally_seconds: 3.0,
  max_rally_seconds: 120,
  target_rally_seconds: 28,
  split_sensitivity: 0.5,
  confidence_min: 0,
  pre_roll: 1.0,
  post_roll: 0.6,
  // 最后一拍打出去之后球还要飞一会儿才落地，所以回合终点 = 最后一拍 + 这个值。
  // 它才是「球落地后还留很久」这个毛病的总开关：太小会吃掉大力高远球的落地
  // 瞬间，太大会把死球后的捡球画面留在回合里。
  hit_tail_seconds: 0.9,
  use_audio: true,
  use_motion: true,
  use_players: true,
  // 姿态辅助默认开：多球场球馆里音频分不出「谁在击球」，姿态能——
  // 我们的球员只在真击球时挥拍，隔壁场地的击球声在我们画面上没有对应动作。
  // 实测多花约 1.5~2 分钟 / 30 分钟素材；失败自动降级，不影响其他分析。
  use_pose: true,
  // 击球归属门控：多球场球馆里，只有「我方画面里确实挥了拍」的击球声才算。
  // 阈值越大剔除越多邻场击球声；保留率越界时默认放行（保护），可勾选强制应用。
  pose_gate_threshold: 0.22,
  pose_gate_window: 0.35,
  pose_gate_one_to_one: true,
  pose_gate_force: false,
  // 羽毛球轨迹跟踪的计算量随「帧数 × 像素数 × 时间窗」增长，长视频能跑到小时级，
  // 所以默认关闭；用户在「AI 分析」里可以按需打开并设置时间预算。
  use_shuttle: false,
  use_scoreboard: false,
  shuttle_fps: 10,
  shuttle_budget_seconds: 420,
  // 语音口令加分默认关：用 faster-whisper 识别「好球」这类短口令并做近音容错，命中给回合加分。
  // speech_fuzzy 打开时还会对回合区间做短窗复核，提升召回；缺依赖 / 缺模型会自动降级（不影响其它分析）。
  use_speech: false,
  speech_phrases: [],
  speech_bonus_points: 10,
  speech_model: 'medium',
  speech_fuzzy: true,
  court_orientation: 'auto',
  // 机位默认自动识别：先标定场地（颜色 + 多边形），再按球员在场地里的
  // 分布判断是底线后 / 边线侧 / 高机位 / 俯拍，然后选对应的先验参数。
  viewpoint: 'auto',
  auto_calibrate: true,
  // 人物框尺寸筛选默认关：分析里的自适应尺寸门限已经能处理「球员是画面里
  // 最大的人」这种常见素材。看台更近、或者全景畸变严重时再手动打开。
  player_size_mode: 'off',
  player_min_height: 0.05,
  player_max_height: 0,
  player_min_area: 0,
  player_max_area: 0,
  // 切分优先用「球员运动 + 静默段」：回合之间球员必然会停下来，
  // 而整帧运动会一直被观众和隔壁场地污染，这就是旧版切不准的根因。
  segment_mode: 'auto',
  // 静默段尺度：这些是「标注 → 优化参数」校准的目标。默认值来自用真实标注
  // 搜出来的较优解（见 scripts/eval_segmentation.py / 标注页「优化参数」）。
  seg_min_quiet: 0.7,
  seg_prominence: 0.18,
  seg_min_rest: 0.8,
  seg_min_core: 1.0,
  max_frames: 0,
  sample_fps: 12,
}

let toastSeq = 0

/** patchRally 的服务端写入防抖（拖滑杆时本地即时、远端合并）。 */
const patchTimers = new Map<string, number>()
const pendingPatches = new Map<string, Partial<Rally>>()

/** setTimeline 的服务端写入防抖：拖片段/滑杆每个 pointermove 都会触发，
 *  不防抖会每秒发几十个 POST 并让服务端反复写盘。
 *
 *  按工程 id 分开存：以前是单一全局变量，A 工程还没落盘的改动会被 B 工程的
 *  一次 setTimeline 覆盖掉，切工程 / 关窗口时最后 300ms 的编辑还会整段丢失。 */
const timelineSaveTimers = new Map<string, number>()
const pendingTimelines = new Map<string, Timeline>()

/** 打开工程的请求序号：快速连点 A、B 时，让先返回的旧响应作废。 */
let openProjectSeq = 0

/** 正在申请 preview 任务的素材：api 调用到 WS 任务事件到达之间用它去重，避免重复提交。 */
const preparingMedia = new Set<string>()

/** 该素材是否有正在排队 / 运行的分析任务（重切分前用它避免被分析结果覆盖）。 */
function mediaIsAnalyzing(mediaId: string): boolean {
  return Object.values(useStore.getState().jobs).some(
    (j) =>
      j.kind === 'analyze' &&
      j.media_id === mediaId &&
      (j.status === 'running' || j.status === 'queued'),
  )
}

/** 取消某工程还没落盘的时间线写入（收到服务端权威时间线、undo/redo 时调用）。 */
function cancelPendingTimeline(pid: string) {
  const timer = timelineSaveTimers.get(pid)
  if (timer !== undefined) {
    window.clearTimeout(timer)
    timelineSaveTimers.delete(pid)
  }
  pendingTimelines.delete(pid)
}

/** 立刻把某工程待写入的时间线发给服务端。 */
function flushPendingTimeline(pid: string) {
  const timer = timelineSaveTimers.get(pid)
  if (timer !== undefined) {
    window.clearTimeout(timer)
    timelineSaveTimers.delete(pid)
  }
  const tl = pendingTimelines.get(pid)
  if (!tl) return
  pendingTimelines.delete(pid)
  api.setTimeline(pid, tl).catch(() => undefined)
}

/** 页面卸载前把所有待写入的时间线用 keepalive 发出去，避免最后 300ms 的编辑丢失。 */
function flushAllPendingTimelines() {
  for (const t of timelineSaveTimers.values()) window.clearTimeout(t)
  timelineSaveTimers.clear()
  for (const [pid, tl] of pendingTimelines) {
    try {
      fetch(`/api/projects/${pid}/timeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-BMS-Lang': getLang() },
        body: JSON.stringify({ timeline: tl }),
        keepalive: true,
      }).catch(() => undefined)
    } catch {
      /* ignore */
    }
  }
  pendingTimelines.clear()
}

/** bootstrap 幂等：StrictMode 下 effect 会跑两次，重复连接 WS / 注册监听会翻倍 */
let bootstrapStarted = false

interface State {
  // ---------------- 会话
  env: EnvInfo | null
  booted: boolean
  view: View
  projects: ProjectSummary[]
  project: Project | null
  mediaId: string | null
  weights: string
  params: AnalysisParams
  filter: RallyFilter
  /** 回合列表范围：当前素材 / 全部素材 */
  rallyScope: RallyScope
  roi: number[] | null
  /** 界面语言：'zh' 中文 / 'en' 英文，持久化在 localStorage */
  lang: Lang

  // ---------------- 播放
  playing: boolean
  currentTime: number
  /** 主动跳转后的短暂屏蔽窗口（毫秒时间戳），期间忽略 video 的 timeupdate */
  seekingUntil: number
  /**
   * 用户正在拖播放头。
   * 用布尔量而不是只靠超时：慢慢拖、或者拖到一半停住几十毫秒再继续时，
   * 超时会过期，而 video 的 timeupdate 仍在用旧时间回调，播放头就被拽回去。
   */
  userSeeking: boolean
  selectedRallyId: string | null
  /** 主选中片段（Inspector、分割、滚动定位都用它） */
  selectedClipId: string | null
  /** 成片轨多选：Ctrl/Shift 点击累加，批量删除作用在这个集合上 */
  selectedClipIds: string[]
  zoom: number // 每秒像素
  scroll: number
  previewMode: 'source' | 'timeline'
  /** 预览时只连续播放「当前筛选出来的回合」，中间的捡球/走动自动跳过 */
  previewFiltered: boolean
  /** 在预览画面上叠出 AI 识别到的球场范围，用于核对标定对不对 */
  showCourtOverlay: boolean
  /** 场地标定窗口是否打开（预览的角标和「AI 分析」里的按钮都会打开它） */
  courtEditorOpen: boolean

  // ---------------- 任务 / 通知
  jobs: Record<string, JobInfo>
  toasts: Toast[]
  busy: Record<string, boolean>

  // ---------------- 撤销
  history: Timeline[]
  future: Timeline[]

  // ---------------- actions
  bootstrap: () => Promise<void>
  /** 重新拉取 /api/env（依赖在 app 运行期间才装好时，能力位可能已经过期） */
  refreshEnv: () => Promise<void>
  setView: (v: View) => void
  /** 切换界面语言并持久化；会重连 WebSocket 以带上新的 lang 参数 */
  setLang: (l: Lang) => void
  refreshProjects: () => Promise<void>
  createProject: (name: string) => Promise<string | null>
  openProject: (id: string) => Promise<void>
  renameProject: (name: string) => Promise<void>
  deleteProject: (id: string) => Promise<void>
  closeProject: () => void
  importMedia: (paths: string[]) => Promise<void>
  importFiles: (files: File[]) => Promise<void>
  /** 确保某个素材的代理/音轨/封面进入生成队列；已有派生资源或已在队列中就跳过 */
  ensurePrepare: (mid: string) => Promise<void>
  removeMedia: (mid: string) => Promise<void>
  /** 批量从工程移除素材；当前选中的素材若被删会自动切到剩下的第一个 */
  removeMediaBulk: (ids: string[]) => Promise<void>
  selectMedia: (mid: string) => void

  // ---------------- 场景预设（跨工程）
  presets: ScenePreset[]
  refreshPresets: () => Promise<void>
  /** 把当前素材的切分参数 + 场地标定 + 播放头那一帧存成预设 */
  savePreset: (input: {
    name: string
    note?: string
    frameTime: number
    params: Partial<AnalysisParams>
    courtPoly?: [number, number][] | null
    fit?: Record<string, number | string>
  }) => Promise<void>
  /** 套用预设：写入参数与场地标定；已有分析结果时自动重切分一次 */
  applyPreset: (preset: ScenePreset) => void
  deletePreset: (id: string) => Promise<void>

  setParams: (patch: Partial<AnalysisParams>) => void
  setWeights: (w: string) => void
  setFilter: (patch: Partial<RallyFilter>) => void
  setRallyScope: (v: RallyScope) => void
  setRoi: (roi: number[] | null) => void
  runAnalysis: () => Promise<void>
  /** 批量分析：不传 ids 表示工程内全部素材；后端串行排队逐个跑 */
  runAnalysisBatch: (mediaIds?: string[]) => Promise<void>
  resegment: (patch?: Partial<AnalysisParams>, weights?: string) => Promise<void>
  /** 旧分析缺「门控前击球」时，用缓存音频重建击球序列（仅音频检测，不重跑视频 AI） */
  rebuildHits: (patch?: Partial<AnalysisParams>) => Promise<void>
  rescore: (w: string, opts?: { silent?: boolean }) => Promise<void>
  /** 全部素材合成一批重算，让分数跨素材可比 */
  rescoreAll: (w: string, opts?: { silent?: boolean }) => Promise<void>
  patchRally: (rid: string, patch: Partial<Rally>) => Promise<void>
  bulkRallies: (patch: Partial<Rally>, opts?: { ids?: string[]; useFilter?: boolean }) => Promise<void>
  selectRally: (rid: string | null) => void
  /** 跨素材切换视频源并定位（供「只看筛选」自动跳素材、跨素材选中用） */
  switchMediaAt: (mid: string, time: number, rid?: string | null) => void
  /** 指定回合归属的素材 id；找不到返回 null */
  rallyOwner: (rid: string) => string | null
  /** 当前范围内的全部回合（未过筛选），scope 感知 */
  allInScope: () => RallyRef[]
  /** 工程内所有已分析素材是否都已按 ``cross:<weight>`` 统一重算 */
  isUnifiedRescore: (weight?: string) => boolean

  autoCut: (opts?: { mode?: 'replace' | 'append' }) => Promise<void>
  clearTimeline: () => Promise<void>
  setTimeline: (tl: Timeline, pushHistory?: boolean) => void
  addClipFromRally: (rally: Rally) => void
  updateClip: (clipId: string, patch: Partial<Clip>, pushHistory?: boolean) => void
  removeClip: (clipId: string) => void
  /** 批量删除成片片段：一次写盘、只占一步撤销；删完自动左移补位 */
  removeClips: (clipIds: string[]) => void
  splitClipAt: (clipId: string, timeOnTimeline: number) => void
  /** 用「原片时间」分割：源片预览模式下按住 S 走这条，内部会换算成成片时间 */
  splitClipAtSourceTime: (clipId: string, sourceTime: number) => void
  moveClipTo: (clipId: string, newStart: number, newTrackIndex?: number) => void
  reorderTrack: () => void
  undo: () => void
  redo: () => void
  /** 把当前时间线压入撤销栈：连续操作开始前调用一次，之后用 pushHistory=false 应用变更 */
  pushHistory: () => void
  /** mode：replace=单选；toggle=Ctrl 点选；range=Shift 连选 */
  selectClip: (id: string | null, mode?: 'replace' | 'toggle' | 'range') => void
  /** 全选成片轨片段，配合批量删除 */
  selectAllClips: () => void

  setPlaying: (p: boolean) => void
  seek: (t: number) => void
  syncTime: (t: number) => void
  setUserSeeking: (v: boolean) => void
  /** 当前预览模式下的总时长（秒）：成片模式是成片时长，源片模式是素材时长 */
  durationForMode: () => number
  /** 当前预览模式下一帧的秒数：成片用成片帧率，源片用代理帧率（回退源帧率） */
  frameStep: () => number
  setZoom: (z: number) => void
  setScroll: (s: number) => void
  setPreviewMode: (m: 'source' | 'timeline') => void
  setPreviewFiltered: (v: boolean) => void
  setShowCourtOverlay: (v: boolean) => void
  /** 保存/清除当前素材的手动场地标定；会写进工程文件，下次打开还在 */
  setCourtPoly: (q: [number, number][] | null) => void
  /** 指定素材的手动场地标定（不传 poly 表示清除） */
  setCourtPolyFor: (mid: string, q: [number, number][] | null) => void
  /**
   * 把一份场地标定（多边形 + 机位）应用到多块素材：
   * 多边形按素材逐个写进工程 ui；机位等参数是工程级的，合并到全局参数。
   * 只发一次 PATCH。
   */
  applyCalibrationTo: (
    mediaIds: string[],
    poly: [number, number][] | null,
    paramPatch?: Partial<AnalysisParams>,
  ) => void
  setCourtEditorOpen: (v: boolean) => void
  previewFilteredRallies: () => void
  /** 「只看筛选」的下一个落点（跨素材）；null=继续播，'end'=结束 */
  nextFilteredTarget: (
    mid: string,
    t: number,
  ) => { media_id: string; rally_id: string; start: number } | 'end' | null

  exportVideo: (
    presetSource: Partial<import('../lib/types').ExportPreset>,
    name?: string,
    opts?: { mode?: 'merge' | 'separate'; outputDir?: string },
  ) => Promise<{ job_id: string; output: string; mode: 'merge' | 'separate' } | null>
  /** 记住本工程的导出目录（写进 project.ui，下次默认沿用） */
  setExportDir: (dir: string) => void
  toast: (t: Omit<Toast, 'id'>) => void
  dismissToast: (id: string) => void
  currentAnalysis: () => AnalysisResult | null
  currentMedia: () => MediaInfo | null
  /** 当前素材的手动场地标定（从工程 ui 里读；没有则 null，表示让 AI 自动识别） */
  currentCourtPoly: () => [number, number][] | null
  visibleRallies: () => RallyRef[]
}

function filterRallies<T extends Rally>(rallies: T[], f: RallyFilter): T[] {
  const q = f.search.trim().toLowerCase()
  return rallies.filter((r) => {
    if (r.scores.total < f.minScore || r.scores.total > f.maxScore) return false
    if (r.duration < f.minDuration || r.duration > f.maxDuration) return false
    if (r.features.shot_count < f.minShots) return false
    if (r.features.confidence < f.minConfidence) return false
    if (f.starredOnly && !r.starred) return false
    if (f.keepOnly && !r.keep) return false
    if (f.tags.length && !f.tags.some((t) => r.tags.includes(t))) return false
    if (q) {
      const hay = `${r.index} ${r.tags.join(' ')} ${r.note} ${r.scores.total.toFixed(0)}`.toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
}

/**
 * 按工程里的素材顺序收集回合并标注归属，供「全部素材」列表与跨素材播放队列共用。
 * 输出顺序 = 素材在工程里的顺序 → 素材内回合顺序，跨素材播放就按这个顺序走。
 */
export function collectRallies(
  project: Project | null,
  scope: RallyScope,
  mediaId: string | null,
): RallyRef[] {
  if (!project) return []
  const out: RallyRef[] = []
  for (const m of project.media) {
    if (scope === 'current' && m.id !== mediaId) continue
    const a = project.analyses[m.id]
    // 未分析 / 分析中的素材没有完整 rallies，跳过，避免下游按空数组误判
    if (!a || a.status !== 'done' || !Array.isArray(a.rallies)) continue
    for (const r of a.rallies) out.push({ ...r, media_id: m.id, media_name: m.name })
  }
  return out
}

/** 素材 id -> 在工程里的顺序，跨素材播放按这个顺序走。 */
function mediaOrder(project: Project | null): Map<string, number> {
  const m = new Map<string, number>()
  project?.media.forEach((x, i) => m.set(x.id, i))
  return m
}

/** 删除素材后，如果选中的片段已经不在时间线里，就清掉选区，避免 Inspector 悬空。 */
function selectionPatch(
  project: Project,
  ids: string[],
  primary: string | null,
): { selectedClipId?: null; selectedClipIds?: [] } {
  const all = new Set(project.timeline.tracks.flatMap((t) => t.clips.map((c) => c.id)))
  const picked = ids.length ? ids : primary ? [primary] : []
  if (!picked.length) return {}
  if (picked.some((id) => all.has(id))) return {}
  return { selectedClipId: null, selectedClipIds: [] }
}

/**
 * 当前范围内的筛选结果，按「素材顺序 → 素材内开始时间」排好，用于跨素材播放队列。
 * 注意不能只按 start 排序：不同素材的源片时间轴互相独立，混排会一跳一跳。
 */
export function orderedInScope(
  project: Project | null,
  scope: RallyScope,
  mediaId: string | null,
  filter: RallyFilter,
): RallyRef[] {
  const list = filterRallies(collectRallies(project, scope, mediaId), filter)
  const order = mediaOrder(project)
  return list.sort((a, b) => {
    const ma = order.get(a.media_id) ?? 0
    const mb = order.get(b.media_id) ?? 0
    return ma !== mb ? ma - mb : a.start - b.start
  })
}

export const useStore = create<State>((set, get) => ({
  env: null,
  booted: false,
  view: 'library',
  projects: [],
  project: null,
  mediaId: null,
  weights: 'balanced',
  params: { ...DEFAULT_PARAMS },
  filter: { ...DEFAULT_FILTER },
  rallyScope: 'current',
  roi: null,
  lang: getLang(),

  playing: false,
  currentTime: 0,
  seekingUntil: 0,
  userSeeking: false,
  selectedRallyId: null,
  selectedClipId: null,
  selectedClipIds: [],
  zoom: 42,
  scroll: 0,
  previewMode: 'source',
  previewFiltered: false,
  showCourtOverlay: false,
  courtEditorOpen: false,

  jobs: {},
  toasts: [],
  busy: {},
  history: [],
  future: [],
  presets: [],

  // ================================================================= 会话
  async bootstrap() {
    if (bootstrapStarted) return
    bootstrapStarted = true
    applyDocumentLang()
    try {
      const env = await api.env()
      set({ env })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('errors.connectFailed'), detail: String(e) })
    }
    // 把已有的后台任务拉回来，否则刷新页面后进度就永远消失了。
    // 按 updated_at 合并而不是整体替换：listJobs 的响应可能比刚到的 WS 推送旧，
    // 整体替换会把更新的状态又刷回去。
    const syncJobs = () => {
      api.listJobs()
        .then((list) => set((s) => {
          const jobs = { ...s.jobs }
          for (const j of list) {
            const cur = jobs[j.id]
            if (!cur || (j.updated_at ?? 0) >= (cur.updated_at ?? 0)) jobs[j.id] = j
          }
          return { jobs }
        }))
        .catch(() => undefined)
    }
    syncJobs()
    // 关窗口 / 刷新前把待写入的时间线用 keepalive 发出去
    window.addEventListener('beforeunload', flushAllPendingTimelines)
    ws.connect()
    ws.subscribe((m) => {
      if (m.type === 'hello') {
        // 首次连接或断线重连：把断线期间错过的任务状态补回来
        syncJobs()
      } else if (m.type === 'job') {
        set((s) => ({ jobs: { ...s.jobs, [m.job.id]: m.job } }))
        const j = m.job
        // 任务到终态时清掉 busy 标记，否则 busy 会随着任务数无限增长
        if (j.status === 'done' || j.status === 'error' || j.status === 'cancelled') {
          set((s) => {
            if (!s.busy[j.id]) return {}
            const busy = { ...s.busy }
            delete busy[j.id]
            return { busy }
          })
        }
        // 分析结果由后端单独广播的 analysis 消息同步（它带着正确的 media_id）。
        // 这里曾经用「当前选中的 mediaId」去拉结果：分析 A 时切到 B 会把 A 的结果
        // 存到 B 上，拿到不完整对象后界面崩溃；也不该把未分析的 stub 存进去。
        if (j.status === 'done' && j.kind === 'prepare') {
          const pid = get().project?.id
          if (pid) {
            // 只刷新工程/素材，不走 openProject：那会把播放位置、选中回合和
            // 撤销历史全部重置，用户正在编辑时任务完成会被打断。
            api.getProject(pid).then((p) => {
              if (get().project?.id !== pid) return
              const cur = get().mediaId
              const mid = cur && p.media.some((x) => x.id === cur) ? cur : p.media[0]?.id ?? null
              set({ project: p, mediaId: mid })
            }).catch(() => undefined)
          }
        }
      } else if (m.type === 'media') {
        set((s) => {
          if (!s.project || s.project.id !== m.project_id) return {}
          const media = s.project.media.map((x) => (x.id === m.media.id ? m.media : x))
          return { project: { ...s.project, media } }
        })
      } else if (m.type === 'timeline') {
        // 重切分后后端会把时间线片段重新挂到新回合上，这里同步过来。
        // 服务端是权威：取消本地还没落盘的防抖写入，否则它稍后会把用户的旧时间线
        // 覆盖回服务端，把这次重挂冲掉。
        cancelPendingTimeline(m.project_id)
        set((s) =>
          s.project && s.project.id === m.project_id ? { project: { ...s.project, timeline: m.timeline } } : {},
        )
      } else if (m.type === 'analysis') {
        set((s) => (s.project && s.project.id === m.project_id
          ? { project: { ...s.project, analyses: { ...s.project.analyses, [m.media_id]: m.result } } }
          : {}))
      }
    })
    await get().refreshProjects()
    // 恢复 URL 里的工程（刷新页面不丢状态）
    const m = /^#\/studio\/([A-Za-z0-9_]+)/.exec(location.hash)
    if (m) {
      await get().openProject(m[1])
    }
    window.addEventListener('hashchange', () => {
      const mm = /^#\/studio\/([A-Za-z0-9_]+)/.exec(location.hash)
      if (mm && mm[1] !== get().project?.id) get().openProject(mm[1])
      else if (!mm && get().project) get().closeProject()
    })
    set({ booted: true })
  },

  async refreshEnv() {
    try {
      set({ env: await api.env() })
    } catch {
      /* ignore */
    }
  },

  setView: (v) => set({ view: v }),

  setLang(l) {
    if (l === get().lang) return
    setRuntimeLang(l)
    set({ lang: l })
    // WebSocket 连接需要带新的 lang；重连一次最简单可靠（后端按连接上下文回文案）。
    ws.reconnect()
  },

  async refreshProjects() {
    try {
      set({ projects: await api.listProjects() })
    } catch {
      /* ignore */
    }
  },

  async createProject(name) {
    try {
      const p = await api.createProject(name)
      await get().refreshProjects()
      set({ project: p, mediaId: null, view: 'studio', selectedRallyId: null, history: [], future: [] })
      // 更新 URL：刷新页面后还能回到刚建的工程，否则会掉回工程库
      if (location.hash !== `#/studio/${p.id}`) history.replaceState(null, '', `#/studio/${p.id}`)
      return p.id
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.createProjectFailed'), detail: String(e) })
      return null
    }
  },

  async openProject(id) {
    const seq = ++openProjectSeq
    const prevPid = get().project?.id
    try {
      const p = await api.getProject(id)
      // 快速连点 / 并发打开时旧请求可能后返回：只认最后一次请求，避免显示错工程
      if (seq !== openProjectSeq) return
      // 切到别的工程前，先把上一个工程还没落盘的时间线写掉
      if (prevPid && prevPid !== id) flushPendingTimeline(prevPid)
      const mid = p.media.length ? p.media[0].id : null
      const sameProject = get().project?.id === id
      set({
        project: p,
        mediaId: get().mediaId && p.media.some((m) => m.id === get().mediaId) ? get().mediaId : mid,
        view: 'studio',
        selectedRallyId: null,
        // 换工程时回到「只看当前素材」；同一工程刷新（批量操作后）保留当前范围
        rallyScope: sameProject ? get().rallyScope : 'current',
        history: [],
        future: [],
        currentTime: 0,
      })
      if (location.hash !== `#/studio/${id}`) {
        history.replaceState(null, '', `#/studio/${id}`)
      }
      // 换工程：把评分口径对齐到该素材上次用的口径（cross:<口径> 去掉前缀），
      // 再确保当前素材是逐素材口径，否则会把用户上次的口径静默换成默认。
      if (!sameProject) {
        const s = get()
        const a = s.mediaId ? s.project?.analyses[s.mediaId] : null
        const raw = typeof a?.stats?.weights === 'string' ? a.stats.weights : ''
        const plain = raw.startsWith('cross:') ? raw.slice('cross:'.length) : raw
        if (plain && WEIGHT_PRESETS_MAP.some((x) => x.value === plain)) set({ weights: plain })
        const s2 = get()
        const a2 = s2.mediaId ? s2.project?.analyses[s2.mediaId] : null
        if (a2?.status === 'done' && a2.stats?.weights !== s2.weights) {
          void s2.rescore(s2.weights, { silent: true })
        }
      }
    } catch (e) {
      if (seq !== openProjectSeq) return
      get().toast({ kind: 'error', title: tr('toast.openProjectFailed'), detail: String(e) })
    }
  },

  async renameProject(name) {
    const p = get().project
    if (!p) return
    try {
      const updated = await api.patchProject(p.id, { name })
      set({ project: updated })
      get().refreshProjects()
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.persistFailed'), detail: String(e) })
    }
  },

  async deleteProject(id) {
    try {
      await api.deleteProject(id)
      if (get().project?.id === id) set({ project: null, mediaId: null, view: 'library' })
      get().refreshProjects()
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.bulkDeleteFailed'), detail: String(e) })
    }
  },

  closeProject() {
    const pid = get().project?.id
    // 关工程前把还没落盘的时间线写掉，否则最后几百毫秒的编辑会丢
    if (pid) flushPendingTimeline(pid)
    set({
      project: null,
      mediaId: null,
      selectedRallyId: null,
      rallyScope: 'current',
      view: 'library',
      history: [],
      future: [],
    })
    if (location.hash) history.replaceState(null, '', location.pathname)
    get().refreshProjects()
  },

  async importMedia(paths) {
    const p = get().project
    if (!p) return
    set((s) => ({ busy: { ...s.busy, import: true } }))
    try {
      const res = await api.addMedia(p.id, paths)
      set({ project: res.project })
      if (res.added.length) {
        set({ mediaId: res.added[0].id })
        get().toast({ kind: 'success', title: tr('toast.importedMedia', { n: res.added.length }) })
        // 每个新素材都排一个 prepare 任务（后端串行执行），否则只有第一个有封面/代理，
        // 其余素材在列表里一直没有预览，HEVC 大文件在播放器里也永远加载不出来。
        for (const m of res.added) get().ensurePrepare(m.id)
      }
      if (res.failed.length) {
        get().toast({ kind: 'warn', title: tr('toast.importFailedCount', { n: res.failed.length }), detail: res.failed[0].error })
      }
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.importFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, import: false } }))
    }
  },

  async importFiles(files) {
    const p = get().project
    if (!p || !files.length) return
    set((s) => ({ busy: { ...s.busy, import: true } }))
    try {
      let okCount = 0
      let firstError = ''
      const addedIds: string[] = []
      for (const f of files) {
        const res = await fetch(`/api/projects/${p.id}/media/upload`, {
          method: 'POST',
          headers: { 'x-filename': encodeURIComponent(f.name), 'X-BMS-Lang': getLang() },
          body: f,
        })
        // 不检查 res.ok 会把「上传失败」也报成导入成功，素材列表却什么都没多。
        if (res.ok) {
          okCount += 1
          try {
            const j = await res.json()
            if (j?.added?.id) addedIds.push(j.added.id)
          } catch {
            /* 拿不到 id 就只是不自动生成预览，不影响导入 */
          }
        } else if (!firstError) firstError = (await res.text().catch(() => '')) || `HTTP ${res.status}`
      }
      await get().openProject(p.id)
      if (okCount) get().toast({ kind: 'success', title: tr('toast.importedFiles', { n: okCount }) })
      const failed = files.length - okCount
      if (failed) get().toast({ kind: 'warn', title: tr('toast.importFailedCount', { n: failed }), detail: firstError })
      for (const id of addedIds) get().ensurePrepare(id)
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.uploadFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, import: false } }))
    }
  },

  async ensurePrepare(mid) {
    const p = get().project
    if (!p || !mid) return
    const m = p.media.find((x) => x.id === mid)
    if (!m) return
    // 代理和封面都有就不必再跑；只有封面（或只有代理）时仍然补齐缺的那项
    if (m.proxy_path && m.poster) return
    const active = Object.values(get().jobs).some(
      (j) =>
        j.kind === 'prepare' &&
        j.media_id === mid &&
        (j.status === 'running' || j.status === 'queued'),
    )
    if (active || preparingMedia.has(mid)) return
    preparingMedia.add(mid)
    try {
      await api.prepareMedia(p.id, mid)
    } catch {
      /* 预览生成失败不打断导入/浏览，界面会继续显示占位 */
    } finally {
      // WS 任务事件通常很快到达；随后由 jobs 里的 active 判断继续去重
      window.setTimeout(() => preparingMedia.delete(mid), 3000)
    }
  },

  async removeMedia(mid) {
    const p = get().project
    if (!p) return
    try {
      const updated = await api.removeMedia(p.id, mid)
      const cur = get().mediaId
      set({
        project: updated,
        // 删的是别的素材时不要切走当前素材；只有当前素材被删才落到第一个
        mediaId: cur && cur !== mid && updated.media.some((m) => m.id === cur) ? cur : updated.media[0]?.id ?? null,
        ...selectionPatch(updated, get().selectedClipIds, get().selectedClipId),
      })
      get().toast({ kind: 'success', title: tr('toast.removedMedia', { n: 1 }) })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.bulkDeleteFailed'), detail: String(e) })
    }
  },

  async removeMediaBulk(ids) {
    const p = get().project
    if (!p || !ids.length) return
    try {
      const res = await api.deleteMediaBulk(p.id, ids)
      const remaining = res.project.media
      const cur = get().mediaId
      set({
        project: res.project,
        // 当前素材被删了就落到剩下的第一个（可能为空 -> 播放器显示空状态）
        mediaId: cur && remaining.some((m) => m.id === cur) ? cur : remaining[0]?.id ?? null,
        ...selectionPatch(res.project, get().selectedClipIds, get().selectedClipId),
      })
      get().toast({ kind: 'success', title: tr('toast.removedMedia', { n: res.removed }) })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.bulkDeleteFailed'), detail: String(e) })
    }
  },

  selectMedia(mid) {
    // 切换素材时清掉片段选区：否则 Inspector / 时间线还挂在上一个素材的片段上
    set({
      mediaId: mid,
      selectedRallyId: null,
      selectedClipId: null,
      selectedClipIds: [],
      currentTime: 0,
      previewMode: 'source',
    })
    // 选中即确保有可播放的代理：老工程里没生成过预览的素材靠这一步补齐
    get().ensurePrepare(mid)
    // 当前素材范围下，切到的素材若还是统一重算口径，恢复成逐素材标准（命中缓存）
    const s = get()
    if (s.rallyScope !== 'current' || s.busy.rescore) return
    const a = s.project?.analyses[mid]
    if (a?.status === 'done' && a.stats?.weights !== s.weights) void s.rescore(s.weights)
  },

  // ================================================================= 场景预设
  async refreshPresets() {
    try {
      set({ presets: await api.listPresets() })
    } catch {
      /* 预设拉不到不影响分析 */
    }
  },

  async savePreset(input) {
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    try {
      const doc = await api.createPreset({
        project_id: p.id,
        media_id: mid,
        name: input.name,
        note: input.note,
        frame_time: input.frameTime,
        params: input.params,
        court_poly: input.courtPoly ?? null,
        fit: input.fit,
      })
      set((s) => ({ presets: [doc, ...s.presets.filter((x) => x.id !== doc.id)] }))
      get().toast({ kind: 'success', title: tr('toast.presetSaved'), detail: doc.name })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.presetSaveFailed'), detail: String(e) })
    }
  },

  applyPreset(preset) {
    set((s) => ({ params: { ...s.params, ...preset.params } }))
    if (preset.court_poly && preset.court_poly.length >= 4) {
      get().setCourtPoly(preset.court_poly)
    }
    // 切分 / 击球门控参数是「即时生效」类：当前素材已有分析结果时，套用后直接重跑一次
    // 重切分（毫秒级），让预设里的邻场抑制等设置立刻体现在回合上。
    const state = get()
    const mid = state.mediaId
    const cur = state.project && mid ? state.project.analyses?.[mid] : undefined
    if (cur && cur.status === 'done') {
      void get().resegment()
    }
    get().toast({
      kind: 'success',
      title: tr('toast.presetApplied', { name: preset.name }),
      detail: tr('toast.presetAppliedDetail'),
    })
  },

  async deletePreset(id) {
    try {
      await api.deletePreset(id)
      set((s) => ({ presets: s.presets.filter((x) => x.id !== id) }))
      get().toast({ kind: 'info', title: tr('toast.presetDeleted') })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.presetDeleteFailed'), detail: String(e) })
    }
  },

  // ================================================================= 分析
  setParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),
  setWeights: (w) => set({ weights: w }),
  setFilter: (patch) => set((s) => ({ filter: { ...s.filter, ...patch } })),
  setRallyScope: (v) => {
    if (get().rallyScope === v) return
    set({ rallyScope: v })
    const s = get()
    if (!s.project || s.busy.rescore) return
    if (v === 'all') {
      // 进入「全部素材」默认统一重算：已算过时后端命中缓存、前端也会直接复用，
      // 来回切换不会重复计算。
      if (!s.isUnifiedRescore()) void s.rescoreAll(s.weights, { silent: true })
    } else {
      // 回到「当前素材」：把当前素材恢复成逐素材评分标准（从缓存还原，不重算）
      const a = s.currentAnalysis()
      if (a && a.stats?.weights !== s.weights) void s.rescore(s.weights)
    }
  },
  setRoi: (roi) => set({ roi }),

  async runAnalysis() {
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    try {
      const poly = get().currentCourtPoly()
      const { job_id } = await api.analyze(p.id, {
        media_id: mid,
        // 手动标过场地就带上：后端会直接用它建标定，不再自动猜测。
        // 多边形（可能多于 4 点）用来描述全景 / 鱼眼弯掉的边界。
        params: { ...get().params, court_poly: poly ?? undefined },
        weights: get().weights,
        roi: get().roi ?? undefined,
      })
      get().toast({ kind: 'info', title: tr('toast.analysisStarted'), detail: tr('toast.analysisStartedDetail') })
      set((s) => ({ busy: { ...s.busy, [job_id]: true }, view: 'studio' }))
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.analysisStartFailed'), detail: String(e) })
    }
  },

  async runAnalysisBatch(mediaIds) {
    const p = get().project
    if (!p) return
    const ids = mediaIds && mediaIds.length ? mediaIds : p.media.map((m) => m.id)
    if (!ids.length) {
      get().toast({ kind: 'warn', title: tr('toast.noAnalyzableMedia') })
      return
    }
    try {
      const { job_ids } = await api.analyzeBatch(p.id, {
        media_ids: ids,
        // 场地标定不放在这里：后端会按每条素材各自已存的标定去填。
        params: get().params,
        weights: get().weights,
      })
      get().toast({
        kind: 'info',
        title: tr('toast.analysisBatchStarted', { n: ids.length }),
        detail: tr('toast.analysisBatchDetail'),
      })
      set((s) => ({
        busy: { ...s.busy, ...Object.fromEntries(job_ids.map((id) => [id, true])) },
        view: 'studio',
      }))
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.analysisBatchFailed'), detail: String(e) })
    }
  },

  async rescore(w, _opts) {
    // 换评分口径只应该重新算分，绝不能重建回合：
    // 重建会让 TimeLine 上所有片段的 rally_id 悬空（颜色丢、标签对不上），
    // 还会清掉用户手调的入点/出点。所以走 /rallies/rescore 而不是 /resegment。
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    set({ weights: w })
    set((s) => ({ busy: { ...s.busy, rescore: true } }))
    try {
      await api.rescore(p.id, w, mid)
      const res = await api.getAnalysis(p.id, mid)
      // 结果回来时用户可能已经切到别的工程，别把旧工程的结果写进新工程
      if (get().project?.id !== p.id) return
      set((s) =>
        s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {},
      )
      // 片段的分数标签跟着一起刷新（有未落盘编辑时跳过，免得覆盖用户正在拖的时间线）
      const proj = get().project
      if (proj && !pendingTimelines.has(p.id)) {
        const byId = new Map(res.rallies.map((r) => [r.id, r]))
        const tl = structuredClone(proj.timeline)
        let touched = false
        for (const t of tl.tracks) {
          for (const c of t.clips) {
            if (!c.rally_id) continue
            const r = byId.get(c.rally_id)
            if (r) {
              const label = tr('clip.label', { index: r.index, score: r.scores.total.toFixed(0) })
              if (c.label !== label) {
                c.label = label
                touched = true
              }
            }
          }
        }
        if (touched) {
          set((s) => (s.project && s.project.id === p.id ? { project: { ...s.project, timeline: tl } } : {}))
          api.setTimeline(p.id, tl).catch(() => undefined)
        }
      }
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.rescoreFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, rescore: false } }))
    }
  },

  /**
   * 全部素材合成一批重算评分。
   * 各素材的分数原本按各自百分位算，跨素材并列时不可比；合批后进入同一分布。
   */
  async rescoreAll(w, opts) {
    const p = get().project
    if (!p) return
    const mids = Object.keys(p.analyses)
    if (!mids.length) return
    set({ weights: w })
    set((s) => ({ busy: { ...s.busy, rescore: true } }))
    try {
      await api.rescore(p.id, w, undefined, true)
      // 合批重算动了所有素材，逐个拉回最新分析结果
      const results = await Promise.all(
        mids.map((mid) => api.getAnalysis(p.id, mid).catch(() => null)),
      )
      // 期间切了工程就丢弃这批结果，别写进别的工程
      if (get().project?.id !== p.id) return
      const analyses: Record<string, AnalysisResult> = { ...(get().project?.analyses ?? {}) }
      results.forEach((res, i) => {
        if (res && Array.isArray((res as { rallies?: unknown }).rallies)) analyses[mids[i]] = res
      })
      set((s) => (s.project ? { project: { ...s.project, analyses } } : {}))
      // 片段的分数标签跟着刷新（可能包含多个素材的片段）
      const proj = get().project
      if (proj && !pendingTimelines.has(p.id)) {
        const byId = new Map<Rally['id'], Rally>()
        for (const mid of mids) {
          const a = analyses[mid]
          if (a) for (const r of a.rallies) byId.set(r.id, r)
        }
        const tl = structuredClone(proj.timeline)
        let touched = false
        for (const t of tl.tracks) {
          for (const c of t.clips) {
            if (!c.rally_id) continue
            const r = byId.get(c.rally_id)
            if (r) {
              const label = tr('clip.label', { index: r.index, score: r.scores.total.toFixed(0) })
              if (c.label !== label) {
                c.label = label
                touched = true
              }
            }
          }
        }
        if (touched) {
          set((s) => (s.project && s.project.id === p.id ? { project: { ...s.project, timeline: tl } } : {}))
          api.setTimeline(p.id, tl).catch(() => undefined)
        }
      }
      const total = Object.values(analyses).reduce((a, x) => a + (x?.rallies?.length ?? 0), 0)
      if (!opts?.silent) {
        get().toast({
          kind: 'success',
          title: tr('toast.rescoreAllDone'),
          detail: tr('toast.rescoreAllDetail', { n: total }),
        })
      }
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.rescoreAllFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, rescore: false } }))
    }
  },

  /**
   * 只重跑「切分 + 评分」，复用已经算好的活跃度曲线，毫秒级出结果。
   * 调切分粒度、间隔判定、留白、评分口径都走这里，不用等几分钟的 AI 分析。
   */
  async resegment(patch, weights) {
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    // 该素材正在分析时不重切分：分析结束会广播 analysis 结果，把刚切好的回合覆盖掉
    if (mediaIsAnalyzing(mid)) {
      get().toast({ kind: 'warn', title: tr('toast.analysisRunningBlocked') })
      return
    }
    const params = { ...get().params, ...(patch ?? {}) }
    if (patch) set({ params })
    set((s) => ({ busy: { ...s.busy, resegment: true } }))
    try {
      const res = await api.resegment(p.id, {
        media_id: mid,
        params,
        weights: weights ?? get().weights,
      })
      if (get().project?.id !== p.id) return
      set((s) =>
        s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {},
      )
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.resegmentFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, resegment: false } }))
    }
  },

  async rebuildHits(patch) {
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    if (mediaIsAnalyzing(mid)) {
      get().toast({ kind: 'warn', title: tr('toast.analysisRunningBlocked') })
      return
    }
    const params = { ...get().params, ...(patch ?? {}) }
    if (patch) set({ params })
    set((s) => ({ busy: { ...s.busy, resegment: true } }))
    try {
      const res = await api.rebuildHits(p.id, {
        media_id: mid,
        params,
        weights: get().weights,
      })
      if (get().project?.id !== p.id) return
      set((s) =>
        s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {},
      )
      const recovered = (res.stats as Record<string, any> | undefined)?.rebuild_hits?.pose_recovered
      get().toast({
        kind: recovered ? 'success' : 'warn',
        title: tr(recovered ? 'toast.rebuildHitsOk' : 'toast.rebuildHitsNoPose'),
      })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.rebuildHitsFailed'), detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, resegment: false } }))
    }
  },

  async patchRally(rid, patch) {
    const p = get().project
    if (!p) return
    // 写盘失败时回滚用的快照；不然界面上改了、服务端没改，刷新后又变回去
    let snapshot: Rally | null = null
    for (const a of Object.values(p.analyses)) {
      const r = a?.rallies?.find((x) => x.id === rid)
      if (r) {
        snapshot = structuredClone(r)
        break
      }
    }
    // 本地立刻生效（拖滑杆要跟手）
    set((s) => {
      if (!s.project) return {}
      const analyses = { ...s.project.analyses }
      for (const k of Object.keys(analyses)) {
        const a = analyses[k]
        const i = a.rallies.findIndex((r) => r.id === rid)
        if (i >= 0) {
          const rallies = [...a.rallies]
          rallies[i] = { ...rallies[i], ...patch }
          analyses[k] = { ...a, rallies }
          break
        }
      }
      return { project: { ...s.project, analyses } }
    })
    // 服务端写入做防抖：拖一次滑杆会触发几十次变更，逐次写盘会卡
    const pending = pendingPatches.get(rid) ?? {}
    Object.assign(pending, patch)
    pendingPatches.set(rid, pending)
    const old = patchTimers.get(rid)
    if (old) window.clearTimeout(old)
    patchTimers.set(
      rid,
      window.setTimeout(async () => {
        patchTimers.delete(rid)
        const body = pendingPatches.get(rid)
        pendingPatches.delete(rid)
        if (!body) return
        try {
          await api.patchRally(p.id, rid, body)
        } catch (e) {
          // 回滚乐观更新：服务端没写成功，本地也就不该显示成改过了
          if (snapshot) {
            set((s) => {
              if (!s.project || s.project.id !== p.id) return {}
              const analyses = { ...s.project.analyses }
              for (const k of Object.keys(analyses)) {
                const a = analyses[k]
                const i = a.rallies.findIndex((r) => r.id === rid)
                if (i >= 0) {
                  const rallies = [...a.rallies]
                  rallies[i] = snapshot as Rally
                  analyses[k] = { ...a, rallies }
                  break
                }
              }
              return { project: { ...s.project, analyses } }
            })
          }
          get().toast({ kind: 'error', title: tr('toast.patchRallyFailed'), detail: String(e) })
        }
      }, 420),
    )
  },

  async bulkRallies(patch, opts) {
    const p = get().project
    if (!p) return
    const body: any = { patch }
    // 「全部素材」范围不传 media_id：后端会遍历所有分析结果，只按 ids 命中，
    // 否则跨素材批量操作会被限制在当前素材上。
    if (get().rallyScope === 'current') body.media_id = get().mediaId ?? undefined
    if (opts?.ids) {
      if (!opts.ids.length) {
        get().toast({ kind: 'warn', title: tr('toast.noRallies') })
        return
      }
      body.ids = opts.ids
    } else if (opts?.useFilter) {
      // 关键：批量操作只能作用在「当前屏幕上看得见的回合」上。
      // 只发筛选条件的话，服务端拿到的是不完整的条件（客户端还按置信度/
      // 仅保留/关键词过滤过），会把用户手动排除掉的回合又复活。
      const ids = get()
        .visibleRallies()
        .map((r) => r.id)
      // 空列表必须当成「什么都不做」：后端把「未传 ids」和「空 ids」都理解为
      // 按筛选匹配，空 ids 会被当成匹配全部，误伤整个工程。
      if (!ids.length) {
        get().toast({ kind: 'warn', title: tr('toast.noFilteredRallies') })
        return
      }
      body.ids = ids
    } else {
      get().toast({ kind: 'warn', title: tr('toast.noRallies') })
      return
    }
    try {
      const res = await api.bulkRallies(p.id, body)
      // 只刷新工程数据，不走 openProject：那会清空撤销历史、把播放头与选中回合重置，
      // 批量操作后编辑上下文就没了。
      const fresh = await api.getProject(p.id)
      if (get().project?.id === p.id) {
        const cur = get().mediaId
        const mid = cur && fresh.media.some((m) => m.id === cur) ? cur : fresh.media[0]?.id ?? null
        set({ project: fresh, mediaId: mid })
      }
      get().toast({ kind: 'success', title: tr('toast.updatedRallies', { n: res.updated }) })
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.bulkFailed'), detail: String(e) })
    }
  },

  selectRally: (rid) => {
    set({ selectedRallyId: rid, selectedClipId: null, selectedClipIds: [] })
    if (!rid) return
    const s = get()
    const owner = s.rallyOwner(rid)
    // 同素材：只更新选中，Player 自己的定位 effect 会 seek。
    if (!owner || owner === s.mediaId) return
    // 跨素材：切到归属素材并定位到回合开头。用 switchMediaAt 统一处理，
    // 它会设置 currentTime，等视频加载完成后同步过去。
    const rally = s.project?.analyses[owner]?.rallies.find((r) => r.id === rid)
    s.switchMediaAt(owner, rally?.start ?? 0, rid)
  },

  rallyOwner(rid) {
    const s = get()
    if (!s.project) return null
    for (const [mid, a] of Object.entries(s.project.analyses)) {
      if (a && Array.isArray(a.rallies) && a.rallies.some((r) => r.id === rid)) return mid
    }
    return null
  },

  allInScope() {
    const s = get()
    return collectRallies(s.project, s.rallyScope, s.mediaId)
  },

  isUnifiedRescore(weight) {
    const s = get()
    if (!s.project) return false
    const w = weight ?? s.weights
    const expect = `cross:${w}`
    const mids = s.project.media
      .filter((m) => s.project?.analyses[m.id]?.status === 'done')
      .map((m) => m.id)
    if (!mids.length) return false
    return mids.every((mid) => s.project?.analyses[mid]?.stats?.weights === expect)
  },

  switchMediaAt(mid, time, rid = null) {
    set({
      mediaId: mid,
      currentTime: time,
      selectedRallyId: rid,
      selectedClipId: null,
      selectedClipIds: [],
      previewMode: 'source',
      // 换源后 video 会重新加载并用旧时间触发 timeupdate，给一个较长的屏蔽窗口，
      // 直到同步 effect 把 currentTime 写进新视频。
      seekingUntil: Date.now() + 1500,
    })
    get().ensurePrepare(mid)
  },

  // ================================================================= 时间线
  async autoCut(opts) {
    const p = get().project
    if (!p) return
    const ids = get()
      .visibleRallies()
      .map((r) => r.id)
    if (!ids.length) {
      get().toast({ kind: 'warn', title: tr('toast.noUsableRallies') })
      return
    }
    // 全部素材范围不传 media_id：后端按 ids 遍历所有分析结果，
    // 会把多个素材的片段按顺序拼进同一条成片轨（导出自动走分段路径）。
    const body: Parameters<typeof api.autoCut>[1] = {
      rally_ids: ids,
      mode: opts?.mode ?? 'replace',
    }
    if (get().rallyScope === 'current') body.media_id = get().mediaId ?? undefined
    try {
      const res = await api.autoCut(p.id, body)
      get().setTimeline(res.timeline)
      if (res.added === 0 && (res.skipped ?? 0) > 0) {
        get().toast({
          kind: 'info',
          title: tr('toast.allAlreadyInFilm'),
          detail: tr('toast.allAlreadyInFilmDetail'),
        })
      } else {
        get().toast({
          kind: 'success',
          title: (opts?.mode ?? 'replace') === 'append' ? tr('toast.appendedToFilm') : tr('toast.generatedFilm'),
          detail:
            tr('toast.autoCutDetail', {
              added: res.added ?? res.clip_count,
              clips: res.clip_count,
              seconds: res.timeline.duration.toFixed(1),
            }) + (res.skipped ? tr('toast.autoCutSkipped', { n: res.skipped }) : ''),
        })
      }
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.autoCutFailed'), detail: String(e) })
    }
  },

  async clearTimeline() {
    const p = get().project
    if (!p) return
    const empty: Timeline = {
      tracks: [{ id: 't_main', name: tr('timeline.trackDefault'), kind: 'video', muted: false, locked: false, clips: [] }],
      duration: 0,
      fps: 30,
      width: 1920,
      height: 1080,
    }
    set({ selectedClipId: null, selectedClipIds: [] })
    get().setTimeline(empty)
  },

  setTimeline(tl, pushHistory = true) {
    set((s) => {
      if (!s.project) return {}
      const hist = pushHistory && s.project.timeline ? [...s.history.slice(-49), s.project.timeline] : s.history
      return { project: { ...s.project, timeline: tl }, history: hist, future: [] }
    })
    const p = get().project
    if (!p) return
    // 本地状态立刻更新，服务端写入做防抖：拖动片段/滑杆时 pointermove 很密集，
    // 每次都 POST 会卡顿，也会让服务端反复整文件写盘。
    pendingTimelines.set(p.id, tl)
    const old = timelineSaveTimers.get(p.id)
    if (old !== undefined) window.clearTimeout(old)
    timelineSaveTimers.set(
      p.id,
      window.setTimeout(() => {
        timelineSaveTimers.delete(p.id)
        const job = pendingTimelines.get(p.id)
        pendingTimelines.delete(p.id)
        if (job) {
          api.setTimeline(p.id, job).catch(() =>
            get().toast({ kind: 'error', title: tr('toast.persistFailed') }),
          )
        }
      }, 300),
    )
  },

  addClipFromRally(rally) {
    const s = get()
    // 归属素材优先：跨素材列表里点「加入成片」时，clip 必须挂到那个素材上，
    // 否则导出会去当前素材里找根本不存在的原片时间。
    const mid = (rally as Partial<RallyRef>).media_id ?? s.rallyOwner(rally.id) ?? s.mediaId
    if (!mid || !s.project) return
    const tl = structuredClone(s.project.timeline)
    if (!tl.tracks.length) {
      tl.tracks = [{ id: 't_main', name: tr('timeline.trackDefault'), kind: 'video', muted: false, locked: false, clips: [] }]
    }
    const track = tl.tracks[0]
    // 一个回合只进成片一次：重复点 + / 双击色块不应该往成片里塞第二段。
    // 已经在成片里就选中那一段（时间线会滚过去），而不是再加一段。
    const cov = rallyFilmCoverage(track.clips, rally, mid)
    if (cov.ratio >= FILM_COVERED_RATIO && cov.best) {
      set({ selectedClipId: cov.best.id, selectedClipIds: [cov.best.id], selectedRallyId: rally.id })
      get().toast({
        kind: 'info',
        title: tr('toast.rallyAlreadyInFilm', { index: rally.index }),
        detail: tr('toast.rallyAlreadyInFilmDetail', { time: timecode(cov.best.tl_start, false) }),
      })
      return
    }
    const start = track.clips.reduce((m, c) => Math.max(m, c.tl_start + (c.src_out - c.src_in) / c.speed), 0)
    const clip: Clip = {
      id: `c_${Math.random().toString(36).slice(2, 12)}`,
      media_id: mid,
      src_in: rally.clip_start,
      src_out: rally.clip_end,
      tl_start: start,
      speed: 1,
      volume: 1,
      transform: { scale: 1, x: 0, y: 0, rotation: 0 },
      rally_id: rally.id,
      label: tr('clip.label', { index: rally.index, score: rally.scores.total.toFixed(0) }),
      vertical_crop: false,
      protected: false,
    }
    track.clips.push(clip)
    tl.duration = track.clips.reduce((m, c) => Math.max(m, c.tl_start + (c.src_out - c.src_in) / c.speed), 0)
    get().setTimeline(tl)
    set({ selectedClipId: clip.id, selectedClipIds: [clip.id] })
    get().toast({
      kind: 'success',
      title: tr('toast.rallyAdded', { index: rally.index }),
      detail: tr('toast.rallyAddedDetail', { time: timecode(start, false), clips: track.clips.length }),
    })
  },

  updateClip(clipId, patch, pushHistory = true) {
    const s = get()
    if (!s.project) return
    const tl = structuredClone(s.project.timeline)
    for (const t of tl.tracks) {
      const i = t.clips.findIndex((c) => c.id === clipId)
      if (i >= 0) {
        t.clips[i] = { ...t.clips[i], ...patch }
        break
      }
    }
    tl.duration = Math.max(
      0,
      ...tl.tracks.flatMap((t) => t.clips.map((c) => c.tl_start + (c.src_out - c.src_in) / c.speed)),
    )
    get().setTimeline(tl, pushHistory)
  },

  removeClip(clipId) {
    get().removeClips([clipId])
  },

  removeClips(clipIds) {
    const s = get()
    if (!s.project || !clipIds.length) return
    const drop = new Set(clipIds)
    const tl = structuredClone(s.project.timeline)
    const track = tl.tracks[0]
    if (!track) return
    // 记下被删片段的位置和长度：后面的片段要按「前面删掉了多少」整体左移，
    // 这样和逐个删除的效果一致，也不会抹平用户有意留出的空隙。
    const removed = track.clips.filter((c) => drop.has(c.id))
    if (!removed.length) return
    track.clips = track.clips.filter((c) => !drop.has(c.id))
    for (const c of track.clips) {
      let shift = 0
      for (const r of removed) {
        if (r.tl_start < c.tl_start) shift += (r.src_out - r.src_in) / r.speed
      }
      c.tl_start = Math.max(0, c.tl_start - shift)
    }
    tl.duration = Math.max(0, ...track.clips.map((c) => c.tl_start + (c.src_out - c.src_in) / c.speed))
    get().setTimeline(tl)
    if (s.selectedClipIds.some((id) => drop.has(id)) || (s.selectedClipId && drop.has(s.selectedClipId))) {
      set({ selectedClipId: null, selectedClipIds: [] })
    }
  },

  splitClipAt(clipId, timeOnTimeline) {
    const s = get()
    if (!s.project) return
    const tl = structuredClone(s.project.timeline)
    const track = tl.tracks[0]
    const i = track.clips.findIndex((c) => c.id === clipId)
    if (i < 0) return
    const c = track.clips[i]
    const dur = (c.src_out - c.src_in) / c.speed
    const offset = timeOnTimeline - c.tl_start
    if (offset <= 0.05 || offset >= dur - 0.05) {
      s.toast({
        kind: 'warn',
        title: tr('toast.cannotSplitHere'),
        detail: tr('toast.cannotSplitTimeline'),
      })
      return
    }
    const srcSplit = c.src_in + offset * c.speed
    const right: Clip = {
      ...c,
      id: `c_${Math.random().toString(36).slice(2, 12)}`,
      src_in: Number(srcSplit.toFixed(3)),
      tl_start: Number((c.tl_start + offset).toFixed(3)),
      label: c.label,
    }
    c.src_out = Number(srcSplit.toFixed(3))
    track.clips.splice(i + 1, 0, right)
    get().setTimeline(tl)
    get().toast({ kind: 'info', title: tr('toast.splitDone') })
  },

  /**
   * 按「原片时间」分割。
   *
   * 默认预览是**源片**模式，播放头读数是原片时间（0~30 分钟），而片段位置是
   * **成片**时间（0~13 分钟）——直接拿原片时间去减 ``tl_start`` 会切在完全
   * 不相干的地方。这里先找出播放头落在片段的哪一时刻，再换算成成片时间。
   */
  splitClipAtSourceTime(clipId, sourceTime) {
    const s = get()
    if (!s.project) return
    const c = s.project.timeline.tracks.flatMap((t) => t.clips).find((x) => x.id === clipId)
    if (!c) return
    if (sourceTime < c.src_in || sourceTime > c.src_out) {
      s.toast({
        kind: 'warn',
        title: tr('toast.cannotSplitHere'),
        detail: tr('toast.cannotSplitSource', {
          time: sourceTime.toFixed(1),
          from: c.src_in.toFixed(1),
          to: c.src_out.toFixed(1),
        }),
      })
      return
    }
    s.splitClipAt(clipId, c.tl_start + (sourceTime - c.src_in) / c.speed)
  },

  moveClipTo(clipId, newStart, newTrackIndex) {
    const s = get()
    if (!s.project) return
    const tl = structuredClone(s.project.timeline)
    let clip: Clip | null = null
    for (const t of tl.tracks) {
      const i = t.clips.findIndex((c) => c.id === clipId)
      if (i >= 0) {
        clip = t.clips.splice(i, 1)[0]
        break
      }
    }
    if (!clip) return
    const ti = Math.min(Math.max(0, newTrackIndex ?? 0), tl.tracks.length - 1)
    clip.tl_start = Math.max(0, Number(newStart.toFixed(3)))
    tl.tracks[ti].clips.push(clip)
    for (const t of tl.tracks) t.clips.sort((a, b) => a.tl_start - b.tl_start)
    tl.duration = Math.max(
      0,
      ...tl.tracks.flatMap((t) => t.clips.map((c) => c.tl_start + (c.src_out - c.src_in) / c.speed)),
    )
    get().setTimeline(tl)
  },

  reorderTrack() {
    const s = get()
    if (!s.project) return
    const tl = structuredClone(s.project.timeline)
    const track = tl.tracks[0]
    track.clips.sort((a, b) => a.tl_start - b.tl_start)
    let cursor = 0
    for (const c of track.clips) {
      c.tl_start = Number(cursor.toFixed(3))
      cursor += (c.src_out - c.src_in) / c.speed
    }
    tl.duration = cursor
    get().setTimeline(tl)
    get().toast({ kind: 'success', title: tr('toast.gapsRemoved') })
  },

  undo() {
    const before = get().project?.timeline
    set((s) => {
      if (!s.history.length || !s.project) return {}
      const history = [...s.history]
      const prev = history.pop()!
      return {
        project: { ...s.project, timeline: prev },
        history,
        future: [s.project.timeline, ...s.future].slice(0, 50),
      }
    })
    const p = get().project
    if (p && p.timeline !== before) {
      // 先取消还没落盘的防抖写入：否则它会在 undo 之后触发，把撤销掉的时间线又写回服务端
      cancelPendingTimeline(p.id)
      api.setTimeline(p.id, p.timeline).catch(() =>
        get().toast({ kind: 'error', title: tr('toast.persistFailed') }),
      )
    }
  },

  redo() {
    const before = get().project?.timeline
    set((s) => {
      if (!s.future.length || !s.project) return {}
      const future = [...s.future]
      const next = future.shift()!
      return {
        project: { ...s.project, timeline: next },
        future,
        history: [...s.history, s.project.timeline].slice(-50),
      }
    })
    const p = get().project
    if (p && p.timeline !== before) {
      cancelPendingTimeline(p.id)
      api.setTimeline(p.id, p.timeline).catch(() =>
        get().toast({ kind: 'error', title: tr('toast.persistFailed') }),
      )
    }
  },

  pushHistory() {
    set((s) => (s.project ? { history: [...s.history.slice(-49), s.project.timeline], future: [] } : {}))
  },

  selectClip: (id, mode = 'replace') => {
    const s = get()
    if (!id) {
      set({ selectedClipId: null, selectedClipIds: [], selectedRallyId: null })
      return
    }
    if (mode === 'toggle') {
      const has = s.selectedClipIds.includes(id)
      const next = has ? s.selectedClipIds.filter((x) => x !== id) : [...s.selectedClipIds, id]
      set({
        selectedClipIds: next,
        selectedClipId: next.length ? (has ? next[next.length - 1] : id) : null,
        selectedRallyId: null,
      })
      return
    }
    if (mode === 'range' && s.selectedClipId) {
      // 连选范围按成片时间轴上的先后顺序取，和视觉顺序一致。
      const ordered = (s.project?.timeline.tracks[0]?.clips ?? []).slice().sort((a, b) => a.tl_start - b.tl_start)
      const ai = ordered.findIndex((c) => c.id === s.selectedClipId)
      const bi = ordered.findIndex((c) => c.id === id)
      if (ai >= 0 && bi >= 0) {
        const [lo, hi] = ai < bi ? [ai, bi] : [bi, ai]
        set({
          selectedClipIds: ordered.slice(lo, hi + 1).map((c) => c.id),
          selectedClipId: id,
          selectedRallyId: null,
        })
        return
      }
    }
    set({ selectedClipId: id, selectedClipIds: [id], selectedRallyId: null })
  },

  selectAllClips: () => {
    const s = get()
    const ids = (s.project?.timeline.tracks.flatMap((t) => t.clips) ?? []).map((c) => c.id)
    set({ selectedClipIds: ids, selectedClipId: ids[ids.length - 1] ?? null, selectedRallyId: null })
  },

  // ================================================================= 播放
  setPlaying: (p) => set({ playing: p }),
  /**
   * 主动跳转（点时间轴、点回合、上一/下一回合）。
   * 会短暂屏蔽 video 的 timeupdate 回写：``<video>.currentTime`` 的赋值是异步的，
   * 屏蔽期内它会继续以旧时间触发 timeupdate，把播放头又拽回去，看起来就是「跳变」。
   */
  seek: (t) => {
    const s = get()
    const dur = s.durationForMode()
    // 夹到可播放范围内：否则「快进」越过末尾后时间码会显示到总时长之外，
    // 要等 video 的 timeupdate 把它拉回来，看上去就是读数乱跳。
    const clamped = dur > 0 ? Math.max(0, Math.min(t, dur)) : Math.max(0, t)
    set({
      currentTime: clamped,
      // 不缩短已有的屏蔽窗口：跨素材切源时 switchMediaAt 会设一个更长的窗口，
      // 随后的定位 seek 不能把它覆盖掉，否则视频还没加载完就被旧 timeupdate 拉回去。
      seekingUntil: Math.max(s.seekingUntil, Date.now() + 420),
    })
  },
  durationForMode() {
    const s = get()
    if (s.previewMode === 'timeline') return s.project?.timeline?.duration ?? 0
    return s.currentMedia()?.duration ?? 0
  },
  frameStep() {
    const s = get()
    if (s.previewMode === 'timeline') return 1 / (s.project?.timeline?.fps || 30)
    const m = s.currentMedia()
    return 1 / (m?.proxy_fps || m?.fps || 30)
  },
  /**
   * video 元素自身推进时间。
   * 拖动播放头期间（以及刚松手的几百毫秒内）一律忽略：
   * ``<video>.currentTime`` 的赋值是异步的，赋值后它还会用旧时间再回调几次
   * ``timeupdate``，直接回写就把播放头拽回去了。
   */
  syncTime: (t) =>
    set((s) =>
      s.userSeeking || Date.now() < s.seekingUntil ? {} : { currentTime: Math.max(0, t) },
    ),
  setUserSeeking: (v) => set(v ? { userSeeking: true } : { userSeeking: false, seekingUntil: Date.now() + 320 }),
  setZoom: (z) => set({ zoom: Math.max(6, Math.min(600, z)) }),
  setScroll: (s) => set({ scroll: Math.max(0, s) }),
  setPreviewMode: (m) => set({ previewMode: m }),
  setPreviewFiltered: (v) => set({ previewFiltered: v }),
  setShowCourtOverlay: (v) => set({ showCourtOverlay: v }),
  setCourtEditorOpen: (v) => set({ courtEditorOpen: v }),

  setCourtPoly(q) {
    const mid = get().mediaId
    if (mid) get().setCourtPolyFor(mid, q)
  },

  setCourtPolyFor(mid, q) {
    // 手动标定写进工程的 ui 里（按素材分开存），下次打开工程还在 ——
    // 所以这里不需要再存一份内存状态，`currentCourtPoly()` 读的就是它，
    // 这样「刷新页面 / 换素材」都不会拿到另一条素材的标定。
    // 一份素材标一次就够，所以刻意做成「跟着素材走」而不是「跟着分析走」。
    const p = get().project
    if (!p || !mid) return
    const ui = { ...(p.ui || {}) } as Record<string, any>
    // 新键 court_polys；同时读的时候兼容旧的 court_quads（只有四个角的年代）
    const polys = { ...((ui.court_polys || ui.court_quads || {}) as Record<string, unknown>) }
    if (q) polys[mid] = q
    else delete polys[mid]
    ui.court_polys = polys
    set({ project: { ...p, ui } })
    api.patchProject(p.id, { ui }).catch(() => {
      // 内存里已生效，但没写进工程文件，刷新就没了——必须告诉用户
      get().toast({ kind: 'error', title: tr('toast.persistFailed') })
    })
  },

  applyCalibrationTo(mediaIds, poly, paramPatch) {
    // 同一机位拍的多个片段没必要逐条重画：把一份标定（多边形）发给多块素材。
    // 多边形按素材存；机位等参数是工程级的，合并进全局 params（会话内生效）。
    const p = get().project
    if (!p || !mediaIds.length) return
    const ids = new Set(mediaIds)
    const ui = { ...(p.ui || {}) } as Record<string, any>
    const polys = { ...((ui.court_polys || ui.court_quads || {}) as Record<string, unknown>) }
    for (const mid of ids) {
      if (poly) polys[mid] = poly
      else delete polys[mid]
    }
    ui.court_polys = polys
    set((s) => ({
      project: { ...p, ui },
      params: paramPatch ? { ...s.params, ...paramPatch } : s.params,
    }))
    api.patchProject(p.id, { ui }).catch(() => {
      get().toast({ kind: 'error', title: tr('toast.persistFailed') })
    })
    get().toast({
      kind: 'success',
      title: tr('toast.calibrationApplied', { n: ids.size }),
      detail: paramPatch?.viewpoint
        ? tr('toast.calibrationDetailWithViewpoint')
        : tr('toast.calibrationDetailNoViewpoint'),
    })
  },

  /** 只看筛选片段：从下一个筛选回合开始连续播放；全部素材模式下播完自动切素材。 */
  previewFilteredRallies() {
    const s = get()
    const list = orderedInScope(s.project, s.rallyScope, s.mediaId, s.filter)
    if (!list.length) {
      get().toast({ kind: 'warn', title: tr('toast.noUsableRallies'), detail: tr('toast.noUsableRalliesHint') })
      return
    }
    const mid = s.mediaId
    const order = mediaOrder(s.project)
    const myIdx = mid ? order.get(mid) ?? 0 : 0
    const t = s.currentTime
    const cur = list.find((r) => r.media_id === mid && t >= r.start && t <= r.end)
    const after = list.find((r) => {
      if (r.media_id === mid) return r.start > t + 0.05
      return (order.get(r.media_id) ?? 0) > myIdx
    })
    const target = cur ?? after ?? list[0]
    set({ previewFiltered: true, previewMode: 'source' })
    // 目标在别的素材上就先切素材（切源），同素材直接 seek
    if (target.media_id !== mid) get().switchMediaAt(target.media_id, target.start, target.id)
    else get().seek(target.start)
    get().setPlaying(true)
    const cross = new Set(list.map((r) => r.media_id)).size > 1
    get().toast({
      kind: 'info',
      title: tr('toast.previewFilteredOn'),
      detail:
        tr('toast.previewFilteredDetail', { n: list.length }) +
        (cross ? tr('toast.previewFilteredCross') : ''),
    })
  },

  nextFilteredTarget(mid, t) {
    const s = get()
    const list = orderedInScope(s.project, s.rallyScope, s.mediaId, s.filter)
    if (!list.length) return 'end'
    const order = mediaOrder(s.project)
    // 当前正处在某个筛选回合里：到尾巴就跳到队列里的下一个（可能换素材）
    const curIdx = list.findIndex((r) => r.media_id === mid && t >= r.start && t <= r.end)
    if (curIdx >= 0) {
      const cur = list[curIdx]
      if (t >= cur.end - 0.06) {
        const next = list[curIdx + 1]
        return next ? { media_id: next.media_id, rally_id: next.id, start: next.start } : 'end'
      }
      return null
    }
    // 不在任何筛选回合里（捡球/走动）：找当前素材里下一个；没有就找后续素材的第一个
    const myIdx = order.get(mid) ?? 0
    const next = list.find((r) => {
      if (r.media_id === mid) return r.start > t + 0.02
      return (order.get(r.media_id) ?? 0) > myIdx
    })
    return next ? { media_id: next.media_id, rally_id: next.id, start: next.start } : 'end'
  },

  // ================================================================= 导出
  async exportVideo(presetSource, name, opts) {
    const p = get().project
    if (!p) return null
    const mode = opts?.mode ?? 'merge'
    try {
      const res = await api.exportProject(p.id, {
        preset: presetSource,
        name,
        mode,
        output_dir: opts?.outputDir?.trim() || undefined,
      })
      if (opts?.outputDir?.trim()) get().setExportDir(opts.outputDir.trim())
      // 不跳页：跳走之后新任务还没完成，用户看到的是空列表，反而以为失败了
      get().toast({
        kind: 'info',
        title: tr('toast.exportStarted'),
        detail: mode === 'separate' ? tr('toast.exportStartedSeparate') : tr('toast.exportStartedMerge'),
      })
      return res
    } catch (e) {
      get().toast({ kind: 'error', title: tr('toast.exportFailed'), detail: String(e) })
      return null
    }
  },

  setExportDir(dir) {
    const p = get().project
    if (!p) return
    const ui = { ...(p.ui || {}) } as Record<string, any>
    ui.export_dir = dir
    set({ project: { ...p, ui } })
    api.patchProject(p.id, { ui }).catch(() => {
      get().toast({ kind: 'error', title: tr('toast.persistFailed') })
    })
  },

  // ================================================================= 通知
  toast(t) {
    const id = `t${++toastSeq}`
    set((s) => ({ toasts: [...s.toasts, { ...t, id }] }))
    const ttl = t.ttl ?? (t.kind === 'error' ? 7000 : 3600)
    window.setTimeout(() => get().dismissToast(id), ttl)
  },
  dismissToast(id) {
    set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) }))
  },

  // ================================================================= 选择器
  currentMedia() {
    const s = get()
    if (!s.project || !s.mediaId) return null
    return s.project.media.find((m) => m.id === s.mediaId) ?? null
  },
  currentAnalysis() {
    const s = get()
    if (!s.project || !s.mediaId) return null
    const a = s.project.analyses[s.mediaId]
    // 后端在「尚未分析」时返回的是 {status:'none'} 这种不完整对象，
    // 它不是 AnalysisResult。当成结果用会让下游 a.rallies 变成 undefined 并崩溃。
    if (!a || !Array.isArray((a as { rallies?: unknown }).rallies)) return null
    return a
  },
  currentCourtPoly() {
    const s = get()
    const ui = s.project?.ui as Record<string, any> | undefined
    // 先找新键；旧工程的 court_quads（只有四角）继续生效
    const raw = ui?.court_polys ?? ui?.court_quads
    const q = s.mediaId ? raw?.[s.mediaId] : null
    if (Array.isArray(q) && q.length >= 4) return q as [number, number][]
    return null
  },
  visibleRallies() {
    const s = get()
    return filterRallies(collectRallies(s.project, s.rallyScope, s.mediaId), s.filter)
  },
}))

export { filterRallies }
