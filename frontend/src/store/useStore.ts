/** 全局状态：会话、工程、分析结果、时间线编辑、播放、任务、通知。 */

import { create } from 'zustand'
import { api } from '../lib/api'
import { timecode } from '../lib/format'
import { ws } from '../lib/ws'
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
  Timeline,
} from '../lib/types'

export type View = 'library' | 'studio' | 'exports' | 'settings'

/**
 * 这个回合在成片里被盖住了多少（0~1），以及盖得最多的那一段。
 *
 * 为什么要按「原片时间范围」而不是只看 ``rally_id``：重新分析或重新切分之后，
 * 成片里的旧片段挂的还是老 id，只看 id 会以为「这个回合还没进成片」，
 * 于是同一个回合被重复加进去——那正是「每个回合只加入一次」要避免的事。
 * 把成片里所有片段与这个回合重叠的长度加起来，够长就算已经加过了。
 */
export function rallyFilmCoverage(clips: Clip[], rally: Rally): { ratio: number; best: Clip | null } {
  const len = Math.max(0.05, rally.clip_end - rally.clip_start)
  let covered = 0
  let best: Clip | null = null
  let bestOverlap = 0
  for (const c of clips) {
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

/**
 * 与后端 ``scoring.PRESETS`` 的键一一对应。
 * ``detail`` 是给用户的原理讲解：总分是五个分项按这里的权重加权求和，
 * 再乘一个「分析置信度」折扣。把配方直接摊开，用户才知道该选哪个。
 */
export const WEIGHT_PRESETS_MAP = [
  {
    value: 'balanced',
    label: '均衡',
    hint: '什么都兼顾，不知道选哪个就用它',
    detail: {
      summary: '五个分项大致平摊，适合先通看一遍。',
      weights: [
        ['强度', 30],
        ['长度', 22],
        ['技术', 20],
        ['精彩', 20],
        ['画面', 8],
      ] as [string, number][],
    },
  },
  {
    value: 'highlight',
    label: '精彩集锦',
    hint: '做高光混剪用，偏向快、猛、有对抗',
    detail: {
      summary: '把「强度」和「技术」的权重拉到最高，长度只占一小部分——短促但对拉凶狠的球会更靠前，画面质量几乎不参与。',
      weights: [
        ['强度', 34],
        ['技术', 26],
        ['精彩', 20],
        ['长度', 16],
        ['画面', 4],
      ] as [string, number][],
    },
  },
  {
    value: 'long_rally',
    label: '多拍回合',
    hint: '找拉锯战，拍数多、时间长的优先',
    detail: {
      summary: '「长度」占近一半权重，拍数与时长的饱和阈值也调高——长回合更容易拿高分，短平快不再占便宜。',
      weights: [
        ['长度', 42],
        ['强度', 20],
        ['精彩', 18],
        ['技术', 14],
        ['画面', 6],
      ] as [string, number][],
    },
  },
  {
    value: 'technique',
    label: '技术动作',
    hint: '复盘动作用，看球速和发力',
    detail: {
      summary: '「技术」权重最高，主要看球速、击球力度与羽毛球在画面中的持续性；长度和强度只是陪跑。',
      weights: [
        ['技术', 42],
        ['强度', 18],
        ['精彩', 18],
        ['长度', 14],
        ['画面', 8],
      ] as [string, number][],
    },
  },
  {
    value: 'training',
    label: '训练复盘',
    hint: '教学材料用，画面清楚、动作完整的优先',
    detail: {
      summary: '唯一把「画面质量」提到两位数权重的口径：清晰度、抖动、主体大小都会影响分数，糊掉或拍歪的片段会被压下去。',
      weights: [
        ['长度', 30],
        ['强度', 26],
        ['技术', 18],
        ['画面', 16],
        ['精彩', 10],
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
  // 羽毛球轨迹跟踪的计算量随「帧数 × 像素数 × 时间窗」增长，长视频能跑到小时级，
  // 所以默认关闭；用户在「AI 分析」里可以按需打开并设置时间预算。
  use_shuttle: false,
  use_scoreboard: false,
  shuttle_fps: 10,
  shuttle_budget_seconds: 420,
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
  max_frames: 0,
  sample_fps: 12,
}

let toastSeq = 0

/** patchRally 的服务端写入防抖（拖滑杆时本地即时、远端合并）。 */
const patchTimers = new Map<string, number>()
const pendingPatches = new Map<string, Partial<Rally>>()

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
  roi: number[] | null

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
  selectedClipId: string | null
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
  setView: (v: View) => void
  refreshProjects: () => Promise<void>
  createProject: (name: string) => Promise<string | null>
  openProject: (id: string) => Promise<void>
  renameProject: (name: string) => Promise<void>
  deleteProject: (id: string) => Promise<void>
  closeProject: () => void
  importMedia: (paths: string[]) => Promise<void>
  importFiles: (files: File[]) => Promise<void>
  removeMedia: (mid: string) => Promise<void>
  selectMedia: (mid: string) => void

  setParams: (patch: Partial<AnalysisParams>) => void
  setWeights: (w: string) => void
  setFilter: (patch: Partial<RallyFilter>) => void
  setRoi: (roi: number[] | null) => void
  runAnalysis: () => Promise<void>
  resegment: (patch?: Partial<AnalysisParams>, weights?: string) => Promise<void>
  rescore: (w: string) => Promise<void>
  patchRally: (rid: string, patch: Partial<Rally>) => Promise<void>
  bulkRallies: (patch: Partial<Rally>, opts?: { ids?: string[]; useFilter?: boolean }) => Promise<void>
  selectRally: (rid: string | null) => void

  autoCut: (opts?: { mode?: 'replace' | 'append' }) => Promise<void>
  clearTimeline: () => Promise<void>
  setTimeline: (tl: Timeline, pushHistory?: boolean) => void
  addClipFromRally: (rally: Rally) => void
  updateClip: (clipId: string, patch: Partial<Clip>, pushHistory?: boolean) => void
  removeClip: (clipId: string) => void
  splitClipAt: (clipId: string, timeOnTimeline: number) => void
  /** 用「原片时间」分割：源片预览模式下按住 S 走这条，内部会换算成成片时间 */
  splitClipAtSourceTime: (clipId: string, sourceTime: number) => void
  moveClipTo: (clipId: string, newStart: number, newTrackIndex?: number) => void
  reorderTrack: () => void
  undo: () => void
  redo: () => void
  selectClip: (id: string | null) => void

  setPlaying: (p: boolean) => void
  seek: (t: number) => void
  syncTime: (t: number) => void
  setUserSeeking: (v: boolean) => void
  setZoom: (z: number) => void
  setScroll: (s: number) => void
  setPreviewMode: (m: 'source' | 'timeline') => void
  setPreviewFiltered: (v: boolean) => void
  setShowCourtOverlay: (v: boolean) => void
  /** 保存/清除手动场地标定；会写进工程文件，下次打开还在 */
  setCourtPoly: (q: [number, number][] | null) => void
  setCourtEditorOpen: (v: boolean) => void
  previewFilteredRallies: () => void

  exportVideo: (
    presetSource: Partial<import('../lib/types').ExportPreset>,
    name?: string,
  ) => Promise<{ job_id: string; output: string } | null>
  toast: (t: Omit<Toast, 'id'>) => void
  dismissToast: (id: string) => void
  currentAnalysis: () => AnalysisResult | null
  currentMedia: () => MediaInfo | null
  /** 当前素材的手动场地标定（从工程 ui 里读；没有则 null，表示让 AI 自动识别） */
  currentCourtPoly: () => [number, number][] | null
  visibleRallies: () => Rally[]
}

function filterRallies(rallies: Rally[], f: RallyFilter): Rally[] {
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
  roi: null,

  playing: false,
  currentTime: 0,
  seekingUntil: 0,
  userSeeking: false,
  selectedRallyId: null,
  selectedClipId: null,
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

  // ================================================================= 会话
  async bootstrap() {
    try {
      const env = await api.env()
      set({ env })
    } catch (e) {
      get().toast({ kind: 'error', title: '无法连接本地服务', detail: String(e) })
    }
    // 把已有的后台任务拉回来，否则刷新页面后进度就永远消失了
    api.listJobs().then((list) => {
      const jobs: Record<string, JobInfo> = {}
      for (const j of list) jobs[j.id] = j
      set({ jobs })
    }).catch(() => undefined)
    ws.connect()
    ws.subscribe((m) => {
      if (m.type === 'job') {
        set((s) => ({ jobs: { ...s.jobs, [m.job.id]: m.job } }))
        const j = m.job
        if (j.status === 'done' && j.kind === 'analyze') {
          const pid = get().project?.id
          const mid = get().mediaId
          if (pid && mid) {
            api.getAnalysis(pid, mid).then((res) => {
              set((s) => (s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {}))
            })
          }
        }
        if (j.status === 'done' && j.kind === 'prepare') {
          const pid = get().project?.id
          if (pid) get().openProject(pid)
        }
      } else if (m.type === 'media') {
        set((s) => {
          if (!s.project || s.project.id !== m.project_id) return {}
          const media = s.project.media.map((x) => (x.id === m.media.id ? m.media : x))
          return { project: { ...s.project, media } }
        })
      } else if (m.type === 'timeline') {
        // 重切分后后端会把时间线片段重新挂到新回合上，这里同步过来
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

  setView: (v) => set({ view: v }),

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
      return p.id
    } catch (e) {
      get().toast({ kind: 'error', title: '新建工程失败', detail: String(e) })
      return null
    }
  },

  async openProject(id) {
    try {
      const p = await api.getProject(id)
      const mid = p.media.length ? p.media[0].id : null
      set({
        project: p,
        mediaId: get().mediaId && p.media.some((m) => m.id === get().mediaId) ? get().mediaId : mid,
        view: 'studio',
        selectedRallyId: null,
        history: [],
        future: [],
        currentTime: 0,
      })
      if (location.hash !== `#/studio/${id}`) {
        history.replaceState(null, '', `#/studio/${id}`)
      }
    } catch (e) {
      get().toast({ kind: 'error', title: '打开工程失败', detail: String(e) })
    }
  },

  async renameProject(name) {
    const p = get().project
    if (!p) return
    const updated = await api.patchProject(p.id, { name })
    set({ project: updated })
    get().refreshProjects()
  },

  async deleteProject(id) {
    await api.deleteProject(id)
    if (get().project?.id === id) set({ project: null, mediaId: null, view: 'library' })
    get().refreshProjects()
  },

  closeProject() {
    set({ project: null, mediaId: null, selectedRallyId: null, view: 'library', history: [], future: [] })
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
        get().toast({ kind: 'success', title: `已导入 ${res.added.length} 个素材` })
        api.prepareMedia(p.id, res.added[0].id).catch(() => undefined)
      }
      if (res.failed.length) {
        get().toast({ kind: 'warn', title: `${res.failed.length} 个文件导入失败`, detail: res.failed[0].error })
      }
    } catch (e) {
      get().toast({ kind: 'error', title: '导入失败', detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, import: false } }))
    }
  },

  async importFiles(files) {
    const p = get().project
    if (!p || !files.length) return
    set((s) => ({ busy: { ...s.busy, import: true } }))
    try {
      for (const f of files) {
        await fetch(`/api/projects/${p.id}/media/upload`, {
          method: 'POST',
          headers: { 'x-filename': encodeURIComponent(f.name) },
          body: f,
        })
      }
      await get().openProject(p.id)
      get().toast({ kind: 'success', title: `已导入 ${files.length} 个文件` })
    } catch (e) {
      get().toast({ kind: 'error', title: '上传失败', detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, import: false } }))
    }
  },

  async removeMedia(mid) {
    const p = get().project
    if (!p) return
    const updated = await api.removeMedia(p.id, mid)
    set({ project: updated, mediaId: updated.media[0]?.id ?? null })
  },

  selectMedia(mid) {
    set({ mediaId: mid, selectedRallyId: null, currentTime: 0, previewMode: 'source' })
  },

  // ================================================================= 分析
  setParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),
  setWeights: (w) => set({ weights: w }),
  setFilter: (patch) => set((s) => ({ filter: { ...s.filter, ...patch } })),
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
      get().toast({ kind: 'info', title: 'AI 分析已开始', detail: '可以在任务面板查看进度' })
      set((s) => ({ busy: { ...s.busy, [job_id]: true }, view: 'studio' }))
    } catch (e) {
      get().toast({ kind: 'error', title: '启动分析失败', detail: String(e) })
    }
  },

  async rescore(w) {
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
      set((s) =>
        s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {},
      )
      // 片段的分数标签跟着一起刷新
      const proj = get().project
      if (proj) {
        const byId = new Map(res.rallies.map((r) => [r.id, r]))
        const tl = structuredClone(proj.timeline)
        let touched = false
        for (const t of tl.tracks) {
          for (const c of t.clips) {
            if (!c.rally_id) continue
            const r = byId.get(c.rally_id)
            if (r) {
              const label = `#${r.index} ${r.scores.total.toFixed(0)}分`
              if (c.label !== label) {
                c.label = label
                touched = true
              }
            }
          }
        }
        if (touched) api.setTimeline(p.id, tl).catch(() => undefined)
      }
    } catch (e) {
      get().toast({ kind: 'error', title: '重新评分失败', detail: String(e) })
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
    const params = { ...get().params, ...(patch ?? {}) }
    if (patch) set({ params })
    set((s) => ({ busy: { ...s.busy, resegment: true } }))
    try {
      const res = await api.resegment(p.id, {
        media_id: mid,
        params,
        weights: weights ?? get().weights,
      })
      set((s) =>
        s.project ? { project: { ...s.project, analyses: { ...s.project.analyses, [mid]: res } } } : {},
      )
    } catch (e) {
      get().toast({ kind: 'error', title: '重新切分失败', detail: String(e) })
    } finally {
      set((s) => ({ busy: { ...s.busy, resegment: false } }))
    }
  },

  async patchRally(rid, patch) {
    const p = get().project
    if (!p) return
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
        } catch {
          /* 本地已更新，静默 */
        }
      }, 420),
    )
  },

  async bulkRallies(patch, opts) {
    const p = get().project
    if (!p) return
    const mid = get().mediaId ?? undefined
    const body: any = { media_id: mid, patch }
    if (opts?.ids) {
      body.ids = opts.ids
    } else if (opts?.useFilter) {
      // 关键：批量操作只能作用在「当前屏幕上看得见的回合」上。
      // 只发筛选条件的话，服务端拿到的是不完整的条件（客户端还按置信度/
      // 仅保留/关键词过滤过），会把用户手动排除掉的回合又复活。
      body.ids = get()
        .visibleRallies()
        .map((r) => r.id)
    }
    if (!body.ids && !opts?.useFilter) {
      get().toast({ kind: 'warn', title: '没有可操作的回合' })
      return
    }
    try {
      const res = await api.bulkRallies(p.id, body)
      await get().openProject(p.id)
      get().toast({ kind: 'success', title: `已更新 ${res.updated} 个回合` })
    } catch (e) {
      get().toast({ kind: 'error', title: '批量操作失败', detail: String(e) })
    }
  },

  selectRally: (rid) => set({ selectedRallyId: rid, selectedClipId: null }),

  // ================================================================= 时间线
  async autoCut(opts) {
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    const ids = get()
      .visibleRallies()
      .map((r) => r.id)
    if (!ids.length) {
      get().toast({ kind: 'warn', title: '当前筛选没有可用回合' })
      return
    }
    try {
      const res = await api.autoCut(p.id, {
        media_id: mid,
        rally_ids: ids,
        mode: opts?.mode ?? 'replace',
      })
      get().setTimeline(res.timeline)
      if (res.added === 0 && (res.skipped ?? 0) > 0) {
        get().toast({
          kind: 'info',
          title: '这些回合都已经在成片里了',
          detail: '每个回合只会排进成片一次，所以这次没有新增内容',
        })
      } else {
        get().toast({
          kind: 'success',
          title: (opts?.mode ?? 'replace') === 'append' ? '已追加到成片' : '已按筛选生成成片',
          detail:
            `本次加入 ${res.added ?? res.clip_count} 个回合 · 成片共 ${res.clip_count} 段 / ${res.timeline.duration.toFixed(1)} 秒` +
            (res.skipped ? `（跳过 ${res.skipped} 个已在成片里的）` : ''),
        })
      }
    } catch (e) {
      get().toast({ kind: 'error', title: '自动剪辑失败', detail: String(e) })
    }
  },

  async clearTimeline() {
    const p = get().project
    if (!p) return
    const empty: Timeline = {
      tracks: [{ id: 't_main', name: '视频轨 1', kind: 'video', muted: false, locked: false, clips: [] }],
      duration: 0,
      fps: 30,
      width: 1920,
      height: 1080,
    }
    get().setTimeline(empty)
  },

  setTimeline(tl, pushHistory = true) {
    set((s) => {
      if (!s.project) return {}
      const hist = pushHistory && s.project.timeline ? [...s.history.slice(-49), s.project.timeline] : s.history
      return { project: { ...s.project, timeline: tl }, history: hist, future: [] }
    })
    const p = get().project
    if (p) api.setTimeline(p.id, tl).catch(() => undefined)
  },

  addClipFromRally(rally) {
    const s = get()
    const mid = s.mediaId
    if (!mid || !s.project) return
    const tl = structuredClone(s.project.timeline)
    if (!tl.tracks.length) {
      tl.tracks = [{ id: 't_main', name: '视频轨 1', kind: 'video', muted: false, locked: false, clips: [] }]
    }
    const track = tl.tracks[0]
    // 一个回合只进成片一次：重复点 + / 双击色块不应该往成片里塞第二段。
    // 已经在成片里就选中那一段（时间线会滚过去），而不是再加一段。
    const cov = rallyFilmCoverage(track.clips, rally)
    if (cov.ratio >= FILM_COVERED_RATIO && cov.best) {
      set({ selectedClipId: cov.best.id, selectedRallyId: rally.id })
      get().toast({
        kind: 'info',
        title: `回合 #${rally.index} 已经在成片里了`,
        detail: `成片 ${timecode(cov.best.tl_start, false)} 处那一段就是它（每个回合只加入一次）`,
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
      label: `#${rally.index} ${rally.scores.total.toFixed(0)}分`,
      vertical_crop: false,
      protected: false,
    }
    track.clips.push(clip)
    tl.duration = track.clips.reduce((m, c) => Math.max(m, c.tl_start + (c.src_out - c.src_in) / c.speed), 0)
    get().setTimeline(tl)
    set({ selectedClipId: clip.id })
    get().toast({
      kind: 'success',
      title: `回合 #${rally.index} 已加入成片`,
      detail: `排在成片 ${timecode(start, false)} 处 · 成片共 ${track.clips.length} 段，可在下方时间线拖动、裁剪或删掉`,
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
    const s = get()
    if (!s.project) return
    const tl = structuredClone(s.project.timeline)
    const track = tl.tracks[0]
    const idx = track.clips.findIndex((c) => c.id === clipId)
    if (idx < 0) return
    const removed = track.clips[idx]
    const removedLen = (removed.src_out - removed.src_in) / removed.speed
    track.clips.splice(idx, 1)
    // 后面的片段整体左移，保持无缝
    for (const c of track.clips) {
      if (c.tl_start > removed.tl_start) c.tl_start = Math.max(0, c.tl_start - removedLen)
    }
    tl.duration = Math.max(0, ...track.clips.map((c) => c.tl_start + (c.src_out - c.src_in) / c.speed))
    get().setTimeline(tl)
    if (s.selectedClipId === clipId) set({ selectedClipId: null })
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
        title: '这里切不了',
        detail: '播放头不在这个片段范围内。请把播放头移到片段中间，或先切到「成片预览」模式。',
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
    get().toast({ kind: 'info', title: '已在播放头处分割' })
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
        title: '这里切不了',
        detail: `播放头（原片 ${sourceTime.toFixed(1)}s）不在选中片段的范围（${c.src_in.toFixed(1)}–${c.src_out.toFixed(1)}s）内`,
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
    get().toast({ kind: 'success', title: '已去空隙（顺序拼接）' })
  },

  undo() {
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
    if (p) api.setTimeline(p.id, p.timeline).catch(() => undefined)
  },

  redo() {
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
    if (p) api.setTimeline(p.id, p.timeline).catch(() => undefined)
  },

  selectClip: (id) => set({ selectedClipId: id, selectedRallyId: null }),

  // ================================================================= 播放
  setPlaying: (p) => set({ playing: p }),
  /**
   * 主动跳转（点时间轴、点回合、上一/下一回合）。
   * 会短暂屏蔽 video 的 timeupdate 回写：``<video>.currentTime`` 的赋值是异步的，
   * 屏蔽期内它会继续以旧时间触发 timeupdate，把播放头又拽回去，看起来就是「跳变」。
   */
  seek: (t) => set({ currentTime: Math.max(0, t), seekingUntil: Date.now() + 420 }),
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
    // 手动标定写进工程的 ui 里（按素材分开存），下次打开工程还在 ——
    // 所以这里不需要再存一份内存状态，`currentCourtPoly()` 读的就是它，
    // 这样「刷新页面 / 换素材」都不会拿到另一条素材的标定。
    // 一份素材标一次就够，所以刻意做成「跟着素材走」而不是「跟着分析走」。
    const p = get().project
    const mid = get().mediaId
    if (!p || !mid) return
    const ui = { ...(p.ui || {}) } as Record<string, any>
    // 新键 court_polys；同时读的时候兼容旧的 court_quads（只有四个角的年代）
    const polys = { ...((ui.court_polys || ui.court_quads || {}) as Record<string, unknown>) }
    if (q) polys[mid] = q
    else delete polys[mid]
    ui.court_polys = polys
    set({ project: { ...p, ui } })
    api.patchProject(p.id, { ui }).catch(() => {
      /* 保存失败不影响本次分析：这次仍然会用内存里的值 */
    })
  },

  /** 只看筛选片段：从下一个筛选回合开始连续播放，中间的捡球、走动自动跳过 */
  previewFilteredRallies() {
    const s = get()
    const list = s
      .visibleRallies()
      .slice()
      .sort((a, b) => a.start - b.start)
    if (!list.length) {
      get().toast({ kind: 'warn', title: '当前筛选没有可用回合', detail: '先把筛选条件放宽一点' })
      return
    }
    const t = s.currentTime
    const cur = list.find((r) => t >= r.start && t <= r.end)
    const target = cur ?? list.find((r) => r.start > t + 0.05) ?? list[0]
    set({ previewFiltered: true, previewMode: 'source' })
    get().seek(target.start)
    get().setPlaying(true)
    get().toast({
      kind: 'info',
      title: '只看筛选片段：已开启',
      detail: `连续播放这 ${list.length} 个回合，中间没被筛选出来的部分会自动跳过`,
    })
  },

  // ================================================================= 导出
  async exportVideo(presetSource, name) {
    const p = get().project
    if (!p) return null
    try {
      const res = await api.exportProject(p.id, { preset: presetSource, name })
      // 不跳页：跳走之后新任务还没完成，用户看到的是空列表，反而以为失败了
      get().toast({ kind: 'info', title: '已开始导出', detail: '进度在右上角「任务」里查看' })
      return res
    } catch (e) {
      get().toast({ kind: 'error', title: '导出失败', detail: String(e) })
      return null
    }
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
    return s.project.analyses[s.mediaId] ?? null
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
    const a = get().currentAnalysis()
    if (!a) return []
    return filterRallies(a.rallies, get().filter)
  },
}))

export { filterRallies }
