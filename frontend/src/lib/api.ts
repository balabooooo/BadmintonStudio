import type {
  AnalysisParams,
  AnalysisResult,
  AnnotationResponse,
  AnnotationSignals,
  EnvInfo,
  OverlayResponse,
  ExportItem,
  ExportPreset,
  JobInfo,
  MediaInfo,
  OptimizeResult,
  PlayerProbe,
  QualityReport,
  Project,
  ProjectSummary,
  Rally,
  ScenePreset,
  Timeline,
} from './types'
import { getLang } from '../i18n'

const BASE = ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/**
 * `timeoutMs` 默认 60s：请求挂死时不该让按钮永远转圈。传 0 表示不限时
 * （标注页的「优化参数」是同步长任务，可能跑好几分钟）。
 */
async function req<T>(path: string, init?: RequestInit, timeoutMs = 60_000): Promise<T> {
  const ctrl = new AbortController()
  let timer: number | undefined
  if (timeoutMs > 0) timer = window.setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(BASE + path, {
      ...init,
      signal: init?.signal ?? ctrl.signal,
      headers: { 'Content-Type': 'application/json', 'X-BMS-Lang': getLang(), ...(init?.headers || {}) },
    })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const j = await res.json()
        detail = j.detail || JSON.stringify(j)
      } catch {
        /* ignore */
      }
      throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
    }
    if (res.status === 204) return undefined as T
    try {
      return (await res.json()) as T
    } catch {
      // 2xx 但响应体不是 JSON（例如被代理拦截）时抛结构化错误，别让 SyntaxError 冒出去
      throw new ApiError(res.status, 'Invalid JSON response')
    }
  } finally {
    if (timer !== undefined) window.clearTimeout(timer)
  }
}

export const api = {
  health: () => req<{ ok: boolean; version: string }>('/api/health'),
  env: () => req<EnvInfo>('/api/env'),

  // ---------------------------------------------------------------- 工程
  listProjects: () => req<ProjectSummary[]>('/api/projects'),
  createProject: (name: string) =>
    req<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ name }) }),
  getProject: (pid: string) => req<Project>(`/api/projects/${pid}`),
  patchProject: (pid: string, patch: Record<string, unknown>) =>
    req<Project>(`/api/projects/${pid}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  deleteProject: (pid: string) => req<{ ok: boolean }>(`/api/projects/${pid}`, { method: 'DELETE' }),
  duplicateProject: (pid: string) =>
    req<Project>(`/api/projects/${pid}/duplicate`, { method: 'POST', body: '{}' }),

  // ---------------------------------------------------------------- 素材
  addMedia: (pid: string, paths: string[]) =>
    req<{ project: Project; added: MediaInfo[]; failed: { path: string; error: string }[] }>(
      `/api/projects/${pid}/media`,
      { method: 'POST', body: JSON.stringify({ paths }) },
    ),
  /** 调起 Windows 原生文件选择框（可多选），拿到本地路径后走 addMedia，避免复制大文件。 */
  pickVideos: () =>
    req<{ paths: string[]; cancelled: boolean }>('/api/dialog/videos', {
      method: 'POST',
      body: '{}',
    }),
  /** 调起 Windows 原生文件夹选择框，后端会递归展开其中的视频文件。 */
  pickFolder: () =>
    req<{ paths: string[]; cancelled: boolean }>('/api/dialog/folder', {
      method: 'POST',
      body: '{}',
    }),
  removeMedia: (pid: string, mid: string) =>
    req<Project>(`/api/projects/${pid}/media/${mid}`, { method: 'DELETE' }),
  /** 批量从工程移除素材（同时取消还在跑的预览任务） */
  deleteMediaBulk: (pid: string, ids: string[]) =>
    req<{ project: Project; removed: number }>(`/api/projects/${pid}/media/bulk-delete`, {
      method: 'POST',
      body: JSON.stringify({ ids }),
    }),
  prepareMedia: (pid: string, mid: string) =>
    req<{ job_id: string }>(`/api/projects/${pid}/media/${mid}/prepare`, {
      method: 'POST',
      body: '{}',
    }),
  sprite: (pid: string, mid: string) =>
    req<{
      path: string
      count: number
      cols: number
      rows: number
      tile_w: number
      tile_h: number
      interval: number
    }>(`/api/projects/${pid}/media/${mid}/sprite`),
  proxyUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/proxy`,
  sourceUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/stream`,
  posterUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/poster`,
  audioUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/audio`,
  assetUrl: (p: string, cache = 3600) => `/api/asset?p=${encodeURIComponent(p)}&cache=${cache}`,

  // ---------------------------------------------------------------- 分析
  analyze: (
    pid: string,
    body: { media_id: string; params?: Partial<AnalysisParams>; weights?: string; roi?: number[] },
  ) => req<{ job_id: string }>(`/api/projects/${pid}/analyze`, { method: 'POST', body: JSON.stringify(body) }),
  /**
   * 批量分析：一次提交多条素材，后端串行排队逐个跑。
   * 不传 `media_ids` 表示工程内全部素材；场地标定由后端按素材各自读取。
   */
  analyzeBatch: (
    pid: string,
    body: { media_ids?: string[]; params?: Partial<AnalysisParams>; weights?: string },
  ) => req<{ job_ids: string[]; count: number }>(`/api/projects/${pid}/analyze-batch`, {
    method: 'POST',
    body: JSON.stringify(body),
  }),
  getAnalysis: (pid: string, mid: string) => req<AnalysisResult>(`/api/projects/${pid}/analysis/${mid}`),
  /**
   * 人物框尺寸试测：取若干帧**只做检测**，秒级返回框高分布。
   * 用来在调「人物框尺寸筛选」时立刻看到「球员的框多大、其他人多大」。
   *
   * - 不给 `times` / `at_time` 时，在整条视频上均匀抽 `count` 帧；
   * - 给了就只取那些时刻，并把**画面本身**一起返回（`frames[].image`），
   *   界面可以把框画在画面上核对。
   */
  playerProbe: (
    pid: string,
    body: {
      media_id: string
      count?: number
      params?: Partial<AnalysisParams>
      court_poly?: number[][]
      /** 指定时刻（秒，最多 40 个） */
      times?: number[]
      /** 只抓这一个时刻 */
      at_time?: number
      /** 是否把帧图存下来（默认 true） */
      save_frames?: boolean
    },
  ) => req<PlayerProbe>(`/api/projects/${pid}/player-probe`, {
    method: 'POST',
    body: JSON.stringify(body),
  }, 300_000),
  resegment: (
    pid: string,
    body: { media_id: string; params?: Partial<AnalysisParams>; weights?: string },
  ) => req<AnalysisResult>(`/api/projects/${pid}/resegment`, { method: 'POST', body: JSON.stringify(body) }),
  rebuildHits: (
    pid: string,
    body: { media_id: string; params?: Partial<AnalysisParams>; weights?: string },
  ) => req<AnalysisResult>(`/api/projects/${pid}/rebuild-hits`, { method: 'POST', body: JSON.stringify(body) }),
  clearAnalysis: (pid: string, mid: string) =>
    req<{ ok: boolean }>(`/api/projects/${pid}/analysis/${mid}`, { method: 'DELETE' }),

  // ---------------------------------------------------------------- 标注 / 参数优化
  getAnnotation: (pid: string, mid: string) =>
    req<AnnotationResponse>(`/api/projects/${pid}/media/${mid}/annotation`),
  saveAnnotation: (
    pid: string,
    mid: string,
    body: {
      rallies: { start: number; end: number; note?: string; source?: string }[]
      hits?: { t: number; ours: boolean }[]
      focus?: number[] | null
      note?: string
    },
  ) =>
    req<{ ok: boolean; count: number; path: string; updated_at: string }>(
      `/api/projects/${pid}/media/${mid}/annotation`,
      { method: 'PUT', body: JSON.stringify(body) },
    ),
  optimizeSegmentation: (
    pid: string,
    mid: string,
    body: { params?: Partial<AnalysisParams> } = {},
  ) =>
    req<OptimizeResult>(
      `/api/projects/${pid}/media/${mid}/annotation/optimize`,
      { method: 'POST', body: JSON.stringify(body) },
      0,
    ),
  annotationCsvUrl: (pid: string, mid: string) =>
    `/api/projects/${pid}/media/${mid}/annotation/export.csv`,
  /** 降采样多轨信号（标注页信号面板） */
  getAnnotationSignals: (pid: string, mid: string) =>
    req<AnnotationSignals>(`/api/projects/${pid}/media/${mid}/annotation/signals`),
  /** 窗口化球员框 + 关键点骨架叠加数据（可传 AbortSignal 做防抖竞态取消） */
  getAnnotationOverlay: (
    pid: string,
    mid: string,
    t0: number,
    t1: number,
    signal?: AbortSignal,
  ) =>
    req<OverlayResponse>(
      `/api/projects/${pid}/media/${mid}/annotation/overlay?t0=${t0.toFixed(2)}&t1=${t1.toFixed(2)}`,
      signal ? { signal } : undefined,
    ),
  /** 标注质量审计（结构/边界证据告警 + 吸附建议，只建议不自动改写） */
  getAnnotationQuality: (pid: string, mid: string, signal?: AbortSignal) =>
    req<QualityReport>(
      `/api/projects/${pid}/media/${mid}/annotation/quality`,
      signal ? { signal } : undefined,
    ),

  // ---------------------------------------------------------------- 回合
  patchRally: (pid: string, rid: string, patch: Partial<Rally>) =>
    req<Rally>(`/api/projects/${pid}/rallies/${rid}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  bulkRallies: (
    pid: string,
    body: { media_id?: string; ids?: string[]; filter?: Record<string, unknown>; patch: Record<string, unknown> },
  ) => req<{ updated: number }>(`/api/projects/${pid}/rallies/bulk`, { method: 'POST', body: JSON.stringify(body) }),
  rescore: (
    pid: string,
    weights: string,
    media_id?: string,
    cross_media?: boolean,
    speech_bonus_points?: number,
  ) =>
    req<{ rescored: number; weights: string; cross_media?: boolean }>(`/api/projects/${pid}/rallies/rescore`, {
      method: 'POST',
      body: JSON.stringify({ weights, media_id, cross_media, speech_bonus_points }),
    }),

  // ---------------------------------------------------------------- 时间线
  autoCut: (
    pid: string,
    body: {
      media_id?: string
      filter?: Record<string, unknown>
      rally_ids?: string[]
      mode?: 'replace' | 'append'
      gap?: number
      pre?: number
      post?: number
    },
  ) => req<{ timeline: Timeline; clip_count: number; added: number; skipped: number }>(
    `/api/projects/${pid}/timeline/auto-cut`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  ),
  setTimeline: (pid: string, timeline: Timeline) =>
    req<Timeline>(`/api/projects/${pid}/timeline`, { method: 'POST', body: JSON.stringify({ timeline }) }),

  // ---------------------------------------------------------------- 导出
  exportPresets: () => req<ExportPreset[]>('/api/export/presets'),
  exportProject: (
    pid: string,
    body: {
      preset: Partial<ExportPreset>
      name?: string
      mode?: 'merge' | 'separate'
      output_dir?: string
    },
  ) =>
    req<{ job_id: string; output: string; mode: 'merge' | 'separate' }>(
      `/api/projects/${pid}/export`,
      { method: 'POST', body: JSON.stringify(body) },
    ),
  listExports: () => req<ExportItem[]>('/api/exports'),
  exportUrl: (id: string) => `/api/exports/by-id/${encodeURIComponent(id)}`,
  revealExport: (id: string) =>
    req<{ ok: boolean }>('/api/exports/reveal', { method: 'POST', body: JSON.stringify({ id }) }),
  /** 原生文件夹选择框：返回空数组表示取消（非 Windows / 不可用时 501）。 */
  pickExportDir: () =>
    req<{ paths: string[]; cancelled: boolean }>('/api/dialog/export-dir', {
      method: 'POST',
      body: '{}',
    }),

  // ---------------------------------------------------------------- 场景预设
  listPresets: () => req<ScenePreset[]>('/api/presets'),
  createPreset: (body: {
    project_id: string
    media_id: string
    name: string
    note?: string
    frame_time: number
    params: Partial<AnalysisParams>
    court_poly?: [number, number][] | null
    fit?: Record<string, number | string>
  }) => req<ScenePreset>('/api/presets', { method: 'POST', body: JSON.stringify(body) }),
  patchPreset: (id: string, body: { name?: string; note?: string }) =>
    req<ScenePreset>(`/api/presets/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deletePreset: (id: string) => req<{ ok: boolean }>(`/api/presets/${id}`, { method: 'DELETE' }),

  // ---------------------------------------------------------------- 任务
  listJobs: () => req<JobInfo[]>('/api/jobs'),
  getJob: (jid: string) => req<JobInfo>(`/api/jobs/${jid}`),
  cancelJob: (jid: string) => req<{ ok: boolean }>(`/api/jobs/${jid}/cancel`, { method: 'POST' }),

  cacheStats: () =>
    req<{ cache: number; proxies: number; thumbs: number; exports: number }>('/api/cache/stats'),
  clearCache: (target: string) =>
    req<{ freed: number }>('/api/cache/clear', { method: 'POST', body: JSON.stringify({ target }) }),
}
