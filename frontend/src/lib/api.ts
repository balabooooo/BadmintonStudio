import type {
  AnalysisParams,
  AnalysisResult,
  EnvInfo,
  ExportPreset,
  JobInfo,
  MediaInfo,
  PlayerProbe,
  Project,
  ProjectSummary,
  Rally,
  Timeline,
} from './types'

const BASE = ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
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
  return (await res.json()) as T
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
  removeMedia: (pid: string, mid: string) =>
    req<Project>(`/api/projects/${pid}/media/${mid}`, { method: 'DELETE' }),
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
  }),
  resegment: (
    pid: string,
    body: { media_id: string; params?: Partial<AnalysisParams>; weights?: string },
  ) => req<AnalysisResult>(`/api/projects/${pid}/resegment`, { method: 'POST', body: JSON.stringify(body) }),
  clearAnalysis: (pid: string, mid: string) =>
    req<{ ok: boolean }>(`/api/projects/${pid}/analysis/${mid}`, { method: 'DELETE' }),

  // ---------------------------------------------------------------- 回合
  patchRally: (pid: string, rid: string, patch: Partial<Rally>) =>
    req<Rally>(`/api/projects/${pid}/rallies/${rid}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  bulkRallies: (
    pid: string,
    body: { media_id?: string; ids?: string[]; filter?: Record<string, unknown>; patch: Record<string, unknown> },
  ) => req<{ updated: number }>(`/api/projects/${pid}/rallies/bulk`, { method: 'POST', body: JSON.stringify(body) }),
  rescore: (pid: string, weights: string, media_id?: string) =>
    req<{ rescored: number; weights: string }>(`/api/projects/${pid}/rallies/rescore`, {
      method: 'POST',
      body: JSON.stringify({ weights, media_id }),
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
  exportProject: (pid: string, body: { preset: Partial<ExportPreset>; name?: string }) =>
    req<{ job_id: string; output: string }>(`/api/projects/${pid}/export`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listExports: () =>
    req<{ name: string; path: string; size: number; mtime: number }[]>('/api/exports'),
  exportUrl: (name: string) => `/api/exports/${encodeURIComponent(name)}`,

  // ---------------------------------------------------------------- 任务
  listJobs: () => req<JobInfo[]>('/api/jobs'),
  getJob: (jid: string) => req<JobInfo>(`/api/jobs/${jid}`),
  cancelJob: (jid: string) => req<{ ok: boolean }>(`/api/jobs/${jid}/cancel`, { method: 'POST' }),

  cacheStats: () =>
    req<{ cache: number; proxies: number; thumbs: number; exports: number }>('/api/cache/stats'),
  clearCache: (target: string) =>
    req<{ freed: number }>('/api/cache/clear', { method: 'POST', body: JSON.stringify({ target }) }),
}
