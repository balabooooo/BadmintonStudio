import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useStore, consumeMediaPick, DEFAULT_PARAMS } from './useStore'
import { api } from '../lib/api'
import type { MediaInfo, Project } from '../lib/types'

// Mock the api module so store actions never hit the network.
vi.mock('../lib/api', () => ({
  api: {
    prepareMedia: vi.fn().mockResolvedValue(undefined),
    rescore: vi.fn().mockResolvedValue(undefined),
    removeMedia: vi.fn().mockResolvedValue({ project: { media: [], analyses: {} }, removed: 1 }),
    resegment: vi.fn().mockResolvedValue({}),
    rebuildHits: vi.fn().mockResolvedValue({}),
    autoCut: vi.fn().mockResolvedValue({}),
    analyze: vi.fn().mockResolvedValue({ job_id: 'job-1' }),
    analyzeBatch: vi.fn().mockResolvedValue({ job_ids: ['job-1'] }),
  },
}))

describe('media picker store actions', () => {
  beforeEach(() => {
    // Reset the singleton store to a known baseline between tests.
    useStore.setState({
      mediaPickerOpen: false,
      mediaPickerMode: 'switch',
      mediaId: null,
      project: null,
      selectedRallyId: null,
      selectedClipId: null,
      selectedClipIds: [],
      currentTime: 5,
      previewMode: 'timeline',
    })
  })

  it('openMediaPicker(switch) opens the dialog in switch mode', () => {
    useStore.getState().openMediaPicker('switch')
    const s = useStore.getState()
    expect(s.mediaPickerOpen).toBe(true)
    expect(s.mediaPickerMode).toBe('switch')
  })

  it('openMediaPicker defaults to switch mode', () => {
    useStore.getState().openMediaPicker()
    expect(useStore.getState().mediaPickerMode).toBe('switch')
  })

  it('openMediaPicker(select, cb) stores the pick callback for later consumption', () => {
    const cb = vi.fn()
    useStore.getState().openMediaPicker('select', cb)
    expect(useStore.getState().mediaPickerOpen).toBe(true)
    expect(useStore.getState().mediaPickerMode).toBe('select')

    const consumed = consumeMediaPick()
    expect(consumed).toBe(cb)
    // consumeMediaPick is one-shot: a second call returns null.
    expect(consumeMediaPick()).toBeNull()
  })

  it('closeMediaPicker hides the dialog without clearing the pending callback', () => {
    const cb = vi.fn()
    useStore.getState().openMediaPicker('select', cb)
    useStore.getState().closeMediaPicker()
    expect(useStore.getState().mediaPickerOpen).toBe(false)
    // ESC / backdrop must not lose the callback; only a confirm consumes it.
    expect(consumeMediaPick()).toBe(cb)
  })

  it('selectMedia updates mediaId and resets per-media selection state', () => {
    useStore.getState().selectMedia('m_abc')
    const s = useStore.getState()
    expect(s.mediaId).toBe('m_abc')
    expect(s.selectedRallyId).toBeNull()
    expect(s.selectedClipId).toBeNull()
    expect(s.selectedClipIds).toEqual([])
    expect(s.currentTime).toBe(0)
    expect(s.previewMode).toBe('source')
  })

  it('selectMedia with no project does not throw', () => {
    expect(() => useStore.getState().selectMedia('m_x')).not.toThrow()
  })

  it('consumeMediaPick returns null when no callback was registered', () => {
    expect(consumeMediaPick()).toBeNull()
  })
})

describe('optimize job locks segmentation actions', () => {
  const optJob = (status: 'queued' | 'running' | 'done', mediaId = 'm1') => ({
    id: `job_${status}`,
    kind: 'optimize' as const,
    title: 'optimize',
    media_id: mediaId,
    status,
    progress: status === 'done' ? 1 : 0.3,
    stage: 'optimize_segment',
  })

  beforeEach(() => {
    vi.clearAllMocks()
    useStore.setState({
      mediaId: 'm1',
      project: { id: 'p1' } as never,
      jobs: {},
      toasts: [],
      busy: {},
    })
  })

  it('mediaIsOptimizing reflects queued/running optimize jobs bound to the media', () => {
    expect(useStore.getState().mediaIsOptimizing('m1')).toBe(false)
    useStore.setState({ jobs: { a: optJob('running') } as never })
    expect(useStore.getState().mediaIsOptimizing('m1')).toBe(true)
    expect(useStore.getState().mediaIsOptimizing('m_other')).toBe(false)
    useStore.setState({ jobs: { a: optJob('queued') } as never })
    expect(useStore.getState().mediaIsOptimizing('m1')).toBe(true)
    useStore.setState({ jobs: { a: optJob('done') } as never })
    expect(useStore.getState().mediaIsOptimizing('m1')).toBe(false)
  })

  it.each([
    ['resegment', (s: ReturnType<typeof useStore.getState>) => s.resegment()],
    ['rebuildHits', (s: ReturnType<typeof useStore.getState>) => s.rebuildHits()],
    ['autoCut', (s: ReturnType<typeof useStore.getState>) => s.autoCut()],
  ])('%s is blocked with a warning toast while optimize runs', async (_name, action) => {
    useStore.setState({ jobs: { a: optJob('running') } as never })
    await action(useStore.getState())
    expect(api.resegment).not.toHaveBeenCalled()
    expect(api.rebuildHits).not.toHaveBeenCalled()
    expect(api.autoCut).not.toHaveBeenCalled()
    const toasts = useStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].kind).toBe('warn')
  })

  it('resegment proceeds once the optimize job is no longer active', async () => {
    useStore.setState({ jobs: { a: optJob('done') } as never })
    await useStore.getState().resegment()
    expect(api.resegment).toHaveBeenCalledTimes(1)
  })
})

/* ---------------------------------------------- manual court calibration gate */

const COURT_POLY: [number, number][] = [
  [0.1, 0.9],
  [0.9, 0.9],
  [0.7, 0.5],
  [0.3, 0.5],
]

const courtMedia = (id: string): MediaInfo => ({
  id,
  path: `/v/${id}.mp4`,
  name: id,
  size: 1024,
  duration: 60,
  fps: 30,
  width: 1920,
  height: 1080,
  rotation: 0,
  vcodec: 'h264',
  acodec: 'aac',
  has_audio: true,
  created_at: 0,
  proxy_path: null,
  proxy_fps: null,
  proxy_width: null,
  proxy_height: null,
  audio_path: null,
  poster: null,
})

function seedCourtProject(polys: Record<string, unknown>) {
  const project: Project = {
    id: 'p1',
    name: 'court',
    created_at: 0,
    updated_at: 0,
    media: [courtMedia('m1'), courtMedia('m2')],
    analyses: {},
    timeline: { tracks: [], duration: 60, fps: 30, width: 1920, height: 1080 },
    ui: { court_polys: polys },
    version: 1,
  }
  useStore.setState({ project, mediaId: 'm1', toasts: [], jobs: {}, busy: {}, lang: 'en' })
}

describe('manual court calibration gates analysis', () => {
  const mockedAnalyze = vi.mocked(api.analyze)
  const mockedAnalyzeBatch = vi.mocked(api.analyzeBatch)

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('uncalibratedMedia lists every media without a saved polygon', () => {
    seedCourtProject({ m1: COURT_POLY })
    expect(useStore.getState().uncalibratedMedia().map((m) => m.id)).toEqual(['m2'])
    expect(useStore.getState().uncalibratedMedia(['m1'])).toEqual([])
    expect(useStore.getState().uncalibratedMedia(['m2']).map((m) => m.id)).toEqual(['m2'])
  })

  it('a polygon with fewer than four points does not count as calibrated', () => {
    seedCourtProject({ m1: [[0.1, 0.9], [0.9, 0.9]] })
    expect(useStore.getState().currentCourtPoly()).toBeNull()
    expect(useStore.getState().uncalibratedMedia(['m1']).map((m) => m.id)).toEqual(['m1'])
  })

  it('runAnalysis refuses to start without a calibration and warns', async () => {
    seedCourtProject({})
    await useStore.getState().runAnalysis()
    expect(mockedAnalyze).not.toHaveBeenCalled()
    const toasts = useStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].kind).toBe('warn')
    expect(toasts[0].title).toBe('Court calibration required first')
  })

  it('runAnalysis starts once the current media is calibrated', async () => {
    seedCourtProject({ m1: COURT_POLY })
    await useStore.getState().runAnalysis()
    expect(mockedAnalyze).toHaveBeenCalledTimes(1)
  })

  it('runAnalysisBatch refuses the whole batch when any selected media is uncalibrated', async () => {
    seedCourtProject({ m1: COURT_POLY })
    await useStore.getState().runAnalysisBatch()
    expect(mockedAnalyzeBatch).not.toHaveBeenCalled()
    const toasts = useStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].kind).toBe('warn')
    expect(toasts[0].title).toBe('1 media still have no court calibration')
  })

  it('runAnalysisBatch starts when every selected media is calibrated', async () => {
    seedCourtProject({ m1: COURT_POLY, m2: COURT_POLY })
    await useStore.getState().runAnalysisBatch()
    expect(mockedAnalyzeBatch).toHaveBeenCalledTimes(1)
  })
})

describe('default analysis parameters', () => {
  it('keeps hit-sound detection off by default', () => {
    // Multi-court venues contaminate the audio hit sequence; visual scoring covers shots instead.
    expect(DEFAULT_PARAMS.use_audio).toBe(false)
    expect(useStore.getState().params.use_audio).toBe(false)
  })
})
