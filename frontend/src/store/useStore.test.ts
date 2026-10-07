import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useStore, consumeMediaPick } from './useStore'
import { api } from '../lib/api'

// Mock the api module so store actions never hit the network.
vi.mock('../lib/api', () => ({
  api: {
    prepareMedia: vi.fn().mockResolvedValue(undefined),
    rescore: vi.fn().mockResolvedValue(undefined),
    removeMedia: vi.fn().mockResolvedValue({ project: { media: [], analyses: {} }, removed: 1 }),
    resegment: vi.fn().mockResolvedValue({}),
    rebuildHits: vi.fn().mockResolvedValue({}),
    autoCut: vi.fn().mockResolvedValue({}),
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
