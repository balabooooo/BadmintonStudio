import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useStore, consumeMediaPick } from './useStore'

// Mock the api module so store actions never hit the network.
vi.mock('../lib/api', () => ({
  api: {
    prepareMedia: vi.fn().mockResolvedValue(undefined),
    rescore: vi.fn().mockResolvedValue(undefined),
    removeMedia: vi.fn().mockResolvedValue({ project: { media: [], analyses: {} }, removed: 1 }),
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
