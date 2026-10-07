import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useAnnotationDraft } from './annotationDraft'
import { api } from '../lib/api'
import { useStore } from './useStore'
import type { AnnotationResponse } from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    getAnnotation: vi.fn(),
    saveAnnotation: vi.fn(),
  },
}))

const mockedGet = vi.mocked(api.getAnnotation)
const mockedSave = vi.mocked(api.saveAnnotation)

function response(mid: string, rallyStarts: number[] = [], duration = 120): AnnotationResponse {
  return {
    media_id: mid,
    media_name: mid,
    duration,
    fps: 30,
    path: 'x',
    rallies: rallyStarts.map((s, i) => ({ start: s, end: s + 5, source: i ? 'manual' : 'auto' })),
    hits: [],
    focus: null,
    note: '',
    auto: [],
    envelope: [],
    envelope_fps: 0,
    hit_times: [],
    hit_times_raw: [],
  }
}

function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const p = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { p, resolve, reject }
}

beforeEach(() => {
  vi.clearAllMocks()
  act(() => useStore.setState({ lang: 'en' }))
})

afterEach(() => {
  act(() => useStore.setState({ lang: 'zh' }))
})

describe('useAnnotationDraft cross-view persistence', () => {
  it('hydrates synchronously from cache on remount with the dirty draft intact', async () => {
    mockedGet.mockResolvedValue(response('mA', [3, 20]))
    const { result, unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_hydrate', mid: 'mA' },
    })
    await waitFor(() => expect(result.current.loaded).toBe(true))
    expect(result.current.rallies).toHaveLength(2)

    act(() => {
      result.current.setRallies((prev) => [...prev, { start: 40, end: 46, id: result.current.nextId() }])
      result.current.markDirty()
    })
    expect(result.current.dirty).toBe(true)

    // Simulate navigating away (page unmount) and back: the component instance is destroyed,
    // but the draft store keeps the entry.
    unmount()
    const second = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_hydrate', mid: 'mA' },
    })
    // First render after remount must already show the unsaved edit — no blank flash.
    expect(second.result.current.rallies).toHaveLength(3)
    expect(second.result.current.rallies[2].start).toBe(40)
    expect(second.result.current.dirty).toBe(true)
    second.unmount()
  })

  it('flushes the pending autosave after unmount against the correct media id', async () => {
    vi.useFakeTimers()
    try {
      mockedGet.mockResolvedValue(response('mA2', [5]))
      mockedSave.mockResolvedValue({ ok: true, count: 1, path: 'x', updated_at: '' })
      const { result, unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
        initialProps: { pid: 'p_flush', mid: 'mA2' },
      })
      await vi.waitFor(() => expect(result.current.loaded).toBe(true))

      act(() => {
        result.current.setRallies([{ start: 9, end: 14, id: result.current.nextId() }])
        result.current.markDirty()
      })
      unmount()
      // Debounced PUT must still fire even though the page that owned the edit is gone.
      await vi.runAllTimersAsync()
      expect(mockedSave).toHaveBeenCalledTimes(1)
      expect(mockedSave).toHaveBeenCalledWith('p_flush', 'mA2', expect.objectContaining({
        rallies: [expect.objectContaining({ start: 9 })],
      }))
    } finally {
      vi.useRealTimers()
    }
  })

  it('drops a stale GET response when a newer request supersedes it', async () => {
    const d1 = deferred<AnnotationResponse>()
    const d2 = deferred<AnnotationResponse>()
    mockedGet.mockReturnValueOnce(d1.p).mockReturnValueOnce(d2.p)

    const { result, unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_stale', mid: 'mA3' },
    })

    // Forced refresh issued before the initial GET landed; do NOT await it (it awaits the
    // pending d2 network promise), just let it bump the request sequence.
    act(() => {
      void result.current.reload()
    })
    expect(mockedGet).toHaveBeenCalledTimes(2)

    // The older response resolves first: it must NOT land.
    d1.resolve(response('mA3', [100]))
    await act(async () => {
      await Promise.resolve()
    })
    expect(result.current.rallies.find((r) => r.start === 100)).toBeUndefined()

    d2.resolve(response('mA3', [200]))
    await waitFor(() => expect(result.current.rallies).toHaveLength(1))
    expect(result.current.rallies[0].start).toBe(200)
    unmount()
  })

  it('does not overwrite a dirty draft with a background refresh on remount', async () => {
    mockedGet.mockResolvedValue(response('mA4', [1]))
    const { result, unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_dirty', mid: 'mA4' },
    })
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => {
      result.current.setRallies([{ start: 70, end: 78, id: result.current.nextId() }])
      result.current.markDirty()
    })
    const callsAfterLoad = mockedGet.mock.calls.length
    unmount()

    const second = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_dirty', mid: 'mA4' },
    })
    // No GET is issued for a dirty draft, and the edit survives untouched.
    expect(mockedGet.mock.calls.length).toBe(callsAfterLoad)
    expect(second.result.current.rallies).toHaveLength(1)
    expect(second.result.current.rallies[0].start).toBe(70)
    expect(second.result.current.dirty).toBe(true)
    second.unmount()
  })

  it('silently refreshes a clean draft from the server on remount', async () => {
    mockedGet.mockResolvedValueOnce(response('mA5', [1])).mockResolvedValueOnce(response('mA5', [1, 2, 3]))
    const { unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_clean', mid: 'mA5' },
    })
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(1))
    unmount()

    const second = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_clean', mid: 'mA5' },
    })
    // Cached content shows immediately...
    expect(second.result.current.rallies).toHaveLength(1)
    // ...and the background refresh lands newer server content.
    await waitFor(() => expect(second.result.current.rallies).toHaveLength(3))
    second.unmount()
  })

  it('keeps each media draft in its own slot when switching clips', async () => {
    mockedGet.mockImplementation((_pid, mid) => Promise.resolve(response(mid, mid === 'mA6' ? [1] : [8, 9])))
    const { result, rerender, unmount } = renderHook(({ pid, mid }) => useAnnotationDraft(pid, mid), {
      initialProps: { pid: 'p_multi', mid: 'mA6' },
    })
    await waitFor(() => expect(result.current.rallies).toHaveLength(1))

    rerender({ pid: 'p_multi', mid: 'mB6' })
    await waitFor(() => expect(result.current.rallies).toHaveLength(2))
    expect(result.current.rallies[0].start).toBe(8)

    // Switching back restores A's slot (not B's data).
    rerender({ pid: 'p_multi', mid: 'mA6' })
    await waitFor(() => expect(result.current.rallies).toHaveLength(1))
    expect(result.current.rallies[0].start).toBe(1)
    unmount()
  })
})
