import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import PoseOverlay from './PoseOverlay'
import {
  COCO_EDGES,
  contentRect,
  selectFrame,
  WINDOW_LEAD,
  WINDOW_TAIL,
} from '../lib/overlay'
import { api } from '../lib/api'
import type { OverlayResponse } from '../lib/types'
import { useStore } from '../store/useStore'

vi.mock('../lib/api', () => ({
  api: {
    getAnnotationOverlay: vi.fn(),
  },
}))

const mockedOverlay = vi.mocked(api.getAnnotationOverlay)

/** Minimal stand-in for the annotated page's <video>: only the fields the overlay reads. */
function fakeVideo(currentTime = 0): HTMLVideoElement {
  const target = new EventTarget()
  return {
    currentTime,
    videoWidth: 960,
    videoHeight: 540,
    clientWidth: 960,
    clientHeight: 540,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  } as unknown as HTMLVideoElement
}

const okResponse: OverlayResponse = {
  t0: 0,
  t1: WINDOW_LEAD + WINDOW_TAIL,
  fps: 12,
  duration: 100,
  boxes_available: true,
  skeletons_available: true,
  frames: [
    { t: 0, boxes: [], skeletons: [] },
    {
      t: 1 / 12,
      boxes: [{ track: 1, xyxy: [0.1, 0.2, 0.3, 0.9] }],
      skeletons: [],
    },
  ],
}

function renderOverlay(video = fakeVideo()) {
  const ref = { current: video }
  return render(<PoseOverlay videoRef={ref} pid="p1" mid="m_abc" />)
}

describe('PoseOverlay geometry helpers', () => {
  it('contentRect fills the element when aspect ratios match', () => {
    expect(contentRect(960, 540, 960, 540)).toEqual({ x: 0, y: 0, w: 960, h: 540 })
  })

  it('contentRect letterboxes a wider element', () => {
    const r = contentRect(960, 540, 1000, 540)
    expect(r).toEqual({ x: 20, y: 0, w: 960, h: 540 })
  })

  it('selectFrame picks the nearest sample and gives up past one frame gap', () => {
    const frames = okResponse.frames
    expect(selectFrame(frames, 0.02, 12)?.t).toBe(0)
    expect(selectFrame(frames, 1 / 12, 12)?.t).toBeCloseTo(1 / 12, 6)
    expect(selectFrame(frames, 5, 12)).toBeNull()
  })

  it('exposes the 16 COCO skeleton edges', () => {
    expect(COCO_EDGES).toHaveLength(16)
  })
})

describe('PoseOverlay component', () => {
  afterEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    act(() => useStore.setState({ lang: 'zh' }))
  })

  it('starts disabled without fetching, then fetches a playhead window when enabled', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    mockedOverlay.mockResolvedValue(okResponse)
    renderOverlay()
    expect(mockedOverlay).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await waitFor(() =>
      expect(mockedOverlay).toHaveBeenCalledWith(
        'p1', 'm_abc', 0, WINDOW_LEAD + WINDOW_TAIL, expect.any(AbortSignal),
      ),
    )
    // The missing-cache hint must not appear for a complete response
    expect(screen.queryByText(/overlay data missing/i)).toBeNull()
  })

  it('surfaces the rebuild hint when neither cache is available', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    mockedOverlay.mockResolvedValue({ ...okResponse, boxes_available: false, skeletons_available: false })
    renderOverlay()
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    expect(await screen.findByText(/overlay data missing/i)).toBeInTheDocument()
  })

  it('aborts the in-flight request when the layer is switched off', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    let captured: AbortSignal | undefined
    mockedOverlay.mockImplementation((_p, _m, _t0, _t1, signal) => {
      captured = signal
      return Promise.resolve(okResponse)
    })
    renderOverlay()
    const btn = screen.getByRole('button', { name: /overlay/i })
    fireEvent.click(btn)
    await waitFor(() => expect(captured).toBeInstanceOf(AbortSignal))
    fireEvent.click(btn)
    expect(captured!.aborted).toBe(true)
  })

  it('refetches after switching to another media', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    mockedOverlay.mockResolvedValue(okResponse)
    const ref = { current: fakeVideo(10) }
    const { rerender } = render(<PoseOverlay videoRef={ref} pid="p1" mid="m_abc" />)
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await waitFor(() => expect(mockedOverlay).toHaveBeenCalledTimes(1))
    rerender(<PoseOverlay videoRef={ref} pid="p1" mid="m_xyz" />)
    await waitFor(() => expect(mockedOverlay).toHaveBeenCalledTimes(2))
    expect(mockedOverlay.mock.calls[1].slice(0, 2)).toEqual(['p1', 'm_xyz'])
    // Window is centered on the current playhead: 10 - 2 = 8
    expect(mockedOverlay.mock.calls[1][2]).toBeCloseTo(8, 6)
    expect(mockedOverlay.mock.calls[1][3]).toBeCloseTo(8 + WINDOW_LEAD + WINDOW_TAIL, 6)
  })

  /* --------------------------------------------------------------- 回归：防抖死锁
   * 旧实现在播放头轮询里每 250ms clearTimeout 重设 350ms 防抖：250 < 350，定时器被
   * 无限重置，取数永不发生——播放头冲出 8s 窗口后叠加层永久空白（真实 bug）。
   * 现实现是节流：边缘区间最多每 500ms 取一次，一定能触发。 */
  it('refetches while the playhead lingers in the edge zone (throttle actually fires)', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    vi.useFakeTimers()
    mockedOverlay.mockResolvedValue(okResponse) // window [0, 8]
    const video = fakeVideo(0)
    renderOverlay(video)
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)

    video.currentTime = 7.5 // 上沿边缘区（t1 - 1.0 < t <= t1）
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250) // 距上次取数仅 250ms：被节流
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300) // 累计 550ms >= 500ms：本轮取数发出
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(2)
    expect(mockedOverlay.mock.calls[1][2]).toBeCloseTo(7.5 - WINDOW_LEAD, 6)
  })

  it('refetches immediately when the playhead jumps outside the window (seek recovery)', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    vi.useFakeTimers()
    mockedOverlay.mockResolvedValue(okResponse) // window [0, 8]
    const video = fakeVideo(0)
    renderOverlay(video)
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)

    video.currentTime = 20 // 完全在窗口外：绕过节流立即重取
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(2)
    expect(mockedOverlay.mock.calls[1][2]).toBeCloseTo(20 - WINDOW_LEAD, 6)
  })

  /* --------------------------------------------------------------- 回归：拖动卡顿
   * 旧实现在进度条拖动时每个 250ms 轮询都 abort 上一个请求再发新请求（拖动一秒 4 次
   * 重 npz 解析 + 4 次 abort），UI 长时间卡死。现在 seeking 期间轮询挂起，seeked 时
   * 合并成一次取数。 */
  it('coalesces a scrub drag (repeated seeking) into one fetch on seeked', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    vi.useFakeTimers()
    mockedOverlay.mockResolvedValue(okResponse) // window [0, 8]
    const video = fakeVideo(0)
    renderOverlay(video)
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)

    // User drags the progress bar: 'seeking' fires repeatedly at new positions.
    for (const t of [5, 12, 24, 30]) {
      video.currentTime = t
      video.dispatchEvent(new window.Event('seeking'))
    }
    // Two poll ticks pass DURING the drag: they must not fetch (old code fired 2 requests here).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)

    // Drag ends -> exactly one fetch, centered on the FINAL playhead position.
    video.dispatchEvent(new window.Event('seeked'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(2)
    expect(mockedOverlay.mock.calls[1][2]).toBeCloseTo(30 - WINDOW_LEAD, 6)
  })

  it('releases the seek lock via the stall timer when seeked never arrives', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    vi.useFakeTimers()
    mockedOverlay.mockResolvedValue(okResponse) // window [0, 8]
    const video = fakeVideo(0)
    renderOverlay(video)
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    video.currentTime = 40
    video.dispatchEvent(new window.Event('seeking'))
    // Polls while seeking: suppressed.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(750)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(1)
    // After the 1s stall safety net, the next poll resumes fetching (150ms out-of-window throttle).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(350) // 750 -> 1100ms: stall fires at 1000ms, one poll catches it
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(2)
    expect(mockedOverlay.mock.calls[1][2]).toBeCloseTo(40 - WINDOW_LEAD, 6)
  })

  it('surfaces a retry hint after two consecutive fetch failures and clears it on success', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    vi.useFakeTimers()
    const err = new Error('network down')
    mockedOverlay
      .mockRejectedValueOnce(err)
      .mockRejectedValueOnce(err)
      .mockResolvedValue(okResponse)
    renderOverlay()
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    // 第一次失败：还不出提示（可能只是偶发）
    expect(screen.queryByText(/retrying/i)).toBeNull()
    // 无数据 -> 轮询按节流间隔持续重试；第二次失败出现提示
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(2)
    expect(screen.getByText(/retrying/i)).toBeInTheDocument()
    // 第三次成功：提示清除（自动恢复）
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(mockedOverlay).toHaveBeenCalledTimes(3)
    expect(screen.queryByText(/retrying/i)).toBeNull()
  })

  it('draws v3 interpolated boxes dimmed (alpha 0.55) and observed boxes opaque', async () => {
    act(() => useStore.setState({ lang: 'en' }))
    // jsdom has no canvas 2d backend; record globalAlpha at every strokeRect to verify dimming.
    const alphas: number[] = []
    const ctx2d = {
      globalAlpha: 1,
      clearRect: vi.fn(),
      strokeRect: vi.fn(() => {
        alphas.push(ctx2d.globalAlpha)
      }),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 10 })),
      setLineDash: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
    }
    const getCtxSpy = vi
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx2d as unknown as CanvasRenderingContext2D)
    let rafCb: FrameRequestCallback | null = null
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCb = cb
      return 1
    })
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})

    const res: OverlayResponse = {
      ...okResponse,
      skeletons_available: false,
      frames: [
        {
          t: 0,
          boxes: [
            { track: 1, xyxy: [0.1, 0.2, 0.3, 0.9], conf: 0.9 },
            { track: 2, xyxy: [0.5, 0.2, 0.7, 0.9], conf: 0.9, interp: true },
          ],
          skeletons: [],
        },
      ],
    }
    mockedOverlay.mockResolvedValue(res)
    renderOverlay()
    fireEvent.click(screen.getByRole('button', { name: /overlay/i }))
    await waitFor(() => expect(mockedOverlay).toHaveBeenCalledTimes(1))
    // Run one draw tick at the sampled position t=0.
    act(() => rafCb?.(0))
    expect(alphas).toContain(1)
    expect(alphas).toContain(0.55)
    getCtxSpy.mockRestore()
    rafSpy.mockRestore()
    cancelSpy.mockRestore()
  })
})
