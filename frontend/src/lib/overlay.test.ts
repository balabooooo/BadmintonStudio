import { describe, expect, it } from 'vitest'
import {
  selectFrame,
  selectFrameTolerant,
  strictGap,
  visibleBoxes,
  visibleDets,
} from './overlay'
import type { OverlayFrame } from './types'

/** 12fps 采样：严格帧距 = min(0.12, 1.5/12) = 0.125 -> 0.12 */
const fps = 12

const frames: OverlayFrame[] = [
  { t: 0, boxes: [], skeletons: [] },
  { t: 1, boxes: [], skeletons: [] },
]

describe('selectFrameTolerant', () => {
  it('with zero tolerance behaves like strict selectFrame', () => {
    expect(selectFrameTolerant(frames, 0.02, fps, 0)).toEqual({ frame: frames[0], age: 0.02 })
    expect(selectFrameTolerant(frames, 5, fps, 0)).toBeNull()
    expect(selectFrame(frames, 5, fps)).toBeNull()
  })

  it('extends the acceptance window by the tolerance and reports the frame age', () => {
    // 最近帧 t=1，帧龄 ≈0.4s：严格 gap 0.12 之外、容差 0.5 之内 -> 丢失容忍（幽灵帧）
    const sel = selectFrameTolerant(frames, 1.4, fps, 0.5)
    expect(sel?.frame).toBe(frames[1])
    expect(sel!.age).toBeCloseTo(0.4, 5)
    expect(sel!.age).toBeGreaterThan(strictGap(fps))
    // 帧龄超过容差 -> 无数据
    expect(selectFrameTolerant(frames, 1.8, fps, 0.5)).toBeNull()
  })

  it('returns empty for empty frames regardless of tolerance', () => {
    expect(selectFrameTolerant([], 1, fps, 1)).toBeNull()
  })
})

describe('visibleBoxes / visibleDets (detection threshold filters)', () => {
  const frame: OverlayFrame = {
    t: 0,
    boxes: [
      { track: 1, xyxy: [0.1, 0.1, 0.2, 0.4], conf: 0.12 },
      { track: 2, xyxy: [0.3, 0.1, 0.4, 0.4], conf: 0.3 },
      { track: 3, xyxy: [0.5, 0.1, 0.6, 0.4] }, // v1 缓存无 conf
    ],
    skeletons: [],
    dets: [
      { xyxy: [0.1, 0.1, 0.2, 0.4], conf: 0.11 },
      { xyxy: [0.7, 0.1, 0.8, 0.4], conf: 0.4 },
    ],
  }

  it('keeps boxes above the threshold and always keeps v1 boxes without conf', () => {
    const kept = visibleBoxes(frame, 0.15)
    expect(kept.map((b) => b.track)).toEqual([2, 3])
    expect(visibleBoxes(frame, 0.1).map((b) => b.track)).toEqual([1, 2, 3])
  })

  it('filters raw detections by the same threshold', () => {
    const kept = visibleDets(frame, 0.15)
    expect(kept).toHaveLength(1)
    expect(kept[0].conf).toBe(0.4)
    expect(visibleDets(frame, 0.1)).toHaveLength(2)
  })

  it('handles frames without the dets field', () => {
    expect(visibleDets({ t: 0, boxes: [], skeletons: [] }, 0.1)).toEqual([])
  })
})

describe('selectFrame binary search parity with the old linear scan', () => {
  /** Reference implementation (pre-performance-fix linear scan, first-min tie kept). */
  function linearNearest(frames: OverlayFrame[], t: number): number {
    let best = 0
    let bestDt = Infinity
    frames.forEach((f, i) => {
      const dt = Math.abs(f.t - t)
      if (dt < bestDt) {
        bestDt = dt
        best = i
      }
    })
    return best
  }

  function makeFrames(ts: number[]): OverlayFrame[] {
    return ts.map((tv) => ({ t: tv, boxes: [], skeletons: [] }))
  }

  it('picks the same frame on dense ascending 12fps-like grids for boundaries and ties', () => {
    const grid = Array.from({ length: 200 }, (_, i) => Math.round(i / 12 * 1000) / 1000)
    const frames = makeFrames(grid)
    const probes = [0, 0.04, 0.5, 0.5 / 12, grid[grid.length - 1], grid[grid.length - 1] + 5]
    for (const q of probes) {
      const li = frames[linearNearest(frames, q)]
      if (Math.abs(li.t - q) <= strictGap(fps)) {
        expect(selectFrame(frames, q, fps)).toBe(li)
      } else {
        expect(selectFrame(frames, q, fps)).toBeNull()
      }
      const tol = selectFrameTolerant(frames, q, fps, 0.35)
      const linAge = Math.abs(li.t - q)
      if (linAge <= Math.max(strictGap(fps), 0.35)) {
        expect(tol?.frame).toBe(li)
        expect(tol?.age).toBeCloseTo(linAge, 8)
      } else {
        expect(tol).toBeNull()
      }
    }
  })

  it('matches linear scan on randomized ascending arrays (incl. wide gaps)', () => {
    let seed = 1234567
    const rand = () => {
      // Deterministic LCG so the property check is reproducible.
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }
    for (let trial = 0; trial < 40; trial++) {
      const ts: number[] = []
      let t = 0
      const count = 1 + Math.floor(rand() * 120)
      for (let i = 0; i < count; i++) {
        // Strictly ascending: backend frames are keyed by unique frame index; add gaps to
        // emulate missing frames as well.
        t += 0.001 + rand() * 0.4
        ts.push(Number(t.toFixed(3)))
      }
      const frames = makeFrames(ts)
      for (let k = 0; k < 10; k++) {
        const q = rand() * (t + 1) - 0.5
        const got = selectFrame(frames, q, fps)
        const want = frames[linearNearest(frames, q)]
        if (Math.abs(want.t - q) <= strictGap(fps)) expect(got).toBe(want)
        else expect(got).toBeNull()
      }
    }
  })
})
