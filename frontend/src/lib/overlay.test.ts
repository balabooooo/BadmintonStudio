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
