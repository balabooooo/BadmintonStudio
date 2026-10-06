/**
 * Pure geometry/selection helpers for the annotation video overlay (PoseOverlay).
 * Kept out of the component file so React fast-refresh stays component-only and the
 * math is unit-testable without rendering.
 */
import type { OverlayBox, OverlayDet, OverlayFrame } from './types'

/** 取数窗口（秒）：播放头前 2s、后 6s，保证正常播放时数据总在前面等着 */
export const WINDOW_LEAD = 2
export const WINDOW_TAIL = 6
/**
 * 播放头冲出舒适区后两次取数的最小间隔（节流而非防抖：防抖会被 250ms 的播放头轮询
 * 不断重置，导致取数永远不发生、叠加层在窗口耗尽后永久空白——已修复的真实 bug）。
 */
export const REFETCH_MIN_INTERVAL_MS = 500
/** 后端单窗上限（见 api/annotations.py OVERLAY_MAX_WINDOW） */
export const MAX_WINDOW = WINDOW_LEAD + WINDOW_TAIL
/** 采样 12fps 下允许的帧—播放头最大时差 */
const MAX_FRAME_GAP = 0.12

/** COCO 17 关键点骨架连线 */
export const COCO_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
]

/** object-fit: contain 的内容区（元素坐标系 px），letterbox 时画外部分不绘制 */
export function contentRect(
  vw: number,
  vh: number,
  ew: number,
  eh: number,
): { x: number; y: number; w: number; h: number } {
  if (!vw || !vh || !ew || !eh) return { x: 0, y: 0, w: ew, h: eh }
  const scale = Math.min(ew / vw, eh / vh)
  const w = vw * scale
  const h = vh * scale
  return { x: (ew - w) / 2, y: (eh - h) / 2, w, h }
}

/** 严格帧距（秒）：采样帧率下的「一帧」容差 */
export function strictGap(fps = 12): number {
  return fps > 0 ? Math.min(MAX_FRAME_GAP, 1.5 / fps) : MAX_FRAME_GAP
}

/** 选距离 t 最近的采样帧；超过一帧间隔认为该时刻无数据 */
export function selectFrame(frames: OverlayFrame[], t: number, fps = 12): OverlayFrame | null {
  if (!frames.length) return null
  const gap = strictGap(fps)
  let best: OverlayFrame | null = null
  let bestDt = Infinity
  for (const f of frames) {
    const dt = Math.abs(f.t - t)
    if (dt < bestDt) {
      bestDt = dt
      best = f
    }
  }
  return bestDt <= gap ? best : null
}

/** selectFrame 的「丢失容忍」版：容差内返回最近帧及其帧龄（用于半透明 LOST 绘制） */
export function selectFrameTolerant(
  frames: OverlayFrame[],
  t: number,
  fps = 12,
  tolerance = 0,
): { frame: OverlayFrame; age: number } | null {
  if (!frames.length) return null
  const limit = Math.max(strictGap(fps), tolerance)
  let best: OverlayFrame | null = null
  let bestDt = Infinity
  for (const f of frames) {
    const dt = Math.abs(f.t - t)
    if (dt < bestDt) {
      bestDt = dt
      best = f
    }
  }
  if (best == null || bestDt > limit) return null
  return { frame: best, age: bestDt }
}

/** 高于检测阈值的跟踪框（v1 缓存无 conf 时不过滤，保持旧行为） */
export function visibleBoxes(frame: OverlayFrame, detThr: number): OverlayBox[] {
  return frame.boxes.filter((b) => b.conf == null || b.conf >= detThr)
}

/** 高于检测阈值的原始检测框（诊断层） */
export function visibleDets(frame: OverlayFrame, detThr: number): OverlayDet[] {
  return (frame.dets ?? []).filter((d) => d.conf >= detThr)
}
