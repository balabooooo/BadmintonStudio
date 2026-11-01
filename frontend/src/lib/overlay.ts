/**
 * Pure geometry/selection helpers for the annotation video overlay (PoseOverlay).
 * Kept out of the component file so React fast-refresh stays component-only and the
 * math is unit-testable without rendering.
 */
import type { OverlayBox, OverlayDet, OverlayFrame, OverlaySkeleton } from './types'

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

/**
 * Nearest-frame index in an ASCENDING-by-``t`` array (binary search).
 * Contract: the backend overlay emits frames sorted by frame index, so ``t`` is monotonic.
 * Returns the insertion position of ``t``; the nearest element is that index or the one before.
 */
function nearestIndex(frames: OverlayFrame[], t: number): number {
  let lo = 0
  let hi = frames.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].t < t) lo = mid + 1
    else hi = mid
  }
  // lo is the first index with frames[lo].t >= t; compare it with lo-1.
  if (lo <= 0) return 0
  if (lo >= frames.length) return frames.length - 1
  return t - frames[lo - 1].t <= frames[lo].t - t ? lo - 1 : lo
}

/** 选距离 t 最近的采样帧；超过一帧间隔认为该时刻无数据 */
export function selectFrame(frames: OverlayFrame[], t: number, fps = 12): OverlayFrame | null {
  if (!frames.length) return null
  const best = frames[nearestIndex(frames, t)]
  return Math.abs(best.t - t) <= strictGap(fps) ? best : null
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
  const best = frames[nearestIndex(frames, t)]
  const dt = Math.abs(best.t - t)
  if (dt > limit) return null
  return { frame: best, age: dt }
}

/**
 * Interpolated frame for drawing on top of a video whose frame rate (usually 30 fps) is
 * higher than the analysis timeline (12 fps).
 *
 * Why: ``selectFrameTolerant`` snaps to the nearest sample. When the user pauses and steps
 * with the arrow keys, the paused video frame sits between two 12 fps samples, so every box
 * is drawn at the wrong sample time and looks offset from the player. Linear interpolation
 * between the two neighbours (matched by track id) puts the boxes back on the presented frame.
 * Gaps wider than one sample interval are never bridged (that is real missing data).
 */
export function selectFrameAt(
  frames: OverlayFrame[],
  t: number,
  fps = 12,
  tolerance = 0,
): { frame: OverlayFrame; age: number } | null {
  if (!frames.length) return null
  const limit = Math.max(strictGap(fps), tolerance)
  const idx = nearestIndex(frames, t)
  const best = frames[idx]
  const dt = Math.abs(best.t - t)
  if (dt > limit) return null

  // Find the immediate pair around t. Interpolation is only valid when the pair is one
  // regular sample interval apart; over a data gap, fall back to the nearest frame.
  let lo = 0
  let hi = frames.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].t < t) lo = mid + 1
    else hi = mid
  }
  const next = frames[Math.min(lo, frames.length - 1)]
  const prev = frames[Math.max(0, lo - 1)]
  if (prev.t >= t || next.t <= t || next.t - prev.t > strictGap(fps)) {
    return { frame: best, age: dt }
  }
  const w = (t - prev.t) / (next.t - prev.t)
  return { frame: interpolateFrame(prev, next, w, t), age: dt }
}

function interpolateFrame(a: OverlayFrame, b: OverlayFrame, w: number, t: number): OverlayFrame {
  const lerp = (x: number, y: number) => x + (y - x) * w
  const boxes = new Map<number, OverlayBox>()
  for (const box of a.boxes) boxes.set(box.track, box)
  for (const box of b.boxes) {
    const prev = boxes.get(box.track)
    if (prev) {
      boxes.set(box.track, {
        ...box,
        xyxy: [
          lerp(prev.xyxy[0], box.xyxy[0]),
          lerp(prev.xyxy[1], box.xyxy[1]),
          lerp(prev.xyxy[2], box.xyxy[2]),
          lerp(prev.xyxy[3], box.xyxy[3]),
        ],
        conf: prev.conf != null && box.conf != null ? lerp(prev.conf, box.conf) : box.conf,
        interp: prev.interp || box.interp,
      })
    } else {
      boxes.set(box.track, box)
    }
  }
  // Tracks that vanished in b keep their a position; that is safer than dropping a box.
  const aBoxes = new Map<number, OverlayBox>()
  for (const box of a.boxes) aBoxes.set(box.track, box)
  for (const [track, box] of aBoxes) {
    if (!boxes.has(track)) boxes.set(track, box)
  }

  const skeletons = new Map<number, OverlaySkeleton>()
  for (const sk of a.skeletons) skeletons.set(sk.track, sk)
  for (const sk of b.skeletons) {
    const prev = skeletons.get(sk.track)
    if (prev && prev.kp.length === sk.kp.length) {
      const kp: [number, number, number][] = sk.kp.map((p, i) => {
        const q = prev.kp[i]
        return [lerp(q[0], p[0]), lerp(q[1], p[1]), lerp(q[2], p[2])]
      })
      skeletons.set(sk.track, { ...sk, kp })
    } else {
      skeletons.set(sk.track, sk)
    }
  }

  return {
    t,
    boxes: [...boxes.values()],
    skeletons: [...skeletons.values()],
    // Raw detections have no track id, so they cannot be matched across frames; using the
    // nearest frame's diagnostics is better than interpolating unrelated boxes.
    dets: (Math.abs(a.t - t) <= Math.abs(b.t - t) ? a : b).dets,
  }
}

/** 高于检测阈值的跟踪框（v1 缓存无 conf 时不过滤，保持旧行为） */
export function visibleBoxes(frame: OverlayFrame, detThr: number): OverlayBox[] {
  return frame.boxes.filter((b) => b.conf == null || b.conf >= detThr)
}

/** 高于检测阈值的原始检测框（诊断层） */
export function visibleDets(frame: OverlayFrame, detThr: number): OverlayDet[] {
  return (frame.dets ?? []).filter((d) => d.conf >= detThr)
}
