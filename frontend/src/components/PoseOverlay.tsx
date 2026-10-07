/**
 * 标注页视频叠加层：在视频画面上绘制 AI 球员框（含归一化尺寸读数）与 COCO 17 点骨架。
 *
 * 数据来自后端窗口化接口 annotation/overlay（只读 data/cache 下的紧凑 npz，不跑 AI）：
 * - 播放时按 500ms 节流拉取「播放头前 2s ~ 后 6s」窗口；播放头完全冲出当前窗口时立即重取
 *   （历史上这里误用了防抖：350ms 防抖被 250ms 轮询不断重置，取数永不发生，叠加层在
 *   首窗耗尽后永久空白——已改为节流修复）；
 * - 切换素材 / 关闭图层时 AbortController 取消在途请求并清空画布（沿用素材卡片的防抖约定）；
 * - 框或骨架缓存缺失时降级为提示条，绝不阻塞标注页其它功能（后端静默降级的前端镜像）。
 *
 * 阈值与诊断（v2 可视化缓存）：
 * - 检测阈值：只显示置信度高于该值的跟踪框（v1 缓存无 conf，不过滤保持旧行为）；
 * - 全部检测：灰色虚线画出尺寸过滤前的原始检测框，用于区分「检测丢」与「跟踪/选人丢」；
 * - 丢失容忍：数据缺口内仍以半透明 + LOST 角标显示最近一帧，避免短暂丢失时画面突兀清空。
 *
 * 坐标全部是相对代理画面的归一化 0~1；canvas 仅覆盖 video 元素 object-fit: contain 的内容区。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Eye, EyeOff, Box, Bone, Ruler, ScanSearch } from 'lucide-react'
import { api } from '../lib/api'
import { createLogger } from '../lib/logger'
import type { OverlayResponse } from '../lib/types'
import { useT } from '../i18n/useT'
import { cn } from '../lib/format'
import {
  COCO_EDGES,
  MAX_WINDOW,
  REFETCH_MIN_INTERVAL_MS,
  WINDOW_LEAD,
  contentRect,
  selectFrameTolerant,
  strictGap,
  visibleBoxes,
  visibleDets,
} from '../lib/overlay'

const overlayLog = createLogger('overlay')

const PLAYHEAD_POLL_MS = 250
/**
 * Minimum gap between two window fetches once the playhead is already OUTSIDE the loaded
 * window. The old code fetched immediately in that case, so dragging the progress bar fired a
 * (heavy) request per 250 ms poll and aborted the previous one every time. 150 ms still feels
 * instant on click-seek but collapses scrub drags into a handful of requests.
 */
const OUT_WINDOW_MIN_INTERVAL_MS = 150
const DEFAULT_CONF = 0.35
const DEFAULT_DET_THR = 0.15
const DET_THR_MIN = 0.1
const DET_THR_MAX = 0.8
const LOST_TOL_MAX = 1.0
/** 连续取数失败几次后提示「加载失败」（轮询本身会一直重试，即自动恢复机制） */
const FETCH_FAIL_HINT_AFTER = 2

/** 按 track 轮换取色（最多 4 名在场球员），与标注页绿/琥珀色板协调 */
const TRACK_COLORS = ['#22c55e', '#38bdf8', '#facc15', '#fb923c', '#a78bfa', '#f472b6']

interface PoseOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  pid: string
  mid: string
}

function clampNum(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v))
}

export default function PoseOverlay({ videoRef, pid, mid }: PoseOverlayProps) {
  const tr = useT()
  const [enabled, setEnabled] = useState(false)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showSize, setShowSize] = useState(true)
  const [showDets, setShowDets] = useState(false)
  const [confThr, setConfThr] = useState(DEFAULT_CONF)
  const [detThr, setDetThr] = useState(DEFAULT_DET_THR)
  const [lostTol, setLostTol] = useState(0)
  const [data, setData] = useState<OverlayResponse | null>(null)
  const [unavailable, setUnavailable] = useState(false)
  const [fetchFailed, setFetchFailed] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const dataRef = useRef<OverlayResponse | null>(null)
  const lastFetchAtRef = useRef(0)
  const failCountRef = useRef(0)
  /** True while the browser is resolving a seek (scrub drag fires 'seeking' repeatedly). */
  const seekingRef = useRef(false)
  /** Safety net: if a 'seeked' event is ever missed, release the seek lock after 1s. */
  const seekStallTimerRef = useRef<number | null>(null)
  const confRef = useRef(confThr)
  const layersRef = useRef({ showBoxes, showSkeleton, showSize, showDets, detThr, lostTol })
  dataRef.current = data
  confRef.current = confThr
  layersRef.current = { showBoxes, showSkeleton, showSize, showDets, detThr, lostTol }

  const fetchWindow = useCallback(
    async (center: number) => {
      const t0 = Math.max(0, center - WINDOW_LEAD)
      const t1 = t0 + MAX_WINDOW
      const now = Date.now()
      lastFetchAtRef.current = now
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      overlayLog.debug('overlay fetch window', { mid, center: Number(center.toFixed(2)), t0: Number(t0.toFixed(2)), t1: Number(t1.toFixed(2)) })
      try {
        const res = await api.getAnnotationOverlay(pid, mid, t0, t1, ctrl.signal)
        if (ctrl.signal.aborted) return
        failCountRef.current = 0
        setFetchFailed(false)
        dataRef.current = res
        setData(res)
        setUnavailable(!res.boxes_available && !res.skeletons_available)
        overlayLog.debug('overlay window applied', { mid, frames: res.frames.length })
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          overlayLog.debug('overlay fetch aborted (superseded)', { mid })
          return
        }
        // 4xx/网络错误：安静降级（图层还在，只是没数据），轮询会持续重试；
        // 连续失败才提示用户，成功一次即清除。
        failCountRef.current += 1
        overlayLog.warn('overlay fetch failed', { mid, fail: failCountRef.current, err: String(err) })
        if (failCountRef.current >= FETCH_FAIL_HINT_AFTER) setFetchFailed(true)
      }
    },
    [pid, mid],
  )
  const fetchWindowRef = useRef(fetchWindow)
  fetchWindowRef.current = fetchWindow

  /* 开关 / 切素材：立即取一次；关闭时取消在途请求并清空 */
  useEffect(() => {
    if (!enabled) return
    const v = videoRef.current
    void fetchWindow(v?.currentTime ?? 0)
    return () => {
      abortRef.current?.abort()
    }
    // videoRef 是稳定引用；播放头变化不需要重新订阅
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pid, mid, fetchWindow])

  /* Seek coalescing: while the user drags the progress bar the video fires 'seeking' many
     times — ignore the 250 ms poll during that burst and fetch exactly once on 'seeked', so a
     whole drag costs one request instead of one per poll tick. */
  useEffect(() => {
    if (!enabled) return
    const v = videoRef.current
    if (!v) return
    const clearStall = () => {
      if (seekStallTimerRef.current !== null) {
        window.clearTimeout(seekStallTimerRef.current)
        seekStallTimerRef.current = null
      }
    }
    const onSeeking = () => {
      if (!seekingRef.current) {
        overlayLog.debug('overlay seek start', { mid, t: Number(v.currentTime.toFixed(2)) })
        // Auto-release if this seek never gets a matching 'seeked' (browser quirks / src swap).
        clearStall()
        seekStallTimerRef.current = window.setTimeout(() => {
          seekingRef.current = false
          seekStallTimerRef.current = null
          overlayLog.debug('overlay seek stall timer released', { mid })
        }, 1000)
      }
      seekingRef.current = true
    }
    const onSeeked = () => {
      clearStall()
      seekingRef.current = false
      overlayLog.debug('overlay seeked -> fetch', { mid, t: Number(v.currentTime.toFixed(2)) })
      void fetchWindowRef.current(v.currentTime)
    }
    v.addEventListener('seeking', onSeeking)
    v.addEventListener('seeked', onSeeked)
    return () => {
      clearStall()
      v.removeEventListener('seeking', onSeeking)
      v.removeEventListener('seeked', onSeeked)
    }
  }, [enabled, mid, videoRef])

  /* 切素材或关闭图层：清掉已绘制数据，避免新窗口返回前画出旧素材的框 */
  useEffect(() => {
    if (!enabled && !mid) return
    seekingRef.current = false
    dataRef.current = null
    setData(null)
    setUnavailable(false)
    setFetchFailed(false)
    failCountRef.current = 0
  }, [enabled, mid])

  /* 播放头轮询：拖动（seeking）期间不取；接近边缘按 500ms 节流；已冲出窗口按 150ms 节流 */
  useEffect(() => {
    if (!enabled) return
    const timer = window.setInterval(() => {
      const v = videoRef.current
      const win = dataRef.current
      if (!v) return
      // A seek burst is in progress: the 'seeked' handler owns the next fetch.
      if (seekingRef.current) return
      const t = v.currentTime
      // Near the clip start t0 is clamped to 0 and can't be extended backward, so the lower-edge
      // trigger only fires when there actually is earlier video to cover.
      const nearLow = win ? t < win.t0 + 0.5 && win.t0 > 0.05 : true
      const nearHigh = win ? t > win.t1 - 1.0 : true
      if (win && !nearLow && !nearHigh) return
      const outOfWindow = win ? t < win.t0 || t > win.t1 : false
      const minInterval = outOfWindow ? OUT_WINDOW_MIN_INTERVAL_MS : REFETCH_MIN_INTERVAL_MS
      // 节流（不是防抖！）：防抖定时器会被本 250ms 轮询不断重置而永不触发。
      if (Date.now() - lastFetchAtRef.current < minInterval) return
      void fetchWindowRef.current(t)
    }, PLAYHEAD_POLL_MS)
    return () => window.clearInterval(timer)
  }, [enabled, videoRef])

  /* 绘制循环：rAF 跟随播放头选最近采样帧（丢失容忍内画半透明 LOST 帧）；jsdom / 无 2d 上下文时不启动 */
  useEffect(() => {
    if (!enabled) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext?.('2d')
    if (!canvas || !ctx) return
    let raf = 0

    const draw = () => {
      raf = requestAnimationFrame(draw)
      const v = videoRef.current
      if (!v || !v.videoWidth) return
      const ew = v.clientWidth
      const eh = v.clientHeight
      if (canvas.width !== Math.round(ew) || canvas.height !== Math.round(eh)) {
        canvas.width = Math.round(ew)
        canvas.height = Math.round(eh)
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const res = dataRef.current
      if (!res) return
      const fps = res.fps || 12
      const { showBoxes: b, showSkeleton: s, showSize: z, showDets: d, detThr: dt, lostTol: lt } =
        layersRef.current
      const sel = selectFrameTolerant(res.frames, v.currentTime, fps, lt)
      const lostTag = tr('annotate.overlayLostTag')
      if (!sel) {
        // 状态监控：窗口内有数据但当前时刻没有任何帧（容差内）——canvas 直接画提示，避免每帧 setState
        if (res.frames.length > 0) {
          const text = tr('annotate.overlayNoData')
          ctx.font = '11px ui-monospace, monospace'
          const tw = ctx.measureText(text).width
          ctx.fillStyle = 'rgba(0,0,0,0.55)'
          ctx.fillRect((canvas.width - tw - 10) / 2, 4, tw + 10, 16)
          ctx.fillStyle = '#fbbf24'
          ctx.textAlign = 'center'
          ctx.fillText(text, canvas.width / 2, 16)
          ctx.textAlign = 'left'
        }
        return
      }
      const frame = sel.frame
      // 帧龄超过严格一帧间隔：处于丢失容忍区，半透明绘制 + LOST 角标
      const ghost = sel.age > strictGap(fps)
      ctx.globalAlpha = ghost ? 0.35 : 1
      const r = contentRect(v.videoWidth, v.videoHeight, ew, eh)
      const colorOf = (track: number) =>
        TRACK_COLORS[((track % TRACK_COLORS.length) + TRACK_COLORS.length) % TRACK_COLORS.length]

      if (d) {
        // 原始检测诊断层：灰色虚线（无 track id），按同一检测阈值过滤
        ctx.setLineDash([4, 3])
        for (const det of visibleDets(frame, dt)) {
          const [x1, y1, x2, y2] = det.xyxy
          const px = r.x + x1 * r.w
          const py = r.y + y1 * r.h
          ctx.strokeStyle = 'rgba(203,213,225,0.85)'
          ctx.lineWidth = 1.5
          ctx.strokeRect(px, py, (x2 - x1) * r.w, (y2 - y1) * r.h)
          ctx.font = '10px ui-monospace, monospace'
          const text = det.conf.toFixed(2)
          const tw = ctx.measureText(text).width
          ctx.fillStyle = 'rgba(0,0,0,0.55)'
          ctx.fillRect(px, py + 2, tw + 6, 13)
          ctx.fillStyle = 'rgba(203,213,225,0.9)'
          ctx.fillText(text, px + 3, py + 12)
        }
        ctx.setLineDash([])
      }

      if (b) {
        for (const box of visibleBoxes(frame, dt)) {
          // Interpolated boxes (v3 cache, short occlusion gaps) are real timeline data but not
          // actual detections — draw them dimmed so the user can tell guessed boxes from hits.
          ctx.globalAlpha = ghost ? 0.35 : box.interp ? 0.55 : 1
          const [x1, y1, x2, y2] = box.xyxy
          const px = r.x + x1 * r.w
          const py = r.y + y1 * r.h
          const pw = (x2 - x1) * r.w
          const ph = (y2 - y1) * r.h
          const color = colorOf(box.track)
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.strokeRect(px, py, pw, ph)
          ctx.font = '11px ui-monospace, monospace'
          let label = tr('annotate.overlayTrack', { id: box.track })
          if (ghost) label += ` · ${lostTag}`
          if (z) {
            const wNorm = ((x2 - x1) * 100).toFixed(1)
            const hNorm = ((y2 - y1) * 100).toFixed(1)
            const text = `${label} ${wNorm}×${hNorm}%`
            const tw = ctx.measureText(text).width
            ctx.fillStyle = 'rgba(0,0,0,0.55)'
            ctx.fillRect(px, py - 15, tw + 8, 15)
            ctx.fillStyle = ghost ? '#fbbf24' : color
            ctx.fillText(text, px + 4, py - 4)
          } else {
            ctx.fillStyle = 'rgba(0,0,0,0.55)'
            const tw = ctx.measureText(label).width
            ctx.fillRect(px, py - 15, tw + 8, 15)
            ctx.fillStyle = ghost ? '#fbbf24' : color
            ctx.fillText(label, px + 4, py - 4)
          }
        }
      }

      if (s) {
        ctx.globalAlpha = ghost ? 0.35 : 1
        for (const sk of frame.skeletons) {
          const color = colorOf(sk.track)
          const pts = sk.kp.map(([x, y, c]) =>
            c >= confRef.current ? { x: r.x + x * r.w, y: r.y + y * r.h } : null,
          )
          ctx.lineWidth = 2
          for (const [a, q] of COCO_EDGES) {
            const pa = pts[a]
            const pq = pts[q]
            if (!pa || !pq) continue
            ctx.strokeStyle = color
            ctx.beginPath()
            ctx.moveTo(pa.x, pa.y)
            ctx.lineTo(pq.x, pq.y)
            ctx.stroke()
          }
          for (const p of pts) {
            if (!p) continue
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }
      ctx.globalAlpha = 1
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [enabled, videoRef, tr])

  return (
    <>
      <canvas
        ref={canvasRef}
        className={cn(
          'pointer-events-none absolute inset-0 h-full w-full',
          enabled ? 'block' : 'hidden',
        )}
      />
      <div className="absolute right-2 bottom-2 flex flex-col items-end gap-1.5">
        {enabled && unavailable && (
          <div className="max-w-[240px] rounded-md bg-black/65 px-2 py-1 text-[10.5px] leading-snug text-amber-glow">
            {tr('annotate.overlayUnavailable')}
          </div>
        )}
        {enabled && fetchFailed && !unavailable && (
          <div className="max-w-[240px] rounded-md bg-black/65 px-2 py-1 text-[10.5px] leading-snug text-amber-glow">
            {tr('annotate.overlayFetchFail')}
          </div>
        )}
        {enabled && (
          <div className="pointer-events-auto flex max-w-[320px] flex-wrap items-center justify-end gap-1 rounded-lg bg-black/55 p-1 backdrop-blur-sm">
            <button
              type="button"
              onClick={() => setShowBoxes((v) => !v)}
              className={cn(
                'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]',
                showBoxes ? 'bg-court-500/30 text-court-200' : 'text-ink-500 hover:text-ink-300',
              )}
              title={tr('annotate.overlayBoxes')}
            >
              <Box size={12} /> {tr('annotate.overlayBoxes')}
            </button>
            <button
              type="button"
              onClick={() => setShowSkeleton((v) => !v)}
              className={cn(
                'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]',
                showSkeleton ? 'bg-court-500/30 text-court-200' : 'text-ink-500 hover:text-ink-300',
              )}
              title={tr('annotate.overlaySkeleton')}
            >
              <Bone size={12} /> {tr('annotate.overlaySkeleton')}
            </button>
            <button
              type="button"
              onClick={() => setShowSize((v) => !v)}
              className={cn(
                'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]',
                showSize ? 'bg-court-500/30 text-court-200' : 'text-ink-500 hover:text-ink-300',
              )}
              title={tr('annotate.overlaySize')}
            >
              <Ruler size={12} /> {tr('annotate.overlaySize')}
            </button>
            <button
              type="button"
              onClick={() => setShowDets((v) => !v)}
              className={cn(
                'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px]',
                showDets ? 'bg-court-500/30 text-court-200' : 'text-ink-500 hover:text-ink-300',
              )}
              title={tr('annotate.overlayShowAllHint')}
            >
              <ScanSearch size={12} /> {tr('annotate.overlayShowAll')}
            </button>
            <label className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-400">
              {tr('annotate.overlayConf')}
              <input
                type="range"
                min={0.1}
                max={0.8}
                step={0.05}
                value={confThr}
                onChange={(e) => setConfThr(Number(e.target.value))}
                className="h-1 w-14 accent-court-400"
              />
              <input
                type="number"
                min={0.1}
                max={0.8}
                step={0.01}
                value={confThr}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v) && e.target.value !== '') setConfThr(clampNum(v, 0.1, 0.8))
                }}
                className="mono w-11 rounded border border-white/10 bg-black/40 px-1 text-right text-[10px] text-ink-300"
              />
            </label>
            <label className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-400">
              {tr('annotate.overlayDetThr')}
              <input
                type="range"
                min={DET_THR_MIN}
                max={DET_THR_MAX}
                step={0.05}
                value={detThr}
                onChange={(e) => setDetThr(Number(e.target.value))}
                className="h-1 w-14 accent-court-400"
              />
              <input
                type="number"
                min={DET_THR_MIN}
                max={DET_THR_MAX}
                step={0.01}
                value={detThr}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v) && e.target.value !== '')
                    setDetThr(clampNum(v, DET_THR_MIN, DET_THR_MAX))
                }}
                className="mono w-11 rounded border border-white/10 bg-black/40 px-1 text-right text-[10px] text-ink-300"
              />
            </label>
            <label className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-ink-400">
              {tr('annotate.overlayLostTol')}
              <input
                type="range"
                min={0}
                max={LOST_TOL_MAX}
                step={0.05}
                value={lostTol}
                onChange={(e) => setLostTol(Number(e.target.value))}
                className="h-1 w-14 accent-court-400"
              />
              <input
                type="number"
                min={0}
                max={LOST_TOL_MAX}
                step={0.05}
                value={lostTol}
                onChange={(e) => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v) && e.target.value !== '')
                    setLostTol(clampNum(v, 0, LOST_TOL_MAX))
                }}
                className="mono w-11 rounded border border-white/10 bg-black/40 px-1 text-right text-[10px] text-ink-300"
              />
            </label>
          </div>
        )}
        <button
          type="button"
          onClick={() => setEnabled((v) => !v)}
          className={cn(
            'pointer-events-auto flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11.5px] shadow',
            enabled ? 'bg-court-500/80 text-white hover:bg-court-400' : 'bg-black/55 text-ink-300 hover:text-white',
          )}
          title={tr('annotate.overlayHint')}
        >
          {enabled ? <Eye size={13} /> : <EyeOff size={13} />}
          {tr('annotate.overlay')}
        </button>
      </div>
    </>
  )
}
