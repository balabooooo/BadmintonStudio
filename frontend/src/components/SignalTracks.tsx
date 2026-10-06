/**
 * 信号轨道：标注页概览下方的多轨诊断画布。
 *
 * 纯投影 GET annotation/signals（降采样曲线，不重跑 AI）：第一轨画融合活动度与
 * 高/低阈值，其后每条非零融合分量各占一轨（球员 / 运动 / 击球声 / 羽毛球 / 画外），
 * 开启 use_shuttle 的新分析额外画一条「在飞」覆盖轨。点击轨道与概览一样 seek。
 */

import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { useT } from '../i18n/useT'
import { useStore } from '../store/useStore'
import { clamp } from '../lib/format'
import type { AnnotationSignals } from '../lib/types'

const COMP_KEYS = ['players', 'motion', 'audio_hits', 'shuttle', 'roi'] as const
const COMP_COLORS: Record<string, string> = {
  players: '#7dd3fc',
  motion: '#c4b5fd',
  audio_hits: '#fca5a5',
  shuttle: '#fde68a',
  roi: '#86efac',
}

const ROW_H = 28
const TOP = 4
const LABEL_W = 46

interface Props {
  projectId: string
  mediaId: string
  time: number
  duration: number
  focus: [number, number]
  onSeek: (t: number) => void
}

/** Normalize a curve to its own max so each track's *shape* stays readable. */
function norm(a: number[]): number[] {
  let m = 0
  for (const v of a) if (v > m) m = v
  if (m <= 1e-9) return a.map(() => 0)
  return a.map((v) => clamp(v / m, 0, 1))
}

export default function SignalTracks({ projectId, mediaId, time, duration, focus, onSeek }: Props) {
  const tr = useT()
  const lang = useStore((s) => s.lang)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [sig, setSig] = useState<AnnotationSignals | null>(null)
  const [failed, setFailed] = useState(false)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    let alive = true
    setSig(null)
    setFailed(false)
    api.getAnnotationSignals(projectId, mediaId)
      .then((d) => { if (alive) setSig(d) })
      .catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [projectId, mediaId])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const update = () => setWidth(el.clientWidth || 600)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const tracks: { key: string; label: string; color: string; data: number[] }[] = []
  if (sig && sig.activity.length) {
    tracks.push({ key: 'activity', label: tr('annotate.signalsActivity'), color: '#e2e8f0', data: norm(sig.activity) })
    for (const k of COMP_KEYS) {
      const data = sig.components?.[k]
      if (data && data.some((v) => v !== 0)) {
        tracks.push({ key: k, label: tr(`annotate.signalsComp.${k}`), color: COMP_COLORS[k], data: norm(data) })
      }
    }
    if (sig.shuttle_in_flight && sig.shuttle_in_flight.some((v) => v !== 0)) {
      tracks.push({
        key: 'in_flight', label: tr('annotate.signalsInFlight'), color: '#fbbf24',
        data: norm(sig.shuttle_in_flight),
      })
    }
  }
  const height = tracks.length * ROW_H + TOP * 2

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !sig || !tracks.length) return
    const dpr = window.devicePixelRatio || 1
    cv.width = width * dpr
    cv.height = height * dpr
    cv.style.width = `${width}px`
    cv.style.height = `${height}px`
    const ctx = cv.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    const dur = duration || sig.duration || 1
    const X = (t: number) => LABEL_W + (t / dur) * (width - LABEL_W)

    // Eval-window shading spans every track.
    const [f0, f1] = focus
    if (f1 > f0) {
      ctx.fillStyle = 'rgba(56,189,248,.08)'
      ctx.fillRect(X(f0), 0, X(f1) - X(f0), height)
    }

    tracks.forEach((tk, i) => {
      const y0 = TOP + i * ROW_H
      // Row separators + label.
      ctx.strokeStyle = 'rgba(255,255,255,.06)'
      ctx.beginPath()
      ctx.moveTo(0, y0 + ROW_H + 0.5)
      ctx.lineTo(width, y0 + ROW_H + 0.5)
      ctx.stroke()
      ctx.font = '10px ui-sans-serif, system-ui'
      ctx.fillStyle = '#8b96ad'
      ctx.textBaseline = 'middle'
      ctx.fillText(tk.label, 3, y0 + ROW_H / 2)

      ctx.beginPath()
      const n = tk.data.length
      for (let j = 0; j < n; j++) {
        const x = LABEL_W + (j / Math.max(1, n - 1)) * (width - LABEL_W)
        const y = y0 + ROW_H - 2 - tk.data[j] * (ROW_H - 6)
        if (j === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.strokeStyle = tk.color
      ctx.globalAlpha = 0.9
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.globalAlpha = 1
    })

    // Activity thresholds live on the first row in the activity's raw 0~1 scale.
    if (sig && tracks[0]?.key === 'activity') {
      const raw = sig.activity
      let m = 0
      for (const v of raw) if (v > m) m = v
      if (m > 0) {
        for (const [thr, style] of [[sig.threshold_hi, 'rgba(244,63,94,.7)'], [sig.threshold_lo, 'rgba(251,146,60,.7)']] as const) {
          const y = TOP + ROW_H - 2 - clamp(thr / m, 0, 1) * (ROW_H - 6)
          ctx.strokeStyle = style
          ctx.setLineDash([3, 3])
          ctx.beginPath()
          ctx.moveTo(LABEL_W, y)
          ctx.lineTo(width, y)
          ctx.stroke()
          ctx.setLineDash([])
        }
      }
    }

    // Playhead across all tracks.
    ctx.fillStyle = '#f43f5e'
    ctx.fillRect(X(time) - 0.5, 0, 1.5, height)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, width, height, time, duration, focus, lang, tracks.length])

  if (failed) return null
  return (
    <div ref={wrapRef} className="px-3 pb-1">
      {!sig ? null : tracks.length === 0 ? (
        <div className="rounded-lg border border-white/8 px-2 py-1.5 text-[10.5px] text-ink-500">
          {tr('annotate.signalsEmpty')}
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          className="block cursor-crosshair rounded-lg border border-white/8"
          onPointerDown={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const t = clamp(
              ((e.clientX - rect.left - LABEL_W) / Math.max(1, rect.width - LABEL_W)) * (duration || sig?.duration || 1),
              0, duration || sig?.duration || 1)
            onSeek(t)
          }}
        />
      )}
    </div>
  )
}
