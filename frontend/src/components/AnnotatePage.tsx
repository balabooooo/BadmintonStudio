/**
 * 标注页：把「人工标注回合」和「用标注优化切分参数」合并进主界面。
 *
 * 与旧的独立标注工具（scripts/annotate）的区别：
 * 1. 直接跑在主服务里，不再需要单独起一个进程 / 另开一个网页；
 * 2. 与工程、素材、分析结果打通：自动草稿来自当前 AI 分析，优化出来的参数
 *    可以一键写回工程并重新切分；
 * 3. **修掉了时间轴跟随 bug**：旧工具的 seek 只改了视图中心、没有重画时间轴，
 *    于是按左右箭头时播放头动了、刻度与色块却留在原地。这里 `seek()` 同时
 *    更新 `time` 与 `viewCenter`，React 会重画整条时间轴，视图始终跟着播放头。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  Download,
  Pause,
  PenLine,
  Play,
  Sparkles,
  Trash2,
  Wand2,
  X,
} from 'lucide-react'
import { api } from '../lib/api'
import { cn, clamp } from '../lib/format'
import { Badge, Button, Empty } from './ui'
import { useStore } from '../store/useStore'
import type { AnnotationDraft, AnnotationRally, OptimizeResult, SegmentMetric } from '../lib/types'

interface AnnRally extends AnnotationRally {
  id: number
}

const fmt = (t: number) => {
  t = Math.max(0, t || 0)
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  const ms = Math.round((t - Math.floor(t)) * 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms)
    .padStart(3, '0')}`
}

const fmtShort = (t: number) => {
  t = Math.max(0, t || 0)
  const m = Math.floor(t / 60)
  const s = t % 60
  return `${m}:${s.toFixed(1).padStart(4, '0')}`
}

/** 描边进度条（优化结果里的 F1 对比） */
function MetricBar({ label, m, color }: { label: string; m: SegmentMetric; color: string }) {
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-ink-300">{label}</span>
        <span className="mono text-[11px]" style={{ color }}>
          F1 {m.f1.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/8">
        <div className="h-full rounded-full" style={{ width: `${clamp(m.f1, 0, 1) * 100}%`, background: color }} />
      </div>
      <div className="mono mt-1 flex gap-2 text-[10px] text-ink-500">
        <span>P {m.precision.toFixed(2)}</span>
        <span>R {m.recall.toFixed(2)}</span>
        <span>n {m.n}</span>
      </div>
    </div>
  )
}

const SEG_LABELS: Record<string, string> = {
  seg_prominence: '静默谷显著度',
  seg_min_core: '最短连续移动',
  seg_min_rest: '最短停顿',
  seg_min_quiet: '最短静默',
  min_rally_seconds: '最短回合',
}

export default function AnnotatePage() {
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const media = useStore((s) => s.currentMedia())
  const params = useStore((s) => s.params)
  const resegment = useStore((s) => s.resegment)
  const toast = useStore((s) => s.toast)

  const pid = project?.id ?? ''
  const mid = mediaId ?? ''

  const videoRef = useRef<HTMLVideoElement>(null)
  const overviewRef = useRef<HTMLCanvasElement>(null)
  const detailRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // ---------- 数据
  const [duration, setDuration] = useState(media?.duration ?? 0)
  const [fps, setFps] = useState(media?.fps ?? 0)
  const [auto, setAuto] = useState<AnnotationDraft[]>([])
  const [ann, setAnn] = useState<AnnRally[]>([])
  const [focus, setFocus] = useState<[number, number]>([0, 0])
  const [note, setNote] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [saveState, setSaveState] = useState('—')
  const [tab, setTab] = useState<'auto' | 'ann'>('auto')

  // ---------- 播放 / 视图
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [loop, setLoop] = useState(false)
  const [viewCenter, setViewCenter] = useState(0)
  const [viewSpan, setViewSpan] = useState(30)
  const [detailW, setDetailW] = useState(600)

  // ---------- 编辑
  const [sel, setSel] = useState<number | null>(null)
  const [pending, setPending] = useState<{ start: number } | null>(null)
  const undoRef = useRef<{ start: number; end: number; source?: string; note?: string }[][]>([])

  // ---------- 优化
  const [opt, setOpt] = useState<OptimizeResult | null>(null)
  const [optimizing, setOptimizing] = useState(false)

  const idSeq = useRef(1)
  const nextId = () => idSeq.current++
  const saveTimer = useRef<number | null>(null)
  const dragging = useRef<null | {
    id: number
    edge: 'start' | 'end' | 'move'
    x0: number
    s0: number
    e0: number
    span: number
    width: number
    moved: boolean
    pushed: boolean
  }>(null)

  // 键盘 / rAF 回调里读最新状态（避免闭包过期，也避免每帧重绑监听器）。
  // 用 effect 而不是在 render 里直接赋值：并发渲染下在 render 期间写 ref 不安全。
  const stateRef = useRef({ ann, auto, sel, pending, duration, viewSpan, viewCenter, focus, tab, loop })
  useEffect(() => {
    stateRef.current = { ann, auto, sel, pending, duration, viewSpan, viewCenter, focus, tab, loop }
  })

  /* ---------------------------------------------------------------- 加载 */
  const reload = useCallback(async () => {
    if (!pid || !mid) return
    try {
      const info = await api.getAnnotation(pid, mid)
      setDuration(info.duration || media?.duration || 0)
      setFps(info.fps || media?.fps || 0)
      setAuto(info.auto || [])
      setAnn((info.rallies || []).map((r) => ({ ...r, id: nextId() })))
      setNote(info.note || '')
      const f = info.focus && info.focus[1] ? (info.focus as [number, number]) : [0, info.duration || 0]
      setFocus(f as [number, number])
      setViewSpan(Math.max(10, Math.min(60, info.duration || 30)))
      setViewCenter(f[0])
      setSel(null)
      setLoaded(true)
      setDirty(false)
      setSaveState(info.rallies?.length ? `已载入 ${info.rallies.length} 回合` : '新建标注')
      undoRef.current = []
    } catch (e) {
      toast({ kind: 'error', title: '读取标注失败', detail: String(e) })
    }
  }, [pid, mid, media?.duration, media?.fps, toast])

  useEffect(() => {
    void reload()
  }, [reload])

  /* ---------------------------------------------------------------- 保存 */
  const doSave = useCallback(async () => {
    if (!pid || !mid) return
    try {
      const res = await api.saveAnnotation(pid, mid, {
        rallies: stateRef.current.ann.map(({ start, end, source, note: n }) => ({
          start, end, source, note: n,
        })),
        focus,
        note,
      })
      setDirty(false)
      setSaveState(`已保存 · ${res.count} 回合`)
    } catch (e) {
      setSaveState('保存失败')
      toast({ kind: 'error', title: '保存失败', detail: String(e) })
    }
  }, [pid, mid, focus, note, toast])

  const markDirty = useCallback(() => {
    setDirty(true)
    setSaveState('未保存…')
    if (saveTimer.current) window.clearTimeout(saveTimer.current)
    saveTimer.current = window.setTimeout(() => void doSave(), 1200)
  }, [doSave])

  const pushUndo = useCallback(() => {
    undoRef.current.push(stateRef.current.ann.map(({ id: _id, ...r }) => r))
    if (undoRef.current.length > 100) undoRef.current.shift()
  }, [])

  /* ---------------------------------------------------------------- 播放控制 */
  const seek = useCallback((t: number, center = true, pause = true) => {
    const v = videoRef.current
    const dur = stateRef.current.duration || v?.duration || 0
    const nt = clamp(t, 0, Math.max(0, dur - 0.001))
    if (v) v.currentTime = nt
    setTime(nt)
    if (center) setViewCenter(nt)
    if (pause) v?.pause()
  }, [])

  const step = useCallback((dt: number) => {
    const v = videoRef.current
    seek((v?.currentTime ?? 0) + dt)
  }, [seek])

  const togglePlay = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }, [])

  /* ---------------------------------------------------------------- 标注操作 */
  const addAnn = useCallback((r: { start: number; end: number; source?: string; note?: string }, select = true) => {
    const obj: AnnRally = { start: r.start, end: r.end, source: r.source || 'manual', note: r.note || '', id: nextId() }
    setAnn((prev) => [...prev, obj].sort((a, b) => a.start - b.start))
    if (select) setSel(obj.id)
    markDirty()
    return obj
  }, [markDirty])

  const newSeg = useCallback(() => {
    const v = videoRef.current
    setPending({ start: v?.currentTime ?? 0 })
    setSel(null)
    setViewCenter(v?.currentTime ?? 0)
  }, [])

  const finishSeg = useCallback(() => {
    const v = videoRef.current
    const t = v?.currentTime ?? 0
    setPending((p) => {
      if (!p) return null
      const st = Math.min(p.start, t)
      const en = Math.max(p.start, t)
      if (en - st >= 0.05) {
        pushUndo()
        addAnn({ start: st, end: en, source: 'manual' }, true)
      }
      return null
    })
  }, [addAnn, pushUndo])

  const delSel = useCallback(() => {
    setSel((cur) => {
      if (cur == null) return cur
      pushUndo()
      setAnn((prev) => prev.filter((r) => r.id !== cur))
      markDirty()
      return null
    })
  }, [markDirty, pushUndo])

  const undo = useCallback(() => {
    const snap = undoRef.current.pop()
    if (!snap) return
    setAnn(snap.map((r) => ({ ...r, id: nextId() })))
    setSel(null)
    markDirty()
  }, [markDirty])

  const clearAll = useCallback(() => {
    if (!stateRef.current.ann.length) return
    if (!window.confirm(`确定清空全部 ${stateRef.current.ann.length} 条标注？`)) return
    pushUndo()
    setAnn([])
    setSel(null)
    markDirty()
  }, [markDirty, pushUndo])

  const seedAll = useCallback(() => {
    if (!stateRef.current.auto.length) {
      toast({ kind: 'warn', title: '没有自动切分结果可导入' })
      return
    }
    if (stateRef.current.ann.length && !window.confirm('将把自动回合追加到现有标注，继续？')) return
    pushUndo()
    setAnn((prev) => [
      ...prev,
      ...stateRef.current.auto.map((a) => ({ start: a.start, end: a.end, source: 'auto', note: '', id: nextId() })),
    ].sort((a, b) => a.start - b.start))
    markDirty()
  }, [markDirty, pushUndo, toast])

  const currentAuto = useCallback((): AnnotationDraft | null => {
    const t = videoRef.current?.currentTime ?? 0
    const list = stateRef.current.auto
    return (
      list.find((r) => t >= r.start - 0.3 && t <= r.end + 0.3) ??
      list.find((r) => r.start > t) ??
      null
    )
  }, [])

  const acceptAuto = useCallback((advance = true) => {
    const a = currentAuto()
    if (!a) return
    const existing = stateRef.current.ann.find(
      (x) => Math.max(0, Math.min(x.end, a.end) - Math.max(x.start, a.start)) >= 0.5 * Math.min(a.end - a.start, x.end - x.start),
    )
    let obj: AnnRally
    if (existing) {
      obj = existing
      setSel(existing.id)
    } else {
      pushUndo()
      obj = addAnn({ start: a.start, end: a.end, source: 'auto' }, true)
    }
    if (advance) {
      const nxt = stateRef.current.auto.find((r) => r.start > obj.end - 0.2)
      seek(nxt ? nxt.start : obj.end, true)
    } else {
      setViewCenter(a.start)
    }
  }, [addAnn, currentAuto, pushUndo, seek])

  const jumpAuto = useCallback((dir: number) => {
    const t = videoRef.current?.currentTime ?? 0
    const list = stateRef.current.auto
    const tgt = dir > 0 ? list.find((r) => r.start > t + 0.01) : [...list].reverse().find((r) => r.start < t - 0.01)
    if (!tgt) return
    seek(tgt.start, true)
    if (dir > 0) void videoRef.current?.play()
  }, [seek])

  const selectNearest = useCallback(() => {
    const t = videoRef.current?.currentTime ?? 0
    let best: AnnRally | null = null
    let bd = Infinity
    for (const r of stateRef.current.ann) {
      const d = t >= r.start && t <= r.end ? 0 : Math.min(Math.abs(t - r.start), Math.abs(t - r.end))
      if (d < bd) {
        bd = d
        best = r
      }
    }
    if (best) {
      setSel(best.id)
      setViewCenter(best.start)
    }
  }, [])

  const exportCsv = useCallback(() => {
    const q = (s: unknown) => '"' + String(s ?? '').replace(/"/g, '""') + '"'
    const lines = ['index,start,end,duration,start_frame,end_frame,source,note']
    const rows = [...stateRef.current.ann].sort((a, b) => a.start - b.start)
    rows.forEach((r, i) => {
      const sf = fps ? Math.round(r.start * fps) : ''
      const ef = fps ? Math.round(r.end * fps) : ''
      lines.push([i + 1, r.start.toFixed(3), r.end.toFixed(3), (r.end - r.start).toFixed(3), sf, ef, q(r.source), q(r.note)].join(','))
    })
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'rally_annotations.csv'
    a.click()
    window.setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }, [fps])

  /* ---------------------------------------------------------------- 键盘 */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      const tag = (el?.tagName || '').toUpperCase()
      if (tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable) return
      if (tag === 'BUTTON' || tag === 'A') el?.blur()
      const big = e.altKey ? 5 : e.shiftKey ? 0.1 : 1
      switch (e.key) {
        case ' ':
          e.preventDefault()
          togglePlay()
          break
        case 'ArrowLeft':
          e.preventDefault()
          step(-big)
          break
        case 'ArrowRight':
          e.preventDefault()
          step(big)
          break
        case '[':
        case 'i':
        case 'I':
          newSeg()
          break
        case ']':
        case 'o':
        case 'O':
          finishSeg()
          break
        case 'c':
        case 'C':
          acceptAuto(!e.shiftKey)
          break
        case 'n':
        case 'N':
          jumpAuto(1)
          break
        case 'p':
        case 'P':
          jumpAuto(-1)
          break
        case 'd':
        case 'D':
          delSel()
          break
        case 'z':
        case 'Z':
          undo()
          break
        case 's':
        case 'S':
          void doSave()
          break
        case 'a':
        case 'A':
          selectNearest()
          break
        case 'Escape':
          setPending(null)
          setSel(null)
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [acceptAuto, delSel, doSave, finishSeg, jumpAuto, newSeg, selectNearest, step, togglePlay, undo])

  /* ---------------------------------------------------------------- rAF：时间同步 + 播放跟随 */
  useEffect(() => {
    let raf = 0
    const tick = () => {
      const v = videoRef.current
      if (v) {
        setTime((prev) => (Math.abs(prev - v.currentTime) > 0.02 ? v.currentTime : prev))
        const { loop: lp, ann: list } = stateRef.current
        if (lp) {
          const r = list.find((x) => x.id === stateRef.current.sel)
          if (r && v.currentTime > r.end) v.currentTime = r.start
        }
        // 播放时如果播放头跑出可视窗，视图跟着滚，避免播放头消失
        if (!v.paused) {
          const span = stateRef.current.viewSpan
          const dur = stateRef.current.duration
          const start = clamp(stateRef.current.viewCenter - span / 2, 0, Math.max(0, dur - span))
          if (v.currentTime > start + span * 0.88 || v.currentTime < start + span * 0.12) {
            setViewCenter(v.currentTime)
          }
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  /* ---------------------------------------------------------------- 尺寸 */
  useEffect(() => {
    const measure = () => {
      const w = detailRef.current?.clientWidth
      if (w) setDetailW(w)
    }
    measure()
    window.addEventListener('resize', measure)
    const ro = new ResizeObserver(measure)
    if (detailRef.current) ro.observe(detailRef.current)
    return () => {
      window.removeEventListener('resize', measure)
      ro.disconnect()
    }
  }, [loaded])

  /* ---------------------------------------------------------------- 概览波形 */
  useEffect(() => {
    const cv = overviewRef.current
    if (!cv) return
    const parentW = cv.parentElement?.clientWidth || 600
    const h = 56
    const dpr = window.devicePixelRatio || 1
    cv.width = parentW * dpr
    cv.height = h * dpr
    cv.style.width = `${parentW}px`
    cv.style.height = `${h}px`
    const ctx = cv.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, parentW, h)
    const dur = duration || 1
    const X = (t: number) => (t / dur) * parentW
    ctx.strokeStyle = '#1c2438'
    ctx.lineWidth = 1
    const step = dur > 1200 ? 300 : dur > 600 ? 120 : 60
    for (let s = 0; s <= dur; s += step) {
      ctx.beginPath()
      ctx.moveTo(X(s), 0)
      ctx.lineTo(X(s), h)
      ctx.stroke()
    }
    const [f0, f1] = focus
    if (f1 > f0) {
      ctx.fillStyle = 'rgba(56,189,248,.08)'
      ctx.fillRect(X(f0), 0, X(f1) - X(f0), h)
    }
    ctx.fillStyle = '#4b5563'
    ctx.globalAlpha = 0.55
    for (const r of auto) {
      const x = X(r.start)
      ctx.fillRect(x, 34, Math.max(1, X(r.end) - x), 16)
    }
    ctx.globalAlpha = 1
    for (const r of ann) {
      const x = X(r.start)
      ctx.fillStyle = r.id === sel ? '#facc15' : '#22c55e'
      ctx.globalAlpha = r.id === sel ? 0.95 : 0.7
      ctx.fillRect(x, 16, Math.max(1, X(r.end) - x), 16)
    }
    ctx.globalAlpha = 1
    if (pending) {
      const x = X(pending.start)
      ctx.fillStyle = '#fb923c'
      ctx.fillRect(x, 16, Math.max(1, X(Math.max(pending.start, time)) - x), 16)
    }
    ctx.fillStyle = '#f43f5e'
    ctx.fillRect(X(time) - 1, 0, 2, h)
  }, [ann, auto, focus, sel, pending, time, duration])

  /* ---------------------------------------------------------------- 详情时间轴几何 */
  const geom = useMemo(() => {
    const span = clamp(viewSpan, 1, Math.max(2, duration || 2))
    const start = clamp(viewCenter - span / 2, 0, Math.max(0, (duration || 0) - span))
    return { W: detailW, span, start, end: start + span }
  }, [viewSpan, viewCenter, duration, detailW])

  const t2x = useCallback((t: number) => ((t - geom.start) / geom.span) * geom.W, [geom])

  const niceStep = (span: number) => {
    const raw = span / 8
    const pow = Math.pow(10, Math.floor(Math.log10(raw)))
    for (const m of [1, 2, 5, 10]) if (raw <= m * pow) return m * pow
    return 10 * pow
  }
  const ticks = useMemo(() => {
    const step = niceStep(geom.span)
    const out: number[] = []
    for (let t = Math.ceil(geom.start / step) * step; t <= geom.end; t += step) out.push(t)
    return out
  }, [geom])

  /* ---------------------------------------------------------------- 时间轴交互 */
  const overviewPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const t = clamp(((e.clientX - rect.left) / rect.width) * (duration || 1), 0, duration)
    seek(t, true, false)
  }

  const onDetailDown = (e: React.PointerEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const bar = (e.target as HTMLElement).closest('.ann-bar') as HTMLElement | null
    if (bar && bar.dataset.id) {
      const id = Number(bar.dataset.id)
      const r = stateRef.current.ann.find((x) => x.id === id)
      if (!r) return
      const edge = ((e.target as HTMLElement).dataset.edge as 'start' | 'end' | undefined) || 'move'
      setSel(id)
      dragging.current = {
        id, edge, x0: e.clientX, s0: r.start, e0: r.end, span: geom.span, width: rect.width, moved: false, pushed: false,
      }
      e.preventDefault()
      return
    }
    // 空白处拖动 = 拖动时间轴
    const t = clamp(geom.start + ((e.clientX - rect.left) / rect.width) * geom.span, 0, duration)
    seek(t, true, false)
    dragging.current = { id: -1, edge: 'move', x0: e.clientX, s0: t, e0: t, span: geom.span, width: rect.width, moved: false, pushed: false }
  }

  useEffect(() => {
    const move = (e: PointerEvent) => {
      const d = dragging.current
      if (!d) return
      if (d.id === -1) {
        const t = clamp(geom.start + ((e.clientX - (detailRef.current?.getBoundingClientRect().left ?? 0)) / d.width) * d.span, 0, duration)
        seek(t, true, false)
        return
      }
      const dt = ((e.clientX - d.x0) / d.width) * d.span
      if (Math.abs(dt) > 1e-6 && !d.pushed) {
        pushUndo()
        d.pushed = true
        d.moved = true
      }
      setAnn((prev) =>
        prev
          .map((r) => {
            if (r.id !== d.id) return r
            if (d.edge === 'start') return { ...r, start: clamp(d.s0 + dt, 0, r.end - 0.05) }
            if (d.edge === 'end') return { ...r, end: clamp(d.e0 + dt, r.start + 0.05, duration) }
            const len = d.e0 - d.s0
            const ns = clamp(d.s0 + dt, 0, Math.max(0, duration - len))
            return { ...r, start: ns, end: ns + len }
          })
          .sort((a, b) => a.start - b.start),
      )
    }
    const up = () => {
      const d = dragging.current
      dragging.current = null
      if (d?.moved) markDirty()
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
    }
  }, [duration, geom, markDirty, pushUndo, seek])

  /* ---------------------------------------------------------------- 优化 */
  const runOptimize = useCallback(async () => {
    if (!pid || !mid) return
    if (!stateRef.current.ann.length) {
      toast({ kind: 'warn', title: '先标注几个回合再优化' })
      return
    }
    setOptimizing(true)
    try {
      // 先把当前标注落盘，优化读的是磁盘上的标注
      await doSave()
      const res = await api.optimizeSegmentation(pid, mid, { params })
      setOpt(res)
      if (!res.best) toast({ kind: 'warn', title: '没有搜到可用参数' })
      else
        toast({
          kind: 'success',
          title: `最优 F1 ${res.best.f1.toFixed(3)}（当前 ${res.baseline.f1.toFixed(3)}）`,
          detail: `试了 ${res.tried} 组参数，点「应用」写回工程并重新切分`,
        })
    } catch (e) {
      toast({ kind: 'error', title: '参数优化失败', detail: String((e as Error)?.message || e) })
    } finally {
      setOptimizing(false)
    }
  }, [doSave, mid, pid, params, toast])

  const applyBest = useCallback(async () => {
    if (!opt?.best) return
    await resegment(opt.best.params)
    toast({ kind: 'success', title: '已应用最优参数并重新切分' })
    await reload()
  }, [opt, reload, resegment, toast])

  const applyParam = useCallback(async (patch: Record<string, number>) => {
    await resegment(patch)
    toast({ kind: 'info', title: '已按这组参数重新切分' })
    await reload()
  }, [reload, resegment, toast])

  /* ---------------------------------------------------------------- 列表点击 */
  const jumpToList = (r: { start: number; id?: number }) => {
    setViewCenter(r.start)
    if (r.id != null) setSel(r.id)
    seek(r.start, true)
  }

  /* ---------------------------------------------------------------- 渲染 */
  if (!project || !media) {
    return (
      <div className="grid h-full place-items-center">
        <Empty icon={<PenLine size={30} />} title="先打开一个工程并导入素材" desc="标注与参数优化都需要一段已分析的视频。" />
      </div>
    )
  }

  const rows = tab === 'auto' ? auto.map((a) => ({ start: a.start, end: a.end, key: `a${a.index}`, meta: `${a.shots} 拍 · ${a.score} 分`, id: undefined as number | undefined })) : ann.map((r) => ({ start: r.start, end: r.end, key: `r${r.id}`, meta: r.source === 'auto' ? '自动采纳' : '手动', id: r.id }))
  const canOptimize = ann.length > 0

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-white/7 px-3 py-2">
        <PenLine size={15} className="text-court-300" />
        <div className="text-[13px] font-semibold text-white">回合标注</div>
        <span className="text-[11px] text-ink-500">
          {media.name} · {fmt(duration)} · {auto.length} 自动 / {ann.length} 标注
        </span>
        <div className="flex-1" />
        <span className={cn('text-[11px]', dirty ? 'text-amber-glow' : 'text-ink-500')}>{saveState}</span>
        <Button size="sm" variant="ghost" onClick={seedAll}>
          <Sparkles size={13} /> 全部采用自动
        </Button>
        <Button size="sm" variant="ghost" onClick={exportCsv}>
          <Download size={13} /> CSV
        </Button>
        <Button size="sm" variant="danger" onClick={clearAll}>
          <Trash2 size={13} /> 清空
        </Button>
        <Button size="sm" variant="primary" onClick={() => void doSave()}>
          <Check size={13} /> 保存
        </Button>
      </div>

      <div className="flex min-h-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col">
          <div className="relative bg-black">
            <video
              ref={videoRef}
              src={api.proxyUrl(pid, mid)}
              className="mx-auto max-h-[44vh] w-full bg-black"
              preload="auto"
              playsInline
              onClick={togglePlay}
              onLoadedMetadata={(e) => {
                const v = e.currentTarget
                if (!duration) setDuration(v.duration)
                if (!fps) setFps(0)
              }}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onRateChange={(e) => setSpeed(e.currentTarget.playbackRate)}
            />
            <div className="mono absolute top-2 left-3 rounded-md bg-black/55 px-2 py-0.5 text-[12px]">
              <b className="text-court-300">{fmt(time)}</b> / {fmt(duration)}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 border-b border-white/7 px-3 py-2">
            <Button size="sm" variant="primary" onClick={togglePlay}>
              {playing ? <Pause size={13} /> : <Play size={13} />}
              {playing ? '暂停' : '播放'}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => step(-1)}>« -1s (←)</Button>
            <Button size="sm" variant="ghost" onClick={() => step(1)}>(→) +1s »</Button>
            <span className="mx-1 text-[11px] text-ink-500">速度</span>
            <input
              type="range" min={0.1} max={2} step={0.1} value={speed} className="w-[80px]"
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (videoRef.current) videoRef.current.playbackRate = v
                setSpeed(v)
              }}
            />
            <span className="mono text-[11px] text-ink-400">{speed.toFixed(1)}x</span>
            <Button size="sm" variant={loop ? 'primary' : 'ghost'} onClick={() => setLoop((v) => !v)}>循环选中</Button>
            <div className="flex-1" />
            <Button size="sm" onClick={newSeg}>[ 起点</Button>
            <Button size="sm" onClick={finishSeg}>] 终点</Button>
            <Button size="sm" variant="subtle" onClick={() => acceptAuto(true)}>c 确认自动</Button>
            <Button size="sm" variant="ghost" onClick={() => jumpAuto(-1)}>p 上一自动</Button>
            <Button size="sm" variant="ghost" onClick={() => jumpAuto(1)}>n 下一自动</Button>
            <Button size="sm" variant="danger" onClick={delSel}>删除选中</Button>
            <Button size="sm" variant="ghost" onClick={undo}>撤销</Button>
          </div>

          <div className="px-3 pt-2">
            <canvas ref={overviewRef} className="block cursor-crosshair rounded-lg border border-white/8" onPointerDown={overviewPointer} />
          </div>

          <div className="px-3 py-1 text-[10.5px] text-ink-500">
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#4b5563]" /> 自动切分</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#22c55e]" /> 我的标注</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#facc15]" /> 选中</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#fb923c]" /> 标记中</span>
            <span className="float-right">
              <button className="mr-1 rounded border border-white/10 px-1.5 hover:border-court-500/60" onClick={() => setViewSpan((s) => clamp(s * 1.4, 2, Math.max(4, duration)))}>缩小</button>
              <button className="rounded border border-white/10 px-1.5 hover:border-court-500/60" onClick={() => setViewSpan((s) => clamp(s * 0.7, 2, Math.max(4, duration)))}>放大</button>
            </span>
          </div>

          <div className="px-3">
            <div ref={detailRef} className="relative h-[96px] select-none overflow-hidden rounded-lg border border-white/8 bg-ink-950/40" onPointerDown={onDetailDown}>
              <div className="absolute inset-x-0 top-0 h-5 border-b border-white/8 bg-white/[0.03]">
                {ticks.map((t) => (
                  <div key={t} className="mono absolute top-0 bottom-0 border-l border-white/10 pl-1 text-[10px] text-ink-500" style={{ left: t2x(t) }}>
                    {fmtShort(t)}
                  </div>
                ))}
              </div>
              <div className="absolute inset-x-0 top-5 bottom-0">
                {auto.map((r, i) => {
                  const x1 = t2x(r.start)
                  const x2 = t2x(r.end)
                  if (x2 < -20 || x1 > geom.W + 20) return null
                  return (
                    <div key={i} className="absolute top-1 h-[26px] rounded bg-[#4b5563]/55" style={{ left: Math.max(-2, x1), width: Math.max(2, x2 - x1) }} />
                  )
                })}
                {ann.map((r) => {
                  const x1 = t2x(r.start)
                  const x2 = t2x(r.end)
                  if (x2 < -20 || x1 > geom.W + 20) return null
                  const active = r.id === sel
                  return (
                    <div
                      key={r.id}
                      data-id={r.id}
                      className={cn(
                        'ann-bar absolute top-1 h-[26px] cursor-ew-resize rounded border',
                        active ? 'border-[#facc15] bg-[#22c55e]/70' : 'border-[#16a34a] bg-[#22c55e]/35',
                      )}
                      style={{ left: Math.max(-2, x1), width: Math.max(2, x2 - x1) }}
                    >
                      <div className="pointer-events-none absolute inset-x-1 top-0.5 truncate text-[10px] text-white/90">
                        {(r.end - r.start).toFixed(1)}s
                      </div>
                      <div data-edge="start" className="absolute top-0 bottom-0 left-0 w-[7px] cursor-ew-resize rounded-l" />
                      <div data-edge="end" className="absolute top-0 right-0 bottom-0 w-[7px] cursor-ew-resize rounded-r" />
                    </div>
                  )
                })}
                {pending && (
                  <div
                    className="absolute top-1 h-[26px] rounded border-2 border-dashed border-[#fb923c]"
                    style={{ left: t2x(Math.min(pending.start, time)), width: Math.max(2, Math.abs(t2x(time) - t2x(pending.start))) }}
                  />
                )}
                <div className="absolute top-0 bottom-0 w-[2px] bg-[#f43f5e]" style={{ left: t2x(time) }} />
              </div>
            </div>
          </div>

          <div className="px-3 py-2 text-[10.5px] leading-relaxed text-ink-500">
            <b>← →</b> 移动时间轴（Shift 0.1s / Alt 5s） · <b>[ ]</b> 标记新回合起止 · <b>c</b> 确认自动回合并跳下一个 ·
            <b> n / p</b> 下一 / 上一自动回合 · <b>d</b> 删除 · <b>z</b> 撤销 · <b>s</b> 保存 · 拖动色块边缘可微调
          </div>
        </section>

        <aside className="flex w-[360px] shrink-0 flex-col border-l border-white/7 bg-ink-950/35">
          <div className="flex gap-1 border-b border-white/7 p-2">
            {([['auto', `自动回合 (${auto.length})`], ['ann', `我的标注 (${ann.length})`]] as const).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={cn('flex-1 rounded-lg py-1.5 text-[12px] transition-colors', tab === id ? 'bg-court-500/15 text-court-200' : 'text-ink-400 hover:text-ink-100')}
              >
                {label}
              </button>
            ))}
          </div>

          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto">
            {rows.length === 0 ? (
              <div className="p-6 text-center text-[12px] text-ink-500">
                {tab === 'auto' ? '没有自动切分数据（先跑一次 AI 分析）' : '还没有标注，按 [ 开始'}
              </div>
            ) : (
              rows.map((r, i) => (
                <div
                  key={r.key}
                  onClick={() => jumpToList(r)}
                  className={cn('flex cursor-pointer items-center gap-2 border-b border-white/5 px-3 py-1.5 text-[12px] hover:bg-white/5', r.id != null && r.id === sel && 'bg-court-500/12')}
                >
                  <span className="w-5 shrink-0 text-[10.5px] text-ink-500">{i + 1}</span>
                  <span className="mono flex-1 truncate">{r.start.toFixed(2)} → {r.end.toFixed(2)} <span className="text-ink-500">({(r.end - r.start).toFixed(2)}s)</span></span>
                  <span className="shrink-0 text-[10.5px] text-ink-500">{r.meta}</span>
                  {r.id != null && (
                    <button
                      className="shrink-0 text-ink-600 hover:text-rose-hot"
                      onClick={(e) => {
                        e.stopPropagation()
                        pushUndo()
                        setAnn((prev) => prev.filter((x) => x.id !== r.id))
                        if (sel === r.id) setSel(null)
                        markDirty()
                      }}
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
              ))
            )}
          </div>

          {/* 参数优化 */}
          <div className="border-t border-white/7 p-3">
            <div className="mb-2 flex items-center gap-2">
              <Wand2 size={13} className="text-court-300" />
              <span className="text-[12px] font-semibold text-white">用标注优化切分参数</span>
              <div className="flex-1" />
              <Button size="sm" variant="primary" loading={optimizing} disabled={!canOptimize} onClick={() => void runOptimize()}>
                优化
              </Button>
            </div>
            {!opt ? (
              <div className="text-[11px] leading-relaxed text-ink-500">
                标好一段（建议 ≥20 个回合）后点「优化」：系统会用标注当标准答案，在参数网格上重跑切分并按 F1 排序。
                选中结果里的任一参数组都能直接「应用」，无需重跑 AI。
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex gap-3">
                  <MetricBar label="当前参数" m={opt.baseline} color="#8b96ad" />
                  {opt.best && <MetricBar label="最优" m={opt.best} color="#38e0a2" />}
                </div>
                <div className="text-[10.5px] text-ink-500">
                  标注 {opt.gt_count} 回合 · 试了 {opt.tried} 组 · IoU≥{opt.iou_threshold}
                </div>
                {opt.best && (
                  <div className="rounded-lg border border-court-500/25 bg-court-500/[0.06] p-2">
                    <div className="mb-1 flex items-center gap-2">
                      <span className="text-[11px] text-court-200">建议参数</span>
                      <div className="flex-1" />
                      <Button size="sm" variant="primary" onClick={() => void applyBest()}>应用并重切分</Button>
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                      {Object.entries(opt.best.params).map(([k, v]) => (
                        <div key={k} className="mono flex justify-between text-[10.5px] text-ink-300">
                          <span className="truncate text-ink-500">{SEG_LABELS[k] || k}</span>
                          <span>{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="max-h-[180px] overflow-y-auto rounded-lg border border-white/8">
                  {opt.results.slice(0, 20).map((m, i) => (
                    <button
                      key={i}
                      onClick={() => applyParam(m.params)}
                      className="flex w-full items-center gap-2 border-b border-white/5 px-2 py-1 text-left text-[11px] last:border-b-0 hover:bg-white/6"
                    >
                      <Badge color={i === 0 ? '#38e0a2' : undefined}>F1 {m.f1.toFixed(3)}</Badge>
                      <span className="mono flex-1 truncate text-ink-500">
                        {Object.entries(m.params).map(([k, v]) => `${SEG_LABELS[k] || k}=${v}`).join(' · ')}
                      </span>
                      <span className="text-ink-500">n{m.n}</span>
                    </button>
                  ))}
                </div>
                <div className="text-[10.5px] leading-relaxed text-ink-500">
                  点某一行会用那组参数立即重切分（复用已存信号，秒级）。不满意可以再选一组。
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
