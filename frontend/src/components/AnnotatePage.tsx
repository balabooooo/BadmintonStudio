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
  Activity,
  AlertTriangle,
  Check,
  Crosshair,
  Download,
  Info,
  Pause,
  PenLine,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'
import { api } from '../lib/api'
import { createLogger } from '../lib/logger'
import { cn, clamp } from '../lib/format'
import { itemCounts, snapPatch, visibleWarnings, warningKey } from '../lib/annotationQuality'
import { Button, Empty, Modal, SpeedMenu, Tooltip, useConfirm } from './ui'
import { stepSpeedValue } from '../lib/playback'
import MediaChip from './MediaChip'
import OptimizePanel from './OptimizePanel'
import PoseOverlay from './PoseOverlay'
import SignalTracks from './SignalTracks'
import { useStore } from '../store/useStore'
import { useAnnotationDraft, type DraftHit as AnnHit, type DraftRally as AnnRally } from '../store/annotationDraft'
import { useT } from '../i18n/useT'
import type {
  AnalysisParams,
  AnnotationDraft,
  QualityItem,
  QualityReport,
  QualityWarning,
} from '../lib/types'

const annLog = createLogger('annotate')

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

export default function AnnotatePage() {
  const tr = useT()
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const media = useStore((s) => s.currentMedia())
  const params = useStore((s) => s.params)
  const resegment = useStore((s) => s.resegment)
  const rebuildHits = useStore((s) => s.rebuildHits)
  const toast = useStore((s) => s.toast)
  const savePreset = useStore((s) => s.savePreset)
  const courtPoly = useStore((s) => s.currentCourtPoly())
  const confirm = useConfirm()

  const pid = project?.id ?? ''
  const mid = mediaId ?? ''

  const videoRef = useRef<HTMLVideoElement>(null)
  const overviewRef = useRef<HTMLCanvasElement>(null)
  const detailRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // ---------- 数据：per-media 草稿保存在全局 draft store，切页（组件卸载）再回来即时 hydrate
  const draft = useAnnotationDraft(pid || null, mid || null, {
    duration: media?.duration,
    fps: media?.fps,
  })
  const {
    loaded,
    duration,
    fps,
    auto,
    rallies: ann,
    hits,
    rawHitTimes,
    focus,
    note,
    dirty,
    saveState,
    nextId,
    setRallies: setAnn,
    setHits,
    setFocus,
    setDuration,
    markDirty,
    save: doSave,
    reload,
  } = draft
  const [videoError, setVideoError] = useState(false)
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
  const undoRef = useRef<{ rallies: Omit<AnnRally, 'id'>[]; hits: { t: number; ours: boolean }[] }[]>([])

  // ---------- 优化结果由全局 store 按 mediaId 维护（任务化；切页/切素材不丢）
  const optimizeResults = useStore((s) => s.optimizeResults)
  const mediaIsOptimizing = useStore((s) => s.mediaIsOptimizing)
  const [showTracks, setShowTracks] = useState(false)
  /** 当前素材的优化结果（store 在 optimize 任务 done 时写入） */
  const opt = mid ? optimizeResults[mid] ?? null : null
  const optimizing = mediaIsOptimizing(mid)
  /** 当前素材的活跃 optimize 任务（进度/阶段/取消用） */
  const optimizeJob = useStore((s) => {
    if (!s.mediaId) return null
    for (const j of Object.values(s.jobs)) {
      if (j.kind === 'optimize' && j.media_id === s.mediaId
        && (j.status === 'queued' || j.status === 'running')) {
        return j
      }
    }
    return null
  })

  // ---------- 标注质量审计（P2-d：只建议，不自动改写）
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [qualityLoading, setQualityLoading] = useState(false)
  const [qOpenIdx, setQOpenIdx] = useState<number | null>(null)
  const [dismissed, setDismissed] = useState<ReadonlySet<string>>(new Set())

  // ---------- 场景预设
  const [presetOpen, setPresetOpen] = useState(false)
  const [presetName, setPresetName] = useState('')
  const [presetNote, setPresetNote] = useState('')
  const [presetSaving, setPresetSaving] = useState(false)

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
  const stateRef = useRef({ ann, auto, hits, sel, pending, duration, viewSpan, viewCenter, focus, tab, loop, note, dirty })
  useEffect(() => {
    stateRef.current = { ann, auto, hits, sel, pending, duration, viewSpan, viewCenter, focus, tab, loop, note, dirty }
  })

  /* ------------------------------------------------- 标注质量审计（P2-d） */
  const loadQuality = useCallback(async (silent = false) => {
    if (!pid || !mid) return
    setQualityLoading(true)
    try {
      setQuality(await api.getAnnotationQuality(pid, mid))
    } catch {
      // Auxiliary audit; a failed refresh keeps whatever report is on screen.
      if (!silent) toast({ kind: 'warn', title: tr('annotate.qualityFailed') })
    } finally {
      setQualityLoading(false)
    }
  }, [pid, mid, toast, tr])

  // Server data landed for this media (first GET / forced reload / clean-draft background
  // refresh): reset viewport, selection and undo history the same way the old in-page reload
  // did, then refresh the auxiliary quality audit. draft.rev identifies each landing.
  const draftRev = draft.rev
  useEffect(() => {
    if (!loaded) return
    setVideoError(false)
    setSel(null)
    undoRef.current = []
    setViewSpan(Math.max(10, Math.min(60, duration || 30)))
    setViewCenter(focus[0])
    setQuality(null)
    setQOpenIdx(null)
    setDismissed(new Set())
    void loadQuality(true)
    annLog.debug('server data applied; view reset', { mid, rev: draftRev })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftRev, mid])

  // Labels changed on the server (autosave landed): refresh the audit so snap suggestions
  // track the new rallies.
  const statusCodeRef = useRef(draft.statusCode)
  useEffect(() => {
    if (draft.statusCode === 'saved' && statusCodeRef.current !== 'saved') void loadQuality(true)
    statusCodeRef.current = draft.statusCode
  }, [draft.statusCode, loadQuality])

  const pushUndo = useCallback(() => {
    undoRef.current.push({
      rallies: stateRef.current.ann.map(({ id: _id, ...r }) => r),
      hits: stateRef.current.hits.map(({ t, ours }) => ({ t, ours })),
    })
    if (undoRef.current.length > 100) undoRef.current.shift()
  }, [])

  /* ---------------------------------------------------------------- 击球标注 */
  const toggleHit = useCallback((id: number) => {
    pushUndo()
    setHits((prev) => prev.map((h) => (h.id === id ? { ...h, ours: !h.ours } : h)))
    markDirty()
  }, [markDirty, pushUndo])

  const markAllHitsOurs = useCallback(() => {
    pushUndo()
    setHits((prev) => prev.map((h) => ({ ...h, ours: true })))
    markDirty()
  }, [markDirty, pushUndo])

  const seedHits = useCallback(() => {
    pushUndo()
    setHits(rawHitTimes.map((t) => ({ t, ours: true, id: nextId() })))
    markDirty()
  }, [markDirty, pushUndo, rawHitTimes])

  const clearHits = useCallback(() => {
    pushUndo()
    setHits([])
    markDirty()
  }, [markDirty, pushUndo])

  // 键盘修正击球标注时的搜索半径（秒）：超出则视为播放头附近没有击球，不做操作，
  // 避免误改远处（可能是几十秒外）的标注。击球间隔通常 <2s。
  const HIT_KEY_RADIUS = 2.0

  const nearestHit = useCallback((t: number): AnnHit | null => {
    let best: AnnHit | null = null
    let bd = Infinity
    for (const h of stateRef.current.hits) {
      const d = Math.abs(h.t - t)
      if (d < bd) {
        bd = d
        best = h
      }
    }
    return best && bd <= HIT_KEY_RADIUS ? best : null
  }, [])

  /** H：切换播放头最近一颗击球的「我方 / 邻场」归属。 */
  const toggleNearestHit = useCallback(() => {
    const h = nearestHit(videoRef.current?.currentTime ?? 0)
    if (!h) return
    pushUndo()
    setHits((prev) => prev.map((x) => (x.id === h.id ? { ...x, ours: !x.ours } : x)))
    markDirty()
  }, [markDirty, nearestHit, pushUndo])

  /** Shift+H：在播放头处补一颗击球（检测漏检时用）；0.2s 内已有则忽略，防手抖加重。 */
  const addHitAtPlayhead = useCallback(() => {
    const t = videoRef.current?.currentTime ?? 0
    if (stateRef.current.hits.some((h) => Math.abs(h.t - t) <= 0.2)) return
    pushUndo()
    setHits((prev) => [...prev, { t, ours: true, id: nextId() }])
    markDirty()
  }, [markDirty, pushUndo])

  /** X：删除播放头最近的一颗击球（误检时用）。 */
  const deleteNearestHit = useCallback(() => {
    const h = nearestHit(videoRef.current?.currentTime ?? 0)
    if (!h) return
    pushUndo()
    setHits((prev) => prev.filter((x) => x.id !== h.id))
    markDirty()
  }, [markDirty, nearestHit, pushUndo])

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

  /** 在预设档位间调倍速：dir=+1 加速，-1 减速。 */
  const stepSpeed = useCallback((dir: number) => {
    setSpeed((prev) => stepSpeedValue(prev, dir))
  }, [])

  // 与工作室播放器一致：speed 是唯一真源，由它写 playbackRate。
  useEffect(() => {
    const v = videoRef.current
    if (v) v.playbackRate = speed
  }, [speed])

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
    const p = stateRef.current.pending
    if (!p) return
    const st = Math.min(p.start, t)
    const en = Math.max(p.start, t)
    setPending(null)
    if (en - st >= 0.05) {
      pushUndo()
      addAnn({ start: st, end: en, source: 'manual' }, true)
    }
  }, [addAnn, pushUndo])

  const delSel = useCallback(() => {
    const cur = stateRef.current.sel
    if (cur == null) return
    pushUndo()
    setAnn((prev) => prev.filter((r) => r.id !== cur))
    setSel(null)
    markDirty()
  }, [markDirty, pushUndo])

  const undo = useCallback(() => {
    const snap = undoRef.current.pop()
    if (!snap) return
    setAnn(snap.rallies.map((r) => ({ ...r, id: nextId() })))
    setHits(snap.hits.map((h) => ({ ...h, id: nextId() })))
    setSel(null)
    markDirty()
  }, [markDirty])

  const clearAll = useCallback(async () => {
    if (!stateRef.current.ann.length) return
    const ok = await confirm({ title: tr('annotate.confirmClear', { n: stateRef.current.ann.length }), danger: true })
    if (!ok) return
    pushUndo()
    setAnn([])
    setSel(null)
    markDirty()
  }, [confirm, markDirty, pushUndo, tr])

  const seedAll = useCallback(async () => {
    if (!stateRef.current.auto.length) {
      toast({ kind: 'warn', title: tr('annotate.noAutoToImport') })
      return
    }
    if (stateRef.current.ann.length) {
      const ok = await confirm({ title: tr('annotate.confirmSeed') })
      if (!ok) return
    }
    pushUndo()
    setAnn((prev) => [
      ...prev,
      ...stateRef.current.auto.map((a) => ({ start: a.start, end: a.end, source: 'auto', note: '', id: nextId() })),
    ].sort((a, b) => a.start - b.start))
    markDirty()
  }, [confirm, markDirty, pushUndo, toast, tr])

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
      // Global media picker owns the keyboard while open (its grid handles
      // Space/arrows/Enter); do not also seek/play the annotate video.
      if (useStore.getState().mediaPickerOpen) return
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
        case 'h':
        case 'H':
          // H 切换最近击球归属，Shift+H 在播放头处补一颗击球。
          if (e.shiftKey) addHitAtPlayhead()
          else toggleNearestHit()
          break
        case 'x':
        case 'X':
          deleteNearestHit()
          break
        case 'Escape':
          setPending(null)
          setSel(null)
          break
        case '-':
        case '_':
          stepSpeed(-1)
          break
        case '=':
        case '+':
          stepSpeed(1)
          break
        default:
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [acceptAuto, addHitAtPlayhead, delSel, deleteNearestHit, doSave, finishSeg, jumpAuto, newSeg, selectNearest, step, stepSpeed, toggleNearestHit, togglePlay, undo])

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
      ctx.fillStyle = 'rgba(56,189,248,.10)'
      ctx.fillRect(X(f0), 0, X(f1) - X(f0), h)
      // Eval-window edges make the optimization/audit span readable at a glance.
      ctx.strokeStyle = 'rgba(56,189,248,.75)'
      ctx.lineWidth = 1
      for (const fe of [f0, f1]) {
        ctx.beginPath()
        ctx.moveTo(X(fe) + 0.5, 0)
        ctx.lineTo(X(fe) + 0.5, h)
        ctx.stroke()
      }
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
    if (mediaIsOptimizing(mid)) return
    if (!stateRef.current.ann.length) {
      toast({ kind: 'warn', title: tr('annotate.needAnnotations') })
      return
    }
    try {
      // Persist current labels first — the optimizer reads the annotation file on disk.
      await doSave()
      const { job_id } = await api.optimizeSegmentation(pid, mid, { params })
      // The report arrives through the WS job "done" event (store.optimizeResults).
      annLog.debug('optimize submitted', { job: job_id, mid })
      toast({ kind: 'info', title: tr('annotate.optimizeStarted') })
    } catch (e) {
      toast({ kind: 'error', title: tr('annotate.optimizeFailed'), detail: String((e as Error)?.message || e) })
    }
  }, [doSave, mediaIsOptimizing, mid, params, pid, toast, tr])

  const cancelOptimize = useCallback(async () => {
    if (!optimizeJob) return
    try {
      await api.cancelJob(optimizeJob.id)
    } catch (e) {
      toast({ kind: 'warn', title: tr('annotate.optimizeCancelFailed'), detail: String(e) })
    }
  }, [optimizeJob, toast, tr])

  // 只要改动了 hit_sensitivity，就必须重跑音频检测（重建击球序列）；否则 resegment 只会
  // 复用按旧灵敏度检测的原始击球，标注标定出的灵敏度等于没生效。
  const applyParamsToProject = useCallback(async (patch: Record<string, number>) => {
    const needAudio = 'hit_sensitivity' in patch && patch.hit_sensitivity !== params.hit_sensitivity
    if (needAudio) await rebuildHits(patch)
    else await resegment(patch)
  }, [params.hit_sensitivity, rebuildHits, resegment])

  const applyBest = useCallback(async () => {
    if (!opt?.best) return
    await applyParamsToProject(opt.best.params)
    toast({ kind: 'success', title: tr('annotate.appliedBest') })
    await reload()
  }, [applyParamsToProject, opt, reload, toast, tr])

  const applyParam = useCallback(async (patch: Record<string, number>) => {
    await applyParamsToProject(patch)
    toast({ kind: 'info', title: tr('annotate.appliedParams') })
    await reload()
  }, [applyParamsToProject, reload, toast, tr])

  /* ---------------------------------------------------------------- 场景预设 */
  // 存哪组参数：当前参数打底，有优化结果就用「最优」覆盖——那才是这次标注得到的结论。
  const collectPresetParams = useCallback((): Partial<AnalysisParams> => {
    const keys = [
      'seg_prominence', 'seg_min_core', 'seg_min_rest', 'seg_min_quiet',
      'min_rally_seconds', 'max_rally_seconds', 'pre_roll', 'post_roll', 'hit_tail_seconds',
      // 击球归属 / 邻场抑制（由标注优化标定）
      'pose_gate_threshold', 'pose_gate_window', 'hit_sensitivity',
    ] as const
    const out: Partial<AnalysisParams> = {}
    for (const k of keys) {
      const v = params[k]
      if (typeof v === 'number' && Number.isFinite(v)) out[k] = v
    }
    out.pose_gate_one_to_one = params.pose_gate_one_to_one
    out.pose_gate_force = params.pose_gate_force
    if (opt?.best?.params) Object.assign(out, opt.best.params)
    return out
  }, [params, opt])

  const collectFit = useCallback((): Record<string, number | string> | undefined => {
    if (!opt) return undefined
    const best = opt.best
    const fit: Record<string, number | string> = {
      source: 'annotation',
      scope: (opt.hit_label_count ?? 0) > 0 ? 'hit' : 'rally',
      baseline_f1: opt.baseline.f1,
      hit_label_count: opt.hit_label_count ?? 0,
    }
    if (best) {
      fit.f1 = best.f1
      fit.precision = best.precision
      fit.recall = best.recall
      if (best.hit) {
        fit.hit_f1 = best.hit.f1
        fit.hit_precision = best.hit.precision
        fit.hit_recall = best.hit.recall
      }
    }
    return fit
  }, [opt])

  const openPresetDialog = useCallback(() => {
    setPresetName(`${media?.name || tr('annotate.sceneFallback')} · ${opt ? tr('annotate.optimizedParams') : tr('annotate.baseline')}`)
    setPresetNote('')
    setPresetOpen(true)
  }, [media?.name, opt, tr])

  const doSavePreset = useCallback(async () => {
    if (!presetName.trim()) return
    setPresetSaving(true)
    try {
      await savePreset({
        name: presetName.trim(),
        note: presetNote.trim(),
        frameTime: time,
        params: collectPresetParams(),
        courtPoly: courtPoly ?? null,
        fit: collectFit(),
      })
      setPresetOpen(false)
    } finally {
      setPresetSaving(false)
    }
  }, [presetName, presetNote, time, collectPresetParams, collectFit, courtPoly, savePreset])

  /* ---------------------------------------------------------------- 列表点击 */
  const jumpToList = (r: { start: number; id?: number }) => {
    setViewCenter(r.start)
    if (r.id != null) setSel(r.id)
    seek(r.start, true)
  }

  /* ----------------------------------------------------- 质量审计交互（P2-d） */
  // Backend items are indexed by the sorted annotation order, which `ann` keeps.
  const qByIndex = useMemo(() => {
    const m = new Map<number, QualityItem>()
    for (const it of quality?.items ?? []) m.set(it.index, it)
    return m
  }, [quality])

  const qualityTotals = useMemo(() => {
    let warn = 0
    let info = 0
    for (const it of quality?.items ?? []) {
      const c = itemCounts(it, dismissed)
      warn += c.warn
      info += c.info
    }
    return { warn, info }
  }, [quality, dismissed])

  const acceptSnap = (r: AnnRally, qi: QualityItem, w: QualityWarning) => {
    const patch = snapPatch(r, w)
    if (!patch) return
    pushUndo()
    setAnn((prev) =>
      prev
        .map((x) => (x.id === r.id ? { ...x, ...patch } : x))
        .sort((a, b) => a.start - b.start),
    )
    setDismissed((prev) => new Set(prev).add(warningKey(qi.index, w)))
    setQOpenIdx(null)
    markDirty()
  }

  const ignoreWarning = (qi: QualityItem, w: QualityWarning) => {
    setDismissed((prev) => new Set(prev).add(warningKey(qi.index, w)))
  }

  const warningText = (w: QualityWarning): string => {
    const side = w.side
      ? tr(w.side === 'start' ? 'annotate.lq.sideStart' : 'annotate.lq.sideEnd')
      : ''
    switch (w.code) {
      case 'duration_outlier':
        return tr('annotate.lq.duration_outlier', { z: (w.z ?? 0).toFixed(1) })
      case 'boundary_off_quiet':
        return tr('annotate.lq.boundary_off_quiet', { side, d: (w.distance ?? 0).toFixed(2) })
      case 'boundary_no_hit':
        return tr('annotate.lq.boundary_no_hit', { side, d: (w.nearest_hit ?? w.distance ?? 0).toFixed(2) })
      case 'evidence_contradiction':
        return tr('annotate.lq.evidence_contradiction', { side })
      default:
        return tr(`annotate.lq.${w.code}`)
    }
  }

  /* --------------------------------------------------------- 评估窗（P2-d） */
  const focusIsFull = focus[0] <= 0 && focus[1] >= (duration || 0) - 1e-6

  const tightenFocus = () => {
    if (!ann.length) return
    const a = Math.min(...ann.map((r) => r.start))
    const b = Math.max(...ann.map((r) => r.end))
    setFocus([Math.max(0, a), Math.min(duration || b, b)])
    setViewCenter((a + b) / 2)
    markDirty()
    toast({ kind: 'success', title: tr('annotate.focusTightened', { n: ann.length }) })
  }

  const resetFocus = () => {
    setFocus([0, duration || 0])
    markDirty()
  }

  /* ---------------------------------------------------------------- 渲染 */
  if (!project || !media) {
    return (
      <div className="grid h-full place-items-center">
        <Empty icon={<PenLine size={30} />} title={tr('annotate.emptyTitle')} desc={tr('annotate.emptyDesc')} />
      </div>
    )
  }

  const rows = tab === 'auto' ? auto.map((a) => ({ start: a.start, end: a.end, key: `a${a.index}`, meta: tr('annotate.autoMeta', { shots: a.shots, score: a.score }), id: undefined as number | undefined })) : ann.map((r) => ({ start: r.start, end: r.end, key: `r${r.id}`, meta: r.source === 'auto' ? tr('annotate.sourceAuto') : tr('annotate.sourceManual'), id: r.id }))
  const canOptimize = ann.length > 0
  const mf = mediaId ? project?.analyses?.[mediaId]?.stats?.match_format : null
  const matchFormatCode = mf?.format && mf.format !== 'unknown' ? String(mf.format) : null

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-white/7 px-3 py-2">
        <PenLine size={15} className="text-court-300" />
        <div className="text-[13px] font-semibold text-white">{tr('annotate.title')}</div>
        <span className="flex items-center gap-1.5 text-[11px] text-ink-500">
          <MediaChip />
          {fmt(duration)} · {tr('annotate.headerMeta', { auto: auto.length, ann: ann.length })}
        </span>
        <div className="flex-1" />
        <span className={cn('text-[11px]', dirty ? 'text-amber-glow' : 'text-ink-500')}>{saveState}</span>
        <Button size="sm" variant="ghost" onClick={() => void seedAll()}>
          <Sparkles size={13} /> {tr('annotate.seedAll')}
        </Button>
        <Button size="sm" variant="ghost" onClick={exportCsv}>
          <Download size={13} /> CSV
        </Button>
        <Button size="sm" variant="danger" onClick={() => void clearAll()}>
          <Trash2 size={13} /> {tr('annotate.clearAll')}
        </Button>
        <Button size="sm" variant="primary" onClick={() => void doSave()}>
          <Check size={13} /> {tr('common.save')}
        </Button>
      </div>

      <div className="flex min-h-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col">
          <div className="relative bg-black">
            <video
              ref={videoRef}
              // key=mid：切换素材时重挂 video（干净加载新片 + 淡入过渡）。PoseOverlay 是
              // 它的兄弟节点不受影响——叠加开关与图层状态跨素材保留。
              key={mid}
              // 加代理路径做缓存键：重新生成代理后 URL 变化，浏览器才会取到新视频。
              src={`${api.proxyUrl(pid, mid)}?v=${encodeURIComponent(media.proxy_path ?? '')}`}
              className="clip-fade-in mx-auto max-h-[44vh] w-full bg-black"
              preload="auto"
              playsInline
              onClick={togglePlay}
              onLoadedMetadata={(e) => {
                const v = e.currentTarget
                if (!duration) setDuration(v.duration)
              }}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onRateChange={(e) => setSpeed(e.currentTarget.playbackRate)}
              onError={() => setVideoError(true)}
            />
            {videoError && (
              <div className="absolute inset-0 grid place-items-center bg-ink-950/85 px-6 text-center">
                <div className="flex max-w-[420px] flex-col items-center gap-1.5 text-[12px] text-ink-300">
                  <AlertTriangle size={20} className="text-amber-glow" />
                  <div className="font-medium text-ink-100">{tr('annotate.videoFailed')}</div>
                  <div className="text-[11px] leading-relaxed text-ink-500">{tr('annotate.videoFailedHint')}</div>
                </div>
              </div>
            )}
            <div className="mono absolute top-2 left-3 rounded-md bg-black/55 px-2 py-0.5 text-[12px]">
              <b className="text-court-300">{fmt(time)}</b> / {fmt(duration)}
            </div>
            <PoseOverlay videoRef={videoRef} pid={pid} mid={mid} />
          </div>

          <div className="flex flex-wrap items-center gap-1.5 border-b border-white/7 px-3 py-2">
            <Tooltip content={playing ? tr('annotate.pause') : tr('annotate.play')} kbd="Space">
              <Button size="sm" variant="primary" onClick={togglePlay}>
                {playing ? <Pause size={13} /> : <Play size={13} />}
                {playing ? tr('annotate.pause') : tr('annotate.play')}
              </Button>
            </Tooltip>
            <Tooltip content={tr('annotate.stepBack')} kbd="←">
              <Button size="sm" variant="ghost" onClick={() => step(-1)}>{tr('annotate.stepBack')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.stepForward')} kbd="→">
              <Button size="sm" variant="ghost" onClick={() => step(1)}>{tr('annotate.stepForward')}</Button>
            </Tooltip>
            <span className="mx-1 text-[11px] text-ink-500">{tr('annotate.speed')}</span>
            <SpeedMenu value={speed} onChange={setSpeed} title={tr('annotate.speed')} direction="down" />
            <Button size="sm" variant={loop ? 'primary' : 'ghost'} onClick={() => setLoop((v) => !v)}>{tr('annotate.loopSelected')}</Button>
            <div className="flex-1" />
            <Tooltip
              content={
                <span>
                  {tr('annotate.markStart')}
                  <br />
                  <span className="text-ink-400">{tr('annotate.partialHintShort')}</span>
                </span>
              }
              kbd={['[', 'i']}
            >
              <Button size="sm" onClick={newSeg}>{tr('annotate.markStart')}</Button>
            </Tooltip>
            <Tooltip
              content={
                <span>
                  {tr('annotate.markEnd')}
                  <br />
                  <span className="text-ink-400">{tr('annotate.partialHintShort')}</span>
                </span>
              }
              kbd={[']', 'o']}
            >
              <Button size="sm" onClick={finishSeg}>{tr('annotate.markEnd')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.confirmAuto')} kbd="c">
              <Button size="sm" variant="subtle" onClick={() => acceptAuto(true)}>{tr('annotate.confirmAuto')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.prevAuto')} kbd="p">
              <Button size="sm" variant="ghost" onClick={() => jumpAuto(-1)}>{tr('annotate.prevAuto')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.nextAuto')} kbd="n">
              <Button size="sm" variant="ghost" onClick={() => jumpAuto(1)}>{tr('annotate.nextAuto')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.deleteSelected')} kbd="d">
              <Button size="sm" variant="danger" onClick={delSel}>{tr('annotate.deleteSelected')}</Button>
            </Tooltip>
            <Tooltip content={tr('annotate.undoScoped')} kbd="z">
              <Button size="sm" variant="ghost" onClick={undo}>{tr('annotate.undoScoped')}</Button>
            </Tooltip>
          </div>

          <div className="px-3 pt-2">
            <canvas ref={overviewRef} className="block cursor-crosshair rounded-lg border border-white/8" onPointerDown={overviewPointer} />
          </div>
          {showTracks && pid && mediaId && (
            <SignalTracks
              projectId={pid}
              mediaId={mediaId}
              time={time}
              duration={duration}
              focus={focus}
              onSeek={(t) => seek(t, true, false)}
            />
          )}

          <div className="px-3 py-1 text-[10.5px] text-ink-500">
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#4b5563]" /> {tr('annotate.legendAuto')}</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#22c55e]" /> {tr('annotate.legendMine')}</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#facc15]" /> {tr('annotate.legendSelected')}</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[#fb923c]" /> {tr('annotate.legendMarking')}</span>
            <span className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-[10px] w-2.5 rounded-sm border border-sky-400/70 bg-sky-400/20" /> {tr('annotate.evalWindow')}</span>
            <span className="float-right inline-flex items-center gap-1">
              <Tooltip content={tr('annotate.signalsToggle')}>
                <button
                  className={cn(
                    'inline-flex items-center gap-1 rounded border px-1.5',
                    showTracks ? 'border-sky-400/60 text-sky-300' : 'border-white/10 hover:border-sky-400/60')}
                  onClick={() => setShowTracks((v) => !v)}
                >
                  <Activity size={11} />
                </button>
              </Tooltip>
              <Tooltip content={tr('annotate.tightenFocus')}>
                <button
                  className="inline-flex items-center gap-1 rounded border border-white/10 px-1.5 hover:border-sky-400/60 disabled:opacity-40"
                  disabled={!ann.length}
                  onClick={tightenFocus}
                >
                  <Crosshair size={11} />
                </button>
              </Tooltip>
              {!focusIsFull && (
                <Tooltip content={tr('annotate.resetFocus')}>
                  <button className="inline-flex items-center gap-1 rounded border border-white/10 px-1.5 hover:border-sky-400/60" onClick={resetFocus}>
                    <RotateCcw size={11} />
                  </button>
                </Tooltip>
              )}
              <button className="ml-1 rounded border border-white/10 px-1.5 hover:border-court-500/60" onClick={() => setViewSpan((s) => clamp(s * 1.4, 2, Math.max(4, duration)))}>{tr('annotate.zoomOut')}</button>
              <button className="rounded border border-white/10 px-1.5 hover:border-court-500/60" onClick={() => setViewSpan((s) => clamp(s * 0.7, 2, Math.max(4, duration)))}>{tr('annotate.zoomIn')}</button>
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
                {focus[1] > focus[0] && (
                  <>
                    {/* Dim everything outside the eval window (focus span). */}
                    <div
                      className="pointer-events-none absolute top-0 bottom-0 z-10 bg-ink-950/55"
                      style={{ left: 0, width: Math.max(0, clamp(t2x(focus[0]), 0, geom.W)) }}
                    />
                    <div
                      className="pointer-events-none absolute top-0 bottom-0 z-10 bg-ink-950/55"
                      style={{ left: Math.min(Math.max(t2x(focus[1]), 0), geom.W), right: 0 }}
                    />
                    {[focus[0], focus[1]].map((fe, k) => {
                      const x = t2x(fe)
                      if (x < 0 || x > geom.W) return null
                      return (
                        <div
                          key={k}
                          title={tr('annotate.evalWindow')}
                          className="pointer-events-none absolute top-0 bottom-0 z-10 w-px bg-sky-400/70"
                          style={{ left: x }}
                        />
                      )
                    })}
                  </>
                )}
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

            {/* 击球标注：点击切换「我方 / 邻场」 */}
            <div className="mt-2">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-ink-300">{tr('annotate.hitTitle')}</span>
                <span className="text-[10px] text-ink-500">{tr('annotate.hitHint')}</span>
                <div className="flex-1" />
                <span className="text-[10px] text-ink-500">
                  {tr('annotate.hitStats', {
                    ours: hits.filter((h) => h.ours).length,
                    other: hits.filter((h) => !h.ours).length,
                  })}
                </span>
                <button className="rounded border border-white/10 px-1.5 py-[1px] text-[10px] hover:border-court-500/60" onClick={markAllHitsOurs}>
                  {tr('annotate.hitAllOurs')}
                </button>
                <button className="rounded border border-white/10 px-1.5 py-[1px] text-[10px] hover:border-court-500/60" onClick={seedHits}>
                  {tr('annotate.hitSeed')}
                </button>
                <button className="rounded border border-white/10 px-1.5 py-[1px] text-[10px] hover:border-rose-hot/60" onClick={clearHits}>
                  {tr('annotate.hitClear')}
                </button>
              </div>
              <div className="relative h-[34px] select-none overflow-hidden rounded-lg border border-white/8 bg-ink-950/40" style={{ width: geom.W }}>
                {hits.map((h) => {
                  const x = t2x(h.t)
                  if (x < -6 || x > geom.W + 6) return null
                  return (
                    <button
                      key={h.id}
                      title={fmt(h.t)}
                      onClick={() => toggleHit(h.id)}
                      className={cn(
                        'absolute top-[4px] h-[26px] w-[3px] rounded-sm transition-colors',
                        h.ours ? 'bg-[#22c55e] hover:bg-[#4ade80]' : 'bg-[#f43f5e] hover:bg-[#fb7185]',
                      )}
                      style={{ left: x }}
                    />
                  )
                })}
                <div className="pointer-events-none absolute top-0 bottom-0 w-[2px] bg-[#f43f5e]/70" style={{ left: t2x(time) }} />
              </div>
            </div>
          </div>

          <div className="px-3 py-2 text-[10.5px] leading-relaxed text-ink-500">
            <div className="mb-1 flex items-start gap-1 text-court-300/90">
              <Info size={12} className="mt-0.5 shrink-0" />
              <span>{tr('annotate.partialHint')}</span>
            </div>
            <b>← →</b> {tr('annotate.helpSeek')} · <b>- / =</b> {tr('annotate.helpSpeed')} · <b>[ ]</b> {tr('annotate.helpMark')} · <b>c</b> {tr('annotate.helpConfirm')} ·
            <b> n / p</b> {tr('annotate.helpJump')} · <b>d</b> {tr('annotate.helpDelete')} · <b>z</b> {tr('annotate.undo')} · <b>s</b> {tr('common.save')} · {tr('annotate.helpDrag')}
            <br />
            <b>h</b> {tr('annotate.helpHitToggle')} · <b>⇧H</b> {tr('annotate.helpHitAdd')} · <b>x</b> {tr('annotate.helpHitDelete')}
          </div>
        </section>

        {/* key=mid：切换素材时右栏内容淡入过渡（回合列表 + 优化面板跟随当前 clip） */}
        <aside key={mid} className="clip-fade-in flex w-[360px] shrink-0 flex-col border-l border-white/7 bg-ink-950/35">
          <div className="flex gap-1 border-b border-white/7 p-2">
            {([['auto', tr('annotate.tabAuto', { n: auto.length })], ['ann', tr('annotate.tabMine', { n: ann.length })]] as const).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={cn('flex-1 rounded-lg py-1.5 text-[12px] transition-colors', tab === id ? 'bg-court-500/15 text-court-200' : 'text-ink-400 hover:text-ink-100')}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === 'ann' && (
            <button
              onClick={() => void loadQuality(false)}
              disabled={qualityLoading || !ann.length}
              className="flex w-full items-center gap-2 border-b border-white/7 px-3 py-1.5 text-left text-[11px] text-ink-400 hover:bg-white/5 disabled:opacity-60"
              title={tr('annotate.qualityHint')}
            >
              {qualityLoading ? (
                <RefreshCw size={12} className="shrink-0 animate-spin text-court-300" />
              ) : (
                <ShieldAlert size={12} className={cn('shrink-0', qualityTotals.warn > 0 ? 'text-amber-glow' : 'text-court-300')} />
              )}
              <span className="shrink-0">{tr('annotate.qualityAudit')}</span>
              {quality ? (
                qualityTotals.warn + qualityTotals.info > 0 ? (
                  <span className="truncate text-ink-500">
                    {tr('annotate.qualityCounts', { warn: qualityTotals.warn, info: qualityTotals.info })}
                  </span>
                ) : (
                  <span className="truncate text-court-300/80">{tr('annotate.qualityEmpty')}</span>
                )
              ) : (
                <span className="truncate text-ink-600">{tr('annotate.qualityHint')}</span>
              )}
            </button>
          )}

          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto">
            {rows.length === 0 ? (
              <div className="p-6 text-center text-[12px] text-ink-500">
                {tab === 'auto' ? tr('annotate.emptyAuto') : tr('annotate.emptyMine')}
              </div>
            ) : (
              rows.map((r, i) => {
                const qi = r.id != null ? qByIndex.get(i) : undefined
                const qc = qi ? itemCounts(qi, dismissed) : null
                const qOpen = qi != null && qOpenIdx === i
                const qVisible = qi && qc && qc.warn + qc.info > 0 ? visibleWarnings(qi, dismissed) : []
                const rally = r.id != null ? ann.find((a) => a.id === r.id) : undefined
                return (
                  <div
                    key={r.key}
                    className={cn(
                      'border-b border-white/5',
                      r.id != null && r.id === sel && 'bg-court-500/12',
                      qOpen && 'bg-white/[0.04]',
                    )}
                  >
                    <div
                      onClick={() => jumpToList(r)}
                      className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-[12px] hover:bg-white/5"
                    >
                      <span className="w-5 shrink-0 text-[10.5px] text-ink-500">{i + 1}</span>
                      <span className="mono flex-1 truncate">{r.start.toFixed(2)} → {r.end.toFixed(2)} <span className="text-ink-500">({(r.end - r.start).toFixed(2)}s)</span></span>
                      <span className="shrink-0 text-[10.5px] text-ink-500">{r.meta}</span>
                      {qi && qc && qc.warn + qc.info > 0 && (
                        <button
                          title={tr('annotate.qualityAudit')}
                          onClick={(e) => {
                            e.stopPropagation()
                            jumpToList(r)
                            setQOpenIdx(qOpen ? null : i)
                          }}
                          className={cn(
                            'inline-flex shrink-0 items-center gap-0.5 rounded px-1 text-[10px]',
                            qc.warn > 0 ? 'text-amber-glow hover:bg-amber-glow/10' : 'text-ink-500 hover:bg-white/10',
                          )}
                        >
                          <ShieldAlert size={11} />
                          {qc.warn > 0 ? qc.warn : qc.info}
                        </button>
                      )}
                      {r.id != null && (
                        <button
                          aria-label={tr('annotate.deleteSelected')}
                          title={tr('annotate.deleteSelected')}
                          className="shrink-0 text-ink-600 hover:text-rose-hot"
                          onClick={(e) => {
                            e.stopPropagation()
                            pushUndo()
                            setAnn((prev) => prev.filter((x) => x.id !== r.id))
                            if (sel === r.id) setSel(null)
                            if (qOpenIdx === i) setQOpenIdx(null)
                            markDirty()
                          }}
                        >
                          <X size={12} />
                        </button>
                      )}
                    </div>
                    {qOpen && rally && qVisible.length > 0 && (
                      <div className="space-y-1 px-3 pb-2 pl-9 pr-2">
                        {qVisible.map((w) => (
                          <div key={warningKey(qi!.index, w)} className="flex items-start gap-1.5 text-[10.5px]">
                            <AlertTriangle
                              size={10}
                              className={cn('mt-[2px] shrink-0', w.severity === 'warn' ? 'text-amber-glow' : 'text-ink-500')}
                            />
                            <span className="flex-1 text-ink-300">{warningText(w)}</span>
                            <div className="flex shrink-0 gap-1">
                              {w.snap_t != null && w.side && snapPatch(rally, w) && (
                                <button
                                  className="rounded border border-court-500/40 px-1 py-[1px] text-court-200 hover:bg-court-500/15"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    acceptSnap(rally, qi!, w)
                                  }}
                                >
                                  {tr('annotate.lq.acceptSnap', { t: w.snap_t!.toFixed(2) })}
                                </button>
                              )}
                              <button
                                className="rounded border border-white/10 px-1 py-[1px] text-ink-500 hover:bg-white/10"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  ignoreWarning(qi!, w)
                                }}
                              >
                                {tr('annotate.lq.ignore')}
                              </button>
                            </div>
                          </div>
                        ))}
                        <div className="text-[10px] text-ink-600">{tr('annotate.qualityHint')}</div>
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>

          {/* 参数优化（per-clip：面板内容绑定当前激活素材，切换 clip 自动跟随） */}
          <OptimizePanel
            opt={opt}
            optimizing={optimizing}
            progress={optimizeJob?.progress ?? 0}
            stage={optimizeJob?.stage ?? ''}
            canOptimize={canOptimize}
            matchFormat={matchFormatCode}
            onRun={() => void runOptimize()}
            onCancel={() => void cancelOptimize()}
            onApplyBest={() => void applyBest()}
            onApplyParam={(patch) => void applyParam(patch)}
            onOpenPreset={openPresetDialog}
          />
        </aside>
      </div>

      {/* 保存为场景预设 */}
      <Modal
        open={presetOpen}
        onClose={() => setPresetOpen(false)}
        title={tr('annotate.presetTitle')}
        subtitle={tr('annotate.presetSubtitle')}
        width={460}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setPresetOpen(false)}>
              {tr('common.cancel')}
            </Button>
            <Button variant="primary" loading={presetSaving} disabled={!presetName.trim()} onClick={() => void doSavePreset()}>
              {tr('common.save')}
            </Button>
          </div>
        }
      >
        <label className="mb-1.5 block text-[12px] text-ink-300">{tr('annotate.presetNameLabel')}</label>
        <input
          autoFocus
          value={presetName}
          onChange={(e) => setPresetName(e.target.value)}
          className="field"
          placeholder={tr('annotate.presetNamePlaceholder')}
        />
        <label className="mt-3 mb-1.5 block text-[12px] text-ink-300">{tr('annotate.presetNoteLabel')}</label>
        <input
          value={presetNote}
          onChange={(e) => setPresetNote(e.target.value)}
          className="field"
          placeholder={tr('annotate.presetNotePlaceholder')}
        />
        <div className="mt-3 space-y-1 text-[11px] leading-relaxed text-ink-500">
          <div>· {tr('annotate.presetParamsLabel')}{opt ? tr('annotate.presetParamsOptimized') : tr('annotate.presetParamsCurrent')}</div>
          <div>· {tr('annotate.presetCourtLabel')}{courtPoly ? tr('annotate.presetCourtManual', { n: courtPoly.length }) : tr('annotate.presetCourtNone')}</div>
          <div>· {tr('annotate.presetFrameLabel', { time: fmt(time) })}</div>
        </div>
      </Modal>
    </div>
  )
}
