import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Film, ListChecks, RefreshCw, Search, Sparkles, Trash2, VolumeX } from 'lucide-react'
import { api } from '../lib/api'
import type { MediaInfo } from '../lib/types'
import { bytes, cn, timecode } from '../lib/format'
import { Badge, Button, Modal, Progress, Tooltip, useConfirm } from './ui'
import ImportVideoButton from './ImportVideoButton'
import { consumeMediaPick, useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

type StatusFilter = 'all' | 'analyzed' | 'not_analyzed' | 'preparing'

const FILTER_KEYS: { value: StatusFilter; key: string }[] = [
  { value: 'all', key: 'mediaPicker.filterAll' },
  { value: 'analyzed', key: 'mediaPicker.filterAnalyzed' },
  { value: 'not_analyzed', key: 'mediaPicker.filterNotAnalyzed' },
  { value: 'preparing', key: 'mediaPicker.filterPreparing' },
]

/** Small status filter dropdown; closes on document-level pointerdown (same pattern as SpeedMenu). */
function FilterMenu({ value, onChange }: { value: StatusFilter; onChange: (v: StatusFilter) => void }) {
  const tr = useT()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  const current = FILTER_KEYS.find((f) => f.value === value)!

  return (
    <div className="relative" ref={ref}>
      <Button variant="ghost" size="sm" onClick={() => setOpen((v) => !v)}>
        {tr(current.key)}
        <ChevronDown size={12} />
      </Button>
      {open && (
        <div className="panel absolute top-full left-0 z-30 mt-1 min-w-[130px] overflow-hidden rounded-lg py-1 shadow-pop">
          {FILTER_KEYS.map((f) => (
            <button
              key={f.value}
              onClick={() => {
                onChange(f.value)
                setOpen(false)
              }}
              className={cn(
                'flex w-full items-center justify-between px-3 py-1.5 text-left text-[12px] transition-colors',
                f.value === value ? 'bg-court-500/12 text-court-200' : 'text-ink-200 hover:bg-white/6',
              )}
            >
              {tr(f.key)}
              {f.value === value && <Check size={12} />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * In-card muted preview. Mounted only while its card is the single active
 * preview: plays the proxy from t=0 at 1x and freezes on the 6 s frame.
 * Unmounting (mouse leave / scroll / dialog close) immediately releases it.
 */
function HoverPreview({ pid, media, onFailed }: { pid: string; media: MediaInfo; onFailed: () => void }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [cur, setCur] = useState(0)

  useEffect(() => {
    const v = ref.current
    if (!v) return
    let stopped = false
    const onTime = () => {
      setCur(v.currentTime)
      // Freeze at 6 s instead of looping or continuing.
      if (v.currentTime >= 6 && !stopped) {
        stopped = true
        v.pause()
      }
    }
    // Seek/play before data is ready aborts the pending play promise, so wait
    // for loadeddata. The element is freshly mounted, so it begins at t=0.
    const start = () => void v.play().catch(() => undefined)
    if (v.readyState >= 2) start()
    else v.addEventListener('loadeddata', start, { once: true })
    v.addEventListener('timeupdate', onTime)
    return () => {
      v.removeEventListener('loadeddata', start)
      v.removeEventListener('timeupdate', onTime)
    }
  }, [])

  const pct = media.duration > 0 ? Math.min(100, (cur / media.duration) * 100) : 0

  return (
    <>
      <video
        ref={ref}
        muted
        playsInline
        preload="auto"
        onError={onFailed}
        className="absolute inset-0 h-full w-full object-cover"
        src={`${api.proxyUrl(pid, media.id)}?v=${encodeURIComponent(media.proxy_path ?? '')}`}
      />
      <span className="absolute top-1.5 right-1.5 flex items-center gap-1 rounded bg-ink-950/70 px-1.5 py-[1px] text-[9.5px] text-ink-200">
        <VolumeX size={9} />
      </span>
      <span className="mono absolute bottom-4 left-2 text-[9.5px] text-white/90 drop-shadow">
        {timecode(cur, false)} / {timecode(media.duration, false)}
      </span>
      <div className="absolute right-0 bottom-0 left-0 h-[3px] bg-ink-950/70">
        <div className="h-full bg-court-400" style={{ width: `${pct}%` }} />
      </div>
    </>
  )
}

export default function MediaPickerDialog() {
  const tr = useT()
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const open = useStore((s) => s.mediaPickerOpen)
  const mode = useStore((s) => s.mediaPickerMode)
  const closeMediaPicker = useStore((s) => s.closeMediaPicker)
  const selectMedia = useStore((s) => s.selectMedia)
  const removeMedia = useStore((s) => s.removeMedia)
  const removeMediaBulk = useStore((s) => s.removeMediaBulk)
  const runAnalysisBatch = useStore((s) => s.runAnalysisBatch)
  const jobs = useStore((s) => s.jobs)
  const confirm = useConfirm()

  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<StatusFilter>('all')
  const [batch, setBatch] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [checked, setChecked] = useState<Set<string>>(new Set())
  // At most one preview video exists at any time (dialog-wide).
  const [activePreviewId, setActivePreviewId] = useState<string | null>(null)

  const gridRef = useRef<HTMLDivElement>(null)
  const activePreviewRef = useRef<string | null>(null)
  // Per-card hover debounce timers.
  const hoverTimers = useRef(new Map<string, number>())

  useEffect(() => {
    activePreviewRef.current = activePreviewId
  }, [activePreviewId])

  useEffect(() => {
    const timers = hoverTimers.current
    return () => {
      timers.forEach((h) => window.clearTimeout(h))
      timers.clear()
    }
  }, [])

  // Close edge: a pending 350 ms hover timer must not start a preview after close.
  useEffect(() => {
    if (!open) {
      hoverTimers.current.forEach((h) => window.clearTimeout(h))
      hoverTimers.current.clear()
    }
  }, [open])

  // Every open is a fresh session: transient UI state must not leak between opens.
  useEffect(() => {
    if (open) {
      setQuery('')
      setFilter('all')
      setBatch(false)
      setChecked(new Set())
      setSelectedId(mediaId)
      setActivePreviewId(null)
    }
    // Open edge only: mediaId changes while open (delete/import) must keep the
    // user's search/filter session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // While open, follow the store selection: importing media selects it, and
  // deleting the current media falls through to the first remaining item.
  useEffect(() => {
    if (open) setSelectedId(mediaId)
  }, [open, mediaId])

  // Live prepare / analyze jobs per media (same derivation as the old studio MediaPanel).
  const activePrepare = useMemo(
    () =>
      Object.values(jobs)
        .filter((j) => j.kind === 'prepare' && (j.status === 'running' || j.status === 'queued'))
        .sort((a, b) => a.created_at - b.created_at),
    [jobs],
  )
  const jobFor = (mid: string) => activePrepare.filter((j) => j.media_id === mid).at(-1)

  const activeAnalyze = useMemo(
    () =>
      Object.values(jobs)
        .filter((j) => j.kind === 'analyze' && (j.status === 'running' || j.status === 'queued'))
        .sort((a, b) => a.created_at - b.created_at),
    [jobs],
  )
  const analyzeFor = (mid: string) => activeAnalyze.filter((j) => j.media_id === mid).at(-1)

  /** Proxy must exist, no prepare/analyze job running, and batch mode off for hover preview. */
  const canPreview = (m: MediaInfo) => !!m.proxy_path && !jobFor(m.id) && !analyzeFor(m.id) && !batch

  const startHover = (id: string, m: MediaInfo) => {
    if (!canPreview(m)) return
    window.clearTimeout(hoverTimers.current.get(id))
    // 350 ms dwell keeps a fast sweep across the grid from mounting videos.
    const h = window.setTimeout(() => {
      hoverTimers.current.delete(id)
      if (canPreview(m)) setActivePreviewId(id)
    }, 350)
    hoverTimers.current.set(id, h)
  }

  const endHover = (id: string) => {
    window.clearTimeout(hoverTimers.current.get(id))
    hoverTimers.current.delete(id)
    setActivePreviewId((cur) => (cur === id ? null : cur))
  }

  const onGridKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement | null
    if (t?.closest('input, textarea, [contenteditable="true"], button')) return
    if (filtered.length === 0) return
    const cols = Math.max(
      1,
      (gridRef.current && window.getComputedStyle(gridRef.current).gridTemplateColumns.split(' ').length) || 1,
    )
    const curId = selectedId ?? mediaId
    let idx = filtered.findIndex((m) => m.id === curId)
    if (idx < 0) idx = 0
    let next = idx
    if (e.key === 'ArrowRight') next = Math.min(filtered.length - 1, idx + 1)
    else if (e.key === 'ArrowLeft') next = Math.max(0, idx - 1)
    else if (e.key === 'ArrowDown') next = Math.min(filtered.length - 1, idx + cols)
    else if (e.key === 'ArrowUp') next = Math.max(0, idx - cols)
    else if (e.code === 'Space') {
      e.preventDefault()
      const m = filtered[idx]
      if (canPreview(m)) setActivePreviewId((cur) => (cur === m.id ? null : m.id))
      return
    } else if (e.key === 'Enter') {
      confirmWith(filtered[idx].id)
      return
    } else {
      return
    }
    e.preventDefault()
    const target = filtered[next]
    setActivePreviewId(null)
    setSelectedId(target.id)
    gridRef.current?.querySelector<HTMLElement>(`[data-card-id="${CSS.escape(target.id)}"]`)?.scrollIntoView({ block: 'nearest' })
  }

  // Filtered, visible card set.
  const filtered = useMemo(() => {
    if (!project) return []
    const q = query.trim().toLowerCase()
    return project.media.filter((m) => {
      if (q && !m.name.toLowerCase().includes(q)) return false
      const done = project.analyses[m.id]?.status === 'done'
      if (filter === 'analyzed') return done
      if (filter === 'not_analyzed') return !done
      if (filter === 'preparing') return !!jobFor(m.id)
      return true
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, query, filter, activePrepare])

  // One dialog-wide observer: scrolling a previewing card out of the viewport
  // (the modal body scrolls) stops its video. The grid is full-height so it
  // cannot be the intersection root; use the browser viewport instead.
  // Re-observes whenever the visible set changes (search/filter, reopen), so
  // observation survives card remounts.
  useEffect(() => {
    if (!open) return
    const grid = gridRef.current
    if (!grid) return
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const id = (e.target as HTMLElement).dataset.cardId
          if (id && !e.isIntersecting && activePreviewRef.current === id) setActivePreviewId(null)
        }
      },
      { threshold: 0.1 },
    )
    grid.querySelectorAll<HTMLElement>('[data-card-id]').forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [open, filtered])

  if (!project) return null

  const currentMedia = project.media.find((m) => m.id === mediaId) ?? null
  const effectiveSelected = selectedId ?? mediaId
  const selectedMedia = effectiveSelected ? project.media.find((m) => m.id === effectiveSelected) : null
  const selectedBusy = effectiveSelected ? !!analyzeFor(effectiveSelected) : false

  const confirmWith = (mid: string) => {
    if (mid === mediaId) return
    if (mode === 'select') {
      const fn = consumeMediaPick()
      closeMediaPicker()
      fn?.(mid)
    } else {
      closeMediaPicker()
      selectMedia(mid)
    }
  }

  const doDelete = async (mid: string) => {
    if (analyzeFor(mid)) return
    const m = project.media.find((x) => x.id === mid)
    const ok = await confirm({
      title: tr('studio.confirmRemoveMedia', { name: m?.name ?? '' }),
      desc: tr('studio.confirmRemoveMediaDesc'),
      danger: true,
    })
    if (!ok) return
    await removeMedia(mid)
    if (selectedId === mid) setSelectedId(null)
  }

  const toggleCheck = (id: string) =>
    setChecked((prev) => {
      if (analyzeFor(id)) return prev
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  // Select-all only covers visible cards that are not being analyzed.
  const checkable = filtered.filter((m) => !analyzeFor(m.id))
  const allCheckableSelected = checkable.length > 0 && checkable.every((m) => checked.has(m.id))
  const toggleAll = () =>
    setChecked(allCheckableSelected ? new Set() : new Set(checkable.map((m) => m.id)))

  const doBulkDelete = async () => {
    const ids = [...checked].filter((id) => !analyzeFor(id))
    if (!ids.length) return
    const ok = await confirm({
      title: tr('studio.confirmRemoveSelected', { n: ids.length }),
      desc: tr('studio.confirmRemoveSelectedDesc'),
      danger: true,
    })
    if (!ok) return
    await removeMediaBulk(ids)
    setChecked(new Set())
    setBatch(false)
  }

  const doBulkAnalyze = async () => {
    const ids = [...checked].filter((id) => !analyzeFor(id))
    if (!ids.length) return
    await runAnalysisBatch(ids)
    setChecked(new Set())
    setBatch(false)
  }

  const enterBatch = () => {
    setBatch(true)
    setChecked(new Set())
    setActivePreviewId(null)
  }

  const renderCard = (m: MediaInfo) => {
    const a = project.analyses[m.id]
    const pj = jobFor(m.id)
    const aj = analyzeFor(m.id)
    const isCurrent = m.id === mediaId
    const isSelected = m.id === effectiveSelected
    const checkedOn = checked.has(m.id)
    const previewActive = !batch && activePreviewId === m.id && !!m.proxy_path && !pj

    return (
      <div
        key={m.id}
        data-card-id={m.id}
        onClick={() => {
          if (batch) toggleCheck(m.id)
          else {
            setSelectedId(m.id)
            gridRef.current?.focus()
          }
        }}
        onDoubleClick={() => {
          if (!batch) confirmWith(m.id)
        }}
        onPointerEnter={() => startHover(m.id, m)}
        onPointerLeave={() => endHover(m.id)}
        className={cn(
          'group cursor-pointer overflow-hidden rounded-xl border transition-all',
          batch
            ? checkedOn
              ? 'border-court-500/60 bg-court-500/[0.1]'
              : 'border-white/7 bg-white/[0.022] hover:border-white/16'
            : isSelected
              ? 'border-court-500/60 bg-court-500/[0.08] ring-1 ring-court-500/40'
              : isCurrent
                ? 'border-court-500/35 bg-court-500/[0.05] hover:border-white/16'
                : 'border-white/7 bg-white/[0.022] hover:border-white/16',
        )}
      >
        <div className="relative h-[104px] bg-ink-850">
          {previewActive ? (
            <HoverPreview pid={project.id} media={m} onFailed={() => endHover(m.id)} />
          ) : m.poster ? (
            <img src={api.assetUrl(m.poster)} className="h-full w-full object-cover" alt="" loading="lazy" />
          ) : (
            <div className="grid h-full place-items-center text-ink-700">
              <Film size={22} />
            </div>
          )}
          {!m.poster && pj && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 bg-ink-950/72 px-3">
              <RefreshCw size={14} className="spin text-court-300" />
              <Progress value={pj.progress} className="w-full" />
              <div className="w-full truncate text-center text-[10px] text-ink-300">
                {pj.stage === 'queued'
                  ? tr('studio.queued')
                  : `${pj.message || tr('studio.generatingPreview')} ${Math.round(pj.progress * 100)}%`}
              </div>
            </div>
          )}
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-ink-950/85 to-transparent" />
          {batch && (
            <div
              onClick={(e) => {
                e.stopPropagation()
                if (!aj) toggleCheck(m.id)
              }}
              title={aj ? tr('mediaPicker.deleteBlockedAnalyzing') : undefined}
              className={cn(
                'absolute top-1.5 left-1.5 grid h-5 w-5 place-items-center rounded-md border transition-colors',
                aj
                  ? 'cursor-not-allowed border-white/20 bg-ink-950/60 opacity-50'
                  : checkedOn
                    ? 'cursor-pointer border-court-400 bg-court-500 text-ink-950'
                    : 'cursor-pointer border-white/45 bg-ink-950/60 hover:border-white/70',
              )}
            >
              {checkedOn && <Check size={13} />}
              {!checkedOn && aj && <span className="text-[8px] text-ink-300">{Math.round(aj.progress * 100)}</span>}
            </div>
          )}
          {!batch && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                void doDelete(m.id)
              }}
              disabled={!!aj}
              aria-label={tr('common.delete')}
              title={aj ? tr('mediaPicker.deleteBlockedAnalyzing') : tr('common.delete')}
              className="absolute top-1.5 right-1.5 grid h-6 w-6 place-items-center rounded-md bg-ink-950/60 text-ink-400 opacity-0 transition-opacity hover:text-rose-hot focus:opacity-100 group-hover:opacity-100 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:text-ink-400"
            >
              <Trash2 size={11} />
            </button>
          )}
          <div className="absolute right-1.5 bottom-1.5 flex gap-1">
            {aj && (
              <Badge color="#c8a4ff">
                <Sparkles size={9} />
                {aj.stage === 'queued' ? tr('studio.queuedShort') : `${Math.round(aj.progress * 100)}%`}
              </Badge>
            )}
            {a?.status === 'done' && (
              <Badge color="#38e0a2">
                <Sparkles size={9} /> {a.rallies.length}
              </Badge>
            )}
            {m.proxy_path && <Badge color="#5c9dff">{tr('studio.previewReady')}</Badge>}
          </div>
          <div className="mono absolute bottom-1.5 left-2 text-[10.5px] text-ink-200">
            {timecode(m.duration, false)}
          </div>
        </div>
        <div className="px-2.5 py-2">
          <div className="truncate text-[12px] font-medium text-ink-100" title={m.path}>
            {m.name}
          </div>
          <div className="mono mt-0.5 truncate text-[10px] text-ink-500">
            {m.width > 0
              ? `${m.width}×${m.height} · ${m.fps.toFixed(0)}fps · ${bytes(m.size)}`
              : tr('mediaPicker.notAnalyzed')}
          </div>
        </div>
      </div>
    )
  }

  return (
    <Modal
      open={open}
      onClose={closeMediaPicker}
      title={tr('mediaPicker.title', { n: project.media.length })}
      width={880}
      footer={
        batch ? (
          <div className="flex items-center gap-2">
            <span className="text-[11.5px] text-ink-400">{tr('studio.selectedCount', { n: checked.size })}</span>
            <button
              onClick={toggleAll}
              className="text-[11px] text-ink-400 transition-colors hover:text-ink-100"
            >
              {allCheckableSelected ? tr('studio.deselectAll') : tr('studio.selectAll')}
            </button>
            <div className="flex-1" />
            <Button variant="primary" size="sm" disabled={!checked.size} onClick={() => void doBulkAnalyze()}>
              <Sparkles size={12} />
              {tr('studio.analyzeSelected')}
            </Button>
            <Button variant="danger" size="sm" disabled={!checked.size} onClick={() => void doBulkDelete()}>
              <Trash2 size={12} />
              {tr('common.delete')}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setBatch(false)
                setChecked(new Set())
              }}
            >
              {tr('studio.done')}
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate text-[11.5px] text-ink-400">
              {currentMedia ? tr('mediaPicker.current', { name: currentMedia.name }) : ''}
            </span>
            <Button
              variant="danger"
              size="sm"
              disabled={!selectedMedia || selectedBusy || effectiveSelected === null}
              onClick={() => effectiveSelected && void doDelete(effectiveSelected)}
            >
              <Trash2 size={12} />
              {tr('common.delete')}
            </Button>
            <Button
              variant="primary"
              size="sm"
              disabled={!selectedMedia || effectiveSelected === mediaId}
              title={effectiveSelected === mediaId ? tr('mediaPicker.alreadyCurrent') : undefined}
              onClick={() => effectiveSelected && confirmWith(effectiveSelected)}
            >
              {mode === 'select' ? tr('mediaPicker.selectThis') : tr('mediaPicker.switchTo')}
            </Button>
          </div>
        )
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[180px] flex-1">
          <Search size={12} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-600" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={tr('mediaPicker.searchPlaceholder')}
            className="w-full rounded-lg border border-white/8 bg-white/[0.03] py-1.5 pr-3 pl-7 text-[12px] text-ink-100 placeholder:text-ink-600 focus:border-court-500/50 focus:outline-none"
          />
        </div>
        <FilterMenu value={filter} onChange={setFilter} />
        <ImportVideoButton size="sm" />
        <ImportVideoButton mode="folder" size="sm" variant="subtle" />
        <Tooltip content={tr('studio.batchSelect')}>
          <Button
            variant={batch ? 'subtle' : 'ghost'}
            size="icon"
            aria-label={tr('studio.batchSelect')}
            aria-pressed={batch}
            onClick={() => (batch ? setBatch(false) : enterBatch())}
          >
            <ListChecks size={13} />
          </Button>
        </Tooltip>
      </div>

      {!batch && filtered.length > 0 && (
        <div className="mb-2 text-[10.5px] text-ink-600">{tr('mediaPicker.previewHint')}</div>
      )}
      {filtered.length === 0 ? (
        <div className="grid place-items-center py-14 text-[12.5px] text-ink-500">{tr('mediaPicker.empty')}</div>
      ) : (
        <div
          ref={gridRef}
          tabIndex={0}
          onKeyDown={onGridKeyDown}
          className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-2.5 rounded-xl outline-none focus-visible:ring-1 focus-visible:ring-court-500/40"
        >
          {filtered.map(renderCard)}
        </div>
      )}
    </Modal>
  )
}
