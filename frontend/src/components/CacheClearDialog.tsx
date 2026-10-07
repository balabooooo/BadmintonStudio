import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Check, CheckCircle2, XCircle, HardDrive } from 'lucide-react'
import { api, type CacheTargetStat } from '../lib/api'
import { bytes, cn } from '../lib/format'
import { Button, Modal, Progress, Spinner, useConfirm } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

type Phase = 'select' | 'running' | 'done'

interface ClearItemResult {
  id: string
  removed: number
  freed: number
  failed: string[]
}
interface ClearResult {
  items: ClearItemResult[]
  freed: number
  timed_out: string[]
}

/**
 * Selective cache cleanup dialog.
 *
 * The scan (GET /api/cache/targets) fills the checklist; submitting starts a cache_clear job and
 * progress/result arrive through the same WebSocket job channel the rest of the app uses. Danger
 * targets (models) get a second confirmation here AND a server-side confirm flag.
 */
export default function CacheClearDialog({
  open,
  onClose,
  onCleared,
}: {
  open: boolean
  onClose: () => void
  onCleared?: () => void
}) {
  const tr = useT()
  const confirm = useConfirm()
  const toast = useStore((s) => s.toast)
  const [targets, setTargets] = useState<CacheTargetStat[] | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [phase, setPhase] = useState<Phase>('select')
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const job = useStore((s) => (jobId ? s.jobs[jobId] : undefined))
  const onClearedRef = useRef(onCleared)
  useEffect(() => {
    onClearedRef.current = onCleared
  }, [onCleared])

  const load = useCallback(async () => {
    try {
      const list = await api.cacheTargets()
      setTargets(list)
      setPicked(new Set(list.filter((x) => x.default).map((x) => x.id)))
    } catch (e) {
      toast({ kind: 'error', title: tr('clearcache.loadFailed'), detail: String(e) })
    }
  }, [tr, toast])

  useEffect(() => {
    if (!open) return
    setPhase('select')
    setJobId(null)
    setTargets(null)
    setPicked(new Set())
    void load()
    // Reload every time the dialog opens (sizes may have changed); load is stable but re-entry resets.
  }, [open, load])

  // WebSocket job transitions the dialog to its result view; refresh the settings-page stats once.
  useEffect(() => {
    if (phase === 'running' && job?.status === 'done') {
      setPhase('done')
      onClearedRef.current?.()
    }
  }, [job?.status, phase])

  const selected = useMemo(
    () => (targets ?? []).filter((x) => picked.has(x.id)),
    [targets, picked],
  )
  const totalBytes = useMemo(() => selected.reduce((s, x) => s + x.bytes, 0), [selected])
  const allChecked = !!targets?.length && picked.size === targets.length

  const toggle = (id: string) =>
    setPicked((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const submit = async () => {
    const danger = picked.has('models')
    const ok = await confirm({
      title: danger ? tr('clearcache.confirmModelsTitle') : tr('clearcache.confirmTitle'),
      desc: danger
        ? tr('clearcache.confirmModelsDesc', { size: bytes(totalBytes) })
        : tr('clearcache.confirmDesc', { size: bytes(totalBytes) }),
      danger: true,
    })
    if (!ok) return
    setSubmitting(true)
    try {
      const ids = (targets ?? []).filter((x) => picked.has(x.id)).map((x) => x.id)
      const r = await api.clearCacheTargets(ids, danger)
      setJobId(r.job_id)
      setPhase('running')
    } catch (e) {
      toast({ kind: 'error', title: tr('clearcache.failed'), detail: String(e) })
    } finally {
      setSubmitting(false)
    }
  }

  const result = (job?.result ?? null) as ClearResult | null
  const failedItems = result?.items.filter((i) => i.failed.length) ?? []
  const webviewLocked = failedItems.some((i) => i.id === 'webview')
  const timedOut = result?.timed_out?.length ? result.timed_out : []

  const footer =
    phase === 'select' ? (
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11.5px] text-ink-400">
          {tr('clearcache.selectedTotal', { size: bytes(totalBytes) })}
        </span>
        <div className="flex shrink-0 gap-2">
          <Button variant="ghost" onClick={onClose}>
            {tr('common.cancel')}
          </Button>
          <Button variant="danger" loading={submitting} disabled={!picked.size} onClick={() => void submit()}>
            <HardDrive size={14} />
            {tr('clearcache.submit')}
          </Button>
        </div>
      </div>
    ) : phase === 'running' ? (
      <div className="flex justify-end gap-2">
        <Button
          variant="outline"
          disabled={!job || job.status !== 'running'}
          onClick={() => jobId && void api.cancelJob(jobId)}
        >
          {tr('clearcache.cancelJob')}
        </Button>
      </div>
    ) : (
      <div className="flex justify-end">
        <Button variant="primary" onClick={onClose}>
          {tr('clearcache.finish')}
        </Button>
      </div>
    )

  return (
    <Modal open={open} onClose={onClose} title={tr('clearcache.title')} subtitle={tr('clearcache.subtitle')} width={620} footer={footer}>
      {phase === 'select' &&
        (targets === null ? (
          <div className="flex items-center justify-center gap-2 py-12 text-[12.5px] text-ink-400">
            <Spinner size={15} /> {tr('clearcache.title')}
          </div>
        ) : (
          <div>
            <div className="mb-2 flex justify-end">
              <Button variant="ghost" size="sm" onClick={() => setPicked(allChecked ? new Set() : new Set(targets.map((x) => x.id)))}>
                {allChecked ? tr('clearcache.deselectAll') : tr('clearcache.selectAll')}
              </Button>
            </div>
            <div className="space-y-1.5">
              {targets.map((x) => {
                const on = picked.has(x.id)
                const dangerRow = x.level === 'danger'
                return (
                  <button
                    key={x.id}
                    type="button"
                    role="checkbox"
                    aria-checked={on}
                    aria-label={tr(`clearcache.t.${x.id}`)}
                    onClick={() => toggle(x.id)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors',
                      on
                        ? dangerRow
                          ? 'border-rose-hot/45 bg-rose-hot/10'
                          : 'border-court-500/45 bg-court-500/10'
                        : 'border-white/8 bg-white/[0.02] hover:border-white/20',
                    )}
                  >
                    <span
                      className={cn(
                        'flex h-4 w-4 shrink-0 items-center justify-center rounded-[5px] border transition-colors',
                        on
                          ? dangerRow
                            ? 'border-rose-hot bg-rose-hot text-white'
                            : 'border-court-400 bg-court-400 text-ink-950'
                          : 'border-white/25 bg-transparent',
                      )}
                    >
                      {on && <Check size={11} strokeWidth={3} />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className={cn('text-[12.5px] font-medium', on ? 'text-white' : 'text-ink-200')}>
                          {tr(`clearcache.t.${x.id}`)}
                        </span>
                        {dangerRow && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-rose-hot/15 px-1.5 py-px text-[10px] text-rose-hot">
                            <AlertTriangle size={9} />
                            {tr('clearcache.dangerBadge')}
                          </span>
                        )}
                      </span>
                      <span className="mt-0.5 block truncate text-[11px] leading-relaxed text-ink-500">
                        {tr(`clearcache.hint.${x.id}`)} · {tr('clearcache.files', { count: x.files })}
                      </span>
                    </span>
                    <span className="mono shrink-0 text-[12px] text-ink-300">{bytes(x.bytes)}</span>
                  </button>
                )
              })}
            </div>
          </div>
        ))}

      {phase === 'running' && (
        <div className="py-6">
          <div className="flex items-center gap-2.5">
            {job?.status === 'cancelled' ? (
              <XCircle size={16} className="text-ink-500" />
            ) : job?.status === 'error' ? (
              <AlertTriangle size={16} className="text-rose-hot" />
            ) : (
              <Spinner size={16} />
            )}
            <span className="text-[13px] font-medium text-ink-100">
              {job?.status === 'cancelled'
                ? tr('clearcache.cancelled')
                : job?.status === 'error'
                  ? tr('clearcache.failed')
                  : job?.message || tr('clearcache.running')}
            </span>
            {job?.status === 'running' && (
              <span className="mono ml-auto text-[12.5px] text-court-300">
                {Math.round((job.progress ?? 0) * 100)}%
              </span>
            )}
          </div>
          {(job?.status === 'running' || job?.status === 'queued') && (
            <Progress value={job?.progress ?? 0} className="mt-3" />
          )}
          {job?.status === 'error' && (
            <div className="mono mt-3 max-h-[140px] overflow-y-auto whitespace-pre-wrap rounded-lg bg-black/30 p-2.5 text-[10.5px] text-rose-hot/85">
              {job.error || job.message}
            </div>
          )}
        </div>
      )}

      {phase === 'done' && result && (
        <div className="py-2">
          <div className="flex items-center gap-2.5 rounded-xl border border-court-500/25 bg-court-500/[0.07] px-4 py-3">
            <CheckCircle2 size={16} className="text-court-400" />
            <span className="text-[13px] font-medium text-court-200">
              {tr('clearcache.freedTotal', { size: bytes(result.freed) })}
            </span>
          </div>
          <div className="mt-3 space-y-1.5">
            {result.items.map((i) => (
              <div key={i.id} className="flex items-center justify-between gap-3 text-[11.5px]">
                <span className="text-ink-300">{tr(`clearcache.t.${i.id}`)}</span>
                <span className={cn('mono', i.failed.length ? 'text-amber-glow' : 'text-ink-500')}>
                  {i.failed.length
                    ? tr('clearcache.failedCount', { count: i.failed.length })
                    : tr('clearcache.removedCount', { count: i.removed })}
                </span>
              </div>
            ))}
          </div>
          {failedItems.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-glow/30 bg-amber-glow/10 px-3 py-2 text-[11px] leading-relaxed text-amber-glow">
              {webviewLocked && <div className="mb-1">{tr('clearcache.webviewLocked')}</div>}
              {tr('clearcache.failHint')}
            </div>
          )}
          {timedOut.length > 0 && (
            <div className="mt-2 text-[11px] leading-relaxed text-ink-400">{tr('clearcache.timeoutHint')}</div>
          )}
        </div>
      )}
    </Modal>
  )
}
