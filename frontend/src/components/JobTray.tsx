import { AnimatePresence, motion } from 'motion/react'
import { useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Activity, CheckCircle2, XCircle, X, Loader2 } from 'lucide-react'
import { api } from '../lib/api'
import { cn, relTime } from '../lib/format'
import { Button, Progress, Tooltip } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
import { jobKindLabel, jobStageLabel } from '../i18n/domain'

const ICONS = {
  running: Loader2,
  queued: Activity,
  done: CheckCircle2,
  error: XCircle,
  cancelled: X,
}

export default function JobTray() {
  const t = useT()
  const jobs = useStore((s) => s.jobs)
  const toast = useStore((s) => s.toast)
  const [open, setOpen] = useState(false)
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())
  const [pos, setPos] = useState<{ top: number; right: number } | null>(null)
  const anchor = useRef<HTMLDivElement>(null)
  const list = Object.values(jobs)
    .filter((j) => !dismissed.has(j.id))
    .sort((a, b) => b.created_at - a.created_at)
  const running = list.filter((j) => j.status === 'running' || j.status === 'queued').length
  const finished = list.filter((j) => j.status === 'done' || j.status === 'error' || j.status === 'cancelled')
  const clearFinished = () => setDismissed((prev) => new Set([...prev, ...finished.map((j) => j.id)]))

  // 顶栏带 backdrop-blur，会变成 absolute/fixed 子元素的包含块，弹层留在里面会
  // 错位（也会被半透明背景透出底下的内容）。所以量好按钮位置后 portal 到 body。
  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      const r = anchor.current?.getBoundingClientRect()
      if (!r) return
      setPos({ top: r.bottom + 8, right: Math.max(8, window.innerWidth - r.right) })
    }
    place()
    window.addEventListener('resize', place)
    // 页面内容滚动时顶栏位置也会变（非 sticky 场景），跟着重新定位。
    window.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open])

  if (!list.length) return null

  return (
    <div className="relative" ref={anchor}>
      <Tooltip content={t('job.tray')} side="bottom">
        <Button variant={running ? 'outline' : 'ghost'} size="sm" onClick={() => { setPos(null); setOpen((v) => !v) }}>
          <Activity size={13} className={running ? 'text-court-300' : ''} />
          {t('job.tray')}
          {running > 0 && (
            <span className="ml-0.5 grid h-4 min-w-4 place-items-center rounded-full bg-court-500 px-1 text-[10px] font-bold text-ink-950">
              {running}
            </span>
          )}
        </Button>
      </Tooltip>

      {createPortal(
        <AnimatePresence>
          {open && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
              <motion.div
                initial={{ opacity: 0, y: -6, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.98 }}
                transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
                className="fixed z-50 max-h-[min(420px,calc(100vh-90px))] w-[360px] overflow-y-auto rounded-[14px] border border-white/10 bg-ink-900 p-2"
                style={{
                  top: pos?.top ?? -9999,
                  right: pos?.right ?? 12,
                  visibility: pos ? 'visible' : 'hidden',
                  boxShadow: 'var(--shadow-pop)',
                }}
              >
              {list.slice(0, 20).map((j) => {
                const Icon = ICONS[j.status] ?? Activity
                const active = j.status === 'running' || j.status === 'queued'
                return (
                  <div key={j.id} className="rounded-lg px-2.5 py-2 transition-colors hover:bg-white/4">
                    <div className="flex items-center gap-2">
                      <Icon
                        size={13}
                        className={cn(
                          active && j.status === 'running' && 'spin text-court-300',
                          j.status === 'done' && 'text-court-400',
                          j.status === 'error' && 'text-rose-hot',
                          j.status === 'cancelled' && 'text-ink-500',
                          j.status === 'queued' && 'text-amber-glow',
                        )}
                      />
                      <span className="min-w-0 flex-1 truncate text-[12px] text-ink-100">
                        {j.title || jobKindLabel(j.kind) || j.kind}
                      </span>
                      {active && (
                        <button
                          onClick={() =>
                            void api
                              .cancelJob(j.id)
                              .catch((e) => toast({ kind: 'error', title: t('job.cancelFailed'), detail: String(e) }))
                          }
                          className="text-[10.5px] text-ink-400 hover:text-rose-hot"
                        >
                          {t('job.cancel')}
                        </button>
                      )}
                      {!active && (
                        <button
                          onClick={() => setDismissed((prev) => new Set([...prev, j.id]))}
                          title={t('job.dismiss')}
                          className="text-ink-600 hover:text-ink-300"
                        >
                          <X size={11} />
                        </button>
                      )}
                      <span className="shrink-0 text-[10px] text-ink-500">{relTime(j.created_at)}</span>
                    </div>
                    {active && (
                      <>
                        <Progress value={j.progress} className="mt-1.5" />
                        <div className="mt-1 flex justify-between text-[10.5px] text-ink-400">
                          <span className="truncate">{j.message || jobStageLabel(j.stage)}</span>
                          <span className="mono shrink-0 pl-2">{Math.round(j.progress * 100)}%</span>
                        </div>
                      </>
                    )}
                    {j.status === 'error' && (
                      <div className="mt-1 line-clamp-3 text-[10.5px] text-rose-hot/90">{j.error || j.message}</div>
                    )}
                    {j.status === 'done' && j.kind === 'export' && (
                      <div className="mt-1 truncate text-[10.5px] text-court-400">{j.message}</div>
                    )}
                  </div>
                )
              })}
              {list.length > 20 && (
                <div className="mt-1.5 border-t border-white/6 pt-1.5 text-center text-[10.5px] text-ink-500">
                  {t('job.moreHidden', { n: list.length - 20 })}
                </div>
              )}
              {finished.length > 0 && (
                <button
                  onClick={clearFinished}
                  className="mt-1 w-full rounded-lg border border-white/8 px-2.5 py-1.5 text-[10.5px] text-ink-400 transition-colors hover:bg-white/6 hover:text-ink-200"
                >
                  {t('job.clearFinished', { n: finished.length })}
                </button>
              )}
            </motion.div>
          </>
        )}
        </AnimatePresence>,
        document.body,
      )}
    </div>
  )
}
