import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { Activity, CheckCircle2, XCircle, X, Loader2 } from 'lucide-react'
import { api } from '../lib/api'
import { cn, relTime } from '../lib/format'
import { Button, Progress, Tooltip } from './ui'
import { useStore } from '../store/useStore'

const ICONS = {
  running: Loader2,
  queued: Activity,
  done: CheckCircle2,
  error: XCircle,
  cancelled: X,
}

/** 把后端的内部阶段名换成中文（原来只在 AI 分析弹窗里翻译过）。 */
const STAGE_LABEL: Record<string, string> = {
  prepare: '准备中',
  proxy: '生成预览副本',
  audio: '提取音轨',
  hits: '检测击球声',
  motion: '分析画面运动',
  players: '检测与跟踪球员',
  shuttle: '跟踪羽毛球',
  segment: '切分回合',
  score: '评分',
  export: '导出',
  done: '完成',
  error: '出错',
  cancelled: '已取消',
}

const KIND_LABEL: Record<string, string> = {
  analyze: 'AI 分析',
  prepare: '准备素材',
  export: '导出',
}

function humanStage(s: string): string {
  return STAGE_LABEL[s] || s
}

export default function JobTray() {
  const jobs = useStore((s) => s.jobs)
  const [open, setOpen] = useState(false)
  const list = Object.values(jobs).sort((a, b) => b.created_at - a.created_at)
  const running = list.filter((j) => j.status === 'running' || j.status === 'queued').length

  if (!list.length) return null

  return (
    <div className="relative">
      <Tooltip content="任务队列" side="bottom">
        <Button variant={running ? 'outline' : 'ghost'} size="sm" onClick={() => setOpen((v) => !v)}>
          <Activity size={13} className={running ? 'text-court-300' : ''} />
          任务
          {running > 0 && (
            <span className="ml-0.5 grid h-4 min-w-4 place-items-center rounded-full bg-court-500 px-1 text-[10px] font-bold text-ink-950">
              {running}
            </span>
          )}
        </Button>
      </Tooltip>

      <AnimatePresence>
        {open && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="panel absolute right-0 z-50 mt-2 max-h-[min(420px,calc(100vh-140px))] w-[360px] overflow-y-auto p-2"
              style={{ boxShadow: 'var(--shadow-pop)' }}
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
                        {j.title || KIND_LABEL[j.kind] || j.kind}
                      </span>
                      {active && (
                        <button
                          onClick={() => api.cancelJob(j.id)}
                          className="text-[10.5px] text-ink-400 hover:text-rose-hot"
                        >
                          取消
                        </button>
                      )}
                      <span className="shrink-0 text-[10px] text-ink-500">{relTime(j.created_at)}</span>
                    </div>
                    {active && (
                      <>
                        <Progress value={j.progress} className="mt-1.5" />
                        <div className="mt-1 flex justify-between text-[10.5px] text-ink-400">
                          <span className="truncate">{j.message || humanStage(j.stage)}</span>
                          <span className="mono shrink-0 pl-2">{Math.round(j.progress * 100)}%</span>
                        </div>
                      </>
                    )}
                    {j.status === 'error' && (
                      <div className="mt-1 line-clamp-3 text-[10.5px] text-rose-hot/90">{j.message}</div>
                    )}
                    {j.status === 'done' && j.kind === 'export' && (
                      <div className="mt-1 truncate text-[10.5px] text-court-400">{j.message}</div>
                    )}
                  </div>
                )
              })}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
