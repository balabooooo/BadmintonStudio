import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { AlertTriangle, Download, Film, FolderOpen, FolderSearch, RefreshCw, Play } from 'lucide-react'
import { api } from '../lib/api'
import { bytes, relTime } from '../lib/format'
import { Button, Card, Empty, Modal, Skeleton } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
import type { ExportItem } from '../lib/types'

export default function ExportsPage() {
  const t = useT()
  const toast = useStore((s) => s.toast)
  const [files, setFiles] = useState<ExportItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  async function load() {
    try {
      const list = await api.listExports()
      setFiles(list)
      setError(null)
    } catch (e) {
      // 不要吞掉错误只显示空列表：那会让人以为导出文件都没了。
      setError(String((e as Error)?.message || e))
      setFiles((prev) => prev ?? [])
    }
  }

  async function refresh() {
    setRefreshing(true)
    try {
      await load()
    } finally {
      setRefreshing(false)
    }
  }

  const reveal = async (id: string) => {
    try {
      await api.revealExport(id)
    } catch (e) {
      toast({ kind: 'error', title: t('exports.revealFailed'), detail: String(e) })
    }
  }

  const copyPath = async (p: string | undefined) => {
    if (!p) return
    try {
      await navigator.clipboard.writeText(p)
      toast({ kind: 'info', title: t('common.copied') })
    } catch (e) {
      toast({ kind: 'error', title: t('exports.copyFailed'), detail: String(e) })
    }
  }

  useEffect(() => {
    load()
    const timer = window.setInterval(load, 4000)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1300px] px-8 py-8">
        <div className="mb-6 flex items-end justify-between gap-4">
          <div>
            <h1 className="text-[24px] font-semibold tracking-tight text-white">{t('exports.title')}</h1>
            <p className="mt-1 text-[12.5px] text-ink-400">{t('exports.subtitle')}</p>
          </div>
          <Button variant="outline" onClick={() => void refresh()} loading={refreshing}>
            <RefreshCw size={13} />
            {t('common.refresh')}
          </Button>
        </div>

        {error && !files?.length ? (
          <Card className="py-6">
            <Empty
              icon={<AlertTriangle size={34} className="text-amber-glow" />}
              title={t('exports.loadFailed')}
              desc={error}
              action={
                <Button variant="outline" onClick={() => void load()}>
                  <RefreshCw size={13} /> {t('common.retry')}
                </Button>
              }
            />
          </Card>
        ) : files === null ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-4">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-[180px]" />
            ))}
          </div>
        ) : files.length === 0 ? (
          <Card className="py-6">
            <Empty icon={<Film size={34} />} title={t('exports.emptyTitle')} desc={t('exports.emptyDesc')} />
          </Card>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-4">
            {files.map((f, i) => (
              <motion.div
                key={f.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.05, 0.35), duration: 0.36 }}
              >
                <Card hover className="overflow-hidden">
                  <div className="relative flex h-[130px] items-center justify-center bg-ink-850">
                    <video
                      src={api.exportUrl(f.id)}
                      className="h-full w-full object-cover"
                      muted
                      preload="metadata"
                      onLoadedData={(e) => {
                        const v = e.currentTarget
                        v.currentTime = Math.min(2, v.duration / 3 || 1)
                      }}
                    />
                    <button
                      onClick={() => setPlaying(f.id)}
                      aria-label={t('exports.play')}
                      className="absolute inset-0 grid place-items-center bg-ink-950/25 opacity-0 transition-opacity hover:opacity-100"
                    >
                      <span className="grid h-12 w-12 place-items-center rounded-full bg-court-500/90 text-ink-950 shadow-lg">
                        <Play size={20} fill="currentColor" />
                      </span>
                    </button>
                  </div>
                  <div className="flex items-center justify-between gap-3 px-3.5 py-3">
                    <div className="min-w-0">
                      <div className="truncate text-[12.5px] font-medium text-white">{f.name}</div>
                      <div className="mt-0.5 text-[11px] text-ink-500">
                        {bytes(f.size)} · {relTime(f.mtime)}
                      </div>
                    </div>
                    <Button
                      variant="subtle"
                      size="sm"
                      onClick={() => {
                        const a = document.createElement('a')
                        a.href = api.exportUrl(f.id)
                        a.download = f.name
                        a.click()
                        toast({ kind: 'info', title: t('exports.downloadStarted'), detail: f.name })
                      }}
                    >
                      <Download size={13} />
                      {t('exports.save')}
                    </Button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      <Modal
        open={!!playing}
        onClose={() => setPlaying(null)}
        title={files?.find((f) => f.id === playing)?.name || ''}
        width={900}
      >
        {playing && (
          <video src={api.exportUrl(playing)} controls autoPlay className="w-full rounded-lg bg-black" />
        )}
        <div className="mt-3 flex items-center justify-between gap-3">
          <div className="mono flex min-w-0 items-center gap-1.5 text-[11.5px] text-ink-400">
            <FolderOpen size={12} className="shrink-0" />
            <span className="truncate" title={files?.find((f) => f.id === playing)?.path}>
              {files?.find((f) => f.id === playing)?.path}
            </span>
          </div>
          <div className="flex shrink-0 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void copyPath(files?.find((f) => f.id === playing)?.path)}
            >
              {t('exports.copyPath')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (playing) void reveal(playing)
              }}
            >
              <FolderSearch size={13} />
              {t('exports.openFolder')}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
