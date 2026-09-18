import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Download, Film, FolderOpen, RefreshCw, Play } from 'lucide-react'
import { api } from '../lib/api'
import { bytes, relTime } from '../lib/format'
import { Button, Card, Empty, Modal, Skeleton } from './ui'
import { useStore } from '../store/useStore'

export default function ExportsPage() {
  const toast = useStore((s) => s.toast)
  const [files, setFiles] = useState<{ name: string; path: string; size: number; mtime: number }[] | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)

  async function load() {
    try {
      setFiles(await api.listExports())
    } catch {
      setFiles([])
    }
  }

  useEffect(() => {
    load()
    const t = window.setInterval(load, 4000)
    return () => window.clearInterval(t)
  }, [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1300px] px-8 py-8">
        <div className="mb-6 flex items-end justify-between gap-4">
          <div>
            <h1 className="text-[24px] font-semibold tracking-tight text-white">导出记录</h1>
            <p className="mt-1 text-[12.5px] text-ink-400">
              成片统一保存在 data/exports 目录（所有工程共用），可直接播放或另存
            </p>
          </div>
          <Button variant="outline" onClick={load}>
            <RefreshCw size={13} />
            刷新
          </Button>
        </div>

        {files === null ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-4">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-[180px]" />
            ))}
          </div>
        ) : files.length === 0 ? (
          <Card className="py-6">
            <Empty icon={<Film size={34} />} title="还没有导出过成片" desc="在剪辑台里自动剪辑后，点击右上角「导出」即可生成成片。" />
          </Card>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(330px,1fr))] gap-4">
            {files.map((f, i) => (
              <motion.div
                key={f.name}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.05, 0.35), duration: 0.36 }}
              >
                <Card hover className="overflow-hidden">
                  <div className="relative flex h-[130px] items-center justify-center bg-ink-850">
                    <video
                      src={api.exportUrl(f.name)}
                      className="h-full w-full object-cover"
                      muted
                      preload="metadata"
                      onLoadedData={(e) => {
                        const v = e.currentTarget
                        v.currentTime = Math.min(2, v.duration / 3 || 1)
                      }}
                    />
                    <button
                      onClick={() => setPlaying(f.name)}
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
                        a.href = api.exportUrl(f.name)
                        a.download = f.name
                        a.click()
                        toast({ kind: 'info', title: '开始下载', detail: f.name })
                      }}
                    >
                      <Download size={13} />
                      保存
                    </Button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      <Modal open={!!playing} onClose={() => setPlaying(null)} title={playing || ''} width={900}>
        {playing && (
          <video src={api.exportUrl(playing)} controls autoPlay className="w-full rounded-lg bg-black" />
        )}
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[11.5px] text-ink-400">
            <FolderOpen size={12} />
            {files?.find((f) => f.name === playing)?.path}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const p = files?.find((f) => f.name === playing)?.path
              if (p) navigator.clipboard.writeText(p)
              toast({ kind: 'info', title: '路径已复制' })
            }}
          >
            复制路径
          </Button>
        </div>
      </Modal>
    </div>
  )
}
