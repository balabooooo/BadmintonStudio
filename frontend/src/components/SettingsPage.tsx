import { useEffect, useState } from 'react'
import { Cpu, HardDrive, Trash2, Zap, CheckCircle2, XCircle, FolderTree } from 'lucide-react'
import { api } from '../lib/api'
import { bytes, cn } from '../lib/format'
import { Button, Card, Progress, SectionTitle, Stat, useConfirm } from './ui'
import { useStore } from '../store/useStore'

export default function SettingsPage() {
  const env = useStore((s) => s.env)
  const toast = useStore((s) => s.toast)
  const confirm = useConfirm()
  const [cache, setCache] = useState<{ cache: number; proxies: number; thumbs: number; exports: number } | null>(null)

  async function loadCache() {
    try {
      setCache(await api.cacheStats())
    } catch {
      /* ignore */
    }
  }
  useEffect(() => {
    loadCache()
  }, [])

  const caps = env?.caps || {}
  const capItems: [string, boolean][] = [
    ['NVIDIA NVENC (H.264)', !!caps.nvenc_h264],
    ['NVIDIA NVENC (HEVC)', !!caps.nvenc_hevc],
    ['Intel QuickSync', !!caps.qsv],
    ['CUDA 硬件解码', !!caps.cuda_decode],
    ['D3D11 硬件解码', !!caps.d3d11va],
    ['libx264 软件编码', !!caps.libx264],
  ]

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1000px] px-8 py-8">
        <h1 className="text-[24px] font-semibold tracking-tight text-white">设置与环境</h1>
        <p className="mt-1 mb-6 text-[12.5px] text-ink-400">分析引擎依赖本地 Python / FFmpeg / GPU，这里可以确认它们是否可用</p>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="p-5">
            <SectionTitle>
              <Cpu size={13} /> 计算设备
            </SectionTitle>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[12.5px] text-ink-300">GPU 加速</span>
                <span className={cn('text-[12.5px] font-medium', env?.gpu?.available ? 'text-court-400' : 'text-amber-glow')}>
                  {env?.gpu?.available ? '已启用' : '不可用'}
                </span>
              </div>
              <div className="text-[12.5px] text-ink-100">{env?.gpu?.name || '未检测到 CUDA 设备'}</div>
              <div className="mono text-[11px] text-ink-500">
                PyTorch {env?.gpu?.torch || '-'}
                {env?.gpu?.capability ? ` · sm_${env.gpu.capability.join('')}` : ''}
              </div>
              <div className="mono text-[11px] text-ink-500">Python {env?.python}</div>
              <div className="mono text-[11px] text-ink-500">{env?.platform}</div>
            </div>
          </Card>

          <Card className="p-5">
            <SectionTitle>
              <Zap size={13} /> FFmpeg 能力
            </SectionTitle>
            <div className="space-y-2">
              {capItems.map(([label, ok]) => (
                <div key={label} className="flex items-center gap-2 text-[12.5px]">
                  {ok ? (
                    <CheckCircle2 size={13} className="text-court-400" />
                  ) : (
                    <XCircle size={13} className="text-ink-600" />
                  )}
                  <span className={ok ? 'text-ink-100' : 'text-ink-500'}>{label}</span>
                </div>
              ))}
            </div>
            <div className="mono mt-3 break-all rounded-lg bg-black/25 px-2.5 py-2 text-[10.5px] text-ink-500">
              {env?.ffmpeg || env?.ffmpeg_error || '未检测到'}
            </div>
          </Card>

          <Card className="p-5 md:col-span-2">
            <SectionTitle
              right={
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={async () => {
                      const ok = await confirm({
                        title: '清理缓存？',
                        desc: '代理视频、缩略图和提取的音轨会被删除。下次分析会重新生成，不影响原始素材。',
                        danger: true,
                      })
                      if (!ok) return
                      const r = await api.clearCache('all')
                      toast({ kind: 'success', title: `已释放 ${bytes(r.freed)}` })
                      loadCache()
                    }}
                  >
                    <Trash2 size={13} />
                    清理全部缓存
                  </Button>
                </div>
              }
            >
              <HardDrive size={13} /> 磁盘占用
            </SectionTitle>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="缓存总计" value={bytes(cache?.cache || 0)} />
              <Stat label="代理视频" value={bytes(cache?.proxies || 0)} hint="AI 分析用的低分辨率副本，可安全重建" />
              <Stat label="缩略图" value={bytes(cache?.thumbs || 0)} />
              <Stat label="导出成片" value={bytes(cache?.exports || 0)} />
            </div>
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between text-[11.5px] text-ink-400">
                <span>缓存占已用空间比例（相对导出目录）</span>
                <span className="mono">
                  {cache && cache.exports + cache.cache > 0
                    ? ((cache.cache / (cache.cache + cache.exports)) * 100).toFixed(0)
                    : 0}
                  %
                </span>
              </div>
              <Progress
                value={
                  cache && cache.cache + cache.exports > 0 ? cache.cache / (cache.cache + cache.exports) : 0
                }
              />
            </div>
          </Card>

          <Card className="p-5 md:col-span-2">
            <SectionTitle>
              <FolderTree size={13} /> 目录位置
            </SectionTitle>
            <div className="space-y-2">
              {[
                ['数据目录', env?.data_dir],
                ['缓存目录', env?.cache_dir],
                ['模型目录', env?.models_dir],
              ].map(([k, v]) => (
                <div key={k as string} className="flex items-center gap-3">
                  <span className="w-20 shrink-0 text-[12px] text-ink-400">{k}</span>
                  <code className="mono min-w-0 flex-1 truncate rounded-md bg-black/25 px-2 py-1 text-[11px] text-ink-300">
                    {v || '-'}
                  </code>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      if (v) navigator.clipboard.writeText(v as string)
                      toast({ kind: 'info', title: '路径已复制' })
                    }}
                  >
                    复制
                  </Button>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
