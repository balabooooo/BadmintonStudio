import { useEffect, useMemo, useState } from 'react'
import { Download, Film, Smartphone, Monitor, Square, Zap, HardDrive, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { api } from '../lib/api'
import { bytes, cn, humanDuration, timecode } from '../lib/format'
import { Button, Modal, Progress, SectionTitle, Segmented, Slider, Toggle } from './ui'
import { useStore } from '../store/useStore'
import type { ExportPreset } from '../lib/types'

const ICONS: Record<string, typeof Film> = {
  yt1080p: Monitor,
  yt4k: Monitor,
  yt720p: Monitor,
  vertical: Smartphone,
  square: Square,
  hevc4k: HardDrive,
  draft: Zap,
}

export default function ExportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const project = useStore((s) => s.project)
  const exportVideo = useStore((s) => s.exportVideo)
  const jobs = useStore((s) => s.jobs)
  const [presets, setPresets] = useState<ExportPreset[]>([])
  const [selected, setSelected] = useState<string>('yt1080p')
  const [name, setName] = useState('')
  const [encoder, setEncoder] = useState<'auto' | 'nvenc' | 'x264'>('auto')
  const [bitrate, setBitrate] = useState<string>('')
  const [reframe, setReframe] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)

  const job = jobId ? jobs[jobId] : null
  const running = job && (job.status === 'running' || job.status === 'queued')

  const timeline = project?.timeline
  const clips = timeline?.tracks?.[0]?.clips ?? []
  const duration = timeline?.duration ?? 0
  const est = useMemo(() => {
    const p = presets.find((x) => x.id === selected)
    if (!p) return 0
    const mbps = parseFloat(p.video_bitrate) || 12
    return ((mbps + 0.2) * 1e6 * duration) / 8
  }, [presets, selected, duration])

  useEffect(() => {
    if (!open) return
    api.exportPresets().then((list) => {
      setPresets(list)
      if (list.length && !list.some((p) => p.id === selected)) setSelected(list[0].id)
    })
    setName(`${project?.name ?? '成片'}_${new Date().toISOString().slice(0, 10)}`)
  }, [open, project?.name, selected])

  const current = presets.find((p) => p.id === selected)

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="导出成片"
      subtitle={`${clips.length} 个片段 · 总时长 ${humanDuration(duration)}`}
      width={780}
      footer={
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 text-[11.5px] text-ink-400">
            {running ? (
              <span className="text-court-300">{job?.message || '正在导出…'}</span>
            ) : job?.status === 'done' ? (
              <span className="text-court-300">导出完成</span>
            ) : (
              <>
                预计体积不超过 <span className="mono text-ink-200">{bytes(est)}</span>
                {current?.crf ? '（按 CRF 质量模式，通常远小于这个上限）' : ''}
                {current?.auto_reframe && ' · 竖屏会自动跟随球员裁切'}
              </>
            )}
          </div>
          <div className="flex shrink-0 gap-2">
            {job?.status === 'done' && job.result?.path ? (
              <>
                <Button
                  variant="subtle"
                  onClick={() => {
                    navigator.clipboard.writeText(String(job.result.path))
                  }}
                >
                  复制路径
                </Button>
                <Button variant="primary" onClick={onClose}>
                  完成
                </Button>
              </>
            ) : (
              <>
                <Button variant="ghost" onClick={onClose} disabled={!!running}>
                  {running ? '导出中…' : '取消'}
                </Button>
                <Button
                  variant="primary"
                  loading={busy || !!running}
                  disabled={!clips.length || !!running}
                  onClick={async () => {
                    if (!current) return
                    setBusy(true)
                    try {
                      const res = await exportVideo(
                        {
                          ...current,
                          encoder,
                          video_bitrate: bitrate || current.video_bitrate,
                          auto_reframe: reframe ?? current.auto_reframe,
                        },
                        name,
                      )
                      if (res?.job_id) setJobId(res.job_id)
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  <Download size={14} />
                  开始导出
                </Button>
              </>
            )}
          </div>
        </div>
      }
    >
      {/* 导出中/完成：就地显示进度与输出位置，不跳页 */}
      {(running || job) && (
        <div className="mb-4 rounded-xl border border-court-500/25 bg-court-500/[0.07] px-4 py-3">
          <div className="flex items-center gap-2">
            {running ? (
              <span className="spin h-3.5 w-3.5 rounded-full border-[2px] border-court-400 border-t-transparent" />
            ) : job?.status === 'error' ? (
              <AlertTriangle size={14} className="text-rose-hot" />
            ) : (
              <CheckCircle2 size={14} className="text-court-400" />
            )}
            <span className="text-[12.5px] font-medium text-court-200">
              {job?.status === 'error' ? '导出失败' : running ? job?.message || '正在导出…' : '导出完成'}
            </span>
            {running && (
              <span className="mono ml-auto text-[12px] text-court-300">
                {Math.round((job?.progress ?? 0) * 100)}%
              </span>
            )}
          </div>
          {running && <Progress value={job?.progress ?? 0} className="mt-2.5" />}
          {job?.status === 'error' && (
            <div className="mono mt-2 max-h-[120px] overflow-y-auto text-[10.5px] whitespace-pre-wrap text-rose-hot/85">
              {job.error || job.message}
            </div>
          )}
          {job?.result?.path && (
            <div className="mt-2">
              <div className="text-[10.5px] text-ink-400">文件已保存到</div>
              <code className="mono mt-1 block truncate rounded-md bg-black/35 px-2 py-1.5 text-[10.5px] text-ink-200">
                {String(job.result.path)}
              </code>
            </div>
          )}
        </div>
      )}

      {!clips.length && (
        <div className="mb-4 rounded-xl border border-amber-glow/30 bg-amber-glow/10 px-4 py-3 text-[12px] text-amber-glow">
          时间线为空。先到「回合」面板用「按筛选自动剪辑」生成时间线，再来导出。
        </div>
      )}

      <SectionTitle>
        <Film size={12} /> 输出预设
      </SectionTitle>
      <div className="mb-5 grid grid-cols-2 gap-2 md:grid-cols-3">
        {presets.map((p) => {
          const Icon = ICONS[p.id] || Film
          const on = p.id === selected
          return (
            <button
              key={p.id}
              onClick={() => {
                setSelected(p.id)
                setReframe(null)
              }}
              className={cn(
                'flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-all',
                on ? 'border-court-500/55 bg-court-500/12' : 'border-white/8 bg-white/[0.025] hover:border-white/18',
              )}
            >
              <Icon size={15} className={cn('mt-0.5 shrink-0', on ? 'text-court-300' : 'text-ink-400')} />
              <div className="min-w-0">
                <div className={cn('truncate text-[12px] font-medium', on ? 'text-court-100' : 'text-ink-100')}>
                  {p.name}
                </div>
                <div className="mono mt-0.5 text-[10.5px] text-ink-400">
                  {p.width}×{p.height} · 码率 {p.video_bitrate.replace('M', '')} Mbps ·{' '}
                  {p.vcodec === 'hevc' ? 'H.265（体积小，老设备可能播不了）' : 'H.264（最通用）'}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <SectionTitle>
        <Zap size={12} /> 编码设置
      </SectionTitle>
      <div className="mb-5 space-y-3">
        <div className="flex items-center justify-between gap-4">
          <span className="text-[12px] text-ink-200">编码器</span>
          <Segmented
            value={encoder}
            onChange={setEncoder}
            options={[
              { value: 'auto', label: '自动', hint: '有独立显卡就用显卡编码' },
              { value: 'nvenc', label: '显卡加速（最快）', hint: '用 NVIDIA 显卡编码，速度最快' },
              { value: 'x264', label: 'CPU 编码（最稳）', hint: '兼容性最好，速度慢一些' },
            ]}
          />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <Slider
            label="码率"
            value={parseFloat(bitrate || current?.video_bitrate || '12')}
            min={2}
            max={80}
            step={1}
            onChange={(v) => setBitrate(`${v}M`)}
            format={(v) => `${v} Mbps`}
          />
          <div className="flex items-end pb-1">
            <Toggle
              checked={reframe ?? !!current?.auto_reframe}
              onChange={setReframe}
              label="竖屏自动跟随裁切"
              hint="把超广角画面裁成竖屏，并让镜头跟着双方球员移动"
            />
          </div>
        </div>
      </div>

      <SectionTitle>文件名</SectionTitle>
      <input value={name} onChange={(e) => setName(e.target.value)} className="field" placeholder="输出文件名" />
      <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
        会保存到 <code className="mono text-ink-300">data\exports\{`{文件名}`}.mp4</code>
        （所有工程的成片都放在这里，可在「导出记录」页查看和另存）
      </div>

      <div className="mt-4 rounded-xl border border-white/7 bg-white/[0.025] px-4 py-3">
        <div className="mb-2 grid grid-cols-[24px_70px_1fr_1fr] gap-3 text-[10.5px] tracking-wide text-ink-500">
          <span>#</span>
          <span>成片时长</span>
          <span>取自原片</span>
          <span>来源回合</span>
        </div>
        <div className="max-h-[160px] space-y-1 overflow-y-auto">
          {clips.map((c, i) => (
            <div key={c.id} className="grid grid-cols-[24px_70px_1fr_1fr] items-center gap-3 text-[11.5px]">
              <span className="mono text-ink-500">{i + 1}</span>
              <span className="mono text-ink-400">{timecode((c.src_out - c.src_in) / c.speed, false)}</span>
              <span className="mono truncate text-ink-300">
                原片 {timecode(c.src_in, false)} → {timecode(c.src_out, false)}
              </span>
              <span className="truncate text-ink-200">
                {c.label}
                {c.speed !== 1 && <span className="mono ml-1 text-court-300">{c.speed}×</span>}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  )
}
