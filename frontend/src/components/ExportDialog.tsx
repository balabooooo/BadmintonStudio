import { useEffect, useMemo, useState } from 'react'
import { Download, Film, Smartphone, Monitor, Square, Zap, HardDrive, AlertTriangle, CheckCircle2, XCircle, FolderOpen, FolderSearch } from 'lucide-react'
import { api } from '../lib/api'
import { bytes, cn, humanDuration, timecode } from '../lib/format'
import { Button, Modal, Progress, SectionTitle, Segmented, Slider, Toggle } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
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
  const tr = useT()
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
  const [mode, setMode] = useState<'merge' | 'separate'>('merge')
  const [outputDir, setOutputDir] = useState('')
  const [existingNames, setExistingNames] = useState<string[]>([])
  const env = useStore((s) => s.env)
  const setExportDir = useStore((s) => s.setExportDir)
  const toast = useStore((s) => s.toast)

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

  // 打开对话框（或工程名变化）时才重置文件名与预设列表。
  // 不能把 selected 放进依赖：那样每次点不同的输出预设，用户手填的文件名
  // 都会被悄悄重置回默认值。
  useEffect(() => {
    if (!open) return
    api.exportPresets().then(setPresets).catch((e) => {
      toast({ kind: 'error', title: tr('exportdlg.presetsFailed'), detail: String(e) })
    })
    api.listExports().then((list) => setExistingNames(list.map((x) => x.name))).catch(() => undefined)
    setName(`${project?.name ?? tr('exportdlg.finalName')}_${new Date().toISOString().slice(0, 10)}`)
    const saved = (project?.ui as Record<string, unknown> | undefined)?.export_dir
    setOutputDir(typeof saved === 'string' && saved ? saved : env?.export_dir ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 只在打开/换工程时初始化，跟随 tr / project.ui 重跑会覆盖用户已填内容
  }, [open, project?.name, project?.id, env?.export_dir])

  // 预设列表就绪后校正一次非法选择，与文件名互不干扰
  useEffect(() => {
    if (!open || !presets.length) return
    if (!presets.some((p) => p.id === selected)) setSelected(presets[0].id)
  }, [open, presets, selected])

  // 关闭时清掉上一次导出的任务状态：否则 jobId 一直指向上一个已完成的任务，
  // 再打开时页脚停在「复制路径 / 完成」，没法导出第二次。放在事件里而不是
  // effect 里，避免 setState-in-effect。
  const handleClose = () => {
    setJobId(null)
    setBusy(false)
    setBitrate('')
    setReframe(null)
    onClose()
  }

  const current = presets.find((p) => p.id === selected)
  // 导出结果：merge 一个文件、separate N 个；统一成路径列表好展示
  const resultPaths: string[] = Array.isArray(job?.result?.paths)
    ? (job!.result.paths as unknown[]).map(String)
    : job?.result?.path
      ? [String(job.result.path)]
      : []
  const revealId = job?.result?.exports?.[0]?.id as string | undefined

  // 与后端 `_safe_name` 同口径，用来推断本次会写出哪个文件名。
  const safeName = (s: string) => s.replace(/[<>:"/\\|?*]/g, '_').trim().slice(0, 120) || 'export'
  const baseName = safeName(name || tr('exportdlg.finalName'))
  const targetName = `${baseName}${mode === 'separate' ? '_01' : ''}.mp4`
  // 与后端 `_unique_export_path` 同口径：目标名已存在时基名追加 _1、_2…（分轨模式再接 _01 段号），
  // 不会覆盖旧导出。这里对照的是本应用登记过的导出名，至少能提前发现最常见的同名冲突。
  const taken = (n: string) => existingNames.some((x) => x.toLowerCase() === n.toLowerCase())
  const conflict = taken(targetName)
  const nextName = (() => {
    for (let k = 1; k < 1000; k++) {
      const cand = mode === 'separate' ? `${baseName}_${k}_01.mp4` : `${baseName}_${k}.mp4`
      if (!taken(cand)) return cand
    }
    return targetName
  })()
  // 后端同名冲突时自动顺延命名；merge 完成区只显示目录，这里推断实际文件名并在变化时明确提示。
  const finalName = resultPaths.length === 1 ? (resultPaths[0].split(/[\\/]/).pop() || '') : ''
  const finalStem = finalName.replace(/\.mp4$/i, '')
  const renamed =
    mode === 'merge' &&
    finalStem.toLowerCase().startsWith(baseName.toLowerCase()) &&
    /^_\d+$/.test(finalStem.slice(baseName.length))

  const copyPath = async () => {
    const p = resultPaths[0]
    if (!p) return
    try {
      await navigator.clipboard.writeText(p)
    } catch (e) {
      toast({ kind: 'error', title: tr('exportdlg.copyFailed'), detail: String(e) })
    }
  }

  const pickDir = async () => {
    try {
      const res = await api.pickExportDir()
      if (!res.cancelled && res.paths[0]) {
        setOutputDir(res.paths[0])
        setExportDir(res.paths[0])
      }
    } catch (e) {
      // 非 Windows / 原生框不可用：用户可直接在输入框里填绝对路径。
      toast({ kind: 'error', title: tr('exportdlg.pickDirFailed'), detail: String(e) })
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={tr('exportdlg.title')}
      subtitle={tr('exportdlg.subtitle', { clips: clips.length, duration: humanDuration(duration) })}
      width={780}
      footer={
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 text-[11.5px] text-ink-400">
            {running ? (
              <span className="text-court-300">{job?.message || tr('exportdlg.exporting')}</span>
            ) : job?.status === 'cancelled' ? (
              <span className="text-ink-400">{tr('exportdlg.cancelled')}</span>
            ) : job?.status === 'error' ? (
              <span className="text-rose-hot">{tr('exportdlg.failed')}</span>
            ) : job?.status === 'done' ? (
              <span className="text-court-300">{tr('exportdlg.done')}</span>
            ) : (
              <>
                {tr('exportdlg.estSizePrefix')} <span className="mono text-ink-200">{bytes(est)}</span>
                {current?.crf ? tr('exportdlg.crfNote') : ''}
                {current?.auto_reframe && ` · ${tr('exportdlg.reframeNote')}`}
              </>
            )}
          </div>
          <div className="flex shrink-0 gap-2">
            {running ? (
              <>
                <Button
                  variant="danger"
                  onClick={() =>
                    job &&
                    void api
                      .cancelJob(job.id)
                      .catch((e) => toast({ kind: 'error', title: tr('exportdlg.cancelFailed'), detail: String(e) }))
                  }
                >
                  {tr('exportdlg.cancelExport')}
                </Button>
                <Button variant="primary" loading disabled>
                  {tr('exportdlg.exportingButton')}
                </Button>
              </>
            ) : job?.status === 'done' && resultPaths.length > 0 ? (
              <>
                <Button variant="subtle" onClick={() => void copyPath()}>
                  {tr('exportdlg.copyPath')}
                </Button>
                <Button variant="outline" onClick={() => setJobId(null)}>
                  {tr('exportdlg.exportAgain')}
                </Button>
                <Button variant="primary" onClick={handleClose}>
                  {tr('exportdlg.finish')}
                </Button>
              </>
            ) : (
              <>
                <Button variant="ghost" onClick={handleClose}>
                  {tr('common.cancel')}
                </Button>
                <Button
                  variant="primary"
                  loading={busy}
                  disabled={!clips.length || !current}
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
                        { mode, outputDir },
                      )
                      if (res?.job_id) setJobId(res.job_id)
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  <Download size={14} />
                  {tr('exportdlg.start')}
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
            ) : job?.status === 'cancelled' ? (
              <XCircle size={14} className="text-ink-500" />
            ) : job?.status === 'error' ? (
              <AlertTriangle size={14} className="text-rose-hot" />
            ) : (
              <CheckCircle2 size={14} className="text-court-400" />
            )}
            <span className="text-[12.5px] font-medium text-court-200">
              {job?.status === 'cancelled'
                ? tr('exportdlg.cancelled')
                : job?.status === 'error'
                  ? tr('exportdlg.failed')
                  : running
                    ? job?.message || tr('exportdlg.exporting')
                    : tr('exportdlg.done')}
            </span>
            {running && (
              <span className="mono ml-auto text-[12px] text-court-300">
                {Math.round((job?.progress ?? 0) * 100)}%
              </span>
            )}
          </div>
          {running && <Progress value={job?.progress ?? 0} className="mt-2.5" />}
          {job?.status === 'cancelled' && job.message && (
            <div className="mt-2 text-[10.5px] text-ink-500">{job.message}</div>
          )}
          {job?.status === 'error' && (
            <div className="mono mt-2 max-h-[120px] overflow-y-auto text-[10.5px] whitespace-pre-wrap text-rose-hot/85">
              {job.error || job.message}
            </div>
          )}
          {job?.status === 'done' && resultPaths.length > 0 && (
            <div className="mt-2">
              <div className="flex items-center gap-2">
                <div className="text-[10.5px] text-ink-400">
                  {resultPaths.length > 1 ? tr('exportdlg.exportedFilesTo', { n: resultPaths.length }) : tr('exportdlg.savedTo')}
                </div>
                {revealId && (
                  <button
                    onClick={() =>
                      void api
                        .revealExport(revealId)
                        .catch((e) => toast({ kind: 'error', title: tr('exportdlg.revealFailed'), detail: String(e) }))
                    }
                    className="inline-flex items-center gap-1 text-[10.5px] text-court-300 hover:underline"
                  >
                    <FolderSearch size={11} /> {tr('exportdlg.openFolder')}
                  </button>
                )}
              </div>
              <code className="mono mt-1 block max-h-[90px] overflow-y-auto break-all rounded-md bg-black/35 px-2 py-1.5 text-[10.5px] text-ink-200">
                {String(job?.result?.dir || outputDir || resultPaths[0])}
              </code>
              {renamed && finalName && (
                <div className="mt-1 flex items-start gap-1.5 text-[10.5px] text-amber-glow">
                  <AlertTriangle size={11} className="mt-[1px] shrink-0" />
                  <span>{tr('exportdlg.renamedNotice', { file: targetName, next: finalName })}</span>
                </div>
              )}
              {resultPaths.length > 1 && (
                <div className="mt-1 max-h-[80px] space-y-0.5 overflow-y-auto text-[10px] text-ink-500">
                  {resultPaths.map((p) => (
                    <div key={p} className="truncate">
                      {p.split(/[\\/]/).pop()}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!clips.length && (
        <div className="mb-4 rounded-xl border border-amber-glow/30 bg-amber-glow/10 px-4 py-3 text-[12px] text-amber-glow">
          {tr('exportdlg.emptyTimeline')}
        </div>
      )}

      <SectionTitle>
        <Film size={12} /> {tr('exportdlg.outputPreset')}
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
                // 换预设就把码率拉回该预设的默认值，避免实际码率与卡片不符。
                setBitrate('')
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
                  {p.width}×{p.height} · {tr('exportdlg.bitrate')} {p.video_bitrate.replace('M', '')} Mbps ·{' '}
                  {p.vcodec === 'hevc' ? tr('exportdlg.codecHevc') : tr('exportdlg.codecH264')}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <SectionTitle>
        <Film size={12} /> {tr('exportdlg.mode')}
      </SectionTitle>
      <div className="mb-5 rounded-xl border border-white/7 bg-white/[0.025] px-4 py-3">
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: 'merge', label: tr('exportdlg.modeMerge'), hint: tr('exportdlg.modeMergeHint') },
            { value: 'separate', label: tr('exportdlg.modeSeparate'), hint: tr('exportdlg.modeSeparateHint') },
          ]}
        />
        <div className="mt-2 text-[10.5px] leading-relaxed text-ink-500">
          {mode === 'merge'
            ? tr('exportdlg.mergeDesc', { duration: humanDuration(duration) })
            : tr('exportdlg.separateDesc', { clips: clips.length, name: name || tr('exportdlg.finalName'), last: String(clips.length).padStart(2, '0') })}
        </div>
      </div>

      <SectionTitle>
        <Zap size={12} /> {tr('exportdlg.encoding')}
      </SectionTitle>
      <div className="mb-5 space-y-3">
        <div className="flex items-center justify-between gap-4">
          <span className="text-[12px] text-ink-200">{tr('exportdlg.encoder')}</span>
          <Segmented
            value={encoder}
            onChange={setEncoder}
            options={[
              { value: 'auto', label: tr('exportdlg.encoderAuto'), hint: tr('exportdlg.encoderAutoHint') },
              { value: 'nvenc', label: tr('exportdlg.encoderNvenc'), hint: tr('exportdlg.encoderNvencHint') },
              { value: 'x264', label: tr('exportdlg.encoderX264'), hint: tr('exportdlg.encoderX264Hint') },
            ]}
          />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <Slider
            label={tr('exportdlg.bitrate')}
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
              label={tr('exportdlg.reframeToggle')}
              hint={tr('exportdlg.reframeHint')}
            />
          </div>
        </div>
      </div>

      <SectionTitle>{tr('exportdlg.filename')}</SectionTitle>
      <input value={name} onChange={(e) => setName(e.target.value)} className="field" placeholder={tr('exportdlg.filenamePlaceholder')} />

      <div className="mt-4">
        <SectionTitle>
          <FolderOpen size={12} /> {tr('exportdlg.location')}
        </SectionTitle>
        <div className="flex items-center gap-2">
          <input
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="field min-w-0 flex-1"
            placeholder={tr('exportdlg.locationPlaceholder')}
          />
          <Button variant="outline" size="sm" onClick={() => void pickDir()}>
            <FolderOpen size={13} />
            {tr('exportdlg.selectFolder')}
          </Button>
        </div>
        <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
          {tr('exportdlg.willSaveTo')}{' '}
          <code className="mono text-ink-300">
            {outputDir || 'data\\exports'}\{name || tr('exportdlg.finalName')}
            {mode === 'separate' ? '_01.mp4 …' : '.mp4'}
          </code>
          {tr('exportdlg.saveHint')}
        </div>
        <div
          className={cn(
            'mt-1 flex items-start gap-1.5 text-[10.5px] leading-relaxed',
            conflict ? 'text-amber-glow' : 'text-ink-500',
          )}
        >
          {conflict && <AlertTriangle size={11} className="mt-[1px] shrink-0" />}
          <span>
            {conflict
              ? tr('exportdlg.overwriteWarn', { file: targetName, next: nextName })
              : tr('exportdlg.overwriteHint', { example: `${baseName}_1.mp4` })}
          </span>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-white/7 bg-white/[0.025] px-4 py-3">
        <div className="mb-2 grid grid-cols-[24px_70px_1fr_1fr] gap-3 text-[10.5px] tracking-wide text-ink-500">
          <span>#</span>
          <span>{tr('exportdlg.tableDuration')}</span>
          <span>{tr('exportdlg.tableSource')}</span>
          <span>{tr('exportdlg.tableRally')}</span>
        </div>
        <div className="max-h-[160px] space-y-1 overflow-y-auto">
          {clips.map((c, i) => (
            <div key={c.id} className="grid grid-cols-[24px_70px_1fr_1fr] items-center gap-3 text-[11.5px]">
              <span className="mono text-ink-500">{i + 1}</span>
              <span className="mono text-ink-400">{timecode((c.src_out - c.src_in) / c.speed, false)}</span>
              <span className="mono truncate text-ink-300">
                {tr('exportdlg.fromSource', { from: timecode(c.src_in, false), to: timecode(c.src_out, false) })}
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
