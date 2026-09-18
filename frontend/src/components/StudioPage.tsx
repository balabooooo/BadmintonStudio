import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  Sparkles,
  Download,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Film,
  ListChecks,
  Upload,
  Trash2,
  RefreshCw,
  HardDrive,
  Crosshair,
} from 'lucide-react'
import { api } from '../lib/api'
import { bytes, cn, timecode } from '../lib/format'
import { Badge, Button, Card, Empty, Progress, Tooltip, useConfirm } from './ui'
import Player from './Player'
import Timeline, { TIMELINE_MIN_H } from './Timeline'
import RallyPanel from './RallyPanel'
import Inspector from './Inspector'
import AnalysisDialog from './AnalysisDialog'
import ExportDialog from './ExportDialog'
import { useStore } from '../store/useStore'

/* ------------------------------------------------------------------ 素材列表 */

function MediaPanel() {
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const selectMedia = useStore((s) => s.selectMedia)
  const removeMedia = useStore((s) => s.removeMedia)
  const importMedia = useStore((s) => s.importMedia)
  const jobs = useStore((s) => s.jobs)
  const confirm = useConfirm()
  const [pathOpen, setPathOpen] = useState(false)
  const [paths, setPaths] = useState('')
  const [dropping, setDropping] = useState(false)

  if (!project) return null

  const prepareJob = Object.values(jobs)
    .filter((j) => j.kind === 'prepare' && j.status === 'running')
    .at(-1)

  return (
    <div
      className="flex h-full flex-col"
      onDragOver={(e) => {
        e.preventDefault()
        setDropping(true)
      }}
      onDragLeave={() => setDropping(false)}
      onDrop={async (e) => {
        e.preventDefault()
        setDropping(false)
        const files = Array.from(e.dataTransfer.files)
        if (files.length) {
          const s = useStore.getState()
          await s.importFiles(files)
        }
      }}
    >
      <div className="flex items-center gap-2 border-b border-white/7 px-3 py-2.5">
        <span className="flex-1 text-[11px] font-semibold tracking-[0.14em] text-ink-400 uppercase">
          素材 ({project.media.length})
        </span>
        <Tooltip content="按文件路径导入">
          <Button variant="ghost" size="icon" onClick={() => setPathOpen((v) => !v)}>
            <Upload size={13} />
          </Button>
        </Tooltip>
      </div>

      <AnimatePresence>
        {pathOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-b border-white/7"
          >
            <div className="p-2.5">
              <textarea
                value={paths}
                onChange={(e) => setPaths(e.target.value)}
                rows={3}
                placeholder={'每行一个视频路径\nD:\\Videos\\match.mp4'}
                className="field mono text-[11px]"
              />
              <div className="mt-1.5 flex gap-1.5">
                <Button
                  variant="primary"
                  size="sm"
                  className="flex-1"
                  disabled={!paths.trim()}
                  onClick={async () => {
                    await importMedia(
                      paths
                        .split(/\r?\n|;/)
                        .map((s) => s.trim())
                        .filter(Boolean),
                    )
                    setPaths('')
                    setPathOpen(false)
                  }}
                >
                  导入
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setPathOpen(false)}>
                  取消
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-2.5">
        {project.media.length === 0 && (
          <div
            className={cn(
              'mt-6 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors',
              dropping ? 'border-court-500/70 bg-court-500/10' : 'border-white/10',
            )}
          >
            <Film size={26} className="mx-auto mb-2 text-ink-600" />
            <div className="text-[12.5px] text-ink-200">把视频拖到这里</div>
            <div className="mt-1 text-[11px] text-ink-500">支持 MP4 / MOV / MKV / AVI 等</div>
          </div>
        )}

        {project.media.map((m) => {
          const active = m.id === mediaId
          const a = project.analyses[m.id]
          return (
            <div
              key={m.id}
              onClick={() => selectMedia(m.id)}
              className={cn(
                'group cursor-pointer overflow-hidden rounded-xl border transition-all',
                active
                  ? 'border-court-500/50 bg-court-500/[0.08]'
                  : 'border-white/7 bg-white/[0.022] hover:border-white/16',
              )}
            >
              <div className="relative h-[86px] bg-ink-850">
                {m.poster ? (
                  <img src={api.assetUrl(m.poster)} className="h-full w-full object-cover" alt="" loading="lazy" />
                ) : (
                  <div className="grid h-full place-items-center text-ink-700">
                    <Film size={22} />
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-ink-950/85 to-transparent" />
                <div className="absolute right-1.5 bottom-1.5 flex gap-1">
                  {a?.status === 'done' && (
                    <Badge color="#38e0a2">
                      <Sparkles size={9} /> {a.rallies.length}
                    </Badge>
                  )}
                  {m.proxy_path && <Badge color="#5c9dff">预览已就绪</Badge>}
                </div>
                <div className="mono absolute bottom-1.5 left-2 text-[10.5px] text-ink-200">
                  {timecode(m.duration, false)}
                </div>
              </div>
              <div className="flex items-start gap-2 px-2.5 py-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[12px] font-medium text-ink-100" title={m.path}>
                    {m.name}
                  </div>
                  <div className="mono mt-0.5 text-[10px] text-ink-500">
                    {m.width}×{m.height} · {m.fps.toFixed(0)}fps · {bytes(m.size)}
                  </div>
                </div>
                <button
                  onClick={async (e) => {
                    e.stopPropagation()
                    const ok = await confirm({ title: `从工程中移除「${m.name}」？`, desc: '不会删除磁盘上的原文件。' })
                    if (ok) removeMedia(m.id)
                  }}
                  className="shrink-0 text-ink-600 opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose-hot"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {prepareJob && (
        <div className="border-t border-white/7 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5 text-[10.5px] text-court-300">
            <RefreshCw size={10} className="spin" />
            {prepareJob.message || '准备素材'}
          </div>
          <Progress value={prepareJob.progress} />
        </div>
      )}

      {project.media.length > 0 && (
        <div className="border-t border-white/7 px-3 py-2 text-[10.5px] text-ink-500">
          提示：直接拖入文件也能导入；右键片段可分割 / 变速
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 主页面 */

export default function StudioPage() {
  const project = useStore((s) => s.project)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const [leftTab, setLeftTab] = useState<'rally' | 'media'>('rally')
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [timelineH, setTimelineH] = useState(218)
  const dragRef = useRef<{ y: number; h: number } | null>(null)

  const jobs = useStore((s) => s.jobs)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const setCourtEditorOpen = useStore((s) => s.setCourtEditorOpen)
  const running = Object.values(jobs).some(
    (j) => j.kind === 'analyze' && (j.status === 'running' || j.status === 'queued'),
  )

  useEffect(() => {
    if (analysis?.status === 'done') setLeftTab('rally')
  }, [analysis?.status])

  // 时间线高度拖拽
  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!dragRef.current) return
      const dy = dragRef.current.y - e.clientY
      // 下限保证标尺 + 回合带 + 成片轨都能露出来（窗口特别矮时才允许继续压扁）
      const minH = Math.min(TIMELINE_MIN_H, Math.max(120, window.innerHeight - 300))
      setTimelineH(Math.max(minH, Math.min(520, dragRef.current.h + dy)))
    }
    const up = () => {
      dragRef.current = null
      document.body.classList.remove('grabbing')
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [])

  const hasMedia = !!media
  const analyzed = analysis?.status === 'done'

  return (
    <div className="flex h-full flex-col">
      {/* 顶部动作条 */}
      <div className="flex items-center gap-2 border-b border-white/7 px-3 py-2">
        <Tooltip content={leftOpen ? '收起左栏' : '展开左栏'}>
          <Button variant="ghost" size="icon" onClick={() => setLeftOpen((v) => !v)}>
            {leftOpen ? <PanelLeftClose size={14} /> : <PanelLeftOpen size={14} />}
          </Button>
        </Tooltip>

        <div className="flex-1" />

        {media && (
          <div className="mr-2 hidden items-center gap-2 text-[11px] text-ink-400 md:flex">
            <HardDrive size={11} />
            <span className="max-w-[260px] truncate" title={media.path}>
              {media.path}
            </span>
          </div>
        )}

        <Button
          variant={manualPoly ? 'subtle' : 'ghost'}
          onClick={() => setCourtEditorOpen(true)}
          disabled={!hasMedia}
          title={
            manualPoly
              ? '已手动标定球场范围，点击修改'
              : '在画面上手动标出球场范围（AI 识别不准时用这个，标一次可一直复用）'
          }
        >
          <Crosshair size={14} />
          {manualPoly ? '已标定场地' : '标定场地'}
        </Button>

        <Button
          variant={running ? 'outline' : analyzed ? 'subtle' : 'primary'}
          onClick={() => setAnalysisOpen(true)}
          disabled={!hasMedia}
        >
          <Sparkles size={14} className={running ? 'spin' : ''} />
          {running ? '分析中…' : 'AI 分析设置'}
          {analyzed && !running && (
            <span className="mono ml-1 text-[11px] text-court-300">{analysis!.rallies.length} 回合</span>
          )}
        </Button>

        <Button
          variant="primary"
          onClick={() => setExportOpen(true)}
          disabled={!project?.timeline.tracks[0]?.clips.length}
        >
          <Download size={14} />
          导出
        </Button>

        <Tooltip content={rightOpen ? '收起右栏' : '展开右栏'}>
          <Button variant="ghost" size="icon" onClick={() => setRightOpen((v) => !v)}>
            {rightOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
          </Button>
        </Tooltip>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* 左栏 */}
        <AnimatePresence initial={false}>
          {leftOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 330, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
              className="shrink-0 overflow-hidden border-r border-white/7 bg-ink-950/35"
            >
              <div className="flex h-full w-[330px] flex-col">
                <div className="flex gap-1 border-b border-white/7 px-2 py-2">
                  {(
                    [
                      ['rally', '回合', ListChecks, analysis?.rallies.length ?? 0],
                      ['media', '素材', Film, project?.media.length ?? 0],
                    ] as const
                  ).map(([id, label, Icon, n]) => (
                    <button
                      key={id}
                      onClick={() => setLeftTab(id)}
                      className={cn(
                        'relative flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-[12px] transition-colors',
                        leftTab === id ? 'text-court-200' : 'text-ink-400 hover:text-ink-100',
                      )}
                    >
                      {leftTab === id && (
                        <motion.span
                          layoutId="left-tab"
                          className="absolute inset-0 rounded-lg border border-court-500/35 bg-court-500/12"
                          transition={{ type: 'spring', stiffness: 520, damping: 36 }}
                        />
                      )}
                      <Icon size={13} className="relative" />
                      <span className="relative font-medium">{label}</span>
                      {n > 0 && (
                        <span className="mono relative rounded bg-white/10 px-1 text-[10px] text-ink-300">{n}</span>
                      )}
                    </button>
                  ))}
                </div>
                <div className="min-h-0 flex-1">
                  {leftTab === 'media' || !analyzed ? <MediaPanel /> : <RallyPanel />}
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* 中间 */}
        <div className="flex min-w-0 flex-1 flex-col">
          {!hasMedia ? (
            <div className="grid min-h-0 flex-1 place-items-center">
              <Card className="max-w-[420px] border-dashed py-2">
                <Empty
                  icon={<Film size={32} />}
                  title="先导入一段羽毛球视频"
                  desc="把文件拖进左侧素材区，或在素材面板里粘贴文件路径。导入后即可运行 AI 分析自动切出回合。"
                />
              </Card>
            </div>
          ) : (
            <>
              <Player />
              <div
                className="mx-auto -mb-0.5 h-1.5 w-full cursor-row-resize px-3"
                onPointerDown={(e) => {
                  dragRef.current = { y: e.clientY, h: timelineH }
                  document.body.classList.add('grabbing')
                }}
              >
                <div className="mx-auto h-1 w-16 rounded-full bg-white/12 transition-colors hover:bg-court-500/60" />
              </div>
              <div style={{ height: timelineH }} className="shrink-0">
                <Timeline />
              </div>
            </>
          )}
        </div>

        {/* 右栏 */}
        <AnimatePresence initial={false}>
          {rightOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 302, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
              className="shrink-0 overflow-hidden border-l border-white/7 bg-ink-950/35"
            >
              <div className="h-full w-full">
                <Inspector />
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>

      <AnalysisDialog open={analysisOpen} onClose={() => setAnalysisOpen(false)} />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />
    </div>
  )
}
