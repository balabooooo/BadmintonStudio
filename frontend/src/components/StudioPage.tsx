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
  Trash2,
  RefreshCw,
  HardDrive,
  Crosshair,
  Check,
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
import ImportVideoButton from './ImportVideoButton'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

/* ------------------------------------------------------------------ 素材列表 */

function MediaPanel({ dropping }: { dropping: boolean }) {
  const tr = useT()
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const selectMedia = useStore((s) => s.selectMedia)
  const removeMedia = useStore((s) => s.removeMedia)
  const removeMediaBulk = useStore((s) => s.removeMediaBulk)
  const runAnalysisBatch = useStore((s) => s.runAnalysisBatch)
  const jobs = useStore((s) => s.jobs)
  const confirm = useConfirm()
  const [selectMode, setSelectMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())

  // 切工程时清空选择，免得把上一个工程的 id 带过来
  useEffect(() => {
    setSelected(new Set())
    setSelectMode(false)
  }, [project?.id])

  if (!project) return null

  // 正在为哪些素材生成预览（后端按 media_id 标记 prepare 任务）
  const activePrepare = Object.values(jobs)
    .filter((j) => j.kind === 'prepare' && (j.status === 'running' || j.status === 'queued'))
    .sort((a, b) => a.created_at - b.created_at)
  const jobFor = (mid: string) => activePrepare.filter((j) => j.media_id === mid).at(-1)

  // 正在分析哪些素材（批量分析时会有多个排队）
  const activeAnalyze = Object.values(jobs)
    .filter((j) => j.kind === 'analyze' && (j.status === 'running' || j.status === 'queued'))
    .sort((a, b) => a.created_at - b.created_at)
  const analyzeFor = (mid: string) => activeAnalyze.filter((j) => j.media_id === mid).at(-1)

  const allSelected = project.media.length > 0 && selected.size === project.media.length
  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const toggleAll = () =>
    setSelected(allSelected ? new Set() : new Set(project.media.map((m) => m.id)))
  const doBulkDelete = async () => {
    const ids = [...selected]
    if (!ids.length) return
    const ok = await confirm({
      title: tr('studio.confirmRemoveSelected', { n: ids.length }),
      desc: tr('studio.confirmRemoveSelectedDesc'),
      danger: true,
    })
    if (!ok) return
    await removeMediaBulk(ids)
    setSelected(new Set())
    setSelectMode(false)
  }

  const doBulkAnalyze = async () => {
    const ids = [...selected]
    if (!ids.length) return
    await runAnalysisBatch(ids)
    setSelected(new Set())
    setSelectMode(false)
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-white/7 px-3 py-2.5">
        <span className="flex-1 text-[11px] font-semibold tracking-[0.14em] text-ink-400 uppercase">
          {tr('studio.mediaWithCount', { n: project.media.length })}
        </span>
        {selectMode ? (
          <>
            <button
              onClick={toggleAll}
              className="text-[10.5px] text-ink-400 transition-colors hover:text-ink-100 active:opacity-70"
            >
              {allSelected ? tr('studio.deselectAll') : tr('studio.selectAll')}
            </button>
            <span className="text-[10.5px] text-ink-500">{tr('studio.selectedCount', { n: selected.size })}</span>
            <Button variant="primary" size="sm" disabled={!selected.size} onClick={() => void doBulkAnalyze()}>
              <Sparkles size={12} />
              {tr('studio.analyzeSelected')}
            </Button>
            <Button variant="danger" size="sm" disabled={!selected.size} onClick={() => void doBulkDelete()}>
              <Trash2 size={12} />
              {tr('common.delete')}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setSelectMode(false)
                setSelected(new Set())
              }}
            >
              {tr('studio.done')}
            </Button>
          </>
        ) : (
          <>
            <ImportVideoButton size="sm" />
            <Tooltip content={tr('studio.batchSelect')}>
              <Button variant="ghost" size="icon" onClick={() => setSelectMode(true)}>
                <ListChecks size={13} />
              </Button>
            </Tooltip>
          </>
        )}
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-2.5">
        {project.media.length === 0 && (
          <div
            className={cn(
              'mt-6 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors',
              dropping ? 'border-court-500/70 bg-court-500/10' : 'border-white/10',
            )}
          >
            <Film size={26} className="mx-auto mb-2 text-ink-600" />
            <div className="text-[12.5px] text-ink-200">{tr('studio.dropVideoHere')}</div>
            <div className="mt-1 text-[11px] text-ink-500">{tr('studio.supportedFormats')}</div>
            <div className="mt-3 flex flex-wrap justify-center gap-1.5">
              <ImportVideoButton size="sm" />
              <ImportVideoButton mode="folder" size="sm" variant="subtle" />
            </div>
          </div>
        )}

        {project.media.map((m) => {
          const active = m.id === mediaId
          const a = project.analyses[m.id]
          const pj = jobFor(m.id)
          const aj = analyzeFor(m.id)
          return (
            <div
              key={m.id}
              onClick={() => (selectMode ? toggle(m.id) : selectMedia(m.id))}
              className={cn(
                'group cursor-pointer overflow-hidden rounded-xl border transition-all',
                selectMode && selected.has(m.id)
                  ? 'border-court-500/60 bg-court-500/[0.1]'
                  : active
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
                {/* 还在生成预览时盖住缩略图，显示进度而不是一个空占位 */}
                {!m.poster && pj && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 bg-ink-950/72 px-3">
                    <RefreshCw size={14} className="spin text-court-300" />
                    <Progress value={pj.progress} className="w-full" />
                    <div className="w-full truncate text-center text-[10px] text-ink-300">
                      {pj.stage === 'queued' ? tr('studio.queued') : `${pj.message || tr('studio.generatingPreview')} ${Math.round(pj.progress * 100)}%`}
                    </div>
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-ink-950/85 to-transparent" />
                {/* 勾选框放在进度/渐变之后，生成预览时也不会被盖住 */}
                {selectMode && (
                  <div
                    className={cn(
                      'absolute top-1.5 left-1.5 grid h-5 w-5 place-items-center rounded-md border transition-colors',
                      selected.has(m.id)
                        ? 'border-court-400 bg-court-500 text-ink-950'
                        : 'border-white/45 bg-ink-950/60',
                    )}
                  >
                    {selected.has(m.id) && <Check size={13} />}
                  </div>
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
              <div className="flex items-start gap-2 px-2.5 py-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[12px] font-medium text-ink-100" title={m.path}>
                    {m.name}
                  </div>
                  <div className="mono mt-0.5 text-[10px] text-ink-500">
                    {m.width}×{m.height} · {m.fps.toFixed(0)}fps · {bytes(m.size)}
                  </div>
                </div>
                {!selectMode && (
                  <button
                    onClick={async (e) => {
                      e.stopPropagation()
                      const ok = await confirm({ title: tr('studio.confirmRemoveMedia', { name: m.name }), desc: tr('studio.confirmRemoveMediaDesc') })
                      if (ok) removeMedia(m.id)
                    }}
                    aria-label={tr('common.delete')}
                    title={tr('common.delete')}
                    className="shrink-0 text-ink-600 opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose-hot"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {activePrepare.length > 0 && (
        <div className="border-t border-white/7 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5 text-[10.5px] text-court-300">
            <RefreshCw size={10} className="spin shrink-0" />
            <span className="min-w-0 flex-1 truncate">
              {activePrepare[0].stage === 'queued' ? tr('studio.queued') : activePrepare[0].message || tr('studio.preparingMedia')}
            </span>
            {activePrepare.length > 1 && (
              <span className="shrink-0 text-ink-400">{tr('studio.pendingItems', { n: activePrepare.length })}</span>
            )}
          </div>
          <Progress value={activePrepare[0].progress} />
        </div>
      )}

      {project.media.length > 0 && (
        <div className="border-t border-white/7 px-3 py-2 text-[10.5px] text-ink-500">
          {tr('studio.mediaHint')}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 主页面 */

const VIDEO_EXT = /\.(mp4|mov|mkv|avi|flv|wmv|m4v|webm|ts|mpg|mpeg)$/i

function isVideoFile(f: File) {
  return f.type.startsWith('video/') || VIDEO_EXT.test(f.name)
}

/** 拖进来的目录会被浏览器伪装成「无类型、无扩展名」的 File。 */
function isLikelyFolder(f: File) {
  return f.type === '' && !/\.[a-z0-9]+$/i.test(f.name)
}

export default function StudioPage() {
  const tr = useT()
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const importFiles = useStore((s) => s.importFiles)
  const [leftTab, setLeftTab] = useState<'rally' | 'media'>('rally')
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [analysisOpen, setAnalysisOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [timelineH, setTimelineH] = useState(218)
  const [dragging, setDragging] = useState(false)
  const dragRef = useRef<{ y: number; h: number } | null>(null)

  const jobs = useStore((s) => s.jobs)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const setCourtEditorOpen = useStore((s) => s.setCourtEditorOpen)
  const running = Object.values(jobs).some(
    (j) => j.kind === 'analyze' && (j.status === 'running' || j.status === 'queued'),
  )

  // 页签自动跟随当前素材：已分析 → 回合页，未分析 → 素材页。
  // 只依赖 status/mediaId，用户手动切页签不会被反复覆盖，未分析时也能切回「回合」看空状态。
  useEffect(() => {
    setLeftTab(analysis?.status === 'done' ? 'rally' : 'media')
  }, [analysis?.status, mediaId])

  // 全屏拖拽导入：往窗口任意位置拖视频都能接住，顺带防止浏览器直接用文件替换页面。
  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files')
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      setDragging(true)
    }
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
    }
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return
      if (e.relatedTarget === null) setDragging(false)
    }
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      setDragging(false)
      // 原生桌面窗口：pywebview 会把拖入文件的完整本地路径推给 __bmsNativeDrop，
      // 由它按路径导入（零拷贝，大文件也能拖）。这里不上传，避免重复导入一份。
      if ((window as unknown as { __bmsNativeDropReady?: boolean }).__bmsNativeDropReady) return
      const s = useStore.getState()
      if (!s.project) {
        s.toast({ kind: 'warn', title: tr('studio.needProject') })
        return
      }
      const all = Array.from(e.dataTransfer?.files ?? [])
      const files = all.filter(isVideoFile)
      if (!files.length) {
        // 目录不是文件，Chromium 里 dataTransfer.files 可能是空的，得看 items 里的 entry。
        const items = Array.from(e.dataTransfer?.items ?? [])
        const dirDropped = items.some((it) => {
          const entry = (
            it as DataTransferItem & { webkitGetAsEntry?: () => { isDirectory: boolean } | null }
          ).webkitGetAsEntry?.()
          return !!entry?.isDirectory
        })
        if (dirDropped || all.some(isLikelyFolder)) {
          s.toast({
            kind: 'warn',
            title: tr('studio.folderDroppedTitle'),
            detail: tr('studio.folderDroppedDetail'),
          })
        } else {
          s.toast({ kind: 'warn', title: tr('studio.noImportableVideo') })
        }
        return
      }
      // 拖进来的文件只能走上传（浏览器拿不到本地路径），大文件会白白复制一份，劝退。
      const MAX_DROP = 2 * 1024 ** 3
      const small = files.filter((f) => f.size <= MAX_DROP)
      const huge = files.filter((f) => f.size > MAX_DROP)
      if (huge.length) {
        s.toast({
          kind: 'warn',
          title: tr('studio.skippedHuge', { n: huge.length }),
          detail: tr('studio.skippedHugeDetail'),
        })
      }
      if (small.length) importFiles(small)
    }
    window.addEventListener('dragenter', onEnter)
    window.addEventListener('dragover', onOver)
    window.addEventListener('dragleave', onLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragenter', onEnter)
      window.removeEventListener('dragover', onOver)
      window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('drop', onDrop)
    }
  }, [importFiles, tr])

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
        <Tooltip content={leftOpen ? tr('studio.collapseLeft') : tr('studio.expandLeft')}>
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
              ? tr('studio.courtCalibrated')
              : tr('studio.courtCalibrateHint')
          }
        >
          <Crosshair size={14} />
          {manualPoly ? tr('studio.courtCalibratedShort') : tr('studio.calibrateCourt')}
        </Button>

        <Button
          variant={running ? 'outline' : analyzed ? 'subtle' : 'primary'}
          onClick={() => setAnalysisOpen(true)}
          disabled={!hasMedia}
        >
          <Sparkles size={14} className={running ? 'spin' : ''} />
          {running ? tr('studio.analyzing') : tr('studio.analysisSettings')}
          {analyzed && !running && (
            <span className="mono ml-1 text-[11px] text-court-300">{tr('studio.rallyCount', { n: analysis!.rallies.length })}</span>
          )}
        </Button>

        <Button
          variant="primary"
          onClick={() => setExportOpen(true)}
          disabled={!project?.timeline.tracks[0]?.clips.length}
        >
          <Download size={14} />
          {tr('studio.export')}
        </Button>

        <Tooltip content={rightOpen ? tr('studio.collapseRight') : tr('studio.expandRight')}>
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
                      ['rally', tr('studio.tabRally'), ListChecks, analysis?.rallies.length ?? 0],
                      ['media', tr('studio.tabMedia'), Film, project?.media.length ?? 0],
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
                  {leftTab === 'media' ? <MediaPanel dropping={dragging} /> : <RallyPanel />}
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
                  title={tr('studio.emptyTitle')}
                  desc={tr('studio.emptyDesc')}
                  action={
                    <div className="flex flex-wrap justify-center gap-2">
                      <ImportVideoButton size="md" />
                      <ImportVideoButton mode="folder" size="md" variant="outline" />
                    </div>
                  }
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

      {/* 拖拽导入遮罩 */}
      <AnimatePresence>
        {dragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="pointer-events-none fixed inset-0 z-[999] grid place-items-center bg-ink-950/70 backdrop-blur-sm"
          >
            <div className="rounded-2xl border-2 border-dashed border-court-500/70 bg-ink-900/85 px-10 py-8 text-center">
              <Film size={34} className="mx-auto mb-2 text-court-300" />
              <div className="text-[15px] font-medium text-white">{tr('studio.releaseToImport')}</div>
              <div className="mt-1 text-[12px] text-ink-400">{tr('studio.releaseToImportDesc')}</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnalysisDialog open={analysisOpen} onClose={() => setAnalysisOpen(false)} />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} />
    </div>
  )
}
