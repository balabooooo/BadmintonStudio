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
  Crosshair,
} from 'lucide-react'
import { Button, Card, Empty, Tooltip } from './ui'
import Player from './Player'
import Timeline, { TIMELINE_MIN_H } from './Timeline'
import RallyPanel from './RallyPanel'
import Inspector from './Inspector'
import AnalysisDialog from './AnalysisDialog'
import ExportDialog from './ExportDialog'
import ImportVideoButton from './ImportVideoButton'
import MediaChip from './MediaChip'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

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
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const importFiles = useStore((s) => s.importFiles)
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
          <div className="mr-2 hidden md:flex">
            <MediaChip />
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
        {/* 左栏：回合列表（素材管理已收进全局素材弹窗） */}
        <AnimatePresence initial={false}>
          {leftOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 330, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
              className="shrink-0 overflow-hidden border-r border-white/7 bg-ink-950/35"
            >
              <div className="h-full w-[330px]">
                <RallyPanel />
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
