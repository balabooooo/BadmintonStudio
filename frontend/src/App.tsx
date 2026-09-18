import { motion } from 'motion/react'
import { Component, useEffect, type ReactNode } from 'react'
import {
  Film,
  FolderOpen,
  Settings2,
  Sparkles,
  Download,
  ChevronLeft,
  CircleDot,
  PenLine,
} from 'lucide-react'
import { cn } from './lib/format'
import { Button, ConfirmProvider, ToastHost, Tooltip } from './components/ui'
import { useStore } from './store/useStore'
import LibraryPage from './components/LibraryPage'
import StudioPage from './components/StudioPage'
import AnnotatePage from './components/AnnotatePage'
import ExportsPage from './components/ExportsPage'
import SettingsPage from './components/SettingsPage'
import JobTray from './components/JobTray'
import CourtEditor from './components/CourtEditor'

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative grid h-8 w-8 place-items-center rounded-[10px] bg-gradient-to-br from-court-300 via-court-500 to-court-700 shadow-[0_6px_20px_-6px_rgb(22_201_138/0.8)]">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="8.2" stroke="#04140e" strokeWidth="1.6" opacity="0.55" />
          <path d="M4.2 9.4h15.6M4.2 14.6h15.6M12 3.8v16.4" stroke="#04140e" strokeWidth="1.4" opacity="0.75" />
          <path d="M8.4 4.6 12 12l3.6-7.4" stroke="#04140e" strokeWidth="1.2" opacity="0.5" />
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-[13.5px] font-semibold tracking-tight text-white">羽毛球智能剪辑台</div>
        <div className="text-[10px] tracking-[0.16em] text-ink-500 uppercase">Badminton Studio</div>
      </div>
    </div>
  )
}

function NavRail() {
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  const project = useStore((s) => s.project)
  const items = [
    { id: 'library' as const, icon: FolderOpen, label: '工程库', hint: '浏览与新建工程' },
    { id: 'studio' as const, icon: Film, label: '剪辑台', hint: '预览 / 时间线 / AI 分析', disabled: !project },
    { id: 'annotate' as const, icon: PenLine, label: '标注', hint: '人工标注回合，并用标注自动优化切分参数', disabled: !project },
    { id: 'exports' as const, icon: Download, label: '导出记录', hint: '查看已导出的成片' },
    { id: 'settings' as const, icon: Settings2, label: '设置', hint: '环境检测与缓存管理' },
  ]
  return (
    <nav className="flex w-[68px] shrink-0 flex-col items-center gap-1.5 border-r border-white/6 bg-ink-950/40 py-3">
      {items.map((it) => {
        const Icon = it.icon
        const active = view === it.id
        return (
          <Tooltip key={it.id} content={`${it.label}\n${it.hint}`} side="right">
            <button
              disabled={it.disabled}
              onClick={() => setView(it.id)}
              className={cn(
                'relative flex h-[54px] w-[54px] flex-col items-center justify-center gap-1 rounded-xl transition-all duration-200',
                it.disabled
                  ? 'cursor-not-allowed text-ink-700'
                  : active
                    ? 'text-court-300'
                    : 'text-ink-400 hover:bg-white/6 hover:text-ink-100',
              )}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-xl border border-court-500/35 bg-court-500/12"
                  transition={{ type: 'spring', stiffness: 520, damping: 36 }}
                />
              )}
              <Icon size={17} className="relative" />
              <span className="relative text-[10px] font-medium">{it.label}</span>
            </button>
          </Tooltip>
        )
      })}
    </nav>
  )
}

function TitleBar() {
  const project = useStore((s) => s.project)
  const closeProject = useStore((s) => s.closeProject)
  const view = useStore((s) => s.view)
  const env = useStore((s) => s.env)
  const jobs = useStore((s) => s.jobs)
  const running = Object.values(jobs).filter((j) => j.status === 'running' || j.status === 'queued')

  return (
    <header className="flex h-[54px] shrink-0 items-center gap-3 border-b border-white/7 bg-ink-950/55 px-4 backdrop-blur-xl">
      {(view === 'studio' || view === 'annotate') && project ? (
        <Button variant="ghost" size="sm" onClick={closeProject} className="-ml-1">
          <ChevronLeft size={14} />
          工程库
        </Button>
      ) : (
        <Logo />
      )}
      {(view === 'studio' || view === 'annotate') && project && (
        <div className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink-200">
          <span className="text-ink-500">/</span> <span className="text-white">{project.name}</span>
        </div>
      )}
      {view !== 'studio' && view !== 'annotate' && <div className="flex-1" />}

      {running.length > 0 && (
        <div className="flex items-center gap-2 rounded-full border border-court-500/30 bg-court-500/10 px-2.5 py-1">
          <CircleDot size={11} className="pulse-ring rounded-full text-court-300" />
          <span className="text-[11px] text-court-300">
            {running[0].title || running[0].kind} {Math.round((running[0].progress || 0) * 100)}%
          </span>
        </div>
      )}

      {env && (
        <Tooltip
          content={
            <div className="space-y-0.5">
              <div>Python {env.python}</div>
              <div>{env.platform}</div>
              <div>
                GPU：{env.gpu?.available ? env.gpu.name : '不可用（将使用 CPU）'}
              </div>
              <div>硬件编码：{env.caps?.nvenc_h264 ? 'NVENC 可用' : '仅 CPU'}</div>
              <div className="max-w-[420px] break-all text-ink-400">{env.ffmpeg || env.ffmpeg_error}</div>
            </div>
          }
        >
          <div className="flex items-center gap-1.5 rounded-full border border-white/8 bg-white/4 px-2.5 py-1 text-[10.5px] text-ink-300">
            <span
              className={cn(
                'h-1.5 w-1.5 rounded-full',
                env.gpu?.available ? 'bg-court-400' : 'bg-amber-glow',
              )}
            />
            {env.gpu?.available ? 'GPU 加速' : 'CPU'}
          </div>
        </Tooltip>
      )}
      <JobTray />
    </header>
  )
}

/** 页面级错误边界：任何一处渲染异常都只影响当前页，不至于整屏空白。 */
class ErrorBoundary extends Component<{ children: ReactNode; view: string }, { err: Error | null }> {
  state = { err: null as Error | null }

  static getDerivedStateFromError(err: Error) {
    return { err }
  }

  componentDidUpdate(prev: { view: string }) {
    if (prev.view !== this.props.view && this.state.err) this.setState({ err: null })
  }

  render() {
    if (this.state.err) {
      return (
        <div className="grid h-full place-items-center p-8">
          <div className="panel max-w-[640px] p-6">
            <div className="text-[14px] font-semibold text-rose-hot">界面渲染出错</div>
            <div className="mt-1 text-[12px] text-ink-300">
              这一页崩了，其他功能不受影响。可以先切到别的页面，或刷新重试。
            </div>
            <pre className="mono mt-3 max-h-[280px] overflow-auto rounded-lg bg-black/40 p-3 text-[11px] whitespace-pre-wrap text-ink-400">
              {String(this.state.err?.stack || this.state.err)}
            </pre>
            <button
              onClick={() => location.reload()}
              className="mt-3 rounded-lg bg-white/10 px-3 py-1.5 text-[12px] text-ink-100 hover:bg-white/16"
            >
              刷新页面
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

function Splash() {  return (
    <div className="flex h-full flex-col items-center justify-center gap-5">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-court-300 via-court-500 to-court-700 shadow-[0_20px_60px_-20px_rgb(22_201_138/0.9)]"
      >
        <Sparkles size={28} className="text-ink-950" />
      </motion.div>
      <div className="text-center">
        <div className="text-[15px] font-semibold text-white">羽毛球智能剪辑台</div>
        <div className="mt-1 text-[12px] text-ink-400">正在连接本地分析引擎…</div>
      </div>
    </div>
  )
}

export default function App() {
  const booted = useStore((s) => s.booted)
  const view = useStore((s) => s.view)
  const project = useStore((s) => s.project)
  const bootstrap = useStore((s) => s.bootstrap)
  const courtEditorOpen = useStore((s) => s.courtEditorOpen)
  const setCourtEditorOpen = useStore((s) => s.setCourtEditorOpen)

  useEffect(() => {
    bootstrap()
  }, [bootstrap])

  // 全局快捷键
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      const typing = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      const s = useStore.getState()
      if (typing) return
      // 这些快捷键只服务于剪辑台。标注页有自己的保存/删除键，选中过片段后
      // 在标注页按 S / Delete / 空格会把时间线片段悄悄切掉或删掉。
      if (s.view !== 'studio') return
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault()
        s.undo()
      } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
        e.preventDefault()
        s.redo()
      } else if (e.code === 'Space') {
        e.preventDefault()
        s.setPlaying(!s.playing)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        s.seek(s.currentTime - (e.shiftKey ? 5 : 1 / 30))
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        s.seek(s.currentTime + (e.shiftKey ? 5 : 1 / 30))
      } else if (e.key === 'Home') {
        s.seek(0)
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && s.selectedClipId) {
        e.preventDefault()
        s.removeClip(s.selectedClipId)
      } else if (e.key.toLowerCase() === 's' && s.selectedClipId) {
        e.preventDefault()
        // 源片模式下播放头读数是原片时间，必须换算，否则会切在完全不相干的位置
        if (s.previewMode === 'source') s.splitClipAtSourceTime(s.selectedClipId, s.currentTime)
        else s.splitClipAt(s.selectedClipId, s.currentTime)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <ConfirmProvider>
      <div className="app-bg" />
      <div className="flex h-full flex-col">
        <TitleBar />
        <div className="flex min-h-0 flex-1">
          <NavRail />
          <main className="relative min-w-0 flex-1 overflow-hidden">
            {/* 用 CSS 过渡而不是 AnimatePresence(mode="wait")：
                后者要等退出动画结束才挂载新页面，动画一旦被节流就会整屏空白。 */}
            <div key={view} className="h-full anim-in">
              <ErrorBoundary view={view}>
                {!booted ? (
                  <Splash />
                ) : view === 'library' ? (
                  <LibraryPage />
                ) : view === 'studio' && project ? (
                  <StudioPage />
                ) : view === 'annotate' && project ? (
                  <AnnotatePage />
                ) : view === 'exports' ? (
                  <ExportsPage />
                ) : view === 'settings' ? (
                  <SettingsPage />
                ) : (
                  <LibraryPage />
                )}
              </ErrorBoundary>
            </div>
          </main>
        </div>
      </div>
      {/* 场地标定是跨页面的：预览角标和「AI 分析」里都能打开，所以挂在应用根上 */}
      <CourtEditor
        open={courtEditorOpen}
        onClose={() => setCourtEditorOpen(false)}
      />
      <ToastHost />
    </ConfirmProvider>
  )
}
