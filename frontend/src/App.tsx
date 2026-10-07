import { motion } from 'motion/react'
import { Component, useEffect, type ErrorInfo, type ReactNode } from 'react'
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
import { useT } from './i18n/useT'
import { t as tr } from './i18n'
import LibraryPage from './components/LibraryPage'
import StudioPage from './components/StudioPage'
import AnnotatePage from './components/AnnotatePage'
import ExportsPage from './components/ExportsPage'
import SettingsPage from './components/SettingsPage'
import JobTray from './components/JobTray'
import CourtEditor from './components/CourtEditor'
import MediaPickerDialog from './components/MediaPickerDialog'
import TourOverlay from './tour/TourOverlay'
import { isFirstRun } from './tour/prefs'
import { useTourStore } from './tour/tourStore'
import { createLogger } from './lib/logger'

function Logo() {
  const t = useT()
  const lang = useStore((s) => s.lang)
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
        <div className="text-[13.5px] font-semibold tracking-tight text-white">{t('app.name')}</div>
        {lang === 'zh' && (
          <div className="text-[10px] tracking-[0.16em] text-ink-500 uppercase">Badminton Studio</div>
        )}
      </div>
    </div>
  )
}

function NavRail() {
  const t = useT()
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  const project = useStore((s) => s.project)
  const items = [
    { id: 'library' as const, icon: FolderOpen, label: t('nav.library'), hint: t('nav.libraryHint') },
    { id: 'studio' as const, icon: Film, label: t('nav.studio'), hint: t('nav.studioHint'), disabled: !project },
    { id: 'annotate' as const, icon: PenLine, label: t('nav.annotate'), hint: t('nav.annotateHint'), disabled: !project },
    { id: 'exports' as const, icon: Download, label: t('nav.exports'), hint: t('nav.exportsHint') },
    { id: 'settings' as const, icon: Settings2, label: t('nav.settings'), hint: t('nav.settingsHint') },
  ]
  return (
    <nav data-tour="nav-rail" className="flex w-[68px] shrink-0 flex-col items-center gap-1.5 border-r border-white/6 bg-ink-950/40 py-3">
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
  const t = useT()
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
          {t('nav.backToLibrary')}
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
              <div>{t('app.gpu.accel')}：{env.gpu?.available ? env.gpu.name : t('app.gpu.unavailable')}</div>
              <div>{t('app.hwEncoding')}：{env.caps?.nvenc_h264 ? t('app.hw.nvenc') : t('app.hw.cpuOnly')}</div>
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
            {env.gpu?.available ? t('app.gpu.accel') : t('app.gpu.cpu')}
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
    return { err } as { err: Error | null }
  }

  componentDidUpdate(prev: { view: string }) {
    if (prev.view !== this.props.view && this.state.err) this.setState({ err: null })
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    createLogger('error-boundary').error(`render error in view=${this.props.view}`, {
      error: err,
      componentStack: info.componentStack,
    })
  }

  render() {
    if (this.state.err) {
      return (
        <div className="grid h-full place-items-center p-8">
          <div className="panel max-w-[640px] p-6">
            <div className="text-[14px] font-semibold text-rose-hot">{tr('app.error.title')}</div>
            <div className="mt-1 text-[12px] text-ink-300">{tr('app.error.desc')}</div>
            <pre className="mono mt-3 max-h-[280px] overflow-auto rounded-lg bg-black/40 p-3 text-[11px] whitespace-pre-wrap text-ink-400">
              {String(this.state.err?.stack || this.state.err)}
            </pre>
            <button
              onClick={() => location.reload()}
              className="mt-3 rounded-lg bg-white/10 px-3 py-1.5 text-[12px] text-ink-100 hover:bg-white/16"
            >
              {tr('app.error.reload')}
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

function Splash() {
  const t = useT()
  return (
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
        <div className="text-[15px] font-semibold text-white">{t('app.name')}</div>
        <div className="mt-1 text-[12px] text-ink-400">{t('app.splash.connecting')}</div>
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

  // First-run guided tour: give the first screen a beat to finish mounting
  // before the overlay starts measuring anchors.
  useEffect(() => {
    if (booted && isFirstRun()) {
      const t = setTimeout(() => useTourStore.getState().start('auto'), 600)
      return () => clearTimeout(t)
    }
  }, [booted])

  // 往窗口里拖文件时别让浏览器直接打开文件把界面顶掉；剪辑台会自己接管导入。
  useEffect(() => {
    const guard = (e: DragEvent) => {
      if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault()
    }
    window.addEventListener('dragover', guard)
    window.addEventListener('drop', guard)
    return () => {
      window.removeEventListener('dragover', guard)
      window.removeEventListener('drop', guard)
    }
  }, [])

  // 全局快捷键
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      const typing =
        !!el &&
        (el.tagName === 'INPUT' ||
          el.tagName === 'TEXTAREA' ||
          el.isContentEditable ||
          !!el.closest('[contenteditable="true"]'))
      const s = useStore.getState()
      if (typing) return
      // 这些快捷键只服务于剪辑台。标注页有自己的保存/删除键，选中过片段后
      // 在标注页按 S / Delete / 空格会把时间线片段悄悄切掉或删掉。
      if (s.view !== 'studio') return
      // 场地标定窗口 / 弹窗 / 右键菜单打开时，别让 S、Delete、空格穿透到底层时间线
      if (s.courtEditorOpen) return
      if (el?.closest('[role="dialog"][aria-modal="true"], [role="menu"]')) return
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault()
        s.undo()
      } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
        e.preventDefault()
        s.redo()
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
        e.preventDefault()
        s.selectAllClips()
      } else if (e.code === 'Space') {
        // 焦点在按钮上时空格是「激活按钮」；别再顺带切换播放，否则一次触发两次
        if (el?.closest('button')) return
        e.preventDefault()
        s.setPlaying(!s.playing)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        s.seek(s.currentTime - (e.shiftKey ? 5 : s.frameStep()))
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        s.seek(s.currentTime + (e.shiftKey ? 5 : s.frameStep()))
      } else if (e.key === 'Home') {
        e.preventDefault()
        s.seek(0)
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && s.selectedClipIds.length) {
        e.preventDefault()
        s.removeClips(s.selectedClipIds)
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === 's' && s.selectedClipId) {
        // 不拦 Ctrl+S：那是浏览器「保存网页」，不该顺带去分割片段
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
      {/* 素材选择弹窗是跨页面的：剪辑台 / 标注页 / AI 分析设置共用同一个实例 */}
      <MediaPickerDialog />
      <ToastHost />
      <TourOverlay />
    </ConfirmProvider>
  )
}
