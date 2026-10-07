import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { useStore } from './store/useStore'
import { t } from './i18n'
import { createLogger } from './lib/logger'

const log = createLogger('global')

// Capture failures that never reach a React boundary / fetch caller, so the in-memory debug log
// (Settings → 导出调试日志) keeps a trace of what happened before the user filed the issue.
window.addEventListener('error', (e) => {
  log.error('uncaught error', { message: e.message, filename: e.filename, lineno: e.lineno, error: e.error })
})
window.addEventListener('unhandledrejection', (e) => {
  log.error('unhandled promise rejection', { reason: e.reason })
})

declare global {
  interface Window {
    /** 桌面外壳注册：把拖入文件的本地路径交给前端（零拷贝、支持大文件） */
    __bmsNativeDrop?: (paths: string[]) => void
    /** pywebview 成功接上原生拖拽后置为 true，浏览器上传兜底据此让路 */
    __bmsNativeDropReady?: boolean
  }
}

// 必须在页面脚本阶段就挂好：pywebview 的 loaded 事件可能在 React 挂载前触发。
window.__bmsNativeDrop = (paths: string[]) => {
  const s = useStore.getState()
  if (!s.project) {
    s.toast({ kind: 'warn', title: t('errors.needProject') })
    return
  }
  const list = (paths ?? []).filter(Boolean)
  if (list.length) s.importMedia(list)
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
