import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { useStore } from './store/useStore'
import { t } from './i18n'

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
