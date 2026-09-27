/** 语言运行时：当前语言 + localStorage 持久化 + 文档级副作用。
 *
 * 这里刻意不依赖 Zustand，避免与 store 形成循环引用；store 只单向调用本模块。
 */

export type Lang = 'zh' | 'en'

export const LANG_STORAGE_KEY = 'bms.lang'

export function normalizeLang(v: string | null | undefined): Lang {
  return v === 'en' ? 'en' : 'zh'
}

function detectInitialLang(): Lang {
  try {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem(LANG_STORAGE_KEY)
      if (saved) return normalizeLang(saved)
    }
  } catch {
    /* ignore */
  }
  return 'zh'
}

let current: Lang = detectInitialLang()

export function getLang(): Lang {
  return current
}

export function isEn(): boolean {
  return current === 'en'
}

/** 应用语言到 <html lang> 与 <title>（英文模式品牌名用 Badminton Studio）。 */
export function applyDocumentLang(): void {
  if (typeof document === 'undefined') return
  document.documentElement.lang = current === 'en' ? 'en' : 'zh-CN'
  document.title = current === 'en' ? 'Badminton Studio' : '羽毛球智能剪辑台 · Badminton Studio'
}

export function setLang(lang: Lang): void {
  current = normalizeLang(lang)
  try {
    if (typeof localStorage !== 'undefined') localStorage.setItem(LANG_STORAGE_KEY, current)
  } catch {
    /* ignore */
  }
  applyDocumentLang()
}
