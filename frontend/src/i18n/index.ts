/** i18n 入口：按当前语言取文案。
 *
 * 组件内优先用 `useT()`（订阅语言变化，切换后自动重渲染）；
 * store action / 非组件逻辑用全局 `t()`（读取当前语言）。
 */

import { zh } from './catalog/zh'
import { en } from './catalog/en'
import { getLang, type Lang } from './runtime'

export type { MsgKey } from './catalog/zh'
import type { MsgKey } from './catalog/zh'

type Params = Record<string, string | number>

const dicts: Record<Lang, Record<string, string>> = {
  zh: zh as Record<string, string>,
  en: en as Record<string, string>,
}

export function translate(lang: Lang, key: string, params?: Params): string {
  const dict = dicts[lang] ?? dicts.zh
  let s = dict[key] ?? dicts.zh[key] ?? key
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      s = s.split(`{${k}}`).join(String(v))
    }
  }
  return s
}

/** 当前语言的翻译（非响应式，调用时读取）。 */
export function t(key: MsgKey | (string & {}), params?: Params): string {
  return translate(getLang(), key, params)
}

export { getLang, setLang, isEn, applyDocumentLang, normalizeLang, LANG_STORAGE_KEY } from './runtime'
export type { Lang } from './runtime'
