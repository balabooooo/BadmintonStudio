import { useMemo } from 'react'
import { useStore } from '../store/useStore'
import { translate, type MsgKey } from './index'

/** 组件内翻译：订阅语言，切换后自动重渲染。 */
export function useT() {
  const lang = useStore((s) => s.lang)
  return useMemo(
    () =>
      (key: MsgKey | (string & {}), params?: Record<string, string | number>) =>
        translate(lang, key, params),
    [lang],
  )
}
