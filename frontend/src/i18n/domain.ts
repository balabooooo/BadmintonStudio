/** 共享枚举/代码的展示名翻译。
 *
 * 后端 stage、rally tags、机位等都是稳定 ASCII 代码，展示时统一走这里，
 * 避免各组件各维护一份 map（旧代码里 JobTray 与 AnalysisDialog 就重复过）。
 */

import { t } from './index'

function trOr(key: string, fallback: string): string {
  const v = t(key)
  return v === key ? fallback : v
}

/** 后端 JobInfo.stage -> 显示名；未知 stage 原样返回。 */
export function jobStageLabel(stage: string): string {
  if (!stage) return ''
  return trOr(`stage.${stage}`, stage)
}

/** 后端 JobInfo.kind -> 显示名；未知 kind 原样返回。 */
export function jobKindLabel(kind: string): string {
  if (!kind) return ''
  return trOr(`kind.${kind}`, kind)
}

/** rally tag 稳定码 -> 显示名；用户自定义语音口令 tag 原样返回。 */
export function tagLabel(tag: string): string {
  if (!tag) return ''
  return trOr(`tag.${tag}`, tag)
}

/** 机位代码 -> 显示名；未知值原样返回。 */
export function viewpointLabel(viewpoint: string): string {
  if (!viewpoint) return ''
  return trOr(`viewpoint.${viewpoint}`, viewpoint)
}

/** stats.match_format.format 稳定码 (single/doubles/unknown) -> 显示名；未知值原样返回。 */
export function matchFormatLabel(code: string): string {
  if (!code) return ''
  return trOr(`matchFormat.${code}`, code)
}
