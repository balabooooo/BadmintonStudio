/**
 * Pure helpers for the label-quality audit UI (P2-d).
 *
 * Warning codes are stable ASCII identifiers (see backend annotation.label_quality);
 * translation happens at render time. Dismissals are UI-only state: they never
 * modify the saved annotation.
 */
import type { QualityItem, QualityWarning } from './types'

/** Minimal structural shape of an annotated rally needed to apply a snap. */
export interface SnapTarget {
  start: number
  end: number
}

/** Minimum post-snap rally length enforced on the client (mirrors backend guard). */
export const MIN_SNAP_LEN = 0.5

export function warningKey(index: number, w: QualityWarning): string {
  return `${index}:${w.code}:${w.side ?? ''}:${w.snap_t ?? ''}`
}

/** Warnings of one item that the user has not dismissed. */
export function visibleWarnings(item: QualityItem, dismissed: ReadonlySet<string>): QualityWarning[] {
  return item.warnings.filter((w) => !dismissed.has(warningKey(item.index, w)))
}

export function itemCounts(item: QualityItem, dismissed: ReadonlySet<string>) {
  let warn = 0
  let info = 0
  for (const w of visibleWarnings(item, dismissed)) {
    if (w.severity === 'warn') warn += 1
    else info += 1
  }
  return { warn, info }
}

/**
 * Boundary patch implied by an accept-snap action, or null when the suggestion
 * is unusable (missing target/side or would shrink below MIN_SNAP_LEN).
 */
export function snapPatch(target: SnapTarget, w: QualityWarning): { start?: number; end?: number } | null {
  if (w.snap_t == null || !Number.isFinite(w.snap_t) || !w.side) return null
  if (w.side === 'start') {
    if (w.snap_t >= target.end - MIN_SNAP_LEN) return null
    return { start: w.snap_t }
  }
  if (w.side === 'end') {
    if (w.snap_t <= target.start + MIN_SNAP_LEN) return null
    return { end: w.snap_t }
  }
  return null
}
