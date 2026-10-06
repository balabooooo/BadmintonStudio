import { describe, expect, it } from 'vitest'
import {
  MIN_SNAP_LEN,
  itemCounts,
  snapPatch,
  visibleWarnings,
  warningKey,
} from './annotationQuality'
import type { QualityItem } from './types'

const item: QualityItem = {
  index: 3,
  start: 10,
  end: 20,
  severity: 'warn',
  warnings: [
    { code: 'boundary_off_quiet', severity: 'warn', side: 'start', distance: 2.4, snap_t: 7.8, snap_kind: 'quiet' },
    { code: 'boundary_no_hit', severity: 'info', side: 'end', distance: 3.1 },
  ],
}

describe('annotationQuality helpers', () => {
  it('builds stable warning keys', () => {
    expect(warningKey(3, item.warnings[0])).toBe('3:boundary_off_quiet:start:7.8')
    expect(warningKey(3, item.warnings[1])).toBe('3:boundary_no_hit:end:')
  })

  it('hides dismissed warnings and counts the rest by severity', () => {
    expect(visibleWarnings(item, new Set())).toHaveLength(2)
    expect(itemCounts(item, new Set())).toEqual({ warn: 1, info: 1 })

    const dismissed = new Set([warningKey(3, item.warnings[0])])
    expect(visibleWarnings(item, dismissed).map((w) => w.code)).toEqual(['boundary_no_hit'])
    expect(itemCounts(item, dismissed)).toEqual({ warn: 0, info: 1 })
  })

  it('applies start/end snaps', () => {
    expect(snapPatch({ start: 10, end: 20 }, item.warnings[0])).toEqual({ start: 7.8 })
    const endSnap = { code: 'boundary_off_quiet', severity: 'warn', side: 'end', snap_t: 22.5 } as const
    expect(snapPatch({ start: 10, end: 20 }, endSnap)).toEqual({ end: 22.5 })
  })

  it('rejects unusable snaps', () => {
    expect(snapPatch({ start: 10, end: 20 }, { code: 'x', severity: 'warn' })).toBeNull()
    // Start snap would make the rally shorter than the minimum length.
    const tooClose = { code: 'boundary_off_quiet', severity: 'warn', side: 'start', snap_t: 19.8 } as const
    expect(snapPatch({ start: 10, end: 20 }, tooClose)).toBeNull()
    expect(MIN_SNAP_LEN).toBeGreaterThan(0)
  })
})
