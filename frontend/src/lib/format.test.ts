import { describe, expect, it } from 'vitest'
import { formatForSpan } from './format'

const segs = [
  { start: 0, end: 60, mode: 'single' },
  { start: 60, end: 600, mode: 'doubles' },
]

describe('formatForSpan', () => {
  it('labels a rally by the mode covering most of it', () => {
    expect(formatForSpan(segs, 10, 30)).toBe('single')
    expect(formatForSpan(segs, 100, 200)).toBe('doubles')
  })

  it('uses the dominant side when a rally straddles a switch', () => {
    // 20 s of singles + 10 s of doubles -> singles
    expect(formatForSpan(segs, 40, 70)).toBe('single')
    // 5 s of singles + 25 s of doubles -> doubles
    expect(formatForSpan(segs, 55, 85)).toBe('doubles')
  })

  it('returns null without usable segments or overlap', () => {
    expect(formatForSpan(null, 0, 10)).toBeNull()
    expect(formatForSpan([], 0, 10)).toBeNull()
    expect(formatForSpan(segs, 1000, 1010)).toBeNull()
    expect(formatForSpan(segs, 10, 10)).toBeNull()
    // An unknown code is not a format label.
    expect(formatForSpan([{ start: 0, end: 10, mode: 'unknown' }], 0, 10)).toBeNull()
  })
})
