import { describe, expect, it } from 'vitest'
import { FRAG as mediaFrag } from './catalog/fragments/media'
import { FRAGMENTS, zhDict, enDict } from './catalog/fragments'

describe('i18n catalog parity', () => {
  it('every media fragment key has a non-empty Chinese and English string', () => {
    for (const [key, zh, en] of mediaFrag) {
      expect(key).toBeTruthy()
      expect(zh.trim().length).toBeGreaterThan(0)
      expect(en.trim().length).toBeGreaterThan(0)
    }
  })

  it('media fragment keys are unique', () => {
    const keys = mediaFrag.map(([k]) => k)
    const dupes = keys.filter((k, i) => keys.indexOf(k) !== i)
    expect(dupes).toEqual([])
  })

  it('full fragment catalog: zh and en dictionaries have identical key sets', () => {
    const zh = zhDict()
    const en = enDict()
    expect(Object.keys(zh).sort()).toEqual(Object.keys(en).sort())
  })

  it('full fragment catalog: no empty translations', () => {
    const zh = zhDict()
    const en = enDict()
    for (const k of Object.keys(zh)) {
      expect(zh[k].trim().length).toBeGreaterThan(0)
      expect(en[k].trim().length).toBeGreaterThan(0)
    }
  })

  it('every fragment entry is a 3-tuple (key, zh, en)', () => {
    for (const entry of FRAGMENTS) {
      expect(entry.length).toBe(3)
    }
  })
})
