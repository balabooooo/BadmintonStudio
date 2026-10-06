import { beforeEach, describe, expect, it } from 'vitest'
import { TOUR_PREFS_KEY, isFirstRun, loadTourPrefs, saveTourPrefs } from './prefs'
import { useTourStore } from './tourStore'

// Reset the singleton store to a known baseline between tests.
function resetTourStore() {
  useTourStore.setState({
    active: false,
    mode: 'auto',
    index: 0,
    stepCount: Number.POSITIVE_INFINITY,
  })
}

describe('tour prefs', () => {
  beforeEach(() => {
    localStorage.removeItem(TOUR_PREFS_KEY)
  })

  it('isFirstRun is true when no prefs are stored', () => {
    expect(isFirstRun()).toBe(true)
    expect(loadTourPrefs()).toBeNull()
  })

  it('isFirstRun is false after the tour is marked seen', () => {
    saveTourPrefs({ seen: true, completed: false })
    expect(isFirstRun()).toBe(false)
    expect(loadTourPrefs()).toEqual({ seen: true, completed: false })
  })

  it('loadTourPrefs returns null on corrupted JSON or invalid shape', () => {
    localStorage.setItem(TOUR_PREFS_KEY, '{not json')
    expect(loadTourPrefs()).toBeNull()
    expect(isFirstRun()).toBe(true)

    localStorage.setItem(TOUR_PREFS_KEY, JSON.stringify({ seen: 'yes' }))
    expect(loadTourPrefs()).toBeNull()
  })
})

describe('tour store', () => {
  beforeEach(() => {
    localStorage.removeItem(TOUR_PREFS_KEY)
    resetTourStore()
  })

  it('start activates the tour at index 0 and records the mode', () => {
    useTourStore.getState().start('auto')
    const s = useTourStore.getState()
    expect(s.active).toBe(true)
    expect(s.index).toBe(0)
    expect(s.mode).toBe('auto')
  })

  it('start ignores stored prefs (the settings re-run entry always activates)', () => {
    saveTourPrefs({ seen: true, completed: true })
    useTourStore.getState().start('manual')
    expect(useTourStore.getState().active).toBe(true)
    expect(useTourStore.getState().mode).toBe('manual')
  })

  it('next/prev clamp at the step bounds', () => {
    useTourStore.setState({ stepCount: 3 })
    useTourStore.getState().start('manual')
    useTourStore.getState().prev()
    expect(useTourStore.getState().index).toBe(0)
    useTourStore.getState().next()
    useTourStore.getState().next()
    expect(useTourStore.getState().index).toBe(2)
    useTourStore.getState().next()
    expect(useTourStore.getState().index).toBe(2)
  })

  it('goTo clamps into the valid index range', () => {
    useTourStore.setState({ stepCount: 3 })
    useTourStore.getState().start('manual')
    useTourStore.getState().goTo(1)
    expect(useTourStore.getState().index).toBe(1)
    useTourStore.getState().goTo(-2)
    expect(useTourStore.getState().index).toBe(0)
    useTourStore.getState().goTo(99)
    expect(useTourStore.getState().index).toBe(2)
  })

  it('next/prev/goTo are no-ops while the tour is inactive', () => {
    useTourStore.getState().start('manual')
    useTourStore.getState().goTo(1)
    useTourStore.getState().skip()
    expect(useTourStore.getState().active).toBe(false)
    const index = useTourStore.getState().index
    useTourStore.getState().next()
    useTourStore.getState().prev()
    useTourStore.getState().goTo(5)
    expect(useTourStore.getState().index).toBe(index)
  })

  it('skip marks the tour seen and deactivates without completed', () => {
    useTourStore.getState().start('auto')
    useTourStore.getState().skip()
    expect(useTourStore.getState().active).toBe(false)
    expect(loadTourPrefs()).toEqual({ seen: true, completed: false })
  })

  it('complete marks the tour seen and completed, then deactivates', () => {
    useTourStore.getState().start('manual')
    useTourStore.getState().complete()
    expect(useTourStore.getState().active).toBe(false)
    expect(loadTourPrefs()).toEqual({ seen: true, completed: true })
  })
})
