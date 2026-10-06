/** Tour session store: drives the guided overlay's navigation state.
 *
 * Keeps only { active, mode, index }; the step table (steps.ts) and the
 * waitFor gating of interactive steps live in the component layer. The
 * overlay registers the step total via ``setState({ stepCount })`` so
 * next/goTo can clamp at the last step; it defaults to Infinity so the
 * store stays usable standalone (tests, early wiring).
 */

import { create } from 'zustand'
import { saveTourPrefs } from './prefs'

export type TourMode = 'auto' | 'manual'

interface TourState {
  active: boolean
  mode: TourMode
  index: number
  /** Total number of tour steps; registered by TourOverlay, Infinity until then. */
  stepCount: number
  /** Enter the tour. Always activates and never reads prefs: the settings
   * re-run entry must start even when the tour was already seen. */
  start: (mode: TourMode) => void
  next: () => void
  prev: () => void
  /** Quit midway: mark seen so the tour never auto-starts again. */
  skip: () => void
  /** Finish on the last step: mark seen + completed. */
  complete: () => void
  goTo: (i: number) => void
}

export const useTourStore = create<TourState>((set, get) => ({
  active: false,
  mode: 'auto',
  index: 0,
  stepCount: Number.POSITIVE_INFINITY,

  start: (mode) => set({ active: true, mode, index: 0 }),

  next: () => {
    const { active, index, stepCount } = get()
    if (!active) return
    set({ index: Math.min(index + 1, stepCount - 1) })
  },

  prev: () => {
    const { active, index } = get()
    if (!active) return
    set({ index: Math.max(index - 1, 0) })
  },

  skip: () => {
    saveTourPrefs({ seen: true, completed: false })
    set({ active: false })
  },

  complete: () => {
    saveTourPrefs({ seen: true, completed: true })
    set({ active: false })
  },

  goTo: (i) => {
    const { active, stepCount } = get()
    if (!active) return
    set({ index: Math.min(Math.max(i, 0), stepCount - 1) })
  },
}))
