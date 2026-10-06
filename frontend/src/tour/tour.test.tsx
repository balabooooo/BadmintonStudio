import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, afterEach, vi } from 'vitest'
import { TOUR_PREFS_KEY, isFirstRun, loadTourPrefs, saveTourPrefs } from './prefs'
import { useTourStore } from './tourStore'
import TourOverlay, { computePlacement } from './TourOverlay'
import { buildSteps } from './steps'
import { api } from '../lib/api'
import { useStore } from '../store/useStore'
import { t as tr } from '../i18n'
import type { MediaInfo, Project } from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    seedSamples: vi.fn(),
    addMedia: vi.fn(),
    createProject: vi.fn(),
    listProjects: vi.fn().mockResolvedValue([]),
    prepareMedia: vi.fn().mockResolvedValue({ job_id: 'j1' }),
  },
}))

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

describe('tour steps', () => {
  it('buildSteps invariants: unique ids, centered ⇔ no target, interactive ⇒ waitFor', () => {
    const steps = buildSteps()
    const ids = steps.map((s) => s.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const step of steps) {
      if (step.kind === 'centered') expect(step.target).toBeUndefined()
      else expect(step.target).toBeDefined()
      if (step.kind === 'interactive') expect(typeof step.waitFor).toBe('function')
    }
  })
})

/* ------------------------------------------------------------------ overlay */

function makeMedia(id: string, name: string): MediaInfo {
  return {
    id,
    path: `D:/videos/${name}.mp4`,
    name,
    size: 12345,
    duration: 120,
    fps: 30,
    width: 1920,
    height: 1080,
    rotation: 0,
    vcodec: 'h264',
    acodec: 'aac',
    has_audio: true,
    created_at: 0,
    proxy_path: `cache/${id}.mp4`,
    proxy_fps: 30,
    proxy_width: 960,
    proxy_height: 540,
    audio_path: 'cache/audio.wav',
    poster: `${id}.jpg`,
  }
}

function makeProject(): Project {
  return {
    id: 'p1',
    name: 'proj',
    created_at: 0,
    updated_at: 0,
    media: [],
    analyses: {},
    timeline: { tracks: [], duration: 240, fps: 30, width: 1920, height: 1080 },
    ui: {},
    version: 1,
  }
}

describe('tour overlay', () => {
  const mockedSeed = vi.mocked(api.seedSamples)
  const mockedAddMedia = vi.mocked(api.addMedia)
  const mockedCreateProject = vi.mocked(api.createProject)

  /** Drop a fake anchor into the DOM so a step resolves without waiting. */
  function addAnchor(target: string): HTMLElement {
    const el = document.createElement('div')
    el.setAttribute('data-tour', target)
    document.body.appendChild(el)
    return el
  }

  beforeEach(() => {
    localStorage.removeItem(TOUR_PREFS_KEY)
    resetTourStore()
    vi.clearAllMocks()
    useStore.setState({ lang: 'en', view: 'library', project: null, mediaId: null, playing: false })
  })

  afterEach(() => {
    vi.useRealTimers()
    document.querySelectorAll('[data-tour]').forEach((n) => n.remove())
    document.querySelectorAll('[data-modal]').forEach((n) => n.remove())
  })

  it('registers the step count on mount and renders the welcome card with dialog semantics', () => {
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    expect(useTourStore.getState().stepCount).toBe(buildSteps().length)

    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog.querySelector('[aria-live="polite"]')).not.toBeNull()
    expect(screen.getByText('Welcome to Badminton Studio')).toBeInTheDocument()
    expect(screen.getByText(/About 4 minutes/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start tour' })).toBeInTheDocument()
  })

  it('Next advances the index and Escape skips the tour (prefs.seen=true)', () => {
    addAnchor('nav-rail')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    expect(screen.getByText('Welcome to Badminton Studio')).toBeInTheDocument()

    act(() => {
      useTourStore.getState().goTo(1)
    })
    expect(screen.getByText('Global navigation')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(useTourStore.getState().index).toBe(2)

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(useTourStore.getState().active).toBe(false)
    expect(loadTourPrefs()).toEqual({ seen: true, completed: false })
  })

  it('interactive step with unsatisfied waitFor keeps Next disabled; skip-this-step advances', () => {
    addAnchor('library-new-project')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(2)
    })
    expect(screen.getByText('Projects')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: 'Skip this step' }))
    expect(useTourStore.getState().index).toBe(3)
  })

  it('falls back to a centered bubble when the target never appears within the cap', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    // nav-rail anchor intentionally missing from the DOM.
    act(() => {
      useTourStore.getState().goTo(1)
    })
    expect(screen.queryByText('Global navigation')).toBeNull()

    act(() => {
      vi.advanceTimersByTime(4200)
    })
    expect(screen.getByText('Global navigation')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next' })).toBeInTheDocument()
    // No `action` on this step, so the retry button must not exist.
    expect(screen.queryByRole('button', { name: 'Reopen' })).toBeNull()
  })

  it('computePlacement flips to the left when the target hugs the right edge', () => {
    const viewport = { width: 800, height: 600 }
    expect(computePlacement({ left: 760, right: 790, top: 200, bottom: 260 }, viewport, 'right')).toBe('left')
    // Roomy target keeps the default placement.
    expect(computePlacement({ left: 300, right: 420, top: 200, bottom: 260 }, { width: 1000, height: 800 })).toBe('bottom')
    // Target near the bottom edge flips to the top.
    expect(computePlacement({ left: 100, right: 400, top: 470, bottom: 560 }, viewport, 'bottom')).toBe('top')
  })

  it('loadSamples failure toasts an error and the step remains skippable', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    mockedSeed.mockRejectedValueOnce(new Error('seed failed'))
    useStore.setState({ project: makeProject() })

    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(3) // studio-empty (samples step)
    })
    act(() => {
      vi.advanceTimersByTime(4200) // fallback centered bubble
    })

    const toastSpy = vi.fn()
    const realToast = useStore.getState().toast
    useStore.setState({ toast: toastSpy })
    try {
      fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'error', title: tr('tour.samples.failed') }),
      )
      // Busy state reset: the button is clickable again...
      expect(screen.getByRole('button', { name: 'Load sample media' })).toBeEnabled()
      // ...and the step can still be skipped.
      fireEvent.click(screen.getByRole('button', { name: 'Skip this step' }))
      expect(useTourStore.getState().index).toBe(4)
    } finally {
      useStore.setState({ toast: realToast })
    }
  })

  it('loadSamples success imports the seeded files into the project', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    const media = makeMedia('m1', 'rally')
    mockedSeed.mockResolvedValueOnce({ files: ['D:/samples/rally.mp4'] })
    const project = makeProject()
    mockedAddMedia.mockResolvedValueOnce({
      project: { ...project, media: [media] },
      added: [media],
      failed: [],
    })
    useStore.setState({ project })

    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(3)
    })
    act(() => {
      vi.advanceTimersByTime(4200)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useStore.getState().mediaId).toBe('m1')
    expect(useStore.getState().currentMedia()).not.toBeNull()
    expect(screen.getByText('Sample media loaded')).toBeInTheDocument()

    // The waitFor (hasMedia) gate re-evaluates on its 100ms interval.
    act(() => {
      vi.advanceTimersByTime(120)
    })
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled()
  })

  it('keyboard cannot bypass an unsatisfied waitFor gate', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    addAnchor('library-new-project')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(2)
    })
    // Gate unsatisfied (no project): both ArrowRight and Enter are no-ops.
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(useTourStore.getState().index).toBe(2)

    // Once the gate is satisfied (the 100ms re-evaluation picks it up),
    // ArrowRight advances again.
    act(() => {
      useStore.setState({ project: makeProject() })
    })
    act(() => {
      vi.advanceTimersByTime(120)
    })
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(useTourStore.getState().index).toBe(3)
  })

  it('Escape with a modal open closes only the dialog: the tour stays on its step', () => {
    addAnchor('nav-rail')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(1)
    })
    expect(screen.getByText('Global navigation')).toBeInTheDocument()

    // Simulate an open Modal (ui.tsx stamps `data-modal` on its root while
    // open): Esc must close just that dialog, so the tour handler backs off
    // and the tour keeps running.
    const modal = document.createElement('div')
    modal.setAttribute('data-modal', '')
    document.body.appendChild(modal)
    try {
      fireEvent.keyDown(window, { key: 'Escape' })
      expect(useTourStore.getState().active).toBe(true)
      expect(useTourStore.getState().index).toBe(1)

      // Without a modal on screen the same keypress skips the whole tour
      // (the pre-existing behavior).
      modal.remove()
      fireEvent.keyDown(window, { key: 'Escape' })
      expect(useTourStore.getState().active).toBe(false)
      expect(loadTourPrefs()).toEqual({ seen: true, completed: false })
    } finally {
      modal.remove()
    }
  })

  it('arrow keys do not navigate while the event target is an editable input', () => {
    addAnchor('nav-rail')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(1)
    })
    expect(screen.getByText('Global navigation')).toBeInTheDocument()

    // Typing a project name (library step): ←/→ move the caret, so the tour
    // must neither navigate nor preventDefault the event.
    const input = document.createElement('input')
    document.body.appendChild(input)
    try {
      const right = new KeyboardEvent('keydown', {
        key: 'ArrowRight',
        bubbles: true,
        cancelable: true,
      })
      input.dispatchEvent(right)
      expect(right.defaultPrevented).toBe(false)
      expect(useTourStore.getState().index).toBe(1)

      const left = new KeyboardEvent('keydown', {
        key: 'ArrowLeft',
        bubbles: true,
        cancelable: true,
      })
      input.dispatchEvent(left)
      expect(left.defaultPrevented).toBe(false)
      expect(useTourStore.getState().index).toBe(1)

      // Control: outside editable targets ArrowLeft still navigates back.
      fireEvent.keyDown(window, { key: 'ArrowLeft' })
      expect(useTourStore.getState().index).toBe(0)
    } finally {
      input.remove()
    }
  })

  it('loadSamples with no project creates one first, then seeds and imports', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    const media = makeMedia('m2', 'sample')
    const newProject = { ...makeProject(), id: 'p_new' }
    mockedCreateProject.mockResolvedValueOnce(newProject)
    mockedSeed.mockResolvedValueOnce({ files: ['D:/samples/rally.mp4'] })
    mockedAddMedia.mockResolvedValueOnce({
      project: { ...newProject, media: [media] },
      added: [media],
      failed: [],
    })
    // Store has no project (beforeEach baseline): the create-first branch runs.

    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(3)
    })
    act(() => {
      vi.advanceTimersByTime(4200)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    // Strict call order: project creation -> seed -> import.
    const [createOrder, seedOrder, addOrder] = [
      mockedCreateProject,
      mockedSeed,
      mockedAddMedia,
    ].map((m) => m.mock.invocationCallOrder[0])
    expect(createOrder).toBeLessThan(seedOrder)
    expect(seedOrder).toBeLessThan(addOrder)

    expect(useStore.getState().project?.id).toBe('p_new')
    expect(useStore.getState().currentMedia()).not.toBeNull()
    expect(screen.getByText('Sample media loaded')).toBeInTheDocument()
  })

  it('loadSamples with a failed project creation does not double-toast', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    mockedCreateProject.mockRejectedValueOnce(new Error('backend down'))

    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(3)
    })
    act(() => {
      vi.advanceTimersByTime(4200)
    })

    const toastSpy = vi.fn()
    const realToast = useStore.getState().toast
    useStore.setState({ toast: toastSpy })
    try {
      fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
      await act(async () => {
        await Promise.resolve()
        await Promise.resolve()
      })
      // The store's own createProject failure toast fires exactly once; the
      // overlay must not stack a second "samples failed" toast on top.
      expect(toastSpy).toHaveBeenCalledTimes(1)
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'error', title: tr('toast.createProjectFailed') }),
      )
      expect(mockedSeed).not.toHaveBeenCalled()
      // Busy reset and the step remains skippable.
      expect(screen.getByRole('button', { name: 'Load sample media' })).toBeEnabled()
      fireEvent.click(screen.getByRole('button', { name: 'Skip this step' }))
      expect(useTourStore.getState().index).toBe(4)
    } finally {
      useStore.setState({ toast: realToast })
    }
  })
})
