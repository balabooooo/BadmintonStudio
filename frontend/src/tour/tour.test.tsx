import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, afterEach, vi } from 'vitest'
import { TOUR_PREFS_KEY, isFirstRun, loadTourPrefs, saveTourPrefs } from './prefs'
import { useTourStore } from './tourStore'
import TourOverlay, { computePlacement } from './TourOverlay'
import { buildSteps } from './steps'
import { api } from '../lib/api'
import { useStore } from '../store/useStore'
import { buildMockAnalysis, clearMockData, MOCK_JOB_ID_PREFIX } from './mockAnalysis'
import { t as tr } from '../i18n'
import type { MediaInfo, Project } from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    seedSamples: vi.fn(),
    addMedia: vi.fn(),
    createProject: vi.fn(),
    listProjects: vi.fn().mockResolvedValue([]),
    prepareMedia: vi.fn().mockResolvedValue({ job_id: 'j1' }),
    analyze: vi.fn().mockResolvedValue({ job_id: 'real-analyze' }),
    analyzeBatch: vi.fn().mockResolvedValue({ job_ids: ['real-batch'] }),
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

  it('inserts the mock-analysis run step right after the parameters step', () => {
    const steps = buildSteps()
    const ids = steps.map((s) => s.id)
    const runIdx = ids.indexOf('dlg-run')
    expect(runIdx).toBeGreaterThan(-1)
    expect(runIdx).toBe(ids.indexOf('dlg-params') + 1)
    expect(ids.indexOf('dlg-close')).toBe(runIdx + 1)
    const run = steps[runIdx]
    expect(run.kind).toBe('interactive')
    expect(run.target).toBe('dlg-run')
    expect(typeof run.waitFor).toBe('function')
  })

  it('every interactive step has a hint line and a known assist kind', () => {
    const assistKinds = ['click', 'createProject', 'play']
    for (const step of buildSteps()) {
      if (step.kind !== 'interactive') continue
      expect(typeof step.hintKey).toBe('string')
      expect((step.hintKey ?? '').length).toBeGreaterThan(0)
      if (step.assist) expect(assistKinds).toContain(step.assist.kind)
    }
  })

  it('mock-analysis waitFor only opens for flagged mock results', () => {
    const runStep = buildSteps().find((s) => s.id === 'dlg-run')!
    useStore.setState({ project: null, mediaId: null })
    expect(runStep.waitFor!()).toBe(false)
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
    useStore.setState({ lang: 'en', view: 'library', project: null, mediaId: null, playing: false, jobs: {}, busy: {} })
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

  it('interactive step with unsatisfied waitFor keeps Next disabled; assist button is offered', async () => {
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
    // The per-step "skip" is gone; the stuck user gets a "Do it for me" assist.
    expect(screen.getByRole('button', { name: tr('tour.common.assist') })).toBeInTheDocument()

    // Once the gate is satisfied, Next enables (the 50ms re-eval picks it up).
    act(() => {
      useStore.setState({ project: makeProject() })
    })
    await act(async () => {
      await new Promise((r) => window.setTimeout(r, 80))
    })
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled()
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

  it('centered bubble does not inherit a stale off-screen transform from the previous anchored step', () => {
    // Regression: when transitioning anchored → centered, React used to reuse
    // the container DOM node and the rAF loop's translate() lingered, pushing
    // the centered bubble off-screen (Next unreachable). Distinct keys + a
    // transform reset in the loop prevent this.
    const origRaf = globalThis.requestAnimationFrame
    let rafCalls = 0
    vi.stubGlobal(
      'requestAnimationFrame',
      (cb: FrameRequestCallback) => {
        // Run the rAF loop a bounded number of times (the loop re-schedules
        // itself each frame); cap it so the synchronous stub doesn't recurse
        // forever while still exercising the transform-write branch.
        if (rafCalls++ < 3) cb(0)
        return 0
      },
    )
    const nav = addAnchor('nav-rail')
    // Non-zero rect so the anchored branch of the rAF loop writes a translate.
    vi.spyOn(nav, 'getBoundingClientRect').mockReturnValue({
      left: 10, right: 60, top: 10, bottom: 40, width: 50, height: 30,
      x: 10, y: 10, toJSON: () => ({}),
    } as DOMRect)

    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(1) // nav-rail (anchored)
    })

    // The anchored container got a translate from the rAF loop.
    const anchored = document.querySelector('[data-tour-lockdown]')?.parentElement
    const anchoredBubbleWrap = anchored?.querySelector('div[style*="transform"]')
    expect(anchoredBubbleWrap).not.toBeNull()

    // Jump to the welcome step (centered, no target).
    act(() => {
      useTourStore.getState().goTo(0)
    })

    // The centered bubble container must NOT carry a translate.
    const dialog = screen.getByRole('dialog')
    const centeredWrap = dialog.querySelector('.absolute.flex')
    expect(centeredWrap).not.toBeNull()
    expect((centeredWrap as HTMLElement).style.transform).toBe('')

    vi.stubGlobal('requestAnimationFrame', origRaf)
  })

  it('computePlacement flips to the left when the target hugs the right edge', () => {
    const viewport = { width: 800, height: 600 }
    expect(computePlacement({ left: 760, right: 790, top: 200, bottom: 260 }, viewport, 'right')).toBe('left')
    // Roomy target keeps the default placement.
    expect(computePlacement({ left: 300, right: 420, top: 200, bottom: 260 }, { width: 1000, height: 800 })).toBe('bottom')
    // Target near the bottom edge flips to the top.
    expect(computePlacement({ left: 100, right: 400, top: 470, bottom: 560 }, viewport, 'bottom')).toBe('top')
  })

  it('loadSamples failure shows an inline error in the bubble (toasts are covered by the lockdown layer)', async () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    // Feedback renders inside the bubble, not as a (covered) toast.
    expect(screen.getByText(tr('tour.samples.failed'))).toBeInTheDocument()
    // Busy state reset: the button is clickable again for a retry.
    expect(screen.getByRole('button', { name: 'Load sample media' })).toBeEnabled()
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

  it('lockdown: keys are blocked outside the hole, but free inside the modal-panel hole (project-name input)', () => {
    // The library step's target sits outside the create-project modal; while
    // the modal is open its whole panel is a second hole so naming the project
    // works normally.
    addAnchor('library-new-project')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(2)
    })
    expect(screen.getByText('Projects')).toBeInTheDocument()

    // Background input (outside every hole): arrow keys are swallowed by the
    // lockdown — caret events neither reach the field nor navigate the tour.
    const bgInput = document.createElement('input')
    document.body.appendChild(bgInput)
    const blocked = new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })
    bgInput.dispatchEvent(blocked)
    expect(blocked.defaultPrevented).toBe(true)
    expect(useTourStore.getState().index).toBe(2)
    bgInput.remove()

    // Input inside the open modal's panel: full native behavior (caret moves).
    const modal = document.createElement('div')
    modal.setAttribute('data-modal', '')
    const panel = document.createElement('div')
    panel.setAttribute('role', 'dialog')
    const nameInput = document.createElement('input')
    panel.appendChild(nameInput)
    modal.appendChild(panel)
    document.body.appendChild(modal)
    try {
      const allowed = new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })
      nameInput.dispatchEvent(allowed)
      expect(allowed.defaultPrevented).toBe(false)
      expect(useTourStore.getState().index).toBe(2)
    } finally {
      modal.remove()
    }

    // Backdrop arrow keys still navigate the tour.
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(useTourStore.getState().index).toBe(1)
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

  it('loadSamples with a failed project creation shows the inline error and never seeds', async () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Load sample media' }))
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    // The failure surfaces inline (toasts live under the lockdown overlay).
    expect(screen.getByText(tr('tour.samples.failed'))).toBeInTheDocument()
    expect(mockedSeed).not.toHaveBeenCalled()
    // Busy reset: the button is clickable again for a retry.
    expect(screen.getByRole('button', { name: 'Load sample media' })).toBeEnabled()
  })

  it('lockdown: renders above all other UI with an evenodd pointer-catcher path', () => {
    addAnchor('nav-rail')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().goTo(1)
    })
    const dialog = screen.getByRole('dialog')
    expect(dialog.className).toContain('z-[95]')
    const catcher = document.querySelector('[data-tour-lockdown]')
    expect(catcher).not.toBeNull()
    const path = catcher!.querySelector('path')
    expect(path).not.toBeNull()
    expect(path).toHaveAttribute('fill-rule', 'evenodd')
  })

  it('lockdown: swallows app-level shortcut keys (space/delete/ctrl+z) from the backdrop', () => {
    addAnchor('nav-rail')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(1)
    })
    useStore.setState({ playing: false })
    for (const init of [{ key: ' ', code: 'Space' }, { key: 'Delete' }, { key: 'z', ctrlKey: true }]) {
      const ev = new KeyboardEvent('keydown', { key: init.key, code: init.code, bubbles: true, cancelable: true, ctrlKey: !!init.ctrlKey })
      window.dispatchEvent(ev)
      expect(ev.defaultPrevented).toBe(true)
    }
    expect(useStore.getState().playing).toBe(false)
  })

  it('lockdown: Tab from the backdrop pulls focus back into the bubble', () => {
    addAnchor('library-new-project')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(2)
    })
    document.body.focus()
    const ev = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true })
    window.dispatchEvent(ev)
    expect(ev.defaultPrevented).toBe(true)
    expect((document.activeElement as HTMLElement | null)?.closest('[data-tour-bubble]')).not.toBeNull()
  })

  it('unsatisfied interactive step shows the hint line, pulsing ring and assist button', async () => {
    addAnchor('library-new-project')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(2)
    })
    expect(screen.getByText(tr('tour.library.hint'))).toBeInTheDocument()
    expect(document.querySelector('[data-tour-pulse]')).not.toBeNull()
    expect(screen.getByRole('button', { name: tr('tour.common.assist') })).toBeInTheDocument()

    // Once the gate is satisfied all three disappear.
    act(() => {
      useStore.setState({ project: makeProject() })
    })
    await act(async () => {
      await new Promise((r) => window.setTimeout(r, 150))
    })
    expect(screen.queryByText(tr('tour.library.hint'))).toBeNull()
    expect(screen.queryByRole('button', { name: tr('tour.common.assist') })).toBeNull()
  })

  it('assist "createProject" creates a project and auto-advances past the library step', async () => {
    addAnchor('library-new-project')
    addAnchor('studio-empty')
    const newProject = { ...makeProject(), id: 'p_assist' }
    mockedCreateProject.mockResolvedValueOnce(newProject)
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(2)
    })

    fireEvent.click(screen.getByRole('button', { name: tr('tour.common.assist') }))
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(mockedCreateProject).toHaveBeenCalledTimes(1)
    expect(useStore.getState().project?.id).toBe('p_assist')
    // createProject switches the view to studio; the tour auto-advances so the
    // step-3 bubble doesn't linger on the new view.
    expect(useTourStore.getState().index).toBe(3)
  })

  it('assist "play" starts playback through the store and satisfies the player step', () => {
    addAnchor('player')
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(buildSteps().findIndex((s) => s.id === 'player'))
    })
    useStore.setState({ playing: false })
    fireEvent.click(screen.getByRole('button', { name: tr('tour.common.assist') }))
    expect(useStore.getState().playing).toBe(true)
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled()
  })

  it('assist "click" programmatically clicks the highlighted target', async () => {
    const anchor = addAnchor('btn-analysis')
    let clicks = 0
    anchor.addEventListener('click', () => {
      clicks += 1
    })
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
      useTourStore.getState().goTo(buildSteps().findIndex((s) => s.id === 'btn-analysis'))
    })
    // Let the step-entry action click settle, then count only the assist click.
    await act(async () => {
      await new Promise((r) => window.setTimeout(r, 30))
    })
    clicks = 0
    fireEvent.click(screen.getByRole('button', { name: tr('tour.common.assist') }))
    expect(clicks).toBe(1)
  })
})

/* ------------------------------------------------------------------ analysis interception */

describe('tour analysis interception', () => {
  const mockedAnalyze = vi.mocked(api.analyze)
  const mockedAnalyzeBatch = vi.mocked(api.analyzeBatch)

  beforeEach(() => {
    localStorage.removeItem(TOUR_PREFS_KEY)
    resetTourStore()
    vi.clearAllMocks()
    useStore.setState({ lang: 'en', view: 'library', project: null, mediaId: null, playing: false, jobs: {}, busy: {} })
  })

  afterEach(() => {
    vi.useRealTimers()
    clearMockData()
    resetTourStore()
  })

  function seededProject() {
    const m1 = { ...makeMedia('m1', 'match'), duration: 48 }
    const m2 = { ...makeMedia('m2', 'rally'), duration: 16 }
    const project = makeProject()
    project.media = [m1, m2]
    useStore.setState({ project, mediaId: 'm1' })
    return { m1, m2 }
  }

  it('runAnalysis while the tour is active never calls the backend and writes mock data', async () => {
    vi.useFakeTimers()
    seededProject()
    act(() => {
      useTourStore.getState().start('manual')
    })

    await act(async () => {
      const p = useStore.getState().runAnalysis()
      await vi.advanceTimersByTimeAsync(4000)
      await p
    })

    expect(mockedAnalyze).not.toHaveBeenCalled()
    const analysis = useStore.getState().project!.analyses.m1
    expect(analysis.status).toBe('done')
    expect(analysis.stats.mock).toBe(true)
    expect(useStore.getState().jobs[`${MOCK_JOB_ID_PREFIX}m1`].status).toBe('done')
  })

  it('runAnalysisBatch while the tour is active mocks every queued media', async () => {
    vi.useFakeTimers()
    seededProject()
    act(() => {
      useTourStore.getState().start('manual')
    })

    await act(async () => {
      const p = useStore.getState().runAnalysisBatch()
      await vi.advanceTimersByTimeAsync(4000)
      await p
    })

    expect(mockedAnalyzeBatch).not.toHaveBeenCalled()
    expect(useStore.getState().project!.analyses.m1.stats.mock).toBe(true)
    expect(useStore.getState().project!.analyses.m2.stats.mock).toBe(true)
  })

  it('runAnalysis outside the tour takes the real backend path', async () => {
    seededProject()
    await act(async () => {
      await useStore.getState().runAnalysis()
    })
    expect(mockedAnalyze).toHaveBeenCalledTimes(1)
    expect(useStore.getState().project!.analyses.m1).toBeUndefined()
  })

  it('the overlay clears mock analyses and jobs when the tour is skipped', () => {
    const { m1 } = seededProject()
    const project = useStore.getState().project!
    project.analyses.m1 = buildMockAnalysis(m1, useStore.getState().params, 'balanced')
    useStore.setState({
      project,
      jobs: { [`${MOCK_JOB_ID_PREFIX}m1`]: { id: `${MOCK_JOB_ID_PREFIX}m1` } as never },
    })
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    expect(useStore.getState().project!.analyses.m1.stats.mock).toBe(true)

    act(() => {
      useTourStore.getState().skip()
    })
    expect(useStore.getState().project!.analyses.m1).toBeUndefined()
    expect(useStore.getState().jobs[`${MOCK_JOB_ID_PREFIX}m1`]).toBeUndefined()
  })

  it('the overlay clears mock data when the tour is completed on the last step', () => {
    const { m1 } = seededProject()
    const project = useStore.getState().project!
    project.analyses.m1 = buildMockAnalysis(m1, useStore.getState().params, 'balanced')
    useStore.setState({ project })
    render(<TourOverlay />)
    act(() => {
      useTourStore.getState().start('manual')
    })
    act(() => {
      useTourStore.getState().complete()
    })
    expect(useStore.getState().project!.analyses.m1).toBeUndefined()
  })
})
