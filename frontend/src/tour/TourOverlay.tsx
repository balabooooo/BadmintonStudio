/** Interactive guided-tour overlay: lockdown layer, spotlight and bubble.
 *
 * Rendered once at the app root (no props); renders nothing while the tour is
 * inactive. While active it OWNS the screen (z-95, above modals/toasts):
 *
 * - A full-screen SVG ``pointer-events:auto`` catcher with an evenodd path
 *   dims everything except cut-out holes. Interactive steps cut a hole around
 *   the ``data-tour`` target; when the target lives outside an open modal
 *   (e.g. the create-project flow on the library step) the modal panel gets a
 *   second hole so that sub-flow stays usable. Info/centered steps cut no
 *   hole at all — the background is visible but not clickable.
 * - Keyboard events are captured on window: the bubble and in-hole elements
 *   keep native behavior, everything else (app shortcuts, tab-to-background)
 *   is swallowed.
 * - The bubble carries the step copy, a concrete hint line, an optional
 *   "do it for me" assist button and the standard Prev/Next/Skip controls.
 *
 * Every step resolves within 4s: when the target never appears the bubble
 * falls back to a centered card, so the tour can never dead-end.
 */

import { useEffect, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { Button } from '../components/ui'
import { cn } from '../lib/format'
import { api } from '../lib/api'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
import { buildSteps, type TourStep } from './steps'
import { useTourStore } from './tourStore'
import { clearMockData } from './mockAnalysis'

const STEPS = buildSteps()

const SPOT_MARGIN = 4
const SPOT_RADIUS = 12
const LOCK_FILL = 'rgba(3,10,8,0.72)'
const BUBBLE_WIDTH = 320
const GAP = 14
const POLL_INTERVAL = 50
/** 50ms × 80 = the 4s cap before an anchored step falls back to centered. */
const POLL_TICKS = 80
const NARROW_VIEWPORT = 640

export type BubblePlacement = 'top' | 'bottom' | 'left' | 'right'

interface RectLike {
  left: number
  top: number
  right: number
  bottom: number
}

interface Viewport {
  width: number
  height: number
}

const OPPOSITE: Record<BubblePlacement, BubblePlacement> = {
  top: 'bottom',
  bottom: 'top',
  left: 'right',
  right: 'left',
}

/** Room the 320px bubble needs on a side before that side counts as usable. */
const ROOM: Record<BubblePlacement, number> = {
  top: 252,
  bottom: 252,
  left: BUBBLE_WIDTH + GAP,
  right: BUBBLE_WIDTH + GAP,
}

/** Pure placement pick: keep the preferred side while it fits, otherwise flip
 * to the opposite side, then to any side that fits. Exported for tests. */
export function computePlacement(
  rect: RectLike,
  viewport: Viewport,
  preferred: BubblePlacement = 'bottom',
): BubblePlacement {
  const fits: Record<BubblePlacement, boolean> = {
    top: rect.top >= ROOM.top,
    bottom: viewport.height - rect.bottom >= ROOM.bottom,
    left: rect.left >= ROOM.left,
    right: viewport.width - rect.right >= ROOM.right,
  }
  if (fits[preferred]) return preferred
  const opposite = OPPOSITE[preferred]
  if (fits[opposite]) return opposite
  const others = (['top', 'bottom', 'left', 'right'] as BubblePlacement[]).filter(
    (s) => s !== preferred && s !== opposite,
  )
  for (const side of others) if (fits[side]) return side
  return preferred // nothing fits (target ~fills the viewport): caller clamps
}

/** evenodd path: a full-viewport rectangle minus one sub-path per hole. */
function lockdownPath(width: number, height: number, holes: RectLike[]): string {
  let d = `M 0 0 H ${width} V ${height} H 0 Z`
  for (const r of holes) {
    d += ` M ${r.left} ${r.top} H ${r.right} V ${r.bottom} H ${r.left} Z`
  }
  return d
}

function expandRect(r: DOMRect, by: number): RectLike {
  return { left: r.left - by, top: r.top - by, right: r.right + by, bottom: r.bottom + by }
}

/** Holes cut into the lockdown SVG for the current interactive step. */
function computeHoles(step: TourStep | null, target: HTMLElement | null): RectLike[] {
  if (!step || step.kind !== 'interactive' || !target) return []
  const r = target.getBoundingClientRect()
  if (r.width === 0 && r.height === 0) return []
  const holes: RectLike[] = [expandRect(r, SPOT_MARGIN + 2)]
  // Target outside any modal: the open modal panel (e.g. the create-project
  // name-and-confirm dialog) is a second hole so its sub-flow stays usable.
  if (!target.closest('[data-modal]')) {
    const panel =
      (document.querySelector('[data-modal] [role="dialog"]') as HTMLElement | null) ??
      (document.querySelector('[data-modal]') as HTMLElement | null)
    if (panel) {
      const mr = panel.getBoundingClientRect()
      if (mr.width > 0 && mr.height > 0) holes.push(expandRect(mr, 8))
    }
  }
  return holes
}

type Mode = 'polling' | 'anchored' | 'centered' | 'fallback'

export default function TourOverlay() {
  const tr = useT()
  const active = useTourStore((s) => s.active)
  const index = useTourStore((s) => s.index)
  const stepCount = useTourStore((s) => s.stepCount)
  const reduced = useReducedMotion() ?? false

  const [mode, setMode] = useState<Mode>('polling')
  const [satisfied, setSatisfied] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [errorKey, setErrorKey] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [narrow, setNarrow] = useState(() => window.innerWidth < NARROW_VIEWPORT)

  const pathRef = useRef<SVGPathElement | null>(null)
  const spotRef = useRef<HTMLDivElement | null>(null)
  const pulseRef = useRef<HTMLDivElement | null>(null)
  const posRef = useRef<HTMLDivElement | null>(null)
  const bubbleRef = useRef<HTMLDivElement | null>(null)
  const targetRef = useRef<HTMLElement | null>(null)
  const stepRef = useRef<TourStep | null>(null)
  /** Mirror of ``satisfied`` so the window-level keydown handler (registered
   * once per activation) can apply the same gate as the Next button. */
  const satisfiedRef = useRef<boolean | null>(null)
  const narrowRef = useRef(narrow)
  const reducedRef = useRef(reduced)

  // Mirror reactive values for the effects below (declared first so the
  // mirrors are fresh by the time the step-entry effect reads them).
  useEffect(() => {
    narrowRef.current = narrow
  }, [narrow])
  useEffect(() => {
    reducedRef.current = reduced
  }, [reduced])

  // Register the real step table size so next/goTo clamp at the last step.
  useEffect(() => {
    useTourStore.setState({ stepCount: buildSteps().length })
  }, [])

  // Tour ending (skip/complete) strips every mock analysis/job so the sample
  // project is back to a clean, real-data state.
  useEffect(() => {
    if (!active) return
    return () => clearMockData()
  }, [active])

  // Step entry: switch view if needed, fire the action click, then poll for
  // the anchor (100ms / 4s cap) before falling back to a centered bubble.
  useEffect(() => {
    if (!active) return
    const step = STEPS[index]
    if (!step) return
    stepRef.current = step
    targetRef.current = null
    setBusy(false)
    setLoaded(false)
    setErrorKey(null)
    const initial = step.waitFor ? step.waitFor() : null
    satisfiedRef.current = initial
    setSatisfied(initial)

    if (step.view !== useStore.getState().view) useStore.getState().setView(step.view)

    if (!step.target) {
      setMode('centered')
      return
    }

    let cancelled = false
    let ticks = 0

    const actionTarget = step.action?.click
    const actionSelector = step.action?.selector
    let actionRaf = 0
    if (actionTarget) {
      actionRaf = requestAnimationFrame(() => {
        if (cancelled) return
        const anchor = document.querySelector(`[data-tour="${actionTarget}"]`)
        const el = actionSelector
          ? anchor?.querySelector<HTMLElement>(actionSelector)
          : (anchor as HTMLElement | null)
        el?.click()
      })
    }

    const poll = (): boolean => {
      if (cancelled) return true
      const el = document.querySelector(`[data-tour="${step.target}"]`)
      if (el) {
        targetRef.current = el as HTMLElement
        ;(el as HTMLElement).scrollIntoView({
          block: 'center',
          behavior: reducedRef.current ? 'auto' : 'smooth',
        })
        setMode('anchored')
        return true
      }
      if (ticks >= POLL_TICKS) {
        setMode('fallback')
        return true
      }
      return false
    }

    setMode('polling')
    if (!poll()) {
      const timer = window.setInterval(() => {
        ticks += 1
        if (poll()) window.clearInterval(timer)
      }, POLL_INTERVAL)
      return () => {
        cancelled = true
        window.clearInterval(timer)
        if (actionRaf) cancelAnimationFrame(actionRaf)
      }
    }
    return () => {
      cancelled = true
      if (actionRaf) cancelAnimationFrame(actionRaf)
    }
  }, [active, index, attempt])

  // Re-evaluate the current step's waitFor gate immediately (used after an
  // assist action) and on a 100ms interval while the bubble is visible, so
  // both store-driven and DOM-driven predicates flip "Next" without wiring.
  const checkGate = () => {
    const st = stepRef.current
    if (!st?.waitFor) return
    const v = st.waitFor()
    satisfiedRef.current = v
    setSatisfied(v)
  }

  useEffect(() => {
    const step = stepRef.current
    if (!active || mode === 'polling' || !step?.waitFor) return
    checkGate()
    const timer = window.setInterval(checkGate, POLL_INTERVAL)
    return () => window.clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, index, mode, attempt])

  // Single rAF loop while the tour is active:
  //  1. write the lockdown path (viewbox minus target/modal holes),
  //  2. follow the target with spotlight ring + pulse (compositor props only),
  //  3. place the anchored bubble.
  useEffect(() => {
    if (!active) return
    let raf = 0
    const frame = () => {
      const el = targetRef.current
      const step = stepRef.current
      const path = pathRef.current
      if (path) {
        const holes = mode === 'anchored' ? computeHoles(step, el) : []
        const d = lockdownPath(window.innerWidth, window.innerHeight, holes)
        if (path.getAttribute('d') !== d) path.setAttribute('d', d)
      }
      const r0 = el?.getBoundingClientRect()
      if (el && r0 && (r0.width > 0 || r0.height > 0)) {
        // Clamp the spotlight box to the viewport so edges (e.g. the nav rail
        // hugging the left/bottom screen edges) are never clipped off-screen.
        const vw = window.innerWidth
        const vh = window.innerHeight
        const rawX = r0.left - SPOT_MARGIN
        const rawY = r0.top - SPOT_MARGIN
        const rawW = r0.width + SPOT_MARGIN * 2
        const rawH = r0.height + SPOT_MARGIN * 2
        const x = Math.max(0, rawX)
        const y = Math.max(0, rawY)
        const w = Math.min(rawW, vw - x)
        const h = Math.min(rawH, vh - y)
        for (const ref of [spotRef, pulseRef]) {
          const node = ref.current
          if (node) {
            node.style.transform = `translate(${x}px, ${y}px)`
            node.style.width = `${w}px`
            node.style.height = `${h}px`
            node.style.opacity = '1'
          }
        }
        const pos = posRef.current
        if (pos && !narrowRef.current) {
          const viewport = { width: window.innerWidth, height: window.innerHeight }
          const placement = computePlacement(r0, viewport, stepRef.current?.placement ?? 'bottom')
          const bh = pos.offsetHeight || 252
          let x: number
          let y: number
          if (placement === 'bottom') {
            x = r0.left + r0.width / 2 - BUBBLE_WIDTH / 2
            y = r0.bottom + GAP
          } else if (placement === 'top') {
            x = r0.left + r0.width / 2 - BUBBLE_WIDTH / 2
            y = r0.top - GAP - bh
          } else if (placement === 'right') {
            x = r0.right + GAP
            y = r0.top + r0.height / 2 - bh / 2
          } else {
            x = r0.left - GAP - BUBBLE_WIDTH
            y = r0.top + r0.height / 2 - bh / 2
          }
          x = Math.min(Math.max(x, 8), Math.max(8, viewport.width - BUBBLE_WIDTH - 8))
          y = Math.min(Math.max(y, 8), Math.max(8, viewport.height - bh - 8))
          pos.style.transform = `translate(${x}px, ${y}px)`
        }
      } else {
        // Target gone (step changed or element unmounted): hide the rings so
        // the previous step's highlight never lingers into the next step.
        for (const ref of [spotRef, pulseRef]) {
          const node = ref.current
          if (node) {
            node.style.width = '0px'
            node.style.height = '0px'
            node.style.opacity = '0'
          }
        }
        // Also clear the anchored bubble's translate so a centered/fallback
        // bubble never inherits a stale off-screen position.
        const pos = posRef.current
        if (pos) pos.style.transform = 'none'
      }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [active, mode])

  // Global keyboard LOCKDOWN (capture phase): keys targeting the bubble or an
  // in-hole element keep their native behavior; every other keypress is
  // stopped here so app shortcuts (space/arrows/Delete/Ctrl+Z) cannot act on
  // the dimmed UI. Tour navigation still works from the backdrop.
  useEffect(() => {
    if (!active) return
    // Node-safe membership: the event target may be window/document (not a
    // Node) when tests or the browser dispatch on those objects directly.
    const isElement = (v: unknown): v is HTMLElement =>
      !!v && typeof (v as Element).closest === 'function'
    const inHole = (el: HTMLElement): boolean => {
      const step = stepRef.current
      if (step?.kind !== 'interactive' || !step.target) return false
      const anchor = document.querySelector(`[data-tour="${step.target}"]`)
      if (anchor && (anchor === el || anchor.contains(el))) return true
      // Second hole: an open modal panel while the target is outside it.
      if (anchor && !anchor.closest('[data-modal]')) {
        const panel = document.querySelector('[data-modal] [role="dialog"]')
        if (panel && (panel === el || panel.contains(el))) return true
      }
      return false
    }
    const focusIntoTour = () => {
      const bubble = bubbleRef.current
      const next = bubble?.querySelector<HTMLButtonElement>('[data-tour-next]:not([disabled])')
      if (next) next.focus()
      else bubble?.focus()
    }
    const navigate = (key: string): boolean => {
      const gated = stepRef.current?.kind === 'interactive' && satisfiedRef.current !== true
      if (key === 'ArrowRight') {
        if (!gated) useTourStore.getState().next()
        return true
      }
      if (key === 'ArrowLeft') {
        useTourStore.getState().prev()
        return true
      }
      return false
    }
    const onKey = (e: KeyboardEvent) => {
      const el = e.target
      // Every Modal also closes itself on a window-level Escape. While a modal
      // is on screen let it win: the dialog-close step's waitFor then opens as
      // designed. A bare Esc skips the whole tour.
      if (e.key === 'Escape') {
        if (document.querySelector('[data-modal]')) return
        e.preventDefault()
        e.stopPropagation()
        useTourStore.getState().skip()
        return
      }
      // Inside the bubble: arrows navigate the tour; Tab is trapped by the
      // bubble's React handler; Enter/Space activate the focused button
      // natively (handling Enter here would double-fire).
      if (isElement(el) && el.closest('[data-tour-bubble]')) {
        if (navigate(e.key)) {
          e.preventDefault()
          e.stopPropagation()
        }
        return
      }
      // Inside a cut-out hole (real UI, including modal inputs): native only.
      if (isElement(el) && inHole(el)) return
      // Backdrop: swallow the event so app shortcuts (space/Delete/Ctrl+Z...)
      // cannot act on the dimmed UI; arrows still drive the tour.
      e.preventDefault()
      e.stopPropagation()
      if (e.key === 'Tab') {
        focusIntoTour()
        return
      }
      navigate(e.key)
      // Enter on the bare backdrop also advances when the gate is open.
      if (e.key === 'Enter') {
        const gated = stepRef.current?.kind === 'interactive' && satisfiedRef.current !== true
        if (!gated) useTourStore.getState().next()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [active])

  // Viewport width class (bottom-docked bubble on narrow windows). Kept
  // attached for the overlay's whole lifetime (it is a root singleton) so
  // `narrow` is also correct when a tour starts after the user resized while
  // the overlay was inactive.
  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth < NARROW_VIEWPORT)
    window.addEventListener('resize', onResize, { passive: true })
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // Move focus into the bubble whenever its content appears/changes.
  useEffect(() => {
    if (!active || mode === 'polling') return
    const bubble = bubbleRef.current
    if (!bubble) return
    const next = bubble.querySelector<HTMLButtonElement>('[data-tour-next]')
    if (next && !next.disabled) next.focus()
    else bubble.focus()
  }, [active, index, mode])

  if (!active) return null
  const step = STEPS[index]
  if (!step || mode === 'polling') return null

  const interactive = step.kind === 'interactive'
  const isLast = index >= STEPS.length - 1
  const nextDisabled = interactive && satisfied !== true
  const anchored = mode === 'anchored'
  const showCue = anchored && interactive && satisfied !== true

  const runLoadSamples = async () => {
    if (busy) return
    setBusy(true)
    setErrorKey(null)
    try {
      if (!useStore.getState().project) {
        const pid = await useStore.getState().createProject(tr('tour.samples.projectName'))
        // createProject swallows its own error and returns undefined: surface
        // it inline (toasts render below the lockdown overlay).
        if (!pid) {
          setErrorKey('tour.samples.failed')
          return
        }
      }
      const { files } = await api.seedSamples()
      await useStore.getState().importMedia(files)
      if (!useStore.getState().currentMedia()) throw new Error('import failed')
      // Stay in the "loaded" state for the rest of this step: media is in, the
      // gate is open, and re-importing would only create duplicates.
      setLoaded(true)
    } catch {
      setErrorKey('tour.samples.failed')
    } finally {
      setBusy(false)
    }
  }

  /** "Do it for me": perform the step's assist action programmatically. */
  const runAssist = async () => {
    const a = step.assist
    if (!a) return
    if (a.kind === 'play') {
      useStore.getState().setPlaying(true)
      checkGate()
      return
    }
    if (a.kind === 'createProject') {
      if (busy) return
      setBusy(true)
      setErrorKey(null)
      try {
        const pid = await useStore.getState().createProject(tr('tour.assist.projectName'))
        if (!pid) {
          setErrorKey('tour.assist.createFailed')
          return
        }
        // createProject switches the view to 'studio'; advance immediately so
        // the step-3 bubble (anchored to the library button) doesn't linger
        // on top of the new studio view.
        checkGate()
        useTourStore.getState().next()
      } catch {
        setErrorKey('tour.assist.createFailed')
      } finally {
        setBusy(false)
      }
      return
    }
    const anchor = step.target ? document.querySelector(`[data-tour="${step.target}"]`) : null
    const el = a.selector ? anchor?.querySelector<HTMLElement>(a.selector) : anchor
    ;(el as HTMLElement | null)?.click()
  }

  const onNext = () => {
    if (isLast) useTourStore.getState().complete()
    else useTourStore.getState().next()
  }

  // Tab cycles inside the bubble only.
  const onBubbleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'Tab') return
    const bubble = bubbleRef.current
    if (!bubble) return
    const focusables = Array.from(bubble.querySelectorAll<HTMLElement>('button:not([disabled])'))
    if (!focusables.length) return
    const first = focusables[0]
    const last = focusables[focusables.length - 1]
    const current = document.activeElement as HTMLElement | null
    if (e.shiftKey && (current === first || !bubble.contains(current))) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && (current === last || !bubble.contains(current))) {
      e.preventDefault()
      first.focus()
    }
  }

  const bubble = (
    <motion.div
      ref={bubbleRef}
      data-tour-bubble
      tabIndex={-1}
      initial={{ opacity: 0, y: reduced ? 0 : 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.12 }}
      onKeyDown={onBubbleKeyDown}
      className={cn(
        'pointer-events-auto rounded-[14px] border border-white/10 bg-ink-900/95 p-4 outline-none',
        'shadow-[0_18px_50px_-12px_rgb(0_0_0/0.7)] backdrop-blur-sm',
        narrow && 'w-full',
      )}
      style={{ width: narrow ? undefined : BUBBLE_WIDTH }}
    >
      <div aria-live="polite">
        <h2 className="text-[14px] font-semibold text-white">{tr(step.titleKey)}</h2>
        <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-300">{tr(step.bodyKey)}</p>
      </div>

      {/* Concrete one-line instruction while the interaction is pending. */}
      {showCue && step.hintKey && (
        <div className="mt-2 rounded-lg border border-court-500/25 bg-court-500/10 px-2.5 py-1.5 text-[11.5px] leading-snug text-court-200">
          <span aria-hidden="true" className="mr-1">👆</span>
          {tr(step.hintKey)}
        </div>
      )}

      {/* Inline error: toasts live below the lockdown overlay and are not
          visible while the tour runs. */}
      {errorKey && (
        <div className="mt-2 rounded-lg border border-rose-hot/30 bg-rose-hot/10 px-2.5 py-1.5 text-[11.5px] text-rose-hot">
          {tr(errorKey)}
        </div>
      )}

      {step.customAction === 'loadSamples' && !loaded && (
        <Button
          variant="primary"
          size="md"
          className="mt-3 w-full"
          loading={busy}
          disabled={busy}
          onClick={runLoadSamples}
        >
          {busy ? tr('tour.samples.loading') : tr('tour.samples.btn')}
        </Button>
      )}
      {step.customAction === 'loadSamples' && loaded && (
        <div className="mt-3 text-center text-[12.5px] font-medium text-court-400">
          {tr('tour.samples.done')}
        </div>
      )}

      {mode === 'fallback' && step.action && (
        <Button
          variant="outline"
          size="sm"
          className="mt-3 w-full"
          onClick={() => setAttempt((a) => a + 1)}
        >
          {tr('tour.common.retry')}
        </Button>
      )}

      <div className="mt-3 flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={() => useTourStore.getState().skip()}>
          {tr('tour.common.skip')}
        </Button>
        <span className="ml-auto text-[11px] text-ink-500 tabular-nums">
          {tr('tour.common.stepOf', { n: index + 1, total: stepCount })}
        </span>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={index === 0}
          onClick={() => useTourStore.getState().prev()}
        >
          {tr('tour.common.prev')}
        </Button>
        {showCue && step.assist && (
          <motion.div
            animate={
              reduced
                ? undefined
                : { y: [0, -3, 0], scale: [1, 1.04, 1] }
            }
            transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
          >
            <Button variant="primary" size="sm" loading={busy} onClick={runAssist}>
              {tr('tour.common.assist')}
            </Button>
          </motion.div>
        )}
        <Button
          data-tour-next=""
          variant="primary"
          size="sm"
          className="ml-auto"
          disabled={nextDisabled}
          onClick={onNext}
        >
          {step.group === 'welcome' ? tr('tour.welcome.start') : tr('tour.common.next')}
        </Button>
      </div>
    </motion.div>
  )

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={tr(step.titleKey)}
      className="fixed inset-0 z-[95]"
      style={{ pointerEvents: 'none' }}
    >
      {/* Lockdown: the path fill catches every pointer/keyboard interaction
          outside the cut-out holes; holes fall through to the real UI. */}
      <svg
        data-tour-lockdown
        aria-hidden="true"
        className="absolute inset-0 h-full w-full"
        style={{ pointerEvents: 'auto' }}
      >
        <path ref={pathRef} d="" fill={LOCK_FILL} fillRule="evenodd" clipRule="evenodd" />
      </svg>

      {anchored && (
        <>
          {/* Static spotlight ring. */}
          <div
            ref={spotRef}
            aria-hidden="true"
            className="pointer-events-none absolute top-0 left-0 rounded-xl border-2 border-court-400/90"
            style={{
              borderRadius: SPOT_RADIUS,
              willChange: 'transform',
              boxShadow: '0 0 18px rgba(56,224,162,0.22)',
            }}
          />
          {/* Pulsing cue ring on the element the user must operate. */}
          {showCue && (
            <motion.div
              ref={pulseRef}
              data-tour-pulse
              aria-hidden="true"
              className="pointer-events-none absolute top-0 left-0 border-2 border-court-400"
              style={{ borderRadius: SPOT_RADIUS, willChange: 'transform' }}
              animate={
                reduced
                  ? undefined
                  : {
                      opacity: [0.9, 0.25, 0.9],
                      boxShadow: [
                        '0 0 0 0 rgba(56,224,162,0.45)',
                        '0 0 0 12px rgba(56,224,162,0)',
                        '0 0 0 0 rgba(56,224,162,0.45)',
                      ],
                    }
              }
              transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
            />
          )}
        </>
      )}

      {anchored && !narrow ? (
        <div key="anchored" ref={posRef} className="absolute top-0 left-0" style={{ willChange: 'transform' }}>
          {bubble}
        </div>
      ) : (
        <div
          key="centered"
          className={cn(
            'absolute flex justify-center',
            // Centered-kind steps (welcome, summary, final) always sit in the
            // middle of the screen — the bottom dock is for anchored fallbacks
            // and info steps on narrow viewports only.
            step.kind === 'centered' ? 'inset-0 items-center' : narrow ? 'inset-x-3 bottom-3' : 'inset-0 items-center',
          )}
        >
          {bubble}
        </div>
      )}
    </div>
  )
}
