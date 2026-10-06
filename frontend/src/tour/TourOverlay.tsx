/** Interactive guided-tour overlay: spotlight + anchored bubble.
 *
 * Rendered once at the app root (no props); renders nothing while the tour is
 * inactive. For anchored steps a spotlight ring (giant box-shadow) follows the
 * ``data-tour`` target via a rAF loop that only writes transform/width/height;
 * the bubble is positioned next to the target and flips to the opposite side
 * when the target hugs a viewport edge (``computePlacement``). Every step
 * resolves within 4s: when the target never appears the bubble falls back to a
 * centered card, so the tour can never dead-end.
 *
 * The overlay never blocks pointer events: the dimming is purely visual, so
 * interactive steps can drive the real UI behind it. Keyboard: ←/→ step,
 * Enter next, Esc quit; Tab is trapped inside the bubble.
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

const STEPS = buildSteps()

const SPOT_MARGIN = 4
const SPOT_RADIUS = 12
const SPOT_SHADOW = '0 0 0 9999px rgba(3,10,8,0.72)'
const BUBBLE_WIDTH = 320
const GAP = 14
const POLL_INTERVAL = 100
/** 100ms × 40 = the 4s cap before an anchored step falls back to centered. */
const POLL_TICKS = 40
const NARROW_VIEWPORT = 640
const SAMPLES_DONE_MS = 1800

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
  const [attempt, setAttempt] = useState(0)
  const [narrow, setNarrow] = useState(() => window.innerWidth < NARROW_VIEWPORT)

  const spotRef = useRef<HTMLDivElement | null>(null)
  const posRef = useRef<HTMLDivElement | null>(null)
  const bubbleRef = useRef<HTMLDivElement | null>(null)
  const targetRef = useRef<HTMLElement | null>(null)
  const stepRef = useRef<TourStep | null>(null)
  const narrowRef = useRef(narrow)
  const reducedRef = useRef(reduced)
  const loadedTimerRef = useRef<number | null>(null)

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
    setSatisfied(step.waitFor ? step.waitFor() : null)

    if (step.view !== useStore.getState().view) useStore.getState().setView(step.view)

    if (!step.target) {
      setMode('centered')
      return
    }

    let cancelled = false
    let ticks = 0

    const actionTarget = step.action?.click
    let actionRaf = 0
    if (actionTarget) {
      actionRaf = requestAnimationFrame(() => {
        if (cancelled) return
        const el = document.querySelector(`[data-tour="${actionTarget}"]`)
        if (el) (el as HTMLElement).click()
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

  // Re-evaluate the step's waitFor gate while the bubble is visible so both
  // store-driven and DOM-driven predicates flip "Next" without wiring here.
  useEffect(() => {
    const step = stepRef.current
    if (!active || mode === 'polling' || !step?.waitFor) return
    const check = () => setSatisfied(step.waitFor ? step.waitFor() : null)
    check()
    const timer = window.setInterval(check, POLL_INTERVAL)
    return () => window.clearInterval(timer)
  }, [active, index, mode, attempt])

  // Spotlight + bubble follow loop: reads the target rect every frame and
  // writes only transform/width/height (compositor friendly).
  const anchored = mode === 'anchored'
  useEffect(() => {
    if (!active || mode !== 'anchored') return
    let raf = 0
    const frame = () => {
      const el = targetRef.current
      const spot = spotRef.current
      if (el && spot) {
        const r = el.getBoundingClientRect()
        if (r.width > 0 || r.height > 0) {
          spot.style.transform = `translate(${r.left - SPOT_MARGIN}px, ${r.top - SPOT_MARGIN}px)`
          spot.style.width = `${r.width + SPOT_MARGIN * 2}px`
          spot.style.height = `${r.height + SPOT_MARGIN * 2}px`
          const pos = posRef.current
          if (pos && !narrowRef.current) {
            const viewport = { width: window.innerWidth, height: window.innerHeight }
            const placement = computePlacement(r, viewport, stepRef.current?.placement ?? 'bottom')
            const bh = pos.offsetHeight || 252
            let x: number
            let y: number
            if (placement === 'bottom') {
              x = r.left + r.width / 2 - BUBBLE_WIDTH / 2
              y = r.bottom + GAP
            } else if (placement === 'top') {
              x = r.left + r.width / 2 - BUBBLE_WIDTH / 2
              y = r.top - GAP - bh
            } else if (placement === 'right') {
              x = r.right + GAP
              y = r.top + r.height / 2 - bh / 2
            } else {
              x = r.left - GAP - BUBBLE_WIDTH
              y = r.top + r.height / 2 - bh / 2
            }
            x = Math.min(Math.max(x, 8), Math.max(8, viewport.width - BUBBLE_WIDTH - 8))
            y = Math.min(Math.max(y, 8), Math.max(8, viewport.height - bh - 8))
            pos.style.transform = `translate(${x}px, ${y}px)`
          }
        }
      }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [active, mode])

  // Global keys while the tour is open. Enter is left to the focused bubble
  // button (it would double-fire next otherwise).
  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const inBubble = !!target?.closest?.('[data-tour-bubble]')
      if (e.key === 'Escape') {
        e.preventDefault()
        useTourStore.getState().skip()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        useTourStore.getState().next()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        useTourStore.getState().prev()
      } else if (e.key === 'Enter' && !inBubble && !target?.closest?.('input, textarea, select')) {
        useTourStore.getState().next()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active])

  // Viewport width class (bottom-docked bubble on narrow windows).
  useEffect(() => {
    if (!active) return
    const onResize = () => setNarrow(window.innerWidth < NARROW_VIEWPORT)
    window.addEventListener('resize', onResize, { passive: true })
    return () => window.removeEventListener('resize', onResize)
  }, [active])

  // Move focus into the bubble whenever its content appears/changes.
  useEffect(() => {
    if (!active || mode === 'polling') return
    const bubble = bubbleRef.current
    if (!bubble) return
    const next = bubble.querySelector<HTMLButtonElement>('[data-tour-next]')
    if (next && !next.disabled) next.focus()
    else bubble.focus()
  }, [active, index, mode])

  // Clear the transient "samples done" timer on unmount.
  useEffect(
    () => () => {
      if (loadedTimerRef.current) window.clearTimeout(loadedTimerRef.current)
    },
    [],
  )

  if (!active) return null
  const step = STEPS[index]
  if (!step || mode === 'polling') return null

  const interactive = step.kind === 'interactive'
  const isLast = index >= STEPS.length - 1
  const nextDisabled = interactive && satisfied !== true

  const runLoadSamples = async () => {
    if (busy) return
    setBusy(true)
    try {
      if (!useStore.getState().project) {
        const pid = await useStore.getState().createProject(tr('tour.samples.projectName'))
        if (!pid) throw new Error('create project failed')
      }
      const { files } = await api.seedSamples()
      await useStore.getState().importMedia(files)
      if (!useStore.getState().currentMedia()) throw new Error('import failed')
      setLoaded(true)
      if (loadedTimerRef.current) window.clearTimeout(loadedTimerRef.current)
      loadedTimerRef.current = window.setTimeout(() => setLoaded(false), SAMPLES_DONE_MS)
    } catch {
      useStore.getState().toast({ kind: 'error', title: tr('tour.samples.failed') })
    } finally {
      setBusy(false)
    }
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
      initial={{ opacity: 0, y: reduced ? 0 : 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.22 }}
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
        {interactive && satisfied !== true && (
          <Button variant="ghost" size="sm" onClick={() => useTourStore.getState().next()}>
            {tr('tour.common.skipStep')}
          </Button>
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
      className="fixed inset-0 z-[75]"
      style={{ pointerEvents: 'none' }}
    >
      {anchored && (
        <div
          ref={spotRef}
          aria-hidden="true"
          className="absolute top-0 left-0"
          style={{
            boxShadow: SPOT_SHADOW,
            borderRadius: SPOT_RADIUS,
            willChange: 'transform',
          }}
        />
      )}
      {anchored && !narrow ? (
        <div ref={posRef} className="absolute top-0 left-0" style={{ willChange: 'transform' }}>
          {bubble}
        </div>
      ) : (
        <div
          className={cn(
            'absolute flex justify-center',
            narrow ? 'inset-x-3 bottom-3' : 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
          )}
        >
          {bubble}
        </div>
      )}
    </div>
  )
}
