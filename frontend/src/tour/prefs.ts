/** Tour preferences persisted in localStorage: first-run detection + re-run flag.
 *
 * Hand-written persistence (no zustand persist middleware), mirroring the
 * i18n runtime's defensive pattern: every localStorage access is wrapped in
 * try/catch and any malformed payload is treated as "no preferences", which
 * makes the tour auto-start again rather than breaking the app.
 */

export const TOUR_PREFS_KEY = 'bms.tour.v1'

export interface TourPrefs {
  /** True once the user has gone through (or dismissed) the tour at least once. */
  seen: boolean
  /** True only when the tour was walked to its final step. */
  completed: boolean
}

function isTourPrefs(v: unknown): v is TourPrefs {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as TourPrefs).seen === 'boolean' &&
    typeof (v as TourPrefs).completed === 'boolean'
  )
}

/** Read the stored tour prefs; null when the key is missing, corrupted, or has an invalid shape. */
export function loadTourPrefs(): TourPrefs | null {
  try {
    if (typeof localStorage === 'undefined') return null
    const raw = localStorage.getItem(TOUR_PREFS_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    return isTourPrefs(parsed) ? parsed : null
  } catch {
    // Corrupted JSON or storage access blocked: treat as first run.
    return null
  }
}

/** Persist tour prefs; failures (private mode, quota) are silently ignored. */
export function saveTourPrefs(p: TourPrefs): void {
  try {
    if (typeof localStorage === 'undefined') return
    localStorage.setItem(TOUR_PREFS_KEY, JSON.stringify(p))
  } catch {
    /* prefs are best-effort */
  }
}

/** True when the tour has never been recorded (missing or unparseable prefs). */
export function isFirstRun(): boolean {
  return loadTourPrefs() === null
}
