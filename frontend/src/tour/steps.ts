/** Guided-tour step table: pure data consumed by the overlay engine (TourOverlay).
 *
 * One step per spec §4 table row (29 rows), with row 12 (analysis-dialog
 * parameters) split into an info step, an interactive "run the (mock)
 * analysis" step and an interactive "close the dialog" step -> 31 steps.
 *
 * Steps reference the ``data-tour`` anchors added to the components by the
 * anchor task; copy lives in the i18n fragment ``tour.*`` (both languages),
 * never inline. Interactive steps gate "Next" behind ``waitFor``; they carry
 * a one-line ``hintKey`` instruction and optionally an ``assist`` action the
 * bubble's "do it for me" button can perform. The user can always skip an
 * individual step from the bubble.
 */

import { useStore } from '../store/useStore'

export type TourKind = 'info' | 'interactive' | 'centered'

/** "Do it for me" action attached to an interactive step.
 * - ``click``: programmatically click the target anchor (or ``selector`` inside it)
 * - ``createProject``: create a project directly, skipping the name-and-confirm modal
 * - ``play``: start playback via the store
 */
export interface TourAssist {
  kind: 'click' | 'createProject' | 'play'
  /** Optional CSS selector scoped inside the target anchor (e.g. a specific tab). */
  selector?: string
}

export interface TourStep {
  id: string
  view: 'library' | 'studio' | 'annotate' | 'exports' | 'settings'
  /** ``data-tour`` anchor; centered steps omit it. */
  target?: string
  placement?: 'top' | 'bottom' | 'left' | 'right'
  kind: TourKind
  group:
    | 'welcome'
    | 'nav'
    | 'library'
    | 'samples'
    | 'topbar'
    | 'dialog'
    | 'main'
    | 'inspector'
    | 'summary'
    | 'settings'
    | 'final'
  titleKey: string
  bodyKey: string
  /** Group summary rendered at the bottom of the bubble (reserved; currently unused). */
  summaryKey?: string
  /** One-line concrete instruction ("click the highlighted X, then Y") shown
   * on interactive steps while their waitFor is unsatisfied. */
  hintKey?: string
  /** "Next" stays disabled until this returns true (interactive steps). */
  waitFor?: () => boolean
  /** Anchor programmatically clicked when the step is entered (e.g. open the analysis dialog). */
  action?: { click: string; selector?: string }
  /** Optional "do it for me" action for users who get stuck. */
  assist?: TourAssist
  customAction?: 'loadSamples'
}

/* ------------------------------------------------------------------ waitFor predicates */

/** True once a project is open (library interactive step). */
const hasProject = () => !!useStore.getState().project

/** True once any media is loaded; already-loaded media satisfies it immediately. */
const hasMedia = () => !!useStore.getState().currentMedia()

/** True while the analysis dialog is mounted. */
const dialogOpen = () => !!document.querySelector('[data-tour="dlg-scope"]')

/** True once the analysis dialog has been closed again. */
const dialogClosed = () => !document.querySelector('[data-tour="dlg-scope"]')

/** True while the player is actually playing. */
const isPlaying = () => useStore.getState().playing

/** True once the flagged mock analysis of the current media is done. A real
 * analysis never satisfies it, even on a replay tour with real data present. */
const mockAnalysisReady = () => {
  const a = useStore.getState().currentAnalysis()
  return !!a && a.status === 'done' && a.stats?.mock === true
}

/* ------------------------------------------------------------------ step table */

/** The full tour: spec §4 rows in table order, row 12 split -> 30 steps. */
export function buildSteps(): TourStep[] {
  return [
    // 1. welcome (centered card)
    {
      id: 'welcome',
      view: 'library',
      kind: 'centered',
      group: 'welcome',
      titleKey: 'tour.welcome.title',
      bodyKey: 'tour.welcome.body',
    },
    // 2. global navigation
    {
      id: 'nav-rail',
      view: 'library',
      target: 'nav-rail',
      kind: 'info',
      group: 'nav',
      titleKey: 'tour.nav.title',
      bodyKey: 'tour.nav.body',
    },
    // 3. library: create/open a project
    {
      id: 'library-new-project',
      view: 'library',
      target: 'library-new-project',
      kind: 'interactive',
      group: 'library',
      titleKey: 'tour.library.title',
      bodyKey: 'tour.library.body',
      hintKey: 'tour.library.hint',
      waitFor: hasProject,
      assist: { kind: 'createProject' },
    },
    // 4. sample media (bubble button runs the seed+import chain)
    {
      id: 'studio-empty',
      view: 'studio',
      target: 'studio-empty',
      kind: 'interactive',
      group: 'samples',
      titleKey: 'tour.samples.title',
      bodyKey: 'tour.samples.body',
      hintKey: 'tour.samples.hint',
      customAction: 'loadSamples',
      waitFor: hasMedia,
    },
    // 5. topbar: court calibration
    {
      id: 'btn-court',
      view: 'studio',
      target: 'btn-court',
      kind: 'info',
      group: 'topbar',
      titleKey: 'tour.court.title',
      bodyKey: 'tour.court.body',
    },
    // 6. topbar: open the analysis dialog (user clicks the highlighted button;
    //     no auto-click so the interaction stays intentional and the button
    //     reads as a normal, enabled control rather than auto-triggering).
    {
      id: 'btn-analysis',
      view: 'studio',
      target: 'btn-analysis',
      kind: 'interactive',
      group: 'topbar',
      titleKey: 'tour.analysis.title',
      bodyKey: 'tour.analysis.body',
      hintKey: 'tour.analysis.hint',
      waitFor: dialogOpen,
      assist: { kind: 'click' },
    },
    // 7. dialog: analysis scope
    {
      id: 'dlg-scope',
      view: 'studio',
      target: 'dlg-scope',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.scope.title',
      bodyKey: 'tour.dlg.scope.body',
    },
    // 8. dialog: module toggles
    {
      id: 'dlg-modules',
      view: 'studio',
      target: 'dlg-modules',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.modules.title',
      bodyKey: 'tour.dlg.modules.body',
    },
    // 9. dialog: viewpoint & calibration
    {
      id: 'dlg-viewpoint',
      view: 'studio',
      target: 'dlg-viewpoint',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.viewpoint.title',
      bodyKey: 'tour.dlg.viewpoint.body',
    },
    // 10. dialog: person-box size filter
    {
      id: 'dlg-size-filter',
      view: 'studio',
      target: 'dlg-size-filter',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.sizeFilter.title',
      bodyKey: 'tour.dlg.sizeFilter.body',
    },
    // 11. dialog: scoring weights
    {
      id: 'dlg-weights',
      view: 'studio',
      target: 'dlg-weights',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.weights.title',
      bodyKey: 'tour.dlg.weights.body',
    },
    // 12a. dialog: key parameters (row 12 split, part 1)
    {
      id: 'dlg-params',
      view: 'studio',
      target: 'dlg-params',
      kind: 'info',
      group: 'dialog',
      titleKey: 'tour.dlg.params.title',
      bodyKey: 'tour.dlg.params.body',
    },
    // 12b. dialog: run the analysis. While the tour is active the store
    // intercepts this button and runs the flagged mock pipeline (no GPU work).
    {
      id: 'dlg-run',
      view: 'studio',
      target: 'dlg-run',
      placement: 'top',
      kind: 'interactive',
      group: 'dialog',
      titleKey: 'tour.dlg.run.title',
      bodyKey: 'tour.dlg.run.body',
      hintKey: 'tour.dlg.run.hint',
      waitFor: mockAnalysisReady,
      assist: { kind: 'click' },
    },
    // 12c. dialog: close it (user clicks the dialog's own close button)
    {
      id: 'dlg-close',
      view: 'studio',
      target: 'dlg-close',
      kind: 'interactive',
      group: 'dialog',
      titleKey: 'tour.dlg.close.title',
      bodyKey: 'tour.dlg.close.body',
      hintKey: 'tour.dlg.close.hint',
      waitFor: dialogClosed,
      assist: { kind: 'click' },
    },
    // 13. left column: rally list
    {
      id: 'rally-panel',
      view: 'studio',
      target: 'rally-panel',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.rallies.title',
      bodyKey: 'tour.rallies.body',
    },
    // 14b. left column: add filtered rallies to the film bar
    {
      id: 'rally-film-bar',
      view: 'studio',
      target: 'rally-film-bar',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.rallyFilm.title',
      bodyKey: 'tour.rallyFilm.body',
    },
    // 14c. left column: filter rallies
    {
      id: 'rally-filter',
      view: 'studio',
      target: 'rally-filter',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.rallyFilter.title',
      bodyKey: 'tour.rallyFilter.body',
    },
    // 15. player: try playing
    {
      id: 'player',
      view: 'studio',
      target: 'player',
      kind: 'interactive',
      group: 'main',
      titleKey: 'tour.player.title',
      bodyKey: 'tour.player.body',
      hintKey: 'tour.player.hint',
      waitFor: isPlaying,
      assist: { kind: 'play' },
    },
    // 16. player: preview mode
    {
      id: 'player-mode',
      view: 'studio',
      target: 'player-mode',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.playerMode.title',
      bodyKey: 'tour.playerMode.body',
    },
    // 16b. player: source / filtered playback switch
    {
      id: 'rally-play-mode',
      view: 'studio',
      target: 'player-mode',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.rallyPlayMode.title',
      bodyKey: 'tour.rallyPlayMode.body',
    },
    // 17. timeline: film track & toolbar
    {
      id: 'timeline',
      view: 'studio',
      target: 'timeline',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.timeline.title',
      bodyKey: 'tour.timeline.body',
    },
    // 18. timeline: zoom & playhead (group summary embedded in the body)
    {
      id: 'timeline-zoom',
      view: 'studio',
      target: 'timeline-zoom',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.timelineZoom.title',
      bodyKey: 'tour.timelineZoom.body',
    },
    // 19. inspector: introduce the three tabs (rally is active by default so
    //     the rally-tab steps that follow resolve immediately)
    {
      id: 'insp-tabs',
      view: 'studio',
      target: 'insp-tabs',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.tabs.title',
      bodyKey: 'tour.insp.tabs.body',
    },
    // 20. inspector / rally: summary
    {
      id: 'insp-rally-summary',
      view: 'studio',
      target: 'insp-rally-summary',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.rallySummary.title',
      bodyKey: 'tour.insp.rallySummary.body',
    },
    // 21. inspector / rally: scores
    {
      id: 'insp-rally-scores',
      view: 'studio',
      target: 'insp-rally-scores',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.rallyScores.title',
      bodyKey: 'tour.insp.rallyScores.body',
    },
    // 22. inspector / rally: trim range
    {
      id: 'insp-rally-range',
      view: 'studio',
      target: 'insp-rally-range',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.rallyRange.title',
      bodyKey: 'tour.insp.rallyRange.body',
    },
    // 22b. inspector / rally: adjust start (in-point)
    {
      id: 'insp-rally-start',
      view: 'studio',
      target: 'insp-rally-start',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.rallyRangeStart.title',
      bodyKey: 'tour.rallyRangeStart.body',
    },
    // 22c. inspector / rally: adjust end (out-point)
    {
      id: 'insp-rally-end',
      view: 'studio',
      target: 'insp-rally-end',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.rallyRangeEnd.title',
      bodyKey: 'tour.rallyRangeEnd.body',
    },
    // 23. inspector / clip: splitting (switch to the clip tab on entry)
    {
      id: 'insp-clip-split',
      view: 'studio',
      target: 'insp-clip-split',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.clipSplit.title',
      bodyKey: 'tour.insp.clipSplit.body',
      action: { click: 'insp-tabs', selector: 'button:nth-of-type(2)' },
    },
    // 24. inspector / clip: speed & volume
    {
      id: 'insp-clip-speed',
      view: 'studio',
      target: 'insp-clip-speed',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.clipSpeed.title',
      bodyKey: 'tour.insp.clipSpeed.body',
    },
    // 25. inspector / info: activity chart (switch to the info tab on entry)
    {
      id: 'insp-info-chart',
      view: 'studio',
      target: 'insp-info-chart',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.infoChart.title',
      bodyKey: 'tour.insp.infoChart.body',
      action: { click: 'insp-tabs', selector: 'button:nth-of-type(3)' },
    },
    // 26. inspector / info: analysis stats (right-column summary embedded in the body)
    {
      id: 'insp-info-stats',
      view: 'studio',
      target: 'insp-info-stats',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.infoStats.title',
      bodyKey: 'tour.insp.infoStats.body',
      // ensure the info tab is active even if the user switched tabs manually
      action: { click: 'insp-tabs', selector: 'button:nth-of-type(3)' },
    },
    // 27. topbar: export (after the rally workflow so the film is ready)
    {
      id: 'btn-export',
      view: 'studio',
      target: 'btn-export',
      kind: 'info',
      group: 'topbar',
      titleKey: 'tour.export.title',
      bodyKey: 'tour.export.body',
    },
    // 28. studio summary (centered card)
    {
      id: 'summary',
      view: 'studio',
      kind: 'centered',
      group: 'summary',
      titleKey: 'tour.center.studioSummary.title',
      bodyKey: 'tour.center.studioSummary.body',
    },
    // 28. settings: tour entry & language
    {
      id: 'settings-guide',
      view: 'settings',
      target: 'settings-guide',
      kind: 'info',
      group: 'settings',
      titleKey: 'tour.settings.title',
      bodyKey: 'tour.settings.body',
    },
    // 29. settings: debug log export for issue reports
    {
      id: 'settings-debug-logs',
      view: 'settings',
      target: 'settings-debug-logs',
      kind: 'info',
      group: 'settings',
      titleKey: 'tour.settings.debugLogsTitle',
      bodyKey: 'tour.settings.debugLogsBody',
    },
    // 30. settings: selective cache cleanup
    {
      id: 'settings-clear-cache',
      view: 'settings',
      target: 'settings-clear-cache',
      kind: 'info',
      group: 'settings',
      titleKey: 'tour.settings.clearCacheTitle',
      bodyKey: 'tour.settings.clearCacheBody',
    },
    // 31. final (centered card)
    {
      id: 'final',
      view: 'settings',
      kind: 'centered',
      group: 'final',
      titleKey: 'tour.final.title',
      bodyKey: 'tour.final.body',
    },
  ]
}
