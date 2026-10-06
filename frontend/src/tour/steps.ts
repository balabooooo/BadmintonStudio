/** Guided-tour step table: pure data consumed by the overlay engine (TourOverlay).
 *
 * One step per spec §4 table row (29 rows), with row 12 (analysis-dialog
 * parameters) split into an info step on the parameters block and an
 * interactive "close the dialog" step -> 30 steps in total.
 *
 * Steps reference the ``data-tour`` anchors added to the components by the
 * anchor task; copy lives in the i18n fragment ``tour.*`` (both languages),
 * never inline. Interactive steps gate "Next" behind ``waitFor``; the user
 * can always skip an individual step from the bubble.
 */

import { useStore } from '../store/useStore'

export type TourKind = 'info' | 'interactive' | 'centered'

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
  /** "Next" stays disabled until this returns true (interactive steps). */
  waitFor?: () => boolean
  /** Anchor programmatically clicked when the step is entered (e.g. open the analysis dialog). */
  action?: { click: string }
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

/**
 * True when the inspector's "clip" tab is the active one.
 *
 * The Segmented control marks the active option with the ``text-ink-950``
 * text-color class on the button itself (no aria-selected/data-active exists),
 * and the clip tab is the 2nd of the three tab buttons inside the anchor.
 * Works whether the anchor lands on the tab-bar container or on the
 * Segmented root: nth-of-type counts within the buttons' own parent.
 */
const clipTabActive = () =>
  !!document.querySelector('[data-tour="insp-tabs"] button:nth-of-type(2).text-ink-950')

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
      waitFor: hasProject,
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
    // 6. topbar: open the analysis dialog
    {
      id: 'btn-analysis',
      view: 'studio',
      target: 'btn-analysis',
      kind: 'interactive',
      group: 'topbar',
      titleKey: 'tour.analysis.title',
      bodyKey: 'tour.analysis.body',
      action: { click: 'btn-analysis' },
      waitFor: dialogOpen,
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
    // 12b. dialog: close it (row 12 split, part 2; user clicks the dialog's own close button)
    {
      id: 'dlg-close',
      view: 'studio',
      target: 'dlg-close',
      kind: 'interactive',
      group: 'dialog',
      titleKey: 'tour.dlg.close.title',
      bodyKey: 'tour.dlg.close.body',
      waitFor: dialogClosed,
    },
    // 13. topbar: export
    {
      id: 'btn-export',
      view: 'studio',
      target: 'btn-export',
      kind: 'info',
      group: 'topbar',
      titleKey: 'tour.export.title',
      bodyKey: 'tour.export.body',
    },
    // 14. left column: rally list
    {
      id: 'rally-panel',
      view: 'studio',
      target: 'rally-panel',
      kind: 'info',
      group: 'main',
      titleKey: 'tour.rallies.title',
      bodyKey: 'tour.rallies.body',
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
      waitFor: isPlaying,
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
    // 19. inspector: switch to the clip tab
    {
      id: 'insp-tabs',
      view: 'studio',
      target: 'insp-tabs',
      kind: 'interactive',
      group: 'inspector',
      titleKey: 'tour.insp.tabs.title',
      bodyKey: 'tour.insp.tabs.body',
      waitFor: clipTabActive,
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
    // 23. inspector / clip: splitting
    {
      id: 'insp-clip-split',
      view: 'studio',
      target: 'insp-clip-split',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.clipSplit.title',
      bodyKey: 'tour.insp.clipSplit.body',
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
    // 25. inspector / info: activity chart
    {
      id: 'insp-info-chart',
      view: 'studio',
      target: 'insp-info-chart',
      kind: 'info',
      group: 'inspector',
      titleKey: 'tour.insp.infoChart.title',
      bodyKey: 'tour.insp.infoChart.body',
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
    },
    // 27. studio summary (centered card)
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
    // 29. final (centered card)
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
