# Unified Media Picker — Design Spec

- Date: 2026-10-05
- Status: Approved (design), pending implementation plan
- Scope: Frontend only (no backend changes)
- Related code: `frontend/src/components/StudioPage.tsx`, `AnnotatePage.tsx`, `AnalysisDialog.tsx`, `frontend/src/store/useStore.ts`

## 1. Problem

Media selection is scattered across two modules and missing from a third:

1. **Studio (剪辑台)** — `StudioPage.tsx` owns the only full media UI: a `MediaPanel`
   hidden behind a "Rallies / Media" tab in the left sidebar (cards with poster,
   duration, analysis badge, prepare progress, import, single delete, batch
   analyze/delete).
2. **AI Analysis settings** — `AnalysisDialog.tsx` only offers an "All media /
   Current media" radio pair; the user cannot see or pick which concrete media is
   "current" from inside the dialog.
3. **Annotate (标注)** — `AnnotatePage.tsx` only renders the current media name as
   static text. Switching media requires: back to Studio via the nav rail → open
   the Media tab → click a card → return to Annotate.

There is a single global `mediaId` in the Zustand store, switched via
`selectMedia()`, but three different surfaces expose (or fail to expose) it.

## 2. Goals / Non-Goals

### Goals

1. One centralized media browsing/selection surface: a **global media picker
   dialog** reachable from everywhere.
2. Fast media switching **without leaving the Annotate page**, without losing
   unsaved annotation edits.
3. Seamless integration with the existing AI Analysis settings dialog and the
   Studio.
4. Fewer navigation jumps; all media management operations (import, delete,
   batch analyze/delete, prepare-progress visibility) available in the one
   surface.

### Non-Goals (YAGNI)

- No media drag-to-reorder, rename, folders, or grouping.
- Hover preview never triggers proxy generation; no preview speed/cycle options.
- No new backend endpoint; no dedicated "media center" full page.
- No changes to scoring, segmentation, export, or analysis parameter semantics.

## 3. Chosen Approach (Approved)

A single **global modal dialog** mounted once at the app root (sibling of
`CourtEditor`), opened through store actions. Grid-card layout (Option A) with
in-card hover preview. The Studio left sidebar loses its Media tab and becomes a
rally-only list; the read-only media path text in the Studio top bar is upgraded
to a clickable media chip. The Analysis dialog gains a media chip with a
"Change" button that opens the picker in select mode.

Decisions confirmed during brainstorming:

- Carrier: **global modal dialog** (not a persistent sidebar, not a new page).
- Capability scope: **full consolidation** — import, single delete, batch
  select → batch analyze/delete, prepare progress all live in the dialog.
- Analysis integration: **media chip + "Change" button** that opens the picker;
  analysis params/weights/ROI are preserved across the pick.
- Hover preview starts at **t = 0**, plays **muted at 1× for 6 seconds**, then
  holds the last frame.
- Keyboard: arrow keys move the selection cursor (no video loading); **Space**
  plays the focused card's preview; **Enter** confirms the switch.

## 4. Interface Layout

### 4.1 Media Picker Dialog (~880px wide, max 86vh)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Media (8)                                                       [X]  │
├──────────────────────────────────────────────────────────────────────┤
│ [🔍 Search file name…]  [All status ▾]   [+ Import video] [☑ Batch]  │
├──────────────────────────────────────────────────────────────────────┤
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│ │ poster   │ │ poster   │ │ poster   │ │ preparing│                   │
│ │ 28:14 ✨36│ │ 41:02 ✨52│ │ 15:30    │ │  ⏳ 64%  │                   │
│ │ name…    │ │ name…    │ │ name…    │ │ name…    │                   │
│ │1080p 60fp│ │ 4K 30fps │ │not analyzed│ │         │                   │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘   (auto-fill     │
│ ┌──────────┐ ┌──────────┐               ...            minmax(180px)) │
│ │ …        │ │ …        │                                           │
├──────────────────────────────────────────────────────────────────────┤
│ ● Current: training_1005.mp4              [Delete] [Switch to media ↵]│
└──────────────────────────────────────────────────────────────────────┘
```

- **Toolbar**: search input (file name substring), status filter dropdown
  (All / Analyzed / Not analyzed / Preparing preview), `ImportVideoButton`
  (files + folder via the existing component's props), batch-mode toggle.
- **Grid**: `grid-template-columns: repeat(auto-fill, minmax(180px, 1fr))`,
  scrollable body. Each card reuses the existing MediaPanel card visuals:
  poster image (`api.assetUrl(m.poster)`) with film-icon fallback, duration
  timecode, ✨ rally-count badge when analysis is done, proxy-ready badge,
  prepare progress overlay while a `prepare` job is active for that media,
  resolution/fps/size meta line, truncating name with full path as `title`.
- **Card states**: default (static poster) / hover-preview (in-card video) /
  selected (green ring) / non-previewable (poster + prepare progress; no video
  mounts).
- **Footer, normal mode**: current-media indicator on the left; on the right a
  destructive Delete (acts on the selected card) plus the primary confirm
  button, whose label is mode-dependent:
  - `mode='switch'` → "Switch to media ⏎", calls `selectMedia(id)` and closes;
    disabled when the selected card is already the current media.
  - `mode='select'` → "Select this media", invokes the stored pick callback and
    closes; likewise disabled for the current media.
- The per-card trash icon (top-right, low-opacity until hover/keyboard focus —
  never exclusively hover-dependent) is a second, equivalent Delete entry for
  that card, matching the existing MediaPanel behavior.
- **Footer, batch mode**: "Selected n" / Select all / Batch analyze / Batch
  delete (confirmation dialog via existing `useConfirm`) / Done. Hover preview
  is disabled in batch mode to avoid conflicting click semantics: a single
  click anywhere on the card toggles its checkbox, double click does not
  switch. Cards with a running/queued analyze job show a disabled checkbox
  (with the live progress % as the reason), so they participate in neither
  batch delete nor duplicate batch-analyze submission.

### 4.2 Media Chip (shared trigger)

A compact button: small poster thumbnail + truncated media name + caret. Opens
the picker in `switch` mode. Placed in:

- **Annotate top bar** — replaces the static `media.name` text.
- **Studio top bar** — replaces the read-only hard-drive path line; the full
  file path moves into the chip's tooltip.
- **Analysis dialog** — rendered next to the "Current media" scope option with a
  "Change" button (opens picker in `select` mode).

### 4.3 Hover Preview Interaction

The preview plays **inside the poster area itself** (`absolute inset-0`
replacement), never as a detached floating layer. This deliberately avoids the
known failure where controls inside a mouseleave-dismissed float cannot be
clicked. Confirm/delete/batch actions are always reachable without hovering.

Card state machine:

| State | Trigger | Behavior |
|---|---|---|
| Default | — | Static poster, persistent badges/meta |
| Previewing | pointer stays over poster ≥ 350 ms and `proxy_path` exists | Mount one `<video muted playsInline preload="none">`, seek to 0, play at 1×; show 3px bottom progress bar and `current / total` timecode; stop after 6 s holding last frame; re-enter replays |
| Selected | single click | Immediately stop/unmount preview, restore poster, green ring |
| Non-previewable | no proxy / prepare job active / analyze running | Keep poster (+ progress overlay); hover does nothing |
| Confirmed | double click / Enter | Switch/select media and close dialog |

Timeline of one hover:

1. `mouseenter` → start 350 ms debounce (rapid sweep across the grid mounts
   nothing).
2. Dwell ≥ 0.35 s with a proxy available → mount the video. The dialog keeps
   at most **one** active preview (`activePreviewId` state in the dialog).
3. Start at t = 0, muted, 1× speed, max 6 s; progress bar + live timecode.
4. `mouseleave`, click, card scrolled out of view (IntersectionObserver), or
   dialog close → pause and unmount the element immediately.
5. No prefetching: only the hovered card's video loads. Closing the dialog
   unmounts all preview videos.

Keyboard: arrow keys move the grid cursor without loading video; Space plays
the focused card preview (same 6 s rule); Enter confirms; Escape closes
(handled by the existing Modal stack, which also correctly stacks above the
Analysis dialog). Touch/touchpad devices have no hover — single tap selects,
all actions remain available.

## 5. Interaction Flows

### Flow 1 — Switch media while annotating (core pain point)

1. Click the media chip in the Annotate top bar → picker opens; the Annotate
   page stays mounted under the modal overlay.
2. Hover-preview / double-click target media (or single click + footer confirm).
   Escape / backdrop click cancels with no change.
3. On confirm, the Annotate page first **flushes the current annotation to the
   old media id immediately** (not waiting for the 1.2 s autosave debounce),
   then `selectMedia(newId)` runs.
4. The existing `reload()` effect keyed on `mid` loads the new media's auto
   drafts / manual rallies / hits / focus independently.

### Flow 2 — Change media from AI Analysis settings

1. Analysis dialog open; scope radio shows "All / Current".
2. Click "Change" on the current-media chip → picker opens stacked above the
   Analysis dialog (select mode).
3. Confirm → `selectMedia(id)` runs, the picker closes, scope is set to
   "Current media". Analysis params, weights, ROI, and dialog state are
   untouched.

### Flow 3 — Batch management

1. "☑ Batch" enters multi-select; hover preview is disabled.
2. Check cards / select all → footer offers Batch analyze / Batch delete
   (confirmed) / Done.
3. Batch analyze calls the existing `runAnalysisBatch(ids)` (serial server
   queue). Deleting the currently selected media falls through to the first
   remaining media (existing store behavior). Media being analyzed cannot be
   deleted.

### Flow 4 — Studio integration

1. Remove the Media tab and `MediaPanel`; the left sidebar renders
   `RallyPanel` only. Remove the effect that auto-switches the tab by analysis
   status.
2. The top-bar media path text becomes the shared `MediaChip` (path in tooltip).
3. Import entry points move into the picker; the empty-project central state's
   import buttons and the window-wide drag-and-drop import remain unchanged, so
   first-media import is unaffected.

## 6. Technical Design

### 6.1 New files

- `frontend/src/components/MediaPickerDialog.tsx` — the global dialog. Built on
  the existing `Modal` (portal, ESC stack, backdrop close already provided).
  Internal subcomponents in the same file: toolbar, `MediaCard`, `HoverPreview`,
  footer (normal / batch). Data comes from `useStore`: `project.media`,
  `project.analyses`, `jobs`, `mediaId`, and the existing actions
  `selectMedia`, `removeMedia`, `removeMediaBulk`, `runAnalysisBatch`,
  `ensurePrepare`, plus `ImportVideoButton` and `useConfirm`.
- `frontend/src/components/MediaChip.tsx` — shared trigger button.
- `frontend/src/i18n/catalog/fragments/media.ts` — `[key, 中文, English]`
  tuples under a new `mediaPicker.*` namespace, registered in
  `i18n/catalog/fragments/index.ts`. Existing `studio.*` / `common.*` keys are
  reused where wording already exists. Both languages must be added; the backend
  catalog is untouched.

### 6.2 Store changes (`useStore.ts`)

Add state: `mediaPickerOpen: boolean`, `mediaPickerMode: 'switch' | 'select'`.
Add actions: `openMediaPicker(mode?: 'switch' | 'select')`,
`closeMediaPicker()`.

For select-mode result delivery, keep a module-level non-reactive callback ref
(`pickResolverRef`) consumed by `openMediaPicker('select', { onPick })`; this
avoids putting a function into React state and causing extra renders.

`selectMedia()` is reused unchanged (selection reset, preview-mode reset,
ensurePrepare, conditional per-media rescore).

### 6.3 Annotate flush-on-switch

In `AnnotatePage.tsx`, subscribe once (mount-only effect) via
`useStore.subscribe((next, prev) => …)`: when `mediaId` changes while
`project.id` stays the same, build the payload from the local `stateRef`
snapshot (ann / hits / focus / note) and issue one
`api.saveAnnotation(prevProjectId, prevMediaId, payload)` directly — without
waiting for the debounced save and without awaiting completion before allowing
the new media's `reload()`. The pending debounce timer for the old id is
cleared. The request targets the previous media id explicitly, so it cannot
clobber the newly loaded media.

### 6.4 Integration edits

- `App.tsx`: mount `<MediaPickerDialog />` once at the root.
- `StudioPage.tsx`: delete `MediaPanel`, the media/rallies tab state and the
  auto-tab effect; sidebar renders `RallyPanel`; replace the path line with
  `<MediaChip />`; keep empty-state import buttons and drag-drop import.
- `AnalysisDialog.tsx`: show the current media poster + name in the "Current
  media" scope option with a "Change" button wired to
  `openMediaPicker('select', { onPick: (mid) => { selectMedia(mid);
  setScope('current') } })`.
- `AnnotatePage.tsx`: chip in the top bar; flush-on-switch subscription
  (6.3).

### 6.5 Error handling

- Picker actions reuse existing store toasts (import failure/partial failure,
  delete failure, rescore failures are already surfaced there).
- Video element `onError` in `HoverPreview` silently falls back to the poster;
  previews are non-essential.
- Switching is disabled when there is no project / no media; the chip is not
  rendered in those states (Annotate already has its empty state).

## 7. Verification

Automated/static:

- `cd frontend && npm run build` (`tsc -b` strict + Vite).
- `cd frontend && npm run lint` (oxlint).
- Backend untouched; `tests/test_core.py` (including i18n key-set parity) is
  unaffected because only frontend catalogs change.

Manual acceptance (desktop app / dev server):

1. In Annotate, edit a rally, immediately switch media via the chip: the last
   edit is persisted against the old media; reopening it shows the edit; the
   new media loads its own independent annotation set.
2. Hover preview: 350 ms dwell starts muted playback at t=0, stops at 6 s,
   leaves on click/leave/scroll; only one preview plays at a time; no preview
   for proxy-missing/preparing media; Space/Enter/arrows work by keyboard.
3. Analysis dialog → Change → pick media: dialog reopens with params/weights/
   ROI intact and scope set to Current.
4. Batch mode: multi-select analyze enqueues jobs; batch delete confirms and
   removes; deleting current media falls through to the first remaining;
   analyzing media cannot be deleted; import from within the dialog works and
   selects the newly imported media.
5. Studio: sidebar is rally-only; chip opens the picker; empty-project import
   and drag-drop import still work.

## 8. Risks / Notes

- The flush-on-switch save races the 1.2 s debounce timer: clear the timer
  before issuing the explicit save so the old snapshot is not written twice or
  against the new id.
- The picker is mounted at app root and reads `currentMedia()`/`jobs` reactively;
  analysis and prepare job badges update live via the existing WS subscription,
  no new polling.
- Hover videos must use the proxy URL with the existing `proxy_path` cache-bust
  query (same pattern as the Annotate `<video>`).
