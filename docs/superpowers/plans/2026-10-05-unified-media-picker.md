# Unified Media Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered media-selection UI with one global media-picker dialog (grid + hover preview) reachable from Studio, Annotate, and the AI Analysis dialog, without losing unsaved annotations on switch.

**Architecture:** Frontend-only. A single `MediaPickerDialog` mounted at the app root is controlled by new Zustand state (`mediaPickerOpen`/`mediaPickerMode` + a module-level select-mode callback). A shared `MediaChip` triggers it from the three surfaces. The Studio `MediaPanel` is deleted; Annotate flushes its autosave snapshot to the previous media id on switch.

**Tech Stack:** React 19 + TypeScript, Zustand, Tailwind v4, `motion/react`, lucide-react; existing `Modal/Button/Badge/Progress/Tooltip/useConfirm` primitives; no new dependencies, no backend changes.

**Spec:** `docs/superpowers/specs/2026-10-05-unified-media-picker-design.md` (read alongside this plan)

## Global Constraints

- The project has **no JS test framework**. Per-task verification is `npm run build` (`tsc -b` strict, treats unused locals as errors in this project) + `npm run lint` (oxlint), plus the listed manual browser checks. Do NOT add a test framework.
- All user-facing strings go through i18n: new strings live in a frontend fragment as `[key, 中文, English]` tuples, BOTH languages filled; never hardcode zh/en in components. Code comments and docstrings stay English.
- Animation library is `motion/react` (not `framer-motion`).
- Reuse store actions as-is: `selectMedia`, `removeMedia`, `removeMediaBulk`, `runAnalysisBatch`, `ensurePrepare`; reuse `ImportVideoButton` and `useConfirm`.
- Proxy `<video src>` must use `` `${api.proxyUrl(pid, mid)}?v=${encodeURIComponent(m.proxy_path ?? '')}` ``; posters use `api.assetUrl(m.poster)`.
- Commits (if the user approves committing during execution): Conventional Commits, English subject/body, only the task's own files staged.
- Frontend commands run from `frontend/`: build `npm run build`, lint `npm run lint`.

## Review Focus

1. **Project switch must not flush a media annotation** — the Annotate `subscribe` listener only saves when `project.id` is unchanged and `mediaId` changed. Pinned by a manual check in Task 4.
2. **Keyboard preview/arrows must not hijack typing or buttons** — the grid key handler ignores events originating from `input/textarea/contenteditable` and from a `button` (Space/Enter activate the focused button natively). Pinned in Task 3.
3. **Rapid media switches must save each old media once** — the flush listener keys every save off `prev.mediaId` and never reads the current `mid`; switching A→B→C fires one save for A and one for B. Pinned in Task 4.
4. **Preview videos must never leak or double-play** — exactly one `activePreviewId`; unmounting on mouseleave/arrows/dialog-close pauses and removes the `<video>` (no detached audio). Pinned in Task 3.
5. **Batch select-all must exclude cards with running/queued analyze jobs** (their checkbox is disabled); batch-deleting the current media falls through to the first remaining media (existing store behavior). Pinned in Task 2.

---

### Task 1: i18n fragment + picker state in the store

**Files:**
- Create: `frontend/src/i18n/catalog/fragments/media.ts`
- Modify: `frontend/src/i18n/catalog/fragments/index.ts`
- Modify: `frontend/src/store/useStore.ts`

**Interfaces:**
- Produces: exported `FRAG` tuple array; store state `mediaPickerOpen: boolean`, `mediaPickerMode: 'switch' | 'select'`; actions `openMediaPicker: (mode?: 'switch' | 'select', onPick?: (mid: string) => void) => void` and `closeMediaPicker: () => void`; module-level handler `mediaPickHandler: ((mid: string) => void) | null` consumed later by the dialog's select-mode confirm.

- [ ] **Step 1: Create `media.ts` fragment with exactly these tuples**

```ts
/** i18n fragment: unified media picker. [key, 中文, English] */
export const FRAG: [string, string, string][] = [
  ['mediaPicker.title', '素材（{n}）', 'Media ({n})'],
  ['mediaPicker.searchPlaceholder', '搜索文件名…', 'Search file name…'],
  ['mediaPicker.filterAll', '全部状态', 'All statuses'],
  ['mediaPicker.filterAnalyzed', '已分析', 'Analyzed'],
  ['mediaPicker.filterNotAnalyzed', '未分析', 'Not analyzed'],
  ['mediaPicker.filterPreparing', '生成预览中', 'Preparing preview'],
  ['mediaPicker.batch', '批量', 'Batch'],
  ['mediaPicker.switchTo', '切换到此素材', 'Switch to media'],
  ['mediaPicker.selectThis', '选择此素材', 'Select this media'],
  ['mediaPicker.current', '当前：{name}', 'Current: {name}'],
  ['mediaPicker.change', '更换', 'Change'],
  ['mediaPicker.notAnalyzed', '未分析', 'Not analyzed'],
  ['mediaPicker.previewHint', '悬停或按空格预览，回车切换', 'Hover or press Space to preview, Enter to switch'],
  ['mediaPicker.deleteBlockedAnalyzing', '分析中的素材不能删除', 'Media being analyzed cannot be deleted'],
  ['mediaPicker.empty', '工程中还没有素材', 'No media in this project yet'],
  ['mediaPicker.alreadyCurrent', '该素材已是当前素材', 'This is already the current media'],
]
```

- [ ] **Step 2: Register the fragment** in `fragments/index.ts`: import `{ FRAG as media }` and spread `...media` into `FRAGMENTS`.

- [ ] **Step 3: Add picker state/actions to the `State` interface and store impl in `useStore.ts`**

Add type export `export type MediaPickerMode = 'switch' | 'select'` near the top. Add a module-level `let mediaPickHandler: ((mid: string) => void) | null = null` beside the other module refs. Add to interface: `mediaPickerOpen: boolean`, `mediaPickerMode: MediaPickerMode`, `openMediaPicker: (mode?: MediaPickerMode, onPick?: (mid: string) => void) => void`, `closeMediaPicker: () => void`. Initial state `mediaPickerOpen: false, mediaPickerMode: 'switch'`. Implementations: `openMediaPicker(mode = 'switch', onPick)` stores `mediaPickHandler = onPick ?? null` and `set({ mediaPickerOpen: true, mediaPickerMode: mode })`; `closeMediaPicker()` sets `mediaPickerOpen: false` (leave the handler until a pick resolves). Also export a plain getter helper `consumeMediaPick(): ((mid: string) => void) | null` returning and **clearing** `mediaPickHandler` (used by Task 2).

- [ ] **Step 4: Verify** — `cd frontend && npm run build && npm run lint`. Both pass (no consumers yet is fine).

- [ ] **Step 5: Commit (only if user approves commits)** — `feat: add media picker i18n strings and store state`

---

### Task 2: MediaChip + MediaPickerDialog core (browse, select, batch)

**Files:**
- Create: `frontend/src/components/MediaChip.tsx`
- Create: `frontend/src/components/MediaPickerDialog.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/StudioPage.tsx` (top-bar chip only this task; `MediaPanel` stays until Task 6)

**Interfaces:**
- Consumes: Task 1 store fields/actions + `consumeMediaPick`; `MediaInfo` type; `Modal`, `Button`, `Badge`, `Progress`, `Tooltip`, `useConfirm` from `./ui`; `ImportVideoButton`; format helpers `bytes`, `timecode` from `../lib/format`; `api.assetUrl`/`api.proxyUrl`.
- Produces: default-exported `MediaPickerDialog` (no props); default-exported `MediaChip` with optional props `{ className?: string }`, renders `null` when no current media.

- [ ] **Step 1: Create `MediaChip.tsx`** — a compact button (own classes, not `Button`) containing a 26×[~36px] rounded poster `<img>` (`api.assetUrl(media.poster)`, fallback `<Film size={13}/>` in a styled box), a truncating span with `media.name`, and `<ChevronDown size={12}/>`; wrapped in the existing `Tooltip` showing `media.path`; `onClick={() => openMediaPicker('switch')}`; reads `currentMedia()` from the store; `className` merges onto the button; returns `null` when media is absent.

- [ ] **Step 2: Create `MediaPickerDialog.tsx` shell + toolbar + grid** (no hover preview yet):

  - Component reads `project`, `mediaId`, `mediaPickerOpen`, `mediaPickerMode`, `closeMediaPicker`, `selectMedia`, `removeMedia`, `removeMediaBulk`, `runAnalysisBatch`, `jobs` from the store.
  - Renders `<Modal open={mediaPickerOpen} onClose={closeMediaPicker} title={tr('mediaPicker.title', { n: project?.media.length ?? 0 })} width={880} footer={…}>`. Return `null` content when no project.
  - Local state: `query: string`, `filter: 'all'|'analyzed'|'not_analyzed'|'preparing'`, `batch: boolean`, `selectedId: string | null` (init/fallback to `mediaId`), `checked: Set<string>`. Reset all transient state when `mediaPickerOpen` transitions to true (an effect keyed on `mediaPickerOpen`).
  - Reuse the MediaPanel logic verbatim (copy from `StudioPage.tsx`) for: `activePrepare`/`jobFor(mid)`, `activeAnalyze`/`analyzeJobFor(mid)`.
  - `filtered` memo: media whose `name.toLowerCase().includes(query.trim().toLowerCase())`; status predicates — analyzed: `project.analyses[m.id]?.status === 'done'`; preparing: `!!jobFor(m.id)`; not_analyzed: analysis status `!== 'done'`.
  - Toolbar: text `<input>` (placeholder `mediaPicker.searchPlaceholder`), a native-button styled dropdown for the four filters (`mediaPicker.filter*` keys), `<ImportVideoButton size="sm" />`, `<ImportVideoButton mode="folder" size="sm" variant="subtle" />` (both keep dialog open — no `onDone`), and a batch toggle button (`mediaPicker.batch`, toggling clears `checked`).
  - Body: responsive grid `grid gap-2.5` with inline/style or arbitrary class `grid-cols-[repeat(auto-fill,minmax(180px,1fr))]`; empty filtered result shows `mediaPicker.empty`.
  - Card markup copied/adapted from the existing MediaPanel card: poster + fallback, duration (`timecode(m.duration, false)`), ✨ rally `Badge`, proxy-ready `Badge` (`studio.previewReady`), prepare overlay with `Progress` and `studio.queued`/`studio.generatingPreview`, name + `${m.width}×${m.height} · ${m.fps.toFixed(0)}fps · ${bytes(m.size)}`; non-analyzed meta text uses `mediaPicker.notAnalyzed`.

- [ ] **Step 3: Card click semantics + footers**

  - Normal mode: single click sets `selectedId`; double click (`onDoubleClick`) calls the same confirm used by the footer. Top-right per-card trash button (stopPropagation; `useConfirm` with `studio.confirmRemoveMedia`/`Desc`; then `removeMedia(m.id)`); trash disabled with tooltip `mediaPicker.deleteBlockedAnalyzing` when `analyzeJobFor(m.id)` exists.
  - Batch mode: click toggles membership in `checked`; checkbox visual overlaid like MediaPanel's select mode; the card checkbox is `disabled` (click no-op) when `analyzeJobFor(m.id)` exists; select-all selects exactly the filtered cards that have no active analyze job (`studio.selectAll`/`studio.deselectAll`, count via `studio.selectedCount`).
  - Footer normal: left `mediaPicker.current` with `{name: currentMedia?.name}`; right: destructive Delete on `selectedId` (same confirm/action as card trash, disabled when analyzing or selected === null) and primary confirm — label `mediaPicker.switchTo` in switch mode, `mediaPicker.selectThis` in select mode; disabled when `selectedId === mediaId` (title `mediaPicker.alreadyCurrent`).
  - Confirm handler: switch mode → `selectMedia(selectedId)` then `closeMediaPicker()`; select mode → `const fn = consumeMediaPick(); fn?.(selectedId)` then close.
  - Footer batch: count + Select-all toggle + `studio.analyzeSelected` (`runAnalysisBatch([...checked])`, then exit batch mode) + Delete (`useConfirm` `studio.confirmRemoveSelected`/`Desc` → `removeMediaBulk([...checked])`, exit batch) + `studio.done`.

- [ ] **Step 4: Mount and wire** — render `<MediaPickerDialog />` once in `App.tsx` next to `<CourtEditor />`. In `StudioPage.tsx` replace the existing read-only hard-drive path block (`HardDrive` icon + path span) with `<MediaChip />` inside the same `hidden md:flex` container; leave `MediaPanel` and the tabs untouched this task.

- [ ] **Step 5: Verify build** — `cd frontend && npm run build && npm run lint` (prune any unused imports the compiler flags).

- [ ] **Step 6: Manual check** (run `pwsh -File scripts/dev.ps1`, open Studio): chip opens the dialog; search/filter narrow results; single click highlights; double click switches media and closes; Analysis/prepare badges match the old MediaPanel; batch select-all skips cards with running analysis; batch analyze enqueues jobs; batch delete prompts and removes; deleting current media lands on the first remaining; import buttons add media without closing the dialog; ESC and backdrop close.

- [ ] **Step 7: Commit (only if approved)** — `feat: add global media picker dialog with media chip`

---

### Task 3: In-card hover preview + keyboard navigation

**Files:**
- Modify: `frontend/src/components/MediaPickerDialog.tsx`

**Interfaces:**
- Produces (internal): `HoverPreview` subcomponent (same file) with props `{ pid: string; media: MediaInfo }`; dialog-level state `activePreviewId: string | null`.

- [ ] **Step 1: Add `HoverPreview`** — rendered in place of the poster children when active: `<video muted playsInline preload="auto" src={…proxy…} />` absolutely filling the poster area (`absolute inset-0 h-full w-full object-cover`), plus a top-right 🔇 tag, bottom-left live `cur / dur` timecode (reuse the `timecode` helper already imported in Task 2), and a 3px bottom progress bar. Effect on mount: play from 0; on reaching 6 s of playback (check in `timeupdate` against `currentTime`) call `video.pause()` and hold the frame; `onError` unmounts back to poster (parent clears `activePreviewId`). No controls, no audio.

- [ ] **Step 2: Hover state machine in the card** — dialog owns `activePreviewId` (at most one). `onPointerEnter` on the poster starts a 350 ms timer that sets `activePreviewId = m.id` only when `m.proxy_path` exists, no `jobFor(m.id)`, and batch mode is off; `onPointerLeave`/card click clears the timer and clears `activePreviewId` when it equals this card. Add an `IntersectionObserver` on the card root that clears `activePreviewId` when the card scrolls out of view. Dialog close resets `activePreviewId` (state reset from Task 2).

- [ ] **Step 3: Keyboard navigation** on the grid container (`tabIndex={0}`, focus ring styles): ArrowRight/Left move `selectedId` ±1 within `filtered`, ArrowDown/Up move by the column count (`Math.round(container.clientWidth / 180)`, clamp to list length); any arrow move clears `activePreviewId` and scrolls the newly selected card into view (`scrollIntoView({ block: 'nearest' })`). Space toggles preview for the selected card (sets/clears `activePreviewId`, `preventDefault`). Enter runs the confirm handler. Handler bails out immediately when the target is inside `input/textarea/[contenteditable]` or closest `button`.

- [ ] **Step 4: Add the hints line** — small text under the toolbar using `mediaPicker.previewHint`; hide it in batch mode.

- [ ] **Step 5: Verify build** — `npm run build && npm run lint`.

- [ ] **Step 6: Manual check** — rapid sweep across cards starts no video; dwelling ~0.35 s on a proxy-ready card plays muted from 0 and freezes at 6 s; moving to another card stops the first and only one video is ever playing (DevTools media/network shows one video); leaving the card, scrolling it out, or closing the dialog stops playback immediately; cards without proxy / preparing show poster only; focus the grid and verify arrows move the ring without loading video, Space previews the ring card, Enter switches, and typing in the search box / tabbing to trash never triggers these keys (Review Focus #2, #4).

- [ ] **Step 7: Commit (only if approved)** — `feat: add hover preview and keyboard nav to media picker`

---

### Task 4: Annotate integration — chip + flush-on-switch

**Files:**
- Modify: `frontend/src/components/AnnotatePage.tsx`

**Interfaces:**
- Consumes: `MediaChip` default export; `api.saveAnnotation(pid, mid, payload)`; Zustand `useStore.subscribe((next, prev) => …)`; existing local `stateRef`, `saveTimer`, `markDirty`/`doSave`.

- [ ] **Step 1: Include `note` in `stateRef`** (it currently omits note): add `note: string` to the ref object, initialize it, and update the sync effect so the flush payload can use one snapshot.

- [ ] **Step 2: Add the mount-only flush subscriber** — in a mount-level effect, `return useStore.subscribe((next, prev) => { … })`. Guard: `if (next.project?.id !== prev.project?.id) return` and require both `prev.mediaId` and `next.mediaId` and inequality. On a same-project media change: clear `saveTimer.current` (set null), then build the payload from `stateRef.current` exactly as `doSave` does (`rallies` mapped to `{start,end,source,note}`, `hits` mapped to `{t,ours}`, plus `focus` and `note` from the snapshot) and fire `api.saveAnnotation(prev.project!.id, prev.mediaId!, payload).catch(() => {})` without awaiting.

- [ ] **Step 3: Put the chip in the header** — replace the static media-name portion of the top-bar meta span (keep `· {duration} · {headerMeta}` text) with `<MediaChip />` beside it, matching existing spacing.

- [ ] **Step 4: Verify build** — `npm run build && npm run lint` in `frontend/`.

- [ ] **Step 5: Manual check** (Review Focus #1, #3) — in Annotate, create/edit a rally and within one second switch media from the chip: reopen the old media → the last edit is saved; the new media shows its own rallies/hits/focus; switch A→B→C quickly → A and B are each saved once (verify via network panel: two POSTs carrying the respective media ids); switch projects via the back button → no annotation save call fires for the old project's media.

- [ ] **Step 6: Commit (only if approved)** — `feat: allow switching media from annotate with flush save`

---

### Task 5: AI Analysis dialog — change media in select mode

**Files:**
- Modify: `frontend/src/components/AnalysisDialog.tsx`

- [ ] **Step 1: Wire the store** — pull `openMediaPicker` and `selectMedia` from the store.

- [ ] **Step 2: Add the current-media row** directly below the two scope-option buttons (only when `scope === 'current' && media`): poster thumbnail (`api.assetUrl(media.poster)`, `Film` fallback), truncated `media.name`, and a small ghost button labeled `mediaPicker.change`. Its onClick: `openMediaPicker('select', (mid) => { selectMedia(mid); setScope('current') })`.

- [ ] **Step 3: Verify build** — `npm run build && npm run lint`.

- [ ] **Step 4: Manual check** — open Analysis settings, choose Current, click Change: picker stacks above the dialog; ESC closes only the picker; choosing a media closes the picker, switches the dialog subtitle/current row to the new media, keeps scope on Current, and leaves all parameter controls, weights, ROI and advanced state untouched; starting analysis runs against the newly chosen media.

- [ ] **Step 5: Commit (only if approved)** — `feat: allow changing target media from analysis dialog`

---

### Task 6: Studio consolidation — remove the duplicated media panel

**Files:**
- Modify: `frontend/src/components/StudioPage.tsx`

- [ ] **Step 1: Delete `MediaPanel`** and remove the media/rallies tab UI: delete `leftTab` state, the tab button row JSX, and the effect that auto-sets `leftTab` by analysis status; the 330 px left sidebar (keep the collapse button and the `AnimatePresence`/`motion.aside` wrapper) renders `<RallyPanel />` directly.

- [ ] **Step 2: Remove the now-dead wiring** — the `dropping` prop passed only to `MediaPanel` (keep the window drag/drop handlers and their full-screen overlay); prune imports that become unused (`api`, `bytes`, `timecode`, `Badge`, `Progress`, `useConfirm`, and icons such as `ListChecks`, `Trash2`, `RefreshCw`, `HardDrive`, `Check` — verify each against remaining usage; keep `Film`, `Sparkles`, `Download`, panel icons, `Empty`, `Card`, `Tooltip`, `ImportVideoButton`).

- [ ] **Step 3: Verify build** — `npm run build && npm run lint`; the compiler must report no unused symbols.

- [ ] **Step 4: Manual check** — Studio left sidebar is rally-only for both analyzed and unanalyzed media; the top-bar chip opens the picker; the empty-project central state still offers both import buttons; dragging video files into the window still imports and still shows the drag overlay.

- [ ] **Step 5: Commit (only if approved)** — `refactor: consolidate studio media UI into global picker`

---

### Task 7: Whole-feature verification

**Files:** none (verification only)

- [ ] **Step 1:** From repo root run the backend-independent static gates: `cd frontend && npm run build && npm run lint`; expect zero errors/warnings beyond the pre-existing baseline.
- [ ] **Step 2:** Run `pwsh -File scripts/dev.ps1` and walk spec §7 acceptance checks 1–5 in order: annotate switch persistence; hover/keyboard preview; analysis-dialog change; batch analyze/delete/import/analyzing-guard; studio rally-only sidebar + empty/drag import.
- [ ] **Step 3:** Toggle language in Settings (zh/en), open the picker from every entry point, and confirm every new string is translated in both languages with no raw key leakage.
- [ ] **Step 4:** If the user asked for commits during the task run, confirm `git status` shows only intended files and the working tree otherwise matches; otherwise leave changes uncommitted.
