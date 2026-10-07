/**
 * Per-media annotation draft cache.
 *
 * Why this exists: the annotation page (`<AnnotatePage>`) is unmounted whenever the user
 * switches views (App renders `<div key={view}>`), and all of its label state used to live in
 * local useState. Navigating annotate -> editor -> annotate therefore remounted the page with
 * empty state until the GET came back, which looked like "the annotations vanished".
 *
 * This module keeps every loaded media's draft in a global zustand store keyed by
 * `${projectId}:${mediaId}`:
 *  - remounting the page hydrates SYNCHRONOUSLY from cache (no blank flash), then refreshes
 *    from the server in the background;
 *  - autosave (1.2 s debounce) and edit sequencing live here too, so a pending PUT still
 *    lands against the correct media after a view / media / project switch — each entry owns
 *    its own timer and its own project id, there is no "current media" ambiguity;
 *  - concurrent GETs are de-duplicated (single-flight) and stale responses are dropped by a
 *    per-entry request sequence number;
 *  - background refreshes never overwrite a dirty local draft.
 */

import { useEffect, useMemo } from 'react'
import { create } from 'zustand'
import { api } from '../lib/api'
import { createLogger } from '../lib/logger'
import { t } from '../i18n'
import { useT } from '../i18n/useT'
import type { AnnotationDraft, AnnotationResponse, AnnotationRally } from '../lib/types'
import { useStore } from './useStore'

const log = createLogger('ann-draft')

/** Client-side rally: wire payload plus a stable React key id. */
export interface DraftRally extends AnnotationRally {
  id: number
}

/** Client-side hit marker: wire payload plus a stable React key id. */
export interface DraftHit {
  id: number
  t: number
  ours: boolean
}

export type DraftStatusCode = 'idle' | 'new' | 'loaded' | 'unsaved' | 'saved' | 'failed'

export interface DraftEntry {
  pid: string
  mid: string
  /** False until the first GET for this entry has landed. */
  loaded: boolean
  /** A GET is currently in flight (first load or forced refresh). */
  loading: boolean
  duration: number
  fps: number
  auto: AnnotationDraft[]
  rallies: DraftRally[]
  hits: DraftHit[]
  /** Ungated hit timestamps; source for "re-mark all hits ours". */
  rawHitTimes: number[]
  focus: [number, number]
  note: string
  /** Local edits not yet acknowledged by a PUT. */
  dirty: boolean
  /** Next client id for rallies/hits. */
  idSeq: number
  /** Bumped on every edit; a PUT response clears dirty only when unchanged meanwhile. */
  editSeq: number
  /** Bumped on every GET; only the newest response may land. */
  reqSeq: number
  /** Bumped whenever server data has been applied (component resets view/selection/undo). */
  rev: number
  statusCode: DraftStatusCode
  /** Rally count carried by loaded/saved status for the label. */
  statusCount: number
}

type EntryKey = string

export function draftKey(pid: string, mid: string): EntryKey {
  return `${pid}:${mid}`
}

interface DraftState {
  entries: Record<EntryKey, DraftEntry>
}

const useDraftStore = create<DraftState>(() => ({ entries: {} }))

// Non-serializable per-entry bookkeeping kept OUT of React state.
const inflight = new Map<EntryKey, Promise<AnnotationResponse>>()
const timers = new Map<EntryKey, number>()
const AUTOSAVE_MS = 1200

function entryOf(key: EntryKey): DraftEntry | undefined {
  return useDraftStore.getState().entries[key]
}

function patchEntry(key: EntryKey, patch: Partial<DraftEntry>): void {
  useDraftStore.setState((s) => {
    const cur = s.entries[key]
    if (!cur) return {}
    return { entries: { ...s.entries, [key]: { ...cur, ...patch } } }
  })
}

function nextClientId(key: EntryKey): number {
  const cur = entryOf(key)
  const id = (cur?.idSeq ?? 0) + 1
  if (cur) patchEntry(key, { idSeq: id })
  return id
}

/** Map a server response into entry state, assigning fresh client ids. */
function applyServer(key: EntryKey, info: AnnotationResponse, seq: number, fallbackDur: number, fallbackFps: number): void {
  const cur = entryOf(key)
  // A newer request was issued meanwhile (A -> B -> C switches, forced reload): drop stale data.
  if (!cur || seq !== cur.reqSeq) {
    log.debug('drop stale annotation response', { key, seq, current: cur?.reqSeq })
    return
  }
  let idSeq = 0
  const nid = () => ++idSeq
  const duration = info.duration || fallbackDur
  const fps = info.fps || fallbackFps
  const rallies = (info.rallies || []).map((r) => ({ ...r, id: nid() }))
  const raw = info.hit_times_raw?.length ? info.hit_times_raw : info.hit_times || []
  const hits = (info.hits || []).length
    ? info.hits.map((h) => ({ ...h, id: nid() }))
    : raw.map((ht) => ({ t: ht, ours: true, id: nid() }))
  const focus: [number, number] =
    info.focus && info.focus[1] ? (info.focus as [number, number]) : [0, duration || 0]

  patchEntry(key, {
    loaded: true,
    loading: false,
    duration,
    fps,
    auto: info.auto || [],
    rallies,
    hits,
    rawHitTimes: raw,
    focus,
    note: info.note || '',
    dirty: false,
    idSeq,
    reqSeq: seq,
    rev: (cur?.rev ?? 0) + 1,
    statusCode: rallies.length ? 'loaded' : 'new',
    statusCount: rallies.length,
  })
  log.debug('annotation applied', { key, rallies: rallies.length, hits: hits.length, seq })
}

function pull(key: EntryKey, pid: string, mid: string, seq: number, force: boolean,
  fallbackDur: number, fallbackFps: number): Promise<AnnotationResponse> {
  if (!force) {
    const shared = inflight.get(key)
    if (shared) {
      log.debug('share in-flight annotation GET', { key })
      return shared
    }
  }
  log.debug('GET annotation', { key, seq, force })
  const p = api.getAnnotation(pid, mid)
  inflight.set(key, p)
  p.then(
    (info) => {
      inflight.delete(key)
      applyServer(key, info, seq, fallbackDur, fallbackFps)
    },
    () => {
      inflight.delete(key)
      patchEntry(key, { loading: false })
    },
  )
  return p
}

async function saveEntry(key: EntryKey): Promise<void> {
  const e = entryOf(key)
  if (!e) return
  const timer = timers.get(key)
  if (timer) {
    window.clearTimeout(timer)
    timers.delete(key)
  }
  const seq = e.editSeq
  log.debug('PUT annotation', { key, seq, rallies: e.rallies.length, dirty: e.dirty })
  try {
    const res = await api.saveAnnotation(e.pid, e.mid, {
      rallies: e.rallies.map(({ start, end, source, note }) => ({ start, end, source, note })),
      hits: e.hits.map(({ t, ours }) => ({ t, ours })),
      focus: e.focus,
      note: e.note,
    })
    const cur = entryOf(key)
    // New edits arrived while the PUT was in flight: let the next save clear dirty instead.
    if (cur && cur.editSeq === seq) {
      patchEntry(key, { dirty: false, statusCode: 'saved', statusCount: res.count })
    }
    log.debug('PUT annotation ok', { key, count: res.count })
  } catch (err) {
    patchEntry(key, { statusCode: 'failed' })
    log.error('PUT annotation failed', { key, err: String(err) })
    useStore.getState().toast({ kind: 'error', title: t('annotate.saveFailed'), detail: String(err) })
  }
}

function scheduleSave(key: EntryKey): void {
  const old = timers.get(key)
  if (old) window.clearTimeout(old)
  const handle = window.setTimeout(() => {
    timers.delete(key)
    void saveEntry(key)
  }, AUTOSAVE_MS)
  timers.set(key, handle)
}

/** Best-effort flush of every dirty draft (tab close / window shutdown). */
function flushAllDirty(): void {
  for (const [key, e] of Object.entries(useDraftStore.getState().entries)) {
    if (e.dirty) void saveEntry(key)
  }
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', flushAllDirty)
}

export interface AnnotationDraftApi {
  loaded: boolean
  loading: boolean
  /** Increments whenever fresh server data lands; mount/forced/background refresh. */
  rev: number
  duration: number
  fps: number
  auto: AnnotationDraft[]
  rallies: DraftRally[]
  hits: DraftHit[]
  rawHitTimes: number[]
  focus: [number, number]
  note: string
  dirty: boolean
  /** Raw status code (for save-side effects). */
  statusCode: DraftStatusCode
  /** Translated status line for the toolbar. */
  saveState: string
  nextId: () => number
  setRallies: (v: DraftRally[] | ((prev: DraftRally[]) => DraftRally[])) => void
  setHits: (v: DraftHit[] | ((prev: DraftHit[]) => DraftHit[])) => void
  setFocus: (v: [number, number]) => void
  setNote: (v: string) => void
  setRawHitTimes: (v: number[]) => void
  setAuto: (v: AnnotationDraft[]) => void
  setDuration: (v: number | ((prev: number) => number)) => void
  /** Mark the current draft edited and (re)schedule autosave. */
  markDirty: () => void
  /** Immediate PUT; used before optimize and via the manual save button. */
  save: () => Promise<void>
  /** Force a server refresh (after applying re-segmented parameters). */
  reload: () => Promise<void>
}

/** Inert API returned while no media is selected; the page renders its own empty gate. */
const INERT_API: AnnotationDraftApi = {
  loaded: false,
  loading: false,
  rev: 0,
  duration: 0,
  fps: 0,
  auto: [],
  rallies: [],
  hits: [],
  rawHitTimes: [],
  focus: [0, 0],
  note: '',
  dirty: false,
  statusCode: 'idle',
  saveState: '—',
  nextId: () => 0,
  setRallies: () => undefined,
  setHits: () => undefined,
  setFocus: () => undefined,
  setNote: () => undefined,
  setRawHitTimes: () => undefined,
  setAuto: () => undefined,
  setDuration: () => undefined,
  markDirty: () => undefined,
  save: async () => undefined,
  reload: async () => undefined,
}

/**
 * Bind the annotation UI to one media's cached draft.
 *
 * Cache hits hydrate synchronously (the component renders stored labels on its very first
 * render after a view switch); a background GET silently refreshes clean drafts.
 */
export function useAnnotationDraft(
  pid: string | undefined | null,
  mid: string | undefined | null,
  fallback?: { duration?: number; fps?: number },
): AnnotationDraftApi {
  const key = pid && mid ? draftKey(pid, mid) : null
  const entry = useDraftStore((s) => (key ? s.entries[key] : undefined))
  const fallbackDur = fallback?.duration ?? 0
  const fallbackFps = fpsGuard(fallback?.fps)

  useEffect(() => {
    if (!pid || !mid) return
    const k = draftKey(pid, mid)
    const store = useDraftStore.getState()
    const existing = store.entries[k]
    if (!existing) {
      // Placeholder so the very first render already has a stable entry to edit into.
      log.debug('create draft entry', { key: k })
      useDraftStore.setState((s) => ({
        entries: {
          ...s.entries,
          [k]: {
            pid,
            mid,
            loaded: false,
            loading: true,
            duration: fallbackDur,
            fps: fallbackFps,
            auto: [],
            rallies: [],
            hits: [],
            rawHitTimes: [],
            focus: [0, fallbackDur],
            note: '',
            dirty: false,
            idSeq: 0,
            editSeq: 0,
            reqSeq: 1,
            rev: 0,
            statusCode: 'idle',
            statusCount: 0,
          },
        },
      }))
      pull(k, pid, mid, 1, false, fallbackDur, fallbackFps).catch((err) => {
        log.error('initial annotation GET failed', { key: k, err: String(err) })
        useStore.getState().toast({ kind: 'error', title: t('annotate.loadFailed'), detail: String(err) })
      })
    } else if (!existing.dirty && !existing.loading) {
      // Returning to the page: show cached data instantly, refresh clean drafts in background.
      const seq = existing.reqSeq + 1
      patchEntry(k, { reqSeq: seq })
      pull(k, pid, mid, seq, false, fallbackDur, fallbackFps).catch(() => undefined)
    } else {
      log.debug('keep dirty/loading draft on hydrate', { key: k, dirty: existing.dirty })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, mid])

  const tr = useT()
  return useMemo<AnnotationDraftApi>(() => {
    if (!key || !pid || !mid) return INERT_API
    const e = entry
    const base = e ?? {
      duration: fallbackDur,
      fps: fallbackFps,
      auto: [] as AnnotationDraft[],
      rallies: [] as DraftRally[],
      hits: [] as DraftHit[],
      rawHitTimes: [] as number[],
      focus: [0, fallbackDur] as [number, number],
      note: '',
      dirty: false,
      loaded: false,
      loading: true,
      rev: 0,
      statusCode: 'idle' as DraftStatusCode,
      statusCount: 0,
    }
    const saveState =
      base.statusCode === 'unsaved'
        ? tr('annotate.unsaved')
        : base.statusCode === 'saved'
          ? tr('annotate.savedCount', { n: base.statusCount })
          : base.statusCode === 'failed'
            ? tr('annotate.saveFailed')
            : base.statusCode === 'loaded'
              ? tr('annotate.loadedRallies', { n: base.statusCount })
              : base.statusCode === 'new'
                ? tr('annotate.newAnnotation')
                : '—'

    const setRallies: AnnotationDraftApi['setRallies'] = (v) => {
      const cur = entryOf(key)
      if (!cur) return
      patchEntry(key, { rallies: typeof v === 'function' ? v(cur.rallies) : v })
    }
    const setHits: AnnotationDraftApi['setHits'] = (v) => {
      const cur = entryOf(key)
      if (!cur) return
      patchEntry(key, { hits: typeof v === 'function' ? v(cur.hits) : v })
    }

    return {
      loaded: base.loaded,
      loading: base.loading,
      rev: base.rev,
      duration: base.duration,
      fps: base.fps,
      auto: base.auto,
      rallies: base.rallies,
      hits: base.hits,
      rawHitTimes: base.rawHitTimes,
      focus: base.focus,
      note: base.note,
      dirty: base.dirty,
      statusCode: base.statusCode,
      saveState,
      nextId: () => nextClientId(key),
      setRallies,
      setHits,
      setFocus: (v) => patchEntry(key, { focus: v }),
      setNote: (v) => patchEntry(key, { note: v }),
      setRawHitTimes: (v) => patchEntry(key, { rawHitTimes: v }),
      setAuto: (v) => patchEntry(key, { auto: v }),
      setDuration: (v) => {
        const cur = entryOf(key)
        if (!cur) return
        patchEntry(key, { duration: typeof v === 'function' ? v(cur.duration) : v })
      },
      markDirty: () => {
        const cur = entryOf(key)
        if (!cur) return
        patchEntry(key, {
          dirty: true,
          editSeq: cur.editSeq + 1,
          statusCode: 'unsaved',
        })
        scheduleSave(key)
      },
      save: () => saveEntry(key),
      reload: async () => {
        const cur = entryOf(key)
        const seq = (cur?.reqSeq ?? 0) + 1
        patchEntry(key, { loading: true, reqSeq: seq })
        try {
          // pull() lands the response itself (stale responses are dropped by seq).
          await pull(key, pid, mid, seq, true, cur?.duration ?? fallbackDur, cur?.fps ?? fallbackFps)
        } catch (err) {
          patchEntry(key, { loading: false })
          log.error('forced annotation GET failed', { key, err: String(err) })
          useStore.getState().toast({ kind: 'error', title: tr('annotate.loadFailed'), detail: String(err) })
        }
      },
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, entry, tr, fallbackDur, fallbackFps, pid, mid])
}

function fpsGuard(v: number | undefined): number {
  return v && Number.isFinite(v) ? v : 0
}
