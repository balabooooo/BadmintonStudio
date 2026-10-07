/** Fake analysis pipeline used only while the guided tour is active.
 *
 * Everything here is in-memory and never touches the backend: no proxy, no
 * GPU work, no project-file writes. ``runMockAnalysis`` drives a local JobInfo
 * through the same stage codes the real pipeline reports (so the analysis
 * dialog's progress UI is pixel-identical) and then writes a complete
 * ``AnalysisResult`` flagged with ``stats.mock = true``. The flag is the
 * single boundary marker: clearMockData() removes every flagged result and
 * mock job when the tour ends (skip/complete) or is aborted mid-run.
 */

import type {
  AnalysisParams,
  AnalysisResult,
  CourtCalibration,
  MediaInfo,
  PlayerSide,
  Rally,
  RallyFeatures,
  RallyScores,
  ShotEvent,
} from '../lib/types'
import type { JobInfo } from '../lib/types'
import { useStore } from '../store/useStore'
import { viewpointLabel } from '../i18n/domain'
import { t } from '../i18n'

export const MOCK_JOB_ID_PREFIX = 'mock-analyze-'

/** Cumulative (stage, progress) points of the ~3s fake run; stage codes are
 * exactly the ones the real backend reports, so jobStageLabel localizes them. */
const STAGE_TICKS: { at: number; stage: string; progress: number }[] = [
  { at: 0, stage: 'prepare', progress: 0.03 },
  { at: 400, stage: 'calibrate', progress: 0.14 },
  { at: 800, stage: 'hits', progress: 0.32 },
  { at: 1200, stage: 'players', progress: 0.5 },
  { at: 1600, stage: 'motion', progress: 0.66 },
  { at: 2000, stage: 'segment', progress: 0.82 },
  { at: 2400, stage: 'score', progress: 0.94 },
]
const DONE_AT = 2900

const SIGNAL_FPS = 4

/* ------------------------------------------------------------------ deterministic pseudo-random */

/** Small deterministic hash -> [0,1): mock data must be identical on every
 * tour run (stable screenshots/expectations), never Math.random(). */
function frac01(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const round1 = (v: number) => Math.round(v * 10) / 10
const round3 = (v: number) => Math.round(v * 1000) / 1000

/* ------------------------------------------------------------------ rally layout */

/** Fixed rally/gap pattern (seconds); cycled until the media runs out.
 * Deterministic and realistic: rallies 5.5–8s with 2.5–3.5s pauses. */
const RALLY_PATTERN = [7, 5.5, 8, 6, 7.5]
const GAP_PATTERN = [3, 2.5, 3.5, 2.8]

function planRallies(duration: number): [number, number][] {
  if (!Number.isFinite(duration) || duration < 1) return []
  const out: [number, number][] = []
  const long = duration >= 12
  let time = long ? 1.5 : round1(Math.min(0.3, duration * 0.1))
  let i = 0
  const tailReserve = long ? 0.5 : 0.2
  while (time < duration - 1) {
    const wanted = long ? RALLY_PATTERN[i % RALLY_PATTERN.length] : 3 + frac01(i) * 4
    const end = round1(Math.min(duration - tailReserve, time + wanted))
    if (end - time < 1.2) break
    out.push([round1(time), end])
    time = round1(end + (long ? GAP_PATTERN[i % GAP_PATTERN.length] : 2 + frac01(i + 9) * 1.5))
    i += 1
  }
  return out
}

const SHOT_KINDS = ['clear', 'drive', 'drop', 'net', 'clear', 'drive', 'smash']

function buildShots(rallyIdx: number, start: number, end: number, count: number): ShotEvent[] {
  const span = Math.max(0.4, end - start - 0.4)
  const shots: ShotEvent[] = []
  for (let k = 0; k < count; k++) {
    const base = count === 1 ? 0.3 : 0.3 + (span * k) / (count - 1)
    const jitter = (frac01(rallyIdx * 31 + k * 7) - 0.5) * Math.min(0.25, span / count)
    const time = round3(clamp(start + base + jitter, start, end))
    const player: PlayerSide = k % 2 === 0 ? 'far' : 'near'
    // Smash only on the finishing shot(s), never the serve.
    const kind = k >= count - 1 && frac01(rallyIdx * 13 + 3) > 0.45
      ? 'smash'
      : SHOT_KINDS[Math.floor(frac01(rallyIdx * 17 + k * 3) * (SHOT_KINDS.length - 1))]
    const speed = kind === 'smash'
      ? round1(70 + frac01(rallyIdx + k * 5) * 24)
      : round1(24 + frac01(rallyIdx * 3 + k * 11) * 30)
    shots.push({
      time,
      player,
      kind,
      confidence: round3(0.72 + frac01(rallyIdx * 5 + k * 13) * 0.27),
      speed,
      airborne: kind === 'smash',
      height: null,
    })
  }
  return shots.sort((a, b) => a.time - b.time)
}

function buildScores(dur: number, f: RallyFeatures, closeness: number): RallyScores {
  const length = round1(clamp(40 + dur * 3.5 + (f.shot_count - 6) * 1.4, 20, 97))
  const intensity = round1(clamp(44 + f.tempo * 17 + f.smash_count * 5 + f.finish_intensity * 8, 20, 97))
  const technique = round1(clamp(54 + f.confidence * 24 + f.shuttle_speed_p95 * 0.2, 20, 97))
  const excitement = round1(clamp(34 + closeness * 34 + f.smash_count * 4 + (f.clutch ? 9 : 0), 20, 97))
  const production = round1(clamp((length + intensity) / 2 + f.finish_intensity * 9 - 18, 20, 97))
  const total = round1(
    clamp(
      intensity * 0.35 + technique * 0.25 + excitement * 0.2 + length * 0.1 + production * 0.1,
      20,
      96,
    ),
  )
  return { total, length, intensity, technique, excitement, production }
}

function buildTags(f: RallyFeatures, score: number, conf: number): string[] {
  const tags: string[] = []
  if (f.shot_count >= 12) tags.push('many_shots')
  if (f.tempo >= 1.15) tags.push('fast_tempo')
  if (f.duration >= 8.4) tags.push('long_rally')
  if (f.shot_count <= 3) tags.push('short_rally')
  if (f.smash_count >= 2) tags.push('smash')
  if (score >= 85) tags.push('high_score')
  if (score >= 88) tags.push('highlight')
  if (f.clutch && score >= 78) tags.push('confrontation')
  if (conf < 0.82) tags.push('low_confidence')
  if (f.finish_intensity > 0.75) tags.push('late_acceleration')
  if (f.motion_energy > 0.55 && f.travel_near + f.travel_far > 0.45) tags.push('high_mobility')
  if (f.shuttle_speed_p95 > 70) tags.push('fast_shuttle')
  return tags
}

/* ------------------------------------------------------------------ public builders */

export function isMockAnalysis(a: AnalysisResult | null | undefined): boolean {
  return !!a && a.stats?.mock === true
}

/** Construct the complete fake AnalysisResult for one media item. */
export function buildMockAnalysis(
  media: MediaInfo,
  params: AnalysisParams,
  weights: string,
): AnalysisResult {
  const spans = planRallies(media.duration)
  const rallies: Rally[] = []
  const allShots: ShotEvent[] = []

  spans.forEach(([start, end], i) => {
    const dur = round1(end - start)
    const seed = i + 1
    const shotCount = Math.round(clamp(dur * 1.15 + frac01(seed * 3) * 4, 2, 16))
    const shots = buildShots(seed, start, end, shotCount)
    allShots.push(...shots)
    const speeds = shots.map((s) => s.speed ?? 0).sort((a, b) => a - b)
    const p50 = speeds.length ? speeds[Math.floor(speeds.length * 0.5)] : 0
    const p95 = speeds.length ? speeds[Math.min(speeds.length - 1, Math.floor(speeds.length * 0.95))] : 0
    const motionEnergy = round3(0.35 + frac01(seed * 19) * 0.25)
    const closeness = round3(0.35 + frac01(seed * 23) * 0.45)
    const confidence = round3(0.75 + frac01(seed * 29) * 0.23)
    const features: RallyFeatures = {
      duration: dur,
      shot_count: shots.length,
      tempo: round3(shots.length / dur),
      shuttle_speed_p50: p50,
      shuttle_speed_p95: p95,
      motion_energy: motionEnergy,
      motion_peak: round3(Math.min(0.98, motionEnergy + 0.2)),
      travel_near: round3(0.1 + frac01(seed * 7) * 0.3),
      travel_far: round3(0.1 + frac01(seed * 11) * 0.3),
      smash_count: shots.filter((s) => s.kind === 'smash').length,
      longest_exchange: shots.length,
      finish_intensity: round3(0.4 + frac01(seed * 37) * 0.5),
      scramble: round3(0.2 + frac01(seed * 41) * 0.5),
      closeness,
      clutch: frac01(seed * 43) > 0.6,
      confidence,
      speech_bonus: 0,
      speech_phrases: [],
    }
    const scores = buildScores(dur, features, closeness)
    const tags = buildTags(features, scores.total, confidence)
    const servePlayer: PlayerSide = i % 2 === 0 ? 'far' : 'near'
    rallies.push({
      id: `mock-rally-${media.id}-${i + 1}`,
      index: i + 1,
      start,
      end,
      clip_start: round3(Math.max(0, start - (params.pre_roll ?? 1.5))),
      clip_end: round3(Math.min(media.duration, end + (params.post_roll ?? 2))),
      serve_time: shots[0]?.time ?? null,
      serve_player: servePlayer,
      receive_time: shots[1]?.time ?? null,
      receive_player: (servePlayer === 'far' ? 'near' : 'far') as PlayerSide,
      shots,
      features,
      scores,
      tags,
      keep: true,
      starred: false,
      note: '',
      winner: i % 3 === 2 ? null : i % 2 === 0 ? 'near' : 'far',
      duration: dur,
    })
  })

  // Activity curve: low baseline between rallies, high energy during them.
  const frameCount = Math.floor(media.duration * SIGNAL_FPS) + 1
  const activity: number[] = []
  for (let f = 0; f < frameCount; f++) {
    const timeAt = f / SIGNAL_FPS
    const inside = spans.some(([s, e]) => timeAt >= s && timeAt <= e)
    const noise = frac01(f * 3.7 + media.duration) * 0.08
    activity.push(round3(clamp((inside ? 0.6 : 0.08) + noise, 0, 1)))
  }
  const hitTimes = allShots.map((s) => s.time).sort((a, b) => a - b)
  // A couple of raw detections the gate would drop (RallyPanel shows the diff).
  const hitTimesRaw = [...hitTimes, ...hitTimes.slice(0, 2).map((v) => round3(v + 0.12))].sort((a, b) => a - b)

  const totalShots = allShots.length
  const activeDuration = rallies.reduce((n, r) => n + r.duration, 0)
  const avgScore = rallies.length ? rallies.reduce((n, r) => n + r.scores.total, 0) / rallies.length : 0
  const polygon = [
    [0.14, 0.93],
    [0.86, 0.93],
    [0.63, 0.45],
    [0.37, 0.45],
  ]
  const calibration: CourtCalibration = {
    ok: true,
    viewpoint: 'rear',
    viewpoint_label: viewpointLabel('rear'),
    confidence: 0.91,
    polygon,
    quad: polygon,
    point_count: 4,
    source: 'auto',
    distortion: 0.03,
    court_color: 'green',
    court_area_ratio: 0.22,
    foreshortening: 0.41,
    in_court_ratio: 0.88,
    roi: null,
    notes: [],
  }

  return {
    media_id: media.id,
    status: 'done',
    stage: 'done',
    message: '',
    progress: 1,
    error: null,
    params: { ...params },
    started_at: Date.now() - DONE_AT,
    finished_at: Date.now(),
    signals: {
      activity,
      threshold_hi: [0.55],
      threshold_lo: [0.25],
      hit_times: hitTimes,
      hit_times_raw: hitTimesRaw,
    },
    signal_fps: SIGNAL_FPS,
    hits: allShots,
    rallies,
    court: null,
    calibration,
    stats: {
      mock: true,
      count: rallies.length,
      active_duration: round1(activeDuration),
      total_shots: totalShots,
      avg_score: round1(avgScore),
      duration: media.duration,
      avg_shots: rallies.length ? round1(totalShots / rallies.length) : 0,
      max_shots: rallies.reduce((m, r) => Math.max(m, r.shots.length), 0),
      high_score_count: rallies.filter((r) => r.scores.total >= 85).length,
      weights,
      player_trace: { active_ids: [0, 1] },
      hit_trace: { count: totalShots },
      shuttle_trace: { disabled: true },
      pose_trace: { keep_ratio: 0.9 },
      speech_trace: { available: false },
      speech: { events: [] },
    },
  }
}

/* ------------------------------------------------------------------ fake run lifecycle */

let runToken = 0

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms))

function mockJob(mid: string): JobInfo {
  const now = Date.now()
  return {
    id: `${MOCK_JOB_ID_PREFIX}${mid}`,
    kind: 'analyze',
    title: t('tour.mock.jobTitle'),
    status: 'running',
    progress: STAGE_TICKS[0].progress,
    stage: STAGE_TICKS[0].stage,
    message: '',
    error: null,
    result: null,
    media_id: mid,
    created_at: now,
    updated_at: now,
  }
}

/** Run the fake analysis pipeline for the given media ids (default: the
 * current media). Aborts silently if clearMockData() invalidates the run. */
export async function runMockAnalysis(mediaIds?: string[]): Promise<void> {
  const s0 = useStore.getState()
  const project0 = s0.project
  if (!project0) return
  const ids = (mediaIds && mediaIds.length ? mediaIds : s0.mediaId ? [s0.mediaId] : []).filter((id) =>
    project0.media.some((m) => m.id === id),
  )
  if (!ids.length) return

  const token = ++runToken
  const jobs: Record<string, JobInfo> = {}
  for (const id of ids) jobs[`${MOCK_JOB_ID_PREFIX}${id}`] = mockJob(id)
  useStore.setState((s) => ({ jobs: { ...s.jobs, ...jobs }, busy: { ...s.busy, ...Object.fromEntries(ids.map((id) => [`${MOCK_JOB_ID_PREFIX}${id}`, true])) } }))

  let elapsed = 0
  for (const tick of STAGE_TICKS.slice(1)) {
    await sleep(tick.at - elapsed)
    if (token !== runToken) return
    elapsed = tick.at
    const now = Date.now()
    useStore.setState((s) => {
      const nextJobs = { ...s.jobs }
      for (const id of ids) {
        const j = nextJobs[`${MOCK_JOB_ID_PREFIX}${id}`]
        if (j) nextJobs[`${MOCK_JOB_ID_PREFIX}${id}`] = { ...j, ...tick, updated_at: now }
      }
      return { jobs: nextJobs }
    })
  }

  await sleep(DONE_AT - elapsed)
  if (token !== runToken) return

  const store = useStore.getState()
  const project = store.project
  if (!project) return
  const analyses = { ...project.analyses }
  for (const id of ids) {
    const media = project.media.find((m) => m.id === id)
    if (media) analyses[id] = buildMockAnalysis(media, store.params, store.weights)
  }
  const doneJobs: Record<string, JobInfo> = {}
  const busy = { ...store.busy }
  const now = Date.now()
  for (const id of ids) {
    const key = `${MOCK_JOB_ID_PREFIX}${id}`
    const prev = store.jobs[key]
    doneJobs[key] = { ...(prev ?? mockJob(id)), status: 'done', progress: 1, stage: 'done', updated_at: now }
    delete busy[key]
  }
  const firstRallyId = analyses[ids[0]]?.rallies[0]?.id ?? null
  useStore.setState({
    project: { ...project, analyses },
    jobs: { ...store.jobs, ...doneJobs },
    busy,
    selectedRallyId: firstRallyId,
    selectedClipId: null,
    selectedClipIds: [],
    view: 'studio',
  })
}

/** Invalidate any running mock pipeline and strip mock jobs/results from the
 * current project. Real analyses and real jobs are left untouched. */
export function clearMockData(): void {
  runToken += 1
  const s = useStore.getState()
  const jobs = Object.fromEntries(Object.entries(s.jobs).filter(([id]) => !id.startsWith(MOCK_JOB_ID_PREFIX)))
  const busy = Object.fromEntries(Object.entries(s.busy).filter(([id]) => !id.startsWith(MOCK_JOB_ID_PREFIX)))
  let project = s.project
  if (project) {
    let changed = false
    const analyses = { ...project.analyses }
    for (const [id, a] of Object.entries(analyses)) {
      if (a?.stats?.mock) {
        delete analyses[id]
        changed = true
      }
    }
    if (changed) project = { ...project, analyses }
  }
  useStore.setState({ jobs, busy, ...(project !== s.project ? { project } : {}) })
}
