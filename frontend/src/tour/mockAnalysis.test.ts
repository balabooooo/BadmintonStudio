import { act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  MOCK_JOB_ID_PREFIX,
  buildMockAnalysis,
  clearMockData,
  isMockAnalysis,
  runMockAnalysis,
} from './mockAnalysis'
import { useTourStore } from './tourStore'
import { useStore } from '../store/useStore'
import type { AnalysisParams, AnalysisResult, MediaInfo, Project } from '../lib/types'

function makeMedia(id: string, duration: number): MediaInfo {
  return {
    id,
    path: `D:/samples/${id}.mp4`,
    name: `${id}.mp4`,
    size: 12345,
    duration,
    fps: 30,
    width: 960,
    height: 540,
    rotation: 0,
    vcodec: 'h264',
    acodec: 'aac',
    has_audio: true,
    created_at: 0,
    proxy_path: null,
    proxy_fps: null,
    proxy_width: null,
    proxy_height: null,
    audio_path: null,
    poster: null,
  }
}

function makeProject(media: MediaInfo[]): Project {
  return {
    id: 'p1',
    name: 'Sample project',
    created_at: 0,
    updated_at: 0,
    media,
    analyses: {},
    timeline: { tracks: [], duration: 240, fps: 30, width: 960, height: 540 },
    ui: {},
    version: 1,
  }
}

const params: AnalysisParams = {
  ...useStore.getState().params,
  pre_roll: 1.5,
  post_roll: 2.0,
}

/** Drive the mock run's timer ladder to completion. */
async function settleMockRun() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(4000)
  })
}

describe('buildMockAnalysis', () => {
  it('marks the result as mock and status done', () => {
    const a = buildMockAnalysis(makeMedia('m1', 48), params, 'balanced')
    expect(a.status).toBe('done')
    expect(a.media_id).toBe('m1')
    expect(a.stats.mock).toBe(true)
    expect(isMockAnalysis(a)).toBe(true)
    expect(isMockAnalysis(null)).toBe(false)
    expect(isMockAnalysis({ stats: {} } as AnalysisResult)).toBe(false)
  })

  it('lays out non-overlapping rallies inside the media duration with clip padding', () => {
    const a = buildMockAnalysis(makeMedia('m1', 48), params, 'balanced')
    expect(a.rallies.length).toBeGreaterThanOrEqual(3)
    let prevEnd = -1
    for (const [i, r] of a.rallies.entries()) {
      expect(r.index).toBe(i + 1)
      expect(r.id).toBeTruthy()
      expect(r.start).toBeGreaterThanOrEqual(0)
      expect(r.end).toBeLessThanOrEqual(48)
      expect(r.end).toBeGreaterThan(r.start)
      expect(r.start).toBeGreaterThan(prevEnd)
      // clip range embeds the configured pre/post roll, clamped to the media
      expect(r.clip_start).toBeCloseTo(Math.max(0, r.start - 1.5), 6)
      expect(r.clip_end).toBeCloseTo(Math.min(48, r.end + 2.0), 6)
      prevEnd = r.end
    }
  })

  it('still yields one rally for a very short sample clip', () => {
    const a = buildMockAnalysis(makeMedia('m2', 2), params, 'balanced')
    expect(a.rallies.length).toBe(1)
    const r = a.rallies[0]
    expect(r.end).toBeLessThanOrEqual(2)
  })

  it('fills shots/features/scores/tags coherently', () => {
    const a = buildMockAnalysis(makeMedia('m1', 48), params, 'balanced')
    const knownTags = new Set([
      'many_shots', 'fast_tempo', 'late_acceleration', 'high_mobility', 'fast_shuttle',
      'long_rally', 'short_rally', 'high_score', 'low_confidence', 'highlight',
      'smash', 'confrontation', 'ultra_long_rally',
    ])
    for (const r of a.rallies) {
      expect(r.shots.length).toBe(r.features.shot_count)
      expect(r.shots.length).toBeGreaterThanOrEqual(2)
      for (const s of r.shots) {
        expect(s.time).toBeGreaterThanOrEqual(r.start)
        expect(s.time).toBeLessThanOrEqual(r.end)
        expect(['near', 'far']).toContain(s.player)
        expect(s.confidence).toBeGreaterThanOrEqual(0)
        expect(s.confidence).toBeLessThanOrEqual(1)
      }
      expect(r.features.tempo).toBeCloseTo(r.features.shot_count / r.duration, 2)
      expect(r.features.speech_bonus).toBe(0)
      expect(r.features.speech_phrases).toEqual([])
      for (const dim of ['total', 'length', 'intensity', 'technique', 'excitement', 'production'] as const) {
        expect(r.scores[dim]).toBeGreaterThanOrEqual(0)
        expect(r.scores[dim]).toBeLessThanOrEqual(100)
      }
      for (const tag of r.tags) expect(knownTags.has(tag)).toBe(true)
    }
  })

  it('builds signal curves and stats aggregates the rallies', () => {
    const media = makeMedia('m1', 48)
    const a = buildMockAnalysis(media, params, 'balanced')
    const fps = a.signal_fps
    expect(fps).toBeGreaterThan(0)
    expect(a.signals.activity.length).toBe(Math.floor(media.duration * fps) + 1)
    for (const v of a.signals.activity) expect(v).toBeGreaterThanOrEqual(0)
    expect(a.signals.threshold_hi[0]).toBeGreaterThan(0)
    // hit_times covers every shot and stays sorted within the media duration
    const allShots = a.rallies.flatMap((r) => r.shots.map((s) => s.time)).sort((x, y) => x - y)
    expect(a.signals.hit_times).toEqual(allShots)
    expect(a.signals.hit_times_raw.length).toBeGreaterThanOrEqual(a.signals.hit_times.length)

    const totalShots = a.rallies.reduce((n, r) => n + r.shots.length, 0)
    const activeDuration = a.rallies.reduce((n, r) => n + r.duration, 0)
    const avgScore = a.rallies.reduce((n, r) => n + r.scores.total, 0) / a.rallies.length
    expect(a.stats.count).toBe(a.rallies.length)
    expect(a.stats.total_shots).toBe(totalShots)
    expect(a.stats.active_duration).toBeCloseTo(activeDuration, 5)
    expect(a.stats.duration).toBe(48)
    expect(a.stats.avg_score).toBeCloseTo(avgScore, 5)
    expect(a.stats.max_shots).toBe(Math.max(...a.rallies.map((r) => r.shots.length)))
    expect(a.stats.weights).toBe('balanced')
    expect(a.stats.hit_trace.count).toBe(totalShots)
    expect(a.stats.shuttle_trace.disabled).toBeTruthy()
  })

  it('includes an in-court calibration so the court overlay can render', () => {
    const a = buildMockAnalysis(makeMedia('m1', 48), params, 'balanced')
    expect(a.calibration?.ok).toBe(true)
    expect(a.calibration?.polygon).toHaveLength(4)
    for (const [x, y] of a.calibration!.polygon) {
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThanOrEqual(1)
      expect(y).toBeGreaterThanOrEqual(0)
      expect(y).toBeLessThanOrEqual(1)
    }
  })
})

describe('runMockAnalysis', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    useTourStore.setState({ active: true, index: 0 })
  })

  afterEach(() => {
    vi.useRealTimers()
    useTourStore.setState({ active: false })
    useStore.setState({ project: null, mediaId: null, jobs: {}, busy: {}, selectedRallyId: null })
  })

  it('is a no-op without a project', async () => {
    useStore.setState({ project: null })
    await act(async () => {
      await runMockAnalysis(['m1'])
    })
    expect(Object.keys(useStore.getState().jobs)).toHaveLength(0)
  })

  it('drives staged job progress and writes the mock result, then selects the first rally', async () => {
    const media = makeMedia('m1', 48)
    useStore.setState({ project: makeProject([media]), mediaId: 'm1', params, weights: 'balanced' })

    let p: Promise<void>
    act(() => {
      p = runMockAnalysis(['m1'])
    })

    // Immediately: a running analyze job exists at the prepare stage.
    const jobId = `${MOCK_JOB_ID_PREFIX}m1`
    expect(useStore.getState().jobs[jobId]).toMatchObject({
      kind: 'analyze',
      status: 'running',
      stage: 'prepare',
    })

    await settleMockRun()
    await p!

    const job = useStore.getState().jobs[jobId]
    expect(job.status).toBe('done')
    expect(job.progress).toBe(1)
    const analysis = useStore.getState().project!.analyses.m1
    expect(analysis.status).toBe('done')
    expect(analysis.stats.mock).toBe(true)
    expect(analysis.stats.weights).toBe('balanced')
    expect(useStore.getState().selectedRallyId).toBe(analysis.rallies[0].id)
    expect(useStore.getState().busy[jobId]).toBeUndefined()
  })

  it('visits the real localized stage codes during progress', async () => {
    const media = makeMedia('m1', 48)
    useStore.setState({ project: makeProject([media]), mediaId: 'm1' })
    act(() => {
      void runMockAnalysis(['m1'])
    })
    const seen: string[] = []
    for (let t = 0; t < 3200; t += 400) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(400)
      })
      seen.push(useStore.getState().jobs[`${MOCK_JOB_ID_PREFIX}m1`].stage)
    }
    for (const stage of ['calibrate', 'hits', 'players', 'segment', 'score']) {
      expect(seen).toContain(stage)
    }
  })

  it('creates one job per media in a batch and writes all results', async () => {
    const m1 = makeMedia('m1', 48)
    const m2 = makeMedia('m2', 16)
    useStore.setState({ project: makeProject([m1, m2]), mediaId: 'm1' })
    act(() => {
      void runMockAnalysis(['m1', 'm2'])
    })
    await settleMockRun()
    const { project, jobs } = useStore.getState()
    expect(project!.analyses.m1.stats.mock).toBe(true)
    expect(project!.analyses.m2.stats.mock).toBe(true)
    expect(jobs[`${MOCK_JOB_ID_PREFIX}m1`].status).toBe('done')
    expect(jobs[`${MOCK_JOB_ID_PREFIX}m2`].status).toBe('done')
  })

  it('defaults to the current media when no ids are given', async () => {
    const m1 = makeMedia('m1', 48)
    useStore.setState({ project: makeProject([m1]), mediaId: 'm1' })
    act(() => {
      void runMockAnalysis()
    })
    await settleMockRun()
    expect(useStore.getState().project!.analyses.m1.stats.mock).toBe(true)
  })

  it('aborts (writes nothing) when clearMockData runs mid-progress', async () => {
    const media = makeMedia('m1', 48)
    useStore.setState({ project: makeProject([media]), mediaId: 'm1' })
    act(() => {
      void runMockAnalysis(['m1'])
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(900)
    })
    expect(useStore.getState().jobs[`${MOCK_JOB_ID_PREFIX}m1`]).toBeDefined()

    act(() => {
      clearMockData()
    })
    await settleMockRun()

    expect(useStore.getState().project!.analyses.m1).toBeUndefined()
    expect(useStore.getState().jobs[`${MOCK_JOB_ID_PREFIX}m1`]).toBeUndefined()
  })

  it('clearMockData removes only mock analyses/jobs and keeps real ones', async () => {
    const m1 = makeMedia('m1', 48)
    const m2 = makeMedia('m2', 16)
    const project = makeProject([m1, m2])
    project.analyses.m1 = buildMockAnalysis(m1, params, 'balanced')
    project.analyses.m2 = { ...buildMockAnalysis(m2, params, 'balanced'), stats: { mock: false } }
    useStore.setState({
      project,
      mediaId: 'm1',
      jobs: {
        [`${MOCK_JOB_ID_PREFIX}m1`]: { id: `${MOCK_JOB_ID_PREFIX}m1` } as never,
        'real-job': { id: 'real-job' } as never,
      },
    })
    act(() => {
      clearMockData()
    })
    const { project: p2, jobs } = useStore.getState()
    expect(p2!.analyses.m1).toBeUndefined()
    expect(p2!.analyses.m2).toBeDefined()
    expect(jobs['real-job']).toBeDefined()
    expect(jobs[`${MOCK_JOB_ID_PREFIX}m1`]).toBeUndefined()
  })
})

// Ensure the timers module under test does not crash when mounted in a tree
// (guards against accidental React dependencies; TourOverlay owns the UI).
describe('mock module surface', () => {
  it('exports a stable job-id prefix used by the cleanup filter', () => {
    expect(MOCK_JOB_ID_PREFIX).toBe('mock-analyze-')
  })
})
