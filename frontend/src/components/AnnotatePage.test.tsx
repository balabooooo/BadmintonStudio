import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import AnnotatePage from './AnnotatePage'
import { api } from '../lib/api'
import { useStore } from '../store/useStore'
import type {
  AnnotationResponse,
  MediaInfo,
  OptimizeResult,
  Project,
  SegmentMetric,
} from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    getAnnotation: vi.fn(),
    saveAnnotation: vi.fn(),
    optimizeSegmentation: vi.fn(),
    getAnnotationQuality: vi.fn(),
    getAnnotationSignals: vi.fn().mockResolvedValue({}),
    getAnnotationOverlay: vi.fn().mockResolvedValue({ frames: [] }),
    proxyUrl: vi.fn(() => '/proxy.mp4'),
    assetUrl: vi.fn(() => '/asset'),
    annotationCsvUrl: vi.fn(() => '/csv'),
  },
}))

const mockedGet = vi.mocked(api.getAnnotation)
const mockedSave = vi.mocked(api.saveAnnotation)
const mockedOptimize = vi.mocked(api.optimizeSegmentation)

function mediaInfo(id: string, name: string): MediaInfo {
  return {
    id,
    path: `D:/videos/${name}.mp4`,
    name,
    size: 12345,
    duration: 120,
    fps: 30,
    width: 1920,
    height: 1080,
    rotation: 0,
    vcodec: 'h264',
    acodec: 'aac',
    has_audio: true,
    created_at: 0,
    proxy_path: `cache/${id}.mp4`,
    proxy_fps: 30,
    proxy_width: 960,
    proxy_height: 540,
    audio_path: 'cache/audio.wav',
    poster: `${id}.jpg`,
  }
}

function makeProject(): Project {
  return {
    id: 'p1',
    name: 'proj',
    created_at: 0,
    updated_at: 0,
    media: [mediaInfo('m_a', 'clipA'), mediaInfo('m_b', 'clipB')],
    analyses: {},
    timeline: { tracks: [], duration: 240, fps: 30, width: 1920, height: 1080 },
    ui: {},
    version: 1,
  }
}

function annotationResponse(mid: string, rallies: { start: number; end: number }[]): AnnotationResponse {
  return {
    media_id: mid,
    media_name: mid,
    duration: 120,
    fps: 30,
    path: 'x',
    rallies: rallies.map((r) => ({ start: r.start, end: r.end, source: 'manual' })),
    hits: [],
    focus: null,
    note: '',
    auto: [],
    envelope: [],
    envelope_fps: 0,
    hit_times: [],
    hit_times_raw: [],
  }
}

const metric = (f1: number, params: Record<string, number> = {}): SegmentMetric => ({
  iou: 0.5,
  n: 9,
  tp: 7,
  fp: 2,
  fn: 5,
  precision: 0.78,
  recall: 0.58,
  f1,
  params,
})

function optimizeResult(f1: number): OptimizeResult {
  return {
    gt_count: 12,
    focus: [0, 120],
    iou_threshold: 0.5,
    baseline: metric(0.4),
    best: metric(f1, { seg_prominence: 0.2 }),
    results: [metric(f1, { seg_prominence: 0.2 })],
    tried: 42,
    search_fields: ['seg_prominence'],
    suggest: {},
  }
}

describe('AnnotatePage optimize panel is bound to the active clip', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStore.setState({ lang: 'en' })
    useStore.setState({ project: makeProject(), mediaId: 'm_a' })
    mockedGet.mockImplementation((_pid, mid) =>
      Promise.resolve(
        mid === 'm_a'
          ? annotationResponse('m_a', [{ start: 5, end: 12 }, { start: 20, end: 30 }])
          : annotationResponse('m_b', []),
      ),
    )
    mockedSave.mockResolvedValue({ ok: true, count: 0, path: 'x', updated_at: '' })
    mockedOptimize.mockResolvedValue(optimizeResult(0.62))
  })

  afterEach(() => {
    useStore.setState({ project: null, mediaId: null, lang: 'zh' })
  })

  it('shows this clip result after optimizing, hides it when switching clips, restores it when switching back', async () => {
    render(<AnnotatePage />)

    // clip A has annotations -> Optimize is enabled
    const optimizeBtn = await screen.findByRole('button', { name: /^optimize$/i })
    await waitFor(() => expect(optimizeBtn).not.toBeDisabled())

    fireEvent.click(optimizeBtn)
    // runOptimize saves first, then posts the optimize request for THIS clip
    await waitFor(() => expect(mockedOptimize).toHaveBeenCalledWith('p1', 'm_a', expect.anything()))
    expect((await screen.findAllByText('F1 0.620')).length).toBeGreaterThanOrEqual(1)

    // switch to clip B: the panel must show the empty state, not clip A's stale numbers
    useStore.setState({ mediaId: 'm_b' })
    await waitFor(() => expect(mockedGet).toHaveBeenCalledWith('p1', 'm_b'))
    await waitFor(() => {
      expect(screen.queryByText('F1 0.620')).toBeNull()
    })
    expect(screen.getByText(/uses your annotations as ground truth/i)).toBeInTheDocument()
    // clip B has no annotations -> button disabled here even though clip A was optimizable
    expect(screen.getByRole('button', { name: /^optimize$/i })).toBeDisabled()

    // switch back to clip A: its stored result is restored (per-clip state, not lost)
    useStore.setState({ mediaId: 'm_a' })
    await screen.findAllByText('F1 0.620')
    expect(mockedOptimize).toHaveBeenCalledTimes(1) // no accidental re-run on switching
  })
})
