import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import MediaPickerDialog from './MediaPickerDialog'
import { useStore } from '../store/useStore'
import type { MediaInfo, Project, AnalysisResult } from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    proxyUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/proxy`,
    assetUrl: (p: string) => `/assets/${p}`,
    removeMedia: vi.fn().mockResolvedValue({ project: { media: [], analyses: {} }, removed: 1 }),
  },
}))

// Suppress the confirm dialog used by delete actions.
vi.mock('./ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./ui')>()
  return {
    ...actual,
    useConfirm: () => vi.fn().mockResolvedValue(true),
  }
})

const makeMedia = (overrides: Partial<MediaInfo> & { id: string; name: string }): MediaInfo => ({
  path: `/v/${overrides.id}.mp4`,
  size: 1024,
  duration: 60,
  fps: 30,
  width: 1920,
  height: 1080,
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
  ...overrides,
})

const makeProject = (media: MediaInfo[]): Project => ({
  id: 'p1',
  name: 'Test',
  created_at: 0,
  updated_at: 0,
  media,
  analyses: {},
  timeline: { tracks: [], duration: 0, fps: 30, width: 1920, height: 1080 },
  ui: {},
  version: 1,
})

const doneAnalysis = (): AnalysisResult =>
  ({
    media_id: '',
    status: 'done',
    stage: '',
    message: '',
    progress: 1,
    error: null,
    params: {} as any,
    started_at: null,
    finished_at: null,
    signals: {},
    signal_fps: 30,
    hits: [],
    rallies: [{ id: 'r1', start: 0, end: 10 } as any],
    court: null,
    calibration: null,
    stats: {},
  }) as unknown as AnalysisResult

function setupStore() {
  const media = [
    makeMedia({ id: 'm1', name: 'match_001.mp4', proxy_path: 'p1' }),
    makeMedia({ id: 'm2', name: 'training_serves.mp4' }),
    makeMedia({ id: 'm3', name: 'highlight.mp4' }),
  ]
  const project = makeProject(media)
  project.analyses['m1'] = doneAnalysis()
  useStore.setState({
    project,
    mediaId: 'm1',
    mediaPickerOpen: true,
    mediaPickerMode: 'switch',
    lang: 'en',
    jobs: {
      j1: {
        id: 'j1', kind: 'prepare', media_id: 'm3', status: 'running',
        stage: 'encoding', message: '', progress: 0.5, created_at: 0,
      } as any,
    },
  })
  return media
}

describe('MediaPickerDialog', () => {
  beforeEach(() => {
    setupStore()
  })

  it('renders all media cards when no search/filter is applied', () => {
    render(<MediaPickerDialog />)
    expect(screen.getByText('match_001.mp4')).toBeInTheDocument()
    expect(screen.getByText('training_serves.mp4')).toBeInTheDocument()
    expect(screen.getByText('highlight.mp4')).toBeInTheDocument()
  })

  it('filters cards by name search (case-insensitive, partial match)', () => {
    render(<MediaPickerDialog />)
    const search = screen.getByPlaceholderText(/search file name/i)
    fireEvent.change(search, { target: { value: 'serve' } })
    expect(screen.getByText('training_serves.mp4')).toBeInTheDocument()
    expect(screen.queryByText('match_001.mp4')).not.toBeInTheDocument()
    expect(screen.queryByText('highlight.mp4')).not.toBeInTheDocument()
  })

  it('shows empty state when search matches nothing', () => {
    render(<MediaPickerDialog />)
    const search = screen.getByPlaceholderText(/search file name/i)
    fireEvent.change(search, { target: { value: 'zzzzz' } })
    expect(screen.getByText(/no media in this project/i)).toBeInTheDocument()
  })

  it('status filter "analyzed" shows only media with done analysis', () => {
    render(<MediaPickerDialog />)
    openFilterMenu()
    fireEvent.click(screen.getByText('Analyzed'))
    expect(screen.getByText('match_001.mp4')).toBeInTheDocument()
    expect(screen.queryByText('training_serves.mp4')).not.toBeInTheDocument()
    expect(screen.queryByText('highlight.mp4')).not.toBeInTheDocument()
  })

  it('status filter "not_analyzed" hides analyzed media but keeps preparing', () => {
    render(<MediaPickerDialog />)
    openFilterMenu()
    fireEvent.click(screen.getByText('Not analyzed'))
    expect(screen.queryByText('match_001.mp4')).not.toBeInTheDocument()
    expect(screen.getByText('training_serves.mp4')).toBeInTheDocument()
    // m3 has a running prepare job but no analysis -> still "not analyzed".
    expect(screen.getByText('highlight.mp4')).toBeInTheDocument()
  })

  it('status filter "preparing" shows only media with an active prepare job', () => {
    render(<MediaPickerDialog />)
    openFilterMenu()
    fireEvent.click(screen.getByText('Preparing preview'))
    expect(screen.queryByText('match_001.mp4')).not.toBeInTheDocument()
    expect(screen.queryByText('training_serves.mp4')).not.toBeInTheDocument()
    expect(screen.getByText('highlight.mp4')).toBeInTheDocument()
  })

  it('renders nothing when project is null', () => {
    useStore.setState({ project: null })
    const { container } = render(<MediaPickerDialog />)
    expect(container.firstChild).toBeNull()
  })

  it('keyboard ArrowRight moves selection to the next card', () => {
    render(<MediaPickerDialog />)
    const gridEl = document.querySelector('[tabindex="0"]') as HTMLElement
    gridEl.focus()
    fireEvent.keyDown(gridEl, { key: 'ArrowRight' })
    const m2 = document.querySelector('[data-card-id="m2"]')
    expect(m2).toHaveClass(/ring-court-500/)
  })

  it('ArrowLeft does not go below the first card', () => {
    render(<MediaPickerDialog />)
    const gridEl = document.querySelector('[tabindex="0"]') as HTMLElement
    gridEl.focus()
    // m1 is first; ArrowLeft stays on m1.
    fireEvent.keyDown(gridEl, { key: 'ArrowLeft' })
    const m1 = document.querySelector('[data-card-id="m1"]')
    expect(m1).toHaveClass(/ring-court-500/)
  })

  it('pressing Enter on a different card calls selectMedia and closes the dialog', () => {
    render(<MediaPickerDialog />)
    const gridEl = document.querySelector('[tabindex="0"]') as HTMLElement
    gridEl.focus()
    fireEvent.keyDown(gridEl, { key: 'ArrowRight' }) // move to m2
    fireEvent.keyDown(gridEl, { key: 'Enter' })
    expect(useStore.getState().mediaId).toBe('m2')
    expect(useStore.getState().mediaPickerOpen).toBe(false)
  })

  it('pressing Enter on the current media does not change selection', () => {
    useStore.setState({ mediaId: 'm1' })
    render(<MediaPickerDialog />)
    const gridEl = document.querySelector('[tabindex="0"]') as HTMLElement
    gridEl.focus()
    fireEvent.keyDown(gridEl, { key: 'Enter' })
    expect(useStore.getState().mediaId).toBe('m1')
  })

  it('batch mode toggles checkboxes on cards', () => {
    render(<MediaPickerDialog />)
    const batchBtn = screen.getByLabelText(/batch select/i)
    fireEvent.click(batchBtn)
    // In batch mode the top-left checkbox overlay appears on every card.
    const checkboxes = document.querySelectorAll('[data-card-id] > div > div.absolute.top-1\\.5.left-1\\.5')
    expect(checkboxes.length).toBe(3)
  })
})

function openFilterMenu() {
  const btn = screen.getAllByRole('button').find((b) => /all statuses/i.test(b.textContent || ''))
  fireEvent.click(btn!)
}
