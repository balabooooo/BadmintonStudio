import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import CourtEditor from './CourtEditor'
import { api } from '../lib/api'
import { useStore } from '../store/useStore'
import type { MediaInfo, Project } from '../lib/types'

// useConfirm is replaced with a controllable mock so the dirty-switch guard can be
// driven deterministically (accept vs. decline) without reaching into the provider.
const { confirmMock } = vi.hoisted(() => ({ confirmMock: vi.fn() }))

vi.mock('../lib/api', () => ({
  api: {
    posterUrl: (pid: string, mid: string) => `/api/projects/${pid}/media/${mid}/poster`,
    patchProject: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock('./ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./ui')>()
  return { ...actual, useConfirm: () => confirmMock }
})

/** The shape the editor starts from when nothing is saved yet (see defaultQuad()). */
const DEFAULT_QUAD: [number, number][] = [
  [0.16, 0.99],
  [0.84, 0.99],
  [0.62, 0.52],
  [0.38, 0.52],
]

const COURT_POLY: [number, number][] = [
  [0.1, 0.9],
  [0.9, 0.9],
  [0.7, 0.5],
  [0.3, 0.5],
]

const makeMedia = (id: string, name: string): MediaInfo => ({
  id,
  path: `/videos/${name}.mp4`,
  name,
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
  poster: `${id}.jpg`,
})

function seedProject(courtPolys: Record<string, unknown> = {}) {
  const project: Project = {
    id: 'p1',
    name: 'court',
    created_at: 0,
    updated_at: 0,
    media: [makeMedia('m1', 'match.mp4'), makeMedia('m2', 'rally.mp4'), makeMedia('m3', 'serve.mp4')],
    analyses: {},
    timeline: { tracks: [], duration: 60, fps: 30, width: 1920, height: 1080 },
    ui: { court_polys: courtPolys },
    version: 1,
  }
  useStore.setState({ project, mediaId: 'm1', lang: 'en', toasts: [], jobs: {}, busy: {} })
}

describe('CourtEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    confirmMock.mockReset()
    confirmMock.mockResolvedValue(true)
    seedProject()
  })

  it('drops the clear / subdivide / delete-point controls and labels the save button Save', () => {
    render(<CourtEditor open onClose={vi.fn()} />)

    // The three non-essential controls were removed entirely.
    expect(screen.queryByText('Clear')).toBeNull()
    expect(screen.queryByText('Add midpoint to every edge')).toBeNull()
    expect(screen.queryByRole('button', { name: /Delete point/ })).toBeNull()
    expect(screen.queryByText('Save and use for next analysis')).toBeNull()

    // Only the core controls remain, and the save label is the plain "Save".
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Reset/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Apply to other media/ })).toBeInTheDocument()
  })

  it('writes the polygon into project.ui.court_polys for the current media on save', () => {
    const onClose = vi.fn()
    render(<CourtEditor open onClose={onClose} />)

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect((useStore.getState().project!.ui as any).court_polys.m1).toEqual(DEFAULT_QUAD)
    expect(api.patchProject).toHaveBeenCalledTimes(1)
    expect((api.patchProject as any).mock.calls[0][0]).toBe('p1')
    expect(onClose).toHaveBeenCalled()
  })

  it('shows a media switcher that marks already-calibrated media', () => {
    seedProject({ m2: COURT_POLY })
    render(<CourtEditor open onClose={vi.fn()} />)

    const select = screen.getByRole('combobox')
    expect(select).toHaveValue('m1')
    const options = within(select).getAllByRole('option').map((o) => o.textContent)
    expect(options).toEqual(['match.mp4', 'rally.mp4', 'serve.mp4'])
  })

  it('switches the current media without confirmation when there are no unsaved edits', () => {
    render(<CourtEditor open onClose={vi.fn()} />)

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'm2' } })

    expect(confirmMock).not.toHaveBeenCalled()
    expect(useStore.getState().mediaId).toBe('m2')
  })

  it('asks for confirmation before switching away with unsaved edits and can be declined', async () => {
    confirmMock.mockResolvedValue(false)
    render(<CourtEditor open onClose={vi.fn()} />)

    // Add a vertex via an edge midpoint -> the canvas is now dirty.
    fireEvent.click(screen.getAllByTitle(/insert a vertex at the midpoint/i)[0])
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'm2' } })

    await waitFor(() => expect(confirmMock).toHaveBeenCalledTimes(1))
    expect(useStore.getState().mediaId).toBe('m1')
  })

  it('switches media after the user accepts discarding unsaved edits', async () => {
    render(<CourtEditor open onClose={vi.fn()} />)

    fireEvent.click(screen.getAllByTitle(/insert a vertex at the midpoint/i)[0])
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'm2' } })

    await waitFor(() => expect(useStore.getState().mediaId).toBe('m2'))
    expect(confirmMock).toHaveBeenCalledTimes(1)
    // The canvas re-initialises for the new (uncalibrated) media, so the edits are gone.
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument()
  })
})
