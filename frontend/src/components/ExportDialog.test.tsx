import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ExportDialog from './ExportDialog'
import { useStore } from '../store/useStore'
import { api } from '../lib/api'
import type { Clip, ExportPreset, JobInfo, Project, Track, Timeline } from '../lib/types'

const PRESET = vi.hoisted(() => ({
  id: 'yt1080p',
  name: 'YouTube 1080p',
  width: 1920,
  height: 1080,
  fps: 30,
  vcodec: 'h264',
  encoder: 'auto',
  video_bitrate: '12M',
  audio_bitrate: '192k',
  crf: null,
  container: 'mp4',
  auto_reframe: false,
})) as unknown as ExportPreset

vi.mock('../lib/api', () => ({
  api: {
    exportPresets: vi.fn(() => Promise.resolve([PRESET])),
    listExports: vi.fn(() => Promise.resolve([] as string[])),
    exportProject: vi.fn(() =>
      Promise.resolve({ job_id: 'j_ex', output: 'D:/exports/out.mp4', mode: 'merge' }),
    ),
    pickExportDir: vi.fn(),
    revealExport: vi.fn(() => Promise.resolve({ ok: true })),
    cancelJob: vi.fn(() => Promise.resolve({ ok: true })),
  },
}))

// Mirrors the dialog's own auto name: `${project.name}_${new Date().toISOString().slice(0, 10)}`
const TODAY = new Date().toISOString().slice(0, 10)
const BASE = `TestProj_${TODAY}`

function makeClip(id: string): Clip {
  return {
    id,
    media_id: 'm1',
    src_in: 0,
    src_out: 2,
    tl_start: 0,
    speed: 1,
    volume: 1,
    transform: { scale: 1, x: 0, y: 0, rotation: 0 },
    rally_id: null,
    label: 'Rally 1',
    vertical_crop: false,
    protected: false,
  }
}

const TRACK: Track = { id: 't1', name: 'V1', kind: 'video', clips: [makeClip('c1')], muted: false, locked: false }
const TIMELINE: Timeline = { tracks: [TRACK], duration: 2, fps: 30, width: 1920, height: 1080 }

function makeProject(): Project {
  return {
    id: 'p_test',
    name: 'TestProj',
    created_at: 1,
    updated_at: 1,
    media: [],
    analyses: {},
    timeline: TIMELINE,
    ui: {},
    version: 1,
  }
}

function renderDialog() {
  render(<ExportDialog open onClose={vi.fn()} />)
}

function doneJob(result: Record<string, unknown>) {
  return {
    id: 'j_ex',
    kind: 'export',
    title: 'export',
    status: 'done',
    progress: 1,
    stage: 'done',
    message: 'done',
    error: null,
    created_at: 1,
    updated_at: 2,
    result,
  } as unknown as JobInfo
}

afterEach(() => {
  act(
    () =>
      useStore.setState({
        lang: 'zh',
        jobs: {},
        toasts: [],
        project: null,
      } as never),
  )
  vi.mocked(api.listExports).mockImplementation(() => Promise.resolve([]))
  vi.mocked(api.exportProject).mockClear()
})

describe('ExportDialog naming-conflict hints', () => {
  it('shows the auto-suffix hint when the target name is free', async () => {
    act(() => useStore.setState({ lang: 'en', project: makeProject() } as never))
    renderDialog()
    await screen.findByText(/1920×1080/)

    // No conflict: neutral hint explaining the auto suffix, no overwrite warning.
    expect(screen.getByText(new RegExp(`e\\.g\\. ${BASE}_1\\.mp4`, 'i'))).toBeInTheDocument()
    expect(screen.queryByText(/already exists; this export will be saved as/i)).not.toBeInTheDocument()
  })

  it('warns with the exact auto-renamed target when the name is taken, bumping past _1', async () => {
    act(() => useStore.setState({ lang: 'en', project: makeProject() } as never))
    vi.mocked(api.listExports).mockImplementation(() =>
      Promise.resolve([
        { id: 'e_a', name: `${BASE}.mp4`, path: `D:/exports/${BASE}.mp4`, size: 1, mtime: 1, group: '', mode: 'merge' },
        { id: 'e_b', name: `${BASE}_1.mp4`, path: `D:/exports/${BASE}_1.mp4`, size: 1, mtime: 2, group: '', mode: 'merge' },
      ] as never),
    )
    renderDialog()
    await screen.findByText(/already exists; this export will be saved as/i)

    // Both the base and _1 names are registered, so the next free slot is _2.
    expect(screen.getByText(new RegExp(`${BASE}\\.mp4 already exists`, 'i'))).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`saved as ${BASE}_2\\.mp4 instead`, 'i'))).toBeInTheDocument()
  })

  it('notices when the backend saved the finished file under a bumped name', async () => {
    act(() => useStore.setState({ lang: 'en', project: makeProject() } as never))
    renderDialog()
    await screen.findByText(/1920×1080/)

    fireEvent.click(screen.getByRole('button', { name: /start export/i }))
    await waitForExportStarted()

    // The file on disk ended up as _1 (backend renamed on conflict).
    act(() =>
      useStore.setState({
        jobs: {
          j_ex: doneJob({
            path: `D:/exports/${BASE}_1.mp4`,
            size: 1024,
            dir: 'D:/exports',
            exports: [{ id: 'e_1', name: `${BASE}_1.mp4`, path: `D:/exports/${BASE}_1.mp4` }],
          }),
        },
      } as never),
    )
    expect(
      await screen.findByText(
        new RegExp(`${BASE}\\.mp4 already existed, so the file was saved as ${BASE}_1\\.mp4`, 'i'),
      ),
    ).toBeInTheDocument()
  })

  it('shows no renamed notice when the file kept the requested name', async () => {
    act(() => useStore.setState({ lang: 'en', project: makeProject() } as never))
    renderDialog()
    await screen.findByText(/1920×1080/)

    fireEvent.click(screen.getByRole('button', { name: /start export/i }))
    await waitForExportStarted()

    act(() =>
      useStore.setState({
        jobs: {
          j_ex: doneJob({
            path: `D:/exports/${BASE}.mp4`,
            size: 1024,
            dir: 'D:/exports',
            exports: [{ id: 'e_1', name: `${BASE}.mp4`, path: `D:/exports/${BASE}.mp4` }],
          }),
        },
      } as never),
    )
    await screen.findByText(/file saved to/i)
    expect(screen.queryByText(/already existed, so the file was saved as/i)).not.toBeInTheDocument()
  })
})

async function waitForExportStarted() {
  await waitFor(() => expect(api.exportProject).toHaveBeenCalled())
  // Flush the microtask chain that carries the resolved job id into the dialog's internal state.
  await act(async () => {
    await Promise.resolve()
  })
}
