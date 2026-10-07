import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import CacheClearDialog from './CacheClearDialog'
import { ConfirmProvider } from './ui'
import { useStore } from '../store/useStore'
import { api } from '../lib/api'
import type { CacheTargetStat } from '../lib/api'
import type { JobInfo } from '../lib/types'

const TARGETS: CacheTargetStat[] = [
  { id: 'proxies', level: 'safe', default: true, files: 2, bytes: 4096 },
  { id: 'thumbs', level: 'safe', default: true, files: 1, bytes: 2048 },
  { id: 'audio', level: 'safe', default: true, files: 1, bytes: 1024 },
  { id: 'frames', level: 'safe', default: true, files: 0, bytes: 0 },
  { id: 'ai', level: 'normal', default: false, files: 3, bytes: 3072 },
  { id: 'eval', level: 'normal', default: false, files: 0, bytes: 0 },
  { id: 'logs', level: 'normal', default: false, files: 4, bytes: 5120 },
  { id: 'webview', level: 'normal', default: false, files: 0, bytes: 0 },
  { id: 'models', level: 'danger', default: false, files: 1, bytes: 1572864 },
]

vi.mock('../lib/api', () => ({
  api: {
    cacheTargets: vi.fn(() => Promise.resolve(TARGETS)),
    clearCacheTargets: vi.fn(() => Promise.resolve({ job_id: 'j_cc' })),
    cancelJob: vi.fn(() => Promise.resolve({ ok: true })),
  },
}))

function renderDialog() {
  const onClose = vi.fn()
  const onCleared = vi.fn()
  render(
    <ConfirmProvider>
      <CacheClearDialog open onClose={onClose} onCleared={onCleared} />
    </ConfirmProvider>,
  )
  return { onClose, onCleared }
}

afterEach(() => {
  act(() => useStore.setState({ lang: 'zh', jobs: {}, toasts: [] } as never))
  vi.mocked(api.clearCacheTargets).mockClear()
})

describe('CacheClearDialog', () => {
  it('lists targets with sizes and pre-checks only the safe defaults', async () => {
    act(() => useStore.setState({ lang: 'en' } as never))
    renderDialog()

    const proxies = await screen.findByRole('checkbox', { name: /proxy videos/i })
    expect(proxies).toBeInTheDocument()
    expect(proxies).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('4.0 KB')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /model files/i })).toHaveAttribute(
      'aria-checked',
      'false',
    )
    expect(screen.getByText('1.5 MB')).toBeInTheDocument()
    expect(screen.getAllByRole('checkbox')).toHaveLength(9)
    // Selected total reflects the four safe defaults (4096+2048+1024).
    expect(screen.getByText(/7\.0 KB/)).toBeInTheDocument()
  })

  it('select-all checks every target and deselect-all clears the selection', async () => {
    act(() => useStore.setState({ lang: 'en' } as never))
    renderDialog()
    await screen.findByRole('checkbox', { name: /proxy videos/i })

    fireEvent.click(screen.getByRole('button', { name: /select all/i }))
    expect(screen.getByRole('checkbox', { name: /model files/i })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    fireEvent.click(screen.getByRole('button', { name: /deselect all/i }))
    expect(screen.getAllByRole('checkbox').every((el) => el.getAttribute('aria-checked') === 'false')).toBe(true)
    expect(screen.getByRole('button', { name: /clear now/i })).toBeDisabled()
  })

  it('requires an extra confirmation for the danger models target, then runs the job and shows results', async () => {
    act(() => useStore.setState({ lang: 'en' } as never))
    const { onCleared } = renderDialog()
    await screen.findByRole('checkbox', { name: /proxy videos/i })

    fireEvent.click(screen.getByRole('checkbox', { name: /model files/i }))
    fireEvent.click(screen.getByRole('button', { name: /clear now/i }))

    // First confirmation explicitly warns about model re-download; declining aborts.
    let confirmModal = await screen.findByRole('dialog', { name: /delete the model files/i })
    fireEvent.click(within(confirmModal).getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(confirmModal).not.toBeInTheDocument())
    expect(api.clearCacheTargets).not.toHaveBeenCalled()

    // Confirm again, this time accept the danger dialog.
    fireEvent.click(screen.getByRole('button', { name: /clear now/i }))
    confirmModal = await screen.findByRole('dialog', { name: /delete the model files/i })
    fireEvent.click(within(confirmModal).getByRole('button', { name: 'OK' }))
    await waitFor(() => expect(api.clearCacheTargets).toHaveBeenCalledWith(
      ['proxies', 'thumbs', 'audio', 'frames', 'models'],
      true,
    ))

    // Job progress from the store is rendered.
    act(() =>
      useStore.setState({
        jobs: {
          j_cc: {
            id: 'j_cc',
            kind: 'cache_clear',
            title: 'Clear cache',
            status: 'running',
            progress: 0.5,
            stage: 'cache_clear',
            message: 'Clearing',
            error: null,
            result: null,
            created_at: 1,
            updated_at: 1,
          } satisfies JobInfo,
        },
      } as never),
    )
    expect(await screen.findByText('50%')).toBeInTheDocument()

    // Done: per-target results, total freed, failure + restart hints, stats refresh callback.
    act(() =>
      useStore.setState({
        jobs: {
          j_cc: {
            id: 'j_cc',
            kind: 'cache_clear',
            title: 'Clear cache',
            status: 'done',
            progress: 1,
            stage: 'done',
            message: 'done',
            error: null,
            created_at: 1,
            updated_at: 2,
            result: {
              freed: 5120,
              timed_out: [],
              items: [
                { id: 'proxies', removed: 2, freed: 4096, failed: [] },
                { id: 'models', removed: 1, freed: 1024, failed: [] },
                { id: 'webview', removed: 0, freed: 0, failed: ['PermissionError: data_0'] },
              ],
            },
          } satisfies JobInfo,
        },
      } as never),
    )
    expect(await screen.findByText(/5\.0 KB/)).toBeInTheDocument()
    expect(screen.getByText(/restart/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /done/i }))
    expect(onCleared).toHaveBeenCalledTimes(1)
  })
})
