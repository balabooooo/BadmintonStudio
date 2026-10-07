import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SettingsPage from './SettingsPage'
import { useStore } from '../store/useStore'
import { clearLogs } from '../lib/logger'

vi.mock('../lib/api', () => ({
  api: {
    cacheStats: vi.fn().mockResolvedValue({ cache: 0, proxies: 0, thumbs: 0, exports: 0 }),
    cacheTargets: vi.fn().mockResolvedValue([
      { id: 'proxies', level: 'safe', default: true, files: 2, bytes: 4096 },
      { id: 'thumbs', level: 'safe', default: true, files: 3, bytes: 2048 },
    ]),
    clearCacheTargets: vi.fn().mockResolvedValue({ job_id: 'j_cc' }),
    cancelJob: vi.fn(),
  },
}))

describe('SettingsPage debug log card', () => {
  afterEach(() => {
    clearLogs()
    act(() => useStore.setState({ lang: 'zh', env: null as never, toasts: [] }))
  })

  it('shows the export button and the backend log path hint', async () => {
    act(() =>
      useStore.setState({
        lang: 'en',
        env: { logs_dir: 'D:\\BadmintonStudio\\data\\logs' } as never,
      }),
    )
    await act(async () => {
      render(<SettingsPage />)
    })
    expect(screen.getByRole('button', { name: /export frontend logs/i })).toBeInTheDocument()
    expect(screen.getByText(/bms_debug_YYYY-MM-DD\.log/)).toBeInTheDocument()
  })

  it('surfaces an info toast when the in-memory buffer is empty', async () => {
    act(() => useStore.setState({ lang: 'en', env: { logs_dir: 'logs' } as never, toasts: [] }))
    await act(async () => {
      render(<SettingsPage />)
    })
    const btn = screen.getByRole('button', { name: /export frontend logs/i })
    // Empty buffer -> friendly notice, no attempted download (jsdom has no object URL sink).
    await act(async () => {
      fireEvent.click(btn)
    })
    await waitFor(() => expect(useStore.getState().toasts.length).toBeGreaterThan(0))
    expect(useStore.getState().toasts[0].kind).toBe('info')
  })

  it('opens the selective cache-clear dialog and lists target sizes', async () => {
    act(() => useStore.setState({ lang: 'en', env: {} as never, toasts: [] }))
    await act(async () => {
      render(<SettingsPage />)
    })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /choose items to clear/i }))
    })
    expect(await screen.findByRole('checkbox', { name: /proxy videos/i })).toBeInTheDocument()
    expect(screen.getByText(/4\.0 KB/)).toBeInTheDocument()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    })
    await waitFor(() =>
      expect(screen.queryByRole('checkbox', { name: /proxy videos/i })).not.toBeInTheDocument(),
    )
  })
})
