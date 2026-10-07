import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import JobTray from './JobTray'
import { useStore } from '../store/useStore'

vi.mock('../lib/api', () => ({
  api: {
    cancelJob: vi.fn().mockResolvedValue(undefined),
  },
}))

import { api } from '../lib/api'

describe('JobTray', () => {
  afterEach(() => {
    act(() => useStore.setState({ lang: 'zh', jobs: {}, toasts: [] }))
  })

  it('renders an optimize job with percentage, stage message and a cancel action', async () => {
    act(() =>
      useStore.setState({
        lang: 'en',
        toasts: [],
        jobs: {
          job_opt_1: {
            id: 'job_opt_1',
            kind: 'optimize',
            title: 'Optimize m1',
            media_id: 'm1',
            status: 'running',
            progress: 0.42,
            stage: 'optimize_segment',
            message: 'Segmenting rallies',
            created_at: Date.now(),
          },
        } as never,
      }),
    )
    render(<JobTray />)
    // Tray toggle shows the active-job badge.
    fireEvent.click(screen.getByRole('button', { name: /jobs/i }))
    expect(screen.getByText('42%')).toBeInTheDocument()
    expect(screen.getByText('Segmenting rallies')).toBeInTheDocument()
    expect(screen.getByText('Optimize m1')).toBeInTheDocument()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    })
    expect(api.cancelJob).toHaveBeenCalledWith('job_opt_1')
  })

  it('hides the tray entirely when there are no jobs', () => {
    act(() => useStore.setState({ lang: 'en', jobs: {} }))
    const { container } = render(<JobTray />)
    expect(container).toBeEmptyDOMElement()
  })
})
