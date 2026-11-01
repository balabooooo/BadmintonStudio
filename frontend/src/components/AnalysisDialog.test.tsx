import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AnalysisDialog from './AnalysisDialog'
import { DEFAULT_PARAMS, useStore } from '../store/useStore'

vi.mock('../lib/api', () => ({
  api: {
    env: vi.fn().mockResolvedValue({ caps: {} }),
    listPresets: vi.fn().mockResolvedValue([]),
  },
}))

function seedProject() {
  useStore.setState({
    lang: 'zh',
    project: {
      id: 'p_hint',
      name: 'hint',
      media: [
        {
          id: 'm_hint',
          path: 'C:/v/hint.mp4',
          name: 'hint',
          size: 0,
          duration: 30,
          fps: 30,
          width: 1280,
          height: 720,
          rotation: 0,
          vcodec: 'h264',
          acodec: 'aac',
          has_audio: true,
          created_at: 0,
        },
      ],
      analyses: {},
      timeline: { tracks: [], duration: 0, fps: 30, width: 1920, height: 1080 },
      ui: {
        court_polys: {
          m_hint: [
            [0.15, 1],
            [0.85, 1],
            [0.7, 0.4],
            [0.3, 0.4],
          ],
        },
      },
    } as never,
    mediaId: 'm_hint',
    env: { caps: {} } as never,
    params: { ...DEFAULT_PARAMS },
    jobs: {},
    presets: [],
    weights: 'balanced',
    toasts: [],
    selectedRallyId: null,
  })
}

describe('AnalysisDialog hit-sound detection', () => {
  beforeEach(seedProject)
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('defaults to off and explains the neighboring-court caution next to the toggle', async () => {
    await act(async () => {
      render(<AnalysisDialog open onClose={() => {}} />)
    })
    expect(useStore.getState().params.use_audio).toBe(false)
    expect(screen.getByText('确保周围无其他场地击球声干扰再开启')).toBeInTheDocument()
  })

  it('exposes one combined player-detection + pose switch', async () => {
    act(() =>
      useStore.setState({
        params: { ...DEFAULT_PARAMS, use_players: false, use_pose: false },
      }),
    )
    await act(async () => {
      render(<AnalysisDialog open onClose={() => {}} />)
    })
    expect(screen.getByText('球员检测与姿态')).toBeInTheDocument()
    expect(screen.queryByText('姿态辅助（击球归属）')).toBeNull()
    expect(screen.queryByText('球员检测与跟踪')).toBeNull()

    const btn = screen.getByRole('button', { name: /球员检测与姿态/ })
    await act(async () => {
      fireEvent.click(btn)
    })
    expect(useStore.getState().params.use_players).toBe(true)
    expect(useStore.getState().params.use_pose).toBe(true)
  })
})
