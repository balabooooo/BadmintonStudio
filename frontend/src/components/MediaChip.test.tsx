import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import MediaChip from './MediaChip'
import { useStore } from '../store/useStore'
import type { MediaInfo, Project } from '../lib/types'

vi.mock('../lib/api', () => ({
  api: {
    assetUrl: (p: string) => `/assets/${p}`,
  },
}))

const makeMedia = (id: string, name: string, poster: string | null = null): MediaInfo => ({
  id, path: `/v/${id}.mp4`, name, size: 1024, duration: 60, fps: 30,
  width: 1920, height: 1080, rotation: 0, vcodec: 'h264', acodec: 'aac',
  has_audio: true, created_at: 0, proxy_path: null, proxy_fps: null,
  proxy_width: null, proxy_height: null, audio_path: null, poster,
})

describe('MediaChip', () => {
  beforeEach(() => {
    useStore.setState({
      mediaId: null,
      project: null,
      lang: 'en',
    })
  })

  it('renders nothing when no media is selected', () => {
    const { container } = render(<MediaChip />)
    expect(container.firstChild).toBeNull()
  })

  it('renders the media name when a media is selected', () => {
    const media = makeMedia('m1', 'match_001.mp4')
    useStore.setState({
      project: { media: [media] } as unknown as Project,
      mediaId: 'm1',
    })
    render(<MediaChip />)
    expect(screen.getByText('match_001.mp4')).toBeInTheDocument()
  })

  it('opens the media picker in switch mode when clicked', () => {
    const media = makeMedia('m1', 'match_001.mp4')
    useStore.setState({
      project: { media: [media] } as unknown as Project,
      mediaId: 'm1',
    })
    render(<MediaChip />)
    const btn = screen.getByRole('button', { name: /switch to media/i })
    fireEvent.click(btn)
    const s = useStore.getState()
    expect(s.mediaPickerOpen).toBe(true)
    expect(s.mediaPickerMode).toBe('switch')
  })

  it('shows the poster image when the media has one', () => {
    const media = makeMedia('m1', 'match_001.mp4', 'poster.jpg')
    useStore.setState({
      project: { media: [media] } as unknown as Project,
      mediaId: 'm1',
    })
    const { container } = render(<MediaChip />)
    const img = container.querySelector('img')
    expect(img).not.toBeNull()
    expect(img!.getAttribute('src')).toContain('poster.jpg')
  })

  it('falls back to a film icon when no poster is available', () => {
    const media = makeMedia('m1', 'match_001.mp4', null)
    useStore.setState({
      project: { media: [media] } as unknown as Project,
      mediaId: 'm1',
    })
    const { container } = render(<MediaChip />)
    // No <img> should be present; the film icon (an SVG) is rendered instead.
    expect(container.querySelector('img')).toBeNull()
  })
})
