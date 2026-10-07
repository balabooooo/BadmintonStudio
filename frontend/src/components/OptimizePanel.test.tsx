import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import OptimizePanel from './OptimizePanel'
import { useStore } from '../store/useStore'
import type { OptimizeResult, SegmentMetric } from '../lib/types'

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

const result: OptimizeResult = {
  gt_count: 12,
  focus: [0, 60],
  iou_threshold: 0.5,
  baseline: metric(0.5),
  best: metric(0.62, { seg_prominence: 0.2 }),
  results: [metric(0.62, { seg_prominence: 0.2 }), metric(0.55, { seg_min_quiet: 0.9 })],
  tried: 42,
  search_fields: ['seg_prominence'],
  stages: {
    segment: { search_fields: ['seg_prominence'], tried: 10, best: metric(0.62) },
  },
  suggest: {},
}

function renderPanel(over: Partial<Parameters<typeof OptimizePanel>[0]> = {}) {
  const props = {
    opt: result,
    optimizing: false,
    progress: 0,
    stage: '',
    canOptimize: true,
    matchFormat: null,
    onRun: vi.fn(),
    onCancel: vi.fn(),
    onApplyBest: vi.fn(),
    onApplyParam: vi.fn(),
    onOpenPreset: vi.fn(),
    ...over,
  }
  const utils = render(<OptimizePanel {...props} />)
  return { ...utils, props }
}

describe('OptimizePanel', () => {
  afterEach(() => {
    act(() => useStore.setState({ lang: 'zh' }))
  })

  it('shows the description placeholder when there is no result for this clip', () => {
    act(() => useStore.setState({ lang: 'en' }))
    const { props } = renderPanel({ opt: null, canOptimize: false })
    // Empty state explains what Optimize does instead of showing another clip's stale numbers.
    expect(screen.getByText(/uses your annotations as ground truth/i)).toBeInTheDocument()
    expect(screen.queryByText('F1 0.620')).toBeNull()
    expect(screen.getByRole('button', { name: /^optimize$/i })).toBeDisabled()
    expect(props.onRun).not.toHaveBeenCalled()
  })

  it('renders the current clip metrics, suggested params and the top list', () => {
    act(() => useStore.setState({ lang: 'en' }))
    renderPanel()
    expect(screen.getByText('F1 0.500')).toBeInTheDocument()
    expect(screen.getAllByText('F1 0.620').length).toBeGreaterThanOrEqual(2) // MetricBar + top list
    expect(screen.getByText(/42 sets tried/i)).toBeInTheDocument()
    // 建议参数表和 top 列表都会出现该参数标签
    expect(screen.getAllByText(/silence-valley prominence/i).length).toBeGreaterThanOrEqual(2)
  })

  it('fires the apply callbacks', () => {
    act(() => useStore.setState({ lang: 'en' }))
    const { props } = renderPanel()
    fireEvent.click(screen.getByRole('button', { name: /apply and re-segment/i }))
    expect(props.onApplyBest).toHaveBeenCalledTimes(1)
    // Top-list rows apply their own parameter patch.
    fireEvent.click(screen.getAllByText(/min quiet/i)[0])
    expect(props.onApplyParam).toHaveBeenCalledWith({ seg_min_quiet: 0.9 })
  })

  it('shows the match format badge only for a known format code', () => {
    act(() => useStore.setState({ lang: 'en' }))
    const { rerender } = renderPanel({ matchFormat: 'single' })
    expect(screen.getByText(/match format/i)).toBeInTheDocument()
    rerender(
      <OptimizePanel
        opt={result}
        optimizing={false}
        progress={0}
        stage=""
        canOptimize
        matchFormat={null}
        onRun={vi.fn()}
        onCancel={vi.fn()}
        onApplyBest={vi.fn()}
        onApplyParam={vi.fn()}
        onOpenPreset={vi.fn()}
      />,
    )
    expect(screen.queryByText(/match format/i)).toBeNull()
  })

  it('shows live percentage + stage while optimizing and locks apply actions', () => {
    act(() => useStore.setState({ lang: 'en' }))
    const { props } = renderPanel({ optimizing: true, progress: 0.426, stage: 'sensitivity' })
    expect(screen.getByText('43%')).toBeInTheDocument()
    expect(screen.getByText(/searching hit sensitivity/i)).toBeInTheDocument()
    // The run button is replaced by a cancel button while the job is active.
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(props.onCancel).toHaveBeenCalledTimes(1)
    // Applying params must be impossible mid-run: best button and result rows are disabled.
    expect(screen.getByRole('button', { name: /apply and re-segment/i })).toBeDisabled()
    const rowButton = screen.getAllByRole('button').find((b) => b.textContent?.includes('F1 0.620'))
    expect(rowButton).toBeDefined()
    expect(rowButton).toBeDisabled()
    expect(props.onApplyParam).not.toHaveBeenCalled()
  })
})
