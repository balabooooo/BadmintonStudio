/**
 * 标注页右下角「用标注优化切分参数」面板。
 *
 * 从 AnnotatePage 抽出的纯展示组件：所有状态（当前素材的 OptimizeResult、loading、
 * 是否可优化）由父组件按 clip 维护并经 props 传入——面板内容永远只反映当前激活
 * clip 的优化结果（切换 clip 时由父组件切换绑定，不在这里自行取数）。
 */

import { Info, Wand2 } from 'lucide-react'
import { Badge, Button } from './ui'
import { useT } from '../i18n/useT'
import { matchFormatLabel } from '../i18n/domain'
import { clamp } from '../lib/format'
import type { OptimizeResult, SegmentMetric } from '../lib/types'

const SEG_LABEL_KEYS: Record<string, string> = {
  seg_prominence: 'annotate.seg.prominence',
  seg_min_core: 'annotate.seg.minCore',
  seg_min_rest: 'annotate.seg.minRest',
  seg_min_quiet: 'annotate.seg.minQuiet',
  min_rally_seconds: 'annotate.seg.minRally',
  pose_gate_threshold: 'annotate.seg.poseGate',
  pose_gate_window: 'annotate.seg.poseGateWindow',
  pose_gate_one_to_one: 'annotate.seg.poseGateOneToOne',
  pose_gate_force: 'annotate.seg.poseGateForce',
  hit_sensitivity: 'annotate.seg.hitSensitivity',
  pre_roll: 'annotate.seg.preRoll',
  post_roll: 'annotate.seg.postRoll',
  fuse_weight_players: 'annotate.seg.fusePlayers',
  fuse_weight_motion: 'annotate.seg.fuseMotion',
  fuse_weight_audio: 'annotate.seg.fuseAudio',
  fuse_weight_shuttle: 'annotate.seg.fuseShuttle',
  fuse_weight_roi: 'annotate.seg.fuseRoi',
}

/** 描边进度条（优化结果里的 F1 对比） */
function MetricBar({ label, m, color }: { label: string; m: SegmentMetric; color: string }) {
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-ink-300">{label}</span>
        <span className="mono text-[11px]" style={{ color }}>
          F1 {m.f1.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/8">
        <div className="h-full rounded-full" style={{ width: `${clamp(m.f1, 0, 1) * 100}%`, background: color }} />
      </div>
      <div className="mono mt-1 flex gap-2 text-[10px] text-ink-500">
        <span>P {m.precision.toFixed(2)}</span>
        <span>R {m.recall.toFixed(2)}</span>
        <span>n {m.n}</span>
      </div>
    </div>
  )
}

interface OptimizePanelProps {
  /** 当前素材的优化结果（per-clip，由父组件按 mediaId 取） */
  opt: OptimizeResult | null
  optimizing: boolean
  canOptimize: boolean
  /** 赛制稳定码（single/doubles），未知时 null 不显示徽标 */
  matchFormat: string | null
  onRun: () => void
  onApplyBest: () => void
  onApplyParam: (patch: Record<string, number>) => void
  onOpenPreset: () => void
}

export default function OptimizePanel({
  opt,
  optimizing,
  canOptimize,
  matchFormat,
  onRun,
  onApplyBest,
  onApplyParam,
  onOpenPreset,
}: OptimizePanelProps) {
  const tr = useT()
  return (
    <div className="border-t border-white/7 p-3">
      <div className="mb-2 flex items-center gap-2">
        <Wand2 size={13} className="text-court-300" />
        <span className="text-[12px] font-semibold text-white">{tr('annotate.optimizeTitle')}</span>
        <div className="flex-1" />
        <Button size="sm" variant="ghost" onClick={onOpenPreset} title={tr('annotate.savePresetTooltip')}>
          {tr('annotate.saveAsPreset')}
        </Button>
        <Button size="sm" variant="primary" loading={optimizing} disabled={!canOptimize} onClick={onRun}>
          {tr('annotate.optimize')}
        </Button>
      </div>
      {matchFormat && matchFormat !== 'unknown' && (
        <div className="mb-2 flex items-center gap-1.5 rounded-lg border border-white/8 bg-white/[0.02] px-2 py-1 text-[10.5px] text-ink-300">
          <Info size={11} className="shrink-0 text-court-300" />
          <span className="text-ink-500">{tr('annotate.matchFormat')}：</span>
          <span className="font-medium">{matchFormatLabel(matchFormat)}</span>
          <span className="ml-auto text-ink-600">{tr('annotate.matchFormatHint')}</span>
        </div>
      )}
      {!opt ? (
        <div className="text-[11px] leading-relaxed text-ink-500">
          {tr('annotate.optimizeDesc')}
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex gap-3">
            <MetricBar label={tr('annotate.baseline')} m={opt.baseline} color="#8b96ad" />
            {opt.best && <MetricBar label={tr('annotate.best')} m={opt.best} color="#38e0a2" />}
          </div>
          <div className="text-[10.5px] text-ink-500">
            {tr('annotate.optStats', { gt: opt.gt_count, tried: opt.tried, iou: opt.iou_threshold })}
          </div>
          {(opt.best?.hit || opt.baseline?.hit) && (
            <div className="rounded-lg border border-white/8 bg-white/[0.02] p-2 text-[10.5px] text-ink-300">
              <div className="mb-0.5 text-ink-500">{tr('annotate.hitMetricsTitle')}</div>
              <div className="mono flex flex-wrap gap-x-3 gap-y-0.5">
                <span>
                  {tr('annotate.hitBaseline')} P {(opt.baseline.hit?.precision ?? 0).toFixed(2)} · R{' '}
                  {(opt.baseline.hit?.recall ?? 0).toFixed(2)} · F1 {(opt.baseline.hit?.f1 ?? 0).toFixed(3)}
                </span>
                {opt.best?.hit && (
                  <span className="text-court-200">
                    {tr('annotate.hitBest')} P {opt.best.hit.precision.toFixed(2)} · R{' '}
                    {opt.best.hit.recall.toFixed(2)} · F1 {opt.best.hit.f1.toFixed(3)}
                  </span>
                )}
              </div>
            </div>
          )}
          {opt.stages && (
            <div className="flex flex-wrap gap-1.5 text-[10px]">
              {(['segment', 'gate', 'sensitivity', 'padding', 'weights'] as const).map((st) => {
                const s = opt.stages?.[st]
                if (!s) return null
                return (
                  <span key={st} className="rounded border border-white/10 px-1.5 py-[1px] text-ink-400">
                    {tr(`annotate.stage.${st}`)}
                    {st === 'weights' ? (
                      <> · {tr('annotate.stage.weightsMoves', { n: s.accepted_moves ?? 0 })}</>
                    ) : (
                      <> · F1 {(s.best?.f1 ?? 0).toFixed(3)}{s.best?.hit ? ` · hitF1 ${s.best.hit.f1.toFixed(3)}` : ''}</>
                    )}
                  </span>
                )
              })}
            </div>
          )}
          {opt.best && (
            <div className="rounded-lg border border-court-500/25 bg-court-500/[0.06] p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[11px] text-court-200">{tr('annotate.suggested')}</span>
                <div className="flex-1" />
                <Button size="sm" variant="primary" onClick={onApplyBest}>{tr('annotate.applyResegment')}</Button>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                {Object.entries(opt.best.params).map(([k, v]) => (
                  <div key={k} className="mono flex justify-between text-[10.5px] text-ink-300">
                    <span className="truncate text-ink-500">{tr(SEG_LABEL_KEYS[k] || k)}</span>
                    <span>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="max-h-[180px] overflow-y-auto rounded-lg border border-white/8">
            {opt.results.slice(0, 20).map((m, i) => (
              <button
                key={i}
                onClick={() => onApplyParam(m.params)}
                className="flex w-full items-center gap-2 border-b border-white/5 px-2 py-1 text-left text-[11px] last:border-b-0 hover:bg-white/6"
              >
                <Badge color={i === 0 ? '#38e0a2' : undefined}>F1 {m.f1.toFixed(3)}</Badge>
                <span className="mono flex-1 truncate text-ink-500">
                  {Object.entries(m.params).map(([k, v]) => `${tr(SEG_LABEL_KEYS[k] || k)}=${v}`).join(' · ')}
                </span>
                <span className="text-ink-500">n{m.n}</span>
              </button>
            ))}
          </div>
          <div className="text-[10.5px] leading-relaxed text-ink-500">
            {tr('annotate.resegmentHint')}
          </div>
        </div>
      )}
    </div>
  )
}
