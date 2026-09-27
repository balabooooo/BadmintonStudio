import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Star, Eye, EyeOff, Plus, Check, Trash2, Scissors, Gauge, Timer, Target, Mic } from 'lucide-react'
import { cn, humanDuration, scoreColor, scoreGrade, tagColor, timecode } from '../lib/format'
import { Badge, Button, ScoreBar, ScoreRing, SectionTitle, Segmented, Slider, Tooltip } from './ui'
import { useStore, WEIGHT_PRESETS_MAP, FILM_COVERED_RATIO, rallyFilmCoverage } from '../store/useStore'
import { useT } from '../i18n/useT'
import { tagLabel } from '../i18n/domain'

/* ------------------------------------------------------------------ 信号图 */

function SignalChart({ height = 92 }: { height?: number }) {
  const tr = useT()
  const ref = useRef<HTMLCanvasElement>(null)
  const analysis = useStore((s) => s.currentAnalysis())
  const media = useStore((s) => s.currentMedia())
  const currentTime = useStore((s) => s.currentTime)
  const seek = useStore((s) => s.seek)
  const selectRally = useStore((s) => s.selectRally)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const [hover, setHover] = useState<number | null>(null)

  const dur = media?.duration || 1

  useEffect(() => {
    const cv = ref.current
    if (!cv || !analysis) return
    const dpr = window.devicePixelRatio || 1
    const w = cv.clientWidth
    const h = height
    cv.width = w * dpr
    cv.height = h * dpr
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)

    const act = analysis.signals['activity'] || []
    if (!act.length) return
    const hi = analysis.signals['threshold_hi']?.[0] ?? 0.5

    // 回合区间背景
    analysis.rallies.forEach((r) => {
      const x0 = (r.start / dur) * w
      const x1 = (r.end / dur) * w
      const c = scoreColor(r.scores.total)
      g.fillStyle = r.id === selectedRallyId ? `${c}44` : `${c}1c`
      g.fillRect(x0, 0, Math.max(1, x1 - x0), h)
      if (r.id === selectedRallyId) {
        g.strokeStyle = c
        g.lineWidth = 1
        g.strokeRect(x0 + 0.5, 0.5, Math.max(1, x1 - x0) - 1, h - 1)
      }
    })

    // 阈值线
    const yOf = (v: number) => h - 6 - v * (h - 12)
    g.strokeStyle = 'rgb(255 84 112 / 0.5)'
    g.setLineDash([3, 3])
    g.beginPath()
    g.moveTo(0, yOf(hi))
    g.lineTo(w, yOf(hi))
    g.stroke()
    g.setLineDash([])

    // 活动曲线
    const grad = g.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, 'rgb(56 224 162 / 0.95)')
    grad.addColorStop(1, 'rgb(56 224 162 / 0.06)')
    // 单点曲线时分母为 0 会让坐标变 NaN，整条曲线画不出来
    const denom = Math.max(1, act.length - 1)
    g.beginPath()
    g.moveTo(0, h)
    for (let i = 0; i < act.length; i++) {
      const x = (i / denom) * w
      g.lineTo(x, yOf(act[i]))
    }
    g.lineTo(w, h)
    g.closePath()
    g.fillStyle = grad
    g.fill()
    g.beginPath()
    for (let i = 0; i < act.length; i++) {
      const x = (i / denom) * w
      if (i === 0) g.moveTo(x, yOf(act[i]))
      else g.lineTo(x, yOf(act[i]))
    }
    g.strokeStyle = 'rgb(111 240 192)'
    g.lineWidth = 1.2
    g.stroke()

    // 播放头
    const px = (currentTime / dur) * w
    g.strokeStyle = '#fff'
    g.lineWidth = 1
    g.beginPath()
    g.moveTo(px, 0)
    g.lineTo(px, h)
    g.stroke()

    // 悬停
    if (hover != null) {
      g.strokeStyle = 'rgb(255 255 255 / 0.4)'
      g.beginPath()
      g.moveTo(hover, 0)
      g.lineTo(hover, h)
      g.stroke()
    }
  }, [analysis, dur, currentTime, hover, height, selectedRallyId])

  if (!analysis) return null

  return (
    <div className="relative">
      <canvas
        ref={ref}
        style={{ height, width: '100%' }}
        className="cursor-crosshair rounded-lg border border-white/7 bg-black/25"
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          setHover(e.clientX - r.left)
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const t = ((e.clientX - r.left) / r.width) * dur
          seek(t)
          const hit = analysis.rallies.find((x) => t >= x.start && t <= x.end)
          if (hit) selectRally(hit.id)
        }}
      />
      <div className="mt-1 flex items-center justify-between text-[10px] text-ink-500">
        <span>{tr('inspector.signalChartCaption')}</span>
        <span className="mono">{timecode(dur, false)}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 检查器 */

/** 把后端的模块状态字典翻译成一句人话。 */
function traceLabel(
  trace: any,
  offText: string,
  tr: (key: string, params?: Record<string, string | number>) => string,
): string {
  if (!trace || typeof trace !== 'object') return offText
  if (trace.disabled) return tr('inspector.trace.disabled')
  if (trace.error) return tr('inspector.trace.error')
  if (trace.skipped) return tr('inspector.trace.skipped')
  if (typeof trace.tracks === 'number') return tr('inspector.trace.tracks', { n: trace.tracks })
  if (typeof trace.count === 'number') return tr('inspector.trace.hits', { n: trace.count })
  if (Array.isArray(trace.active_ids)) return tr('inspector.trace.activeIds', { n: trace.active_ids.length })
  return tr('inspector.trace.enabled')
}

export default function Inspector() {
  const tr = useT()
  const analysis = useStore((s) => s.currentAnalysis())
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const selectedClipId = useStore((s) => s.selectedClipId)
  const mediaId = useStore((s) => s.mediaId)
  const patchRally = useStore((s) => s.patchRally)
  const addClip = useStore((s) => s.addClipFromRally)
  const project = useStore((s) => s.project)
  const seek = useStore((s) => s.seek)
  const updateClip = useStore((s) => s.updateClip)
  const pushHistory = useStore((s) => s.pushHistory)
  const removeClip = useStore((s) => s.removeClip)
  const splitClipAt = useStore((s) => s.splitClipAt)
  const splitClipAtSourceTime = useStore((s) => s.splitClipAtSourceTime)
  const previewMode = useStore((s) => s.previewMode)
  const currentTime = useStore((s) => s.currentTime)
  const [tab, setTab] = useState<'rally' | 'clip' | 'info'>('rally')
  // 一旦用户自己点过页签，就不再自动切页签。
  // 切回合常常只是想顺手看看别的回合，右侧停在「信号」上比被拽回「回合」有用得多。
  const pinnedTab = useRef(false)
  const pickTab = (v: 'rally' | 'clip' | 'info') => {
    pinnedTab.current = true
    setTab(v)
  }

  const rally = useMemo(
    () => analysis?.rallies.find((r) => r.id === selectedRallyId) ?? null,
    [analysis, selectedRallyId],
  )
  const clip = useMemo(
    () => project?.timeline.tracks.flatMap((t) => t.clips).find((c) => c.id === selectedClipId) ?? null,
    [project, selectedClipId],
  )
  /** 这个回合在成片里是否已经被覆盖（与 RallyPanel 同口径：按原片时间范围算覆盖率） */
  const inFilm = useMemo(() => {
    if (!rally) return false
    const clips = project?.timeline.tracks.flatMap((t) => t.clips) ?? []
    return rallyFilmCoverage(clips, rally, mediaId ?? undefined).ratio >= FILM_COVERED_RATIO
  }, [project, rally, mediaId])

  const weightValue = analysis?.stats?.weights as string | undefined
  const weightPreset = WEIGHT_PRESETS_MAP.find((w) => w.value === weightValue)
  const weightLabel = weightPreset ? tr(weightPreset.labelKey) : weightValue || tr('weight.balanced.label')
  const traceOffLabels = [tr('inspector.trace.off'), tr('inspector.trace.disabled')]

  // 切换工程时解除页签锁定，让页签重新跟随选择走。
  useEffect(() => {
    pinnedTab.current = false
  }, [project?.id])

  useEffect(() => {
    if (pinnedTab.current) return
    if (selectedClipId) setTab('clip')
    else if (selectedRallyId) setTab('rally')
  }, [selectedClipId, selectedRallyId])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-white/7 px-3 py-2">
        <Segmented
          size="sm"
          value={tab}
          onChange={pickTab}
          options={[
            { value: 'rally', label: tr('inspector.tab.rally') },
            { value: 'clip', label: tr('inspector.tab.clip') },
            { value: 'info', label: tr('inspector.tab.info') },
          ]}
        />
        <div className="mt-1.5 text-[10px] leading-relaxed text-ink-500">
          {tr('inspector.tabsHint')}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {tab === 'rally' &&
          (rally ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <ScoreRing score={rally.scores.total} size={58} color={scoreColor(rally.scores.total)} label={scoreGrade(rally.scores.total)} />
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] font-semibold text-white">{tr('inspector.rallyTitle', { index: rally.index })}</div>
                  <div className="mono mt-0.5 text-[11px] text-ink-400">
                    {timecode(rally.start, false)} → {timecode(rally.end, false)}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {rally.tags.map((t) => (
                      <Badge key={t} color={tagColor(t)}>
                        {tagLabel(t)}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex gap-1.5">
                <Button
                  variant={rally.starred ? 'outline' : 'ghost'}
                  size="sm"
                  className="flex-1"
                  onClick={() => patchRally(rally.id, { starred: !rally.starred })}
                >
                  <Star size={12} fill={rally.starred ? 'currentColor' : 'none'} />
                  {rally.starred ? tr('inspector.starred') : tr('inspector.star')}
                </Button>
                <Button variant="ghost" size="sm" className="flex-1" onClick={() => patchRally(rally.id, { keep: !rally.keep })}>
                  {rally.keep ? <Eye size={12} /> : <EyeOff size={12} />}
                  {rally.keep ? tr('inspector.keep') : tr('inspector.excluded')}
                </Button>
                <Tooltip
                  width={300}
                  className="flex-1"
                  content={
                    inFilm ? (
                      <span>
                        <b className="text-court-300">{tr('inspector.alreadyInFilmTitle')}</b>
                        {'\n\n'}
                        {tr('inspector.alreadyInFilmDesc')}
                      </span>
                    ) : (
                      <span>
                        <b className="text-court-300">{tr('inspector.addToFilmTitle')}</b>
                        {'\n\n'}
                        {tr('inspector.addToFilmDesc')}
                        {'\n\n'}
                        {tr('inspector.addToFilmDesc2')}
                      </span>
                    )
                  }
                >
                  <Button
                    variant={inFilm ? 'subtle' : 'outline'}
                    size="sm"
                    className="w-full"
                    onClick={() => addClip(rally)}
                  >
                    {inFilm ? <Check size={12} /> : <Plus size={12} />}
                    {inFilm ? tr('inspector.inFilm') : tr('inspector.addToFilm')}
                  </Button>
                </Tooltip>
              </div>

              <div>
                <SectionTitle>
                  {tr('inspector.scoreBreakdownTitle')}
                </SectionTitle>
                <div className="mb-1.5 text-[10.5px] leading-relaxed text-ink-500">
                  {tr('inspector.scoreBreakdownHint')}
                </div>
                <div className="space-y-2.5">
                  {(
                    [
                      ['weight.dim.length', rally.scores.length, tr('inspector.dim.lengthHint')],
                      ['weight.dim.intensity', rally.scores.intensity, tr('inspector.dim.intensityHint')],
                      ['weight.dim.technique', rally.scores.technique, tr('inspector.dim.techniqueHint')],
                      ['weight.dim.excitement', rally.scores.excitement, tr('inspector.dim.excitementHint')],
                      ['weight.dim.production', rally.scores.production, tr('inspector.dim.productionHint')],
                    ] as const
                  ).map(([k, v, hint]) => (
                    <Tooltip key={k} content={hint} side="left" width={280} block>
                      <div className="w-full cursor-help">
                        <div className="mb-1 flex justify-between text-[11px]">
                          <span className="text-ink-300 underline decoration-dotted decoration-ink-600 underline-offset-2">
                            {tr(k)}
                          </span>
                          <span className="mono" style={{ color: scoreColor(v) }}>
                            {v.toFixed(0)}
                          </span>
                        </div>
                        <ScoreBar value={v} color={scoreColor(v)} />
                      </div>
                    </Tooltip>
                  ))}
                </div>
                <Tooltip
                  side="left"
                  width={300}
                  content={
                    <span>
                      {tr('inspector.totalFormula1')}
                      {'\n\n'}
                      {tr('inspector.totalFormula2')}
                      {'\n\n'}
                      {tr('inspector.totalFormula3')}
                    </span>
                  }
                >
                  <div className="mt-2 cursor-help text-[10.5px] text-ink-500 underline decoration-dotted underline-offset-2">
                    {tr('inspector.totalHow', { score: rally.scores.total.toFixed(1) })}
                  </div>
                </Tooltip>
              </div>

              <div>
                <SectionTitle>{tr('inspector.objectiveTitle')}</SectionTitle>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    [tr('inspector.metric.duration'), `${rally.duration.toFixed(2)}s`, Timer],
                    [tr('inspector.metric.shots'), `${rally.features.shot_count}`, Target],
                    [tr('inspector.metric.tempo'), `${rally.features.tempo.toFixed(2)}/s`, Gauge],
                    [tr('inspector.metric.confidence'), `${(rally.features.confidence * 100).toFixed(0)}%`, Activity],
                    ...(rally.features.speech_bonus > 0
                      ? [[tr('inspector.metric.speechBonus'), tr('inspector.metric.speechBonusValue', { n: rally.features.speech_bonus, phrases: (rally.features.speech_phrases || []).join(tr('common.listSeparator')) }), Mic]]
                      : []),
                  ].map(([k, v, Icon]: any) => (
                    <div key={k} className="panel-flat flex items-center gap-2 px-2.5 py-2">
                      <Icon size={13} className="text-ink-500" />
                      <div className="min-w-0">
                        <div className="text-[10px] text-ink-500">{k}</div>
                        <div className="mono text-[12.5px] text-ink-100">{v}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {rally.shots.length > 0 && (
                <div>
                  <SectionTitle right={<span className="text-[10.5px] text-ink-500">{tr('inspector.clickToSeek')}</span>}>
                    {tr('inspector.shotTimelineTitle')}
                  </SectionTitle>
                  <div className="flex flex-wrap gap-1">
                    {rally.shots.map((s, i) => (
                      <Tooltip key={i} content={tr('inspector.shotTooltip', { n: i + 1, time: s.time.toFixed(2), conf: (s.confidence * 100).toFixed(0) })}>
                        <button
                          onClick={() => seek(s.time)}
                          className={cn(
                            'mono rounded px-1.5 py-[2px] text-[10.5px] transition-colors',
                            i === 0
                              ? 'bg-amber-glow/20 text-amber-glow'
                              : i === 1
                                ? 'bg-flux-400/20 text-flux-400'
                                : 'bg-white/7 text-ink-300 hover:bg-white/14',
                          )}
                        >
                          {i === 0 ? tr('inspector.serve') : i === 1 ? tr('inspector.receive') : i + 1}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                  <div className="mt-1.5 text-[10.5px] text-ink-500">
                    {tr('inspector.serveTime', { time: rally.serve_time ? `${rally.serve_time.toFixed(2)}s` : '—' })} ·{' '}
                    {tr('inspector.receiveTime', { time: rally.receive_time ? `${rally.receive_time.toFixed(2)}s` : '—' })}
                  </div>
                </div>
              )}

              <div>
                <SectionTitle>{tr('inspector.clipRangeTitle')}</SectionTitle>
                <div className="mb-1.5 text-[10.5px] leading-relaxed text-ink-500">
                  {tr('inspector.clipRangeHint1')}
                  <br />
                  {tr('inspector.clipRangeHint2')}
                </div>
                <div className="space-y-2">
                  <Slider
                    label={tr('inspector.inPoint')}
                    value={rally.clip_start}
                    min={0}
                    max={Math.max(0.1, rally.clip_end - 0.05)}
                    step={0.05}
                    onChange={(v) => patchRally(rally.id, { clip_start: Math.max(0, Math.min(v, rally.clip_end - 0.05)) })}
                    format={(v) => timecode(v, false)}
                  />
                  <Slider
                    label={tr('inspector.outPoint')}
                    value={rally.clip_end}
                    min={Math.max(0, rally.clip_start + 0.05)}
                    max={(analysis?.stats?.duration as number) ?? rally.clip_end + 30}
                    step={0.05}
                    onChange={(v) => patchRally(rally.id, { clip_end: Math.max(v, rally.clip_start + 0.05) })}
                    format={(v) => timecode(v, false)}
                  />
                  <div className="flex gap-1.5 pt-0.5">
                    {(
                      [
                        [tr('inspector.presetTight'), 0.2, 0.4, tr('inspector.presetTightHint')],
                        [tr('inspector.presetStandard'), 0.8, 1.2, tr('inspector.presetStandardHint')],
                        [tr('inspector.presetLoose'), 1.8, 2.6, tr('inspector.presetLooseHint')],
                      ] as const
                    ).map(([label, pre, post, hint]) => (
                      <Tooltip key={label} content={hint} width={220}>
                        <button
                          onClick={() =>
                            patchRally(rally.id, {
                              clip_start: Math.max(0, rally.start - pre),
                              clip_end: Math.min(
                                (analysis?.stats?.duration as number) ?? rally.end + post,
                                rally.end + post,
                              ),
                            })
                          }
                          className="rounded-md bg-white/6 px-2 py-1 text-[11px] text-ink-300 transition-colors hover:bg-court-500/20 hover:text-court-200"
                        >
                          {label}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              </div>

              <div>
                <SectionTitle>{tr('inspector.noteTitle')}</SectionTitle>
                <textarea
                  value={rally.note}
                  onChange={(e) => patchRally(rally.id, { note: e.target.value })}
                  rows={2}
                  placeholder={tr('inspector.notePlaceholder')}
                  className="field text-[12px]"
                />
              </div>
            </div>
          ) : (
            <div className="px-2 py-10 text-center text-[12px] text-ink-500">
              {tr('inspector.emptyRally')}
            </div>
          ))}

        {tab === 'clip' &&
          (clip ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-flux-400/25 bg-flux-400/[0.06] px-3 py-2.5">
                <div className="text-[11.5px] font-medium text-flux-400">{tr('inspector.clipCardTitle')}</div>
                <div className="mt-1 text-[10.5px] leading-relaxed text-ink-400">
                  {tr('inspector.clipCardDesc1')}
                  <br />
                  {tr('inspector.clipCardDesc2')}
                </div>
              </div>

              <div>
                <div className="text-[13.5px] font-semibold text-white">{clip.label || tr('inspector.tab.clip')}</div>
                <div className="mono mt-0.5 text-[11px] text-ink-400">
                  {tr('inspector.clipSource', { from: timecode(clip.src_in, false), to: timecode(clip.src_out, false) })}
                  {' · '}
                  {tr('inspector.clipMaterialDuration', { duration: (clip.src_out - clip.src_in).toFixed(2) })}
                </div>
                <div className="mono mt-0.5 text-[11px] text-ink-500">
                  {tr('inspector.clipTimelinePos', { from: timecode(clip.tl_start, false), to: timecode(clip.tl_start + (clip.src_out - clip.src_in) / clip.speed, false) })}
                  {' · '}
                  {tr('inspector.clipOccupies', { duration: ((clip.src_out - clip.src_in) / clip.speed).toFixed(2) })}
                </div>
              </div>

              <div className="flex gap-1.5">
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1"
                  onClick={() =>
                    previewMode === 'source'
                      ? splitClipAtSourceTime(clip.id, currentTime)
                      : splitClipAt(clip.id, currentTime)
                  }
                >
                  <Scissors size={12} />
                  {tr('inspector.split')}
                </Button>
                <Button variant="ghost" size="sm" className="flex-1" onClick={() => removeClip(clip.id)}>
                  <Trash2 size={12} className="text-rose-hot/80" />
                  {tr('common.delete')}
                </Button>
              </div>

              <div className="space-y-2.5">
                <Slider
                  label={tr('inspector.playbackSpeed')}
                  value={clip.speed}
                  min={0.25}
                  max={4}
                  step={0.05}
                  onStart={() => pushHistory()}
                  onChange={(v) => updateClip(clip.id, { speed: Number(v.toFixed(2)) }, false)}
                  format={(v) => `${v.toFixed(2)}×`}
                  hint={tr('inspector.playbackSpeedHint')}
                />
                <Slider
                  label={tr('inspector.volume')}
                  value={clip.volume}
                  min={0}
                  max={2}
                  step={0.05}
                  onStart={() => pushHistory()}
                  onChange={(v) => updateClip(clip.id, { volume: Number(v.toFixed(2)) }, false)}
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                />
              </div>

              <div>
                <div className="mb-1.5 text-[10.5px] text-ink-500">{tr('inspector.speedPresets')}</div>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    [0.35, tr('inspector.speed.superSlow')],
                    [0.5, tr('inspector.speed.slow')],
                    [1, tr('inspector.speed.normal')],
                    [1.5, tr('inspector.speed.fast15')],
                    [2, tr('inspector.speed.fast2')],
                  ].map(([s, label]) => (
                    <button
                      key={label as string}
                      onClick={() => updateClip(clip.id, { speed: s as number })}
                      className={cn(
                        'rounded-md px-2 py-1 text-[11px] transition-colors',
                        Math.abs(clip.speed - (s as number)) < 0.01
                          ? 'bg-court-500/20 text-court-300'
                          : 'bg-white/6 text-ink-300 hover:bg-white/12',
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
                  {tr('inspector.dragHint')}
                </div>
              </div>
            </div>
          ) : (
            <div className="px-2 py-10 text-center text-[12px] text-ink-500">{tr('inspector.emptyClip')}</div>
          ))}

        {tab === 'info' && (
          <div className="space-y-4">
            <div>
              <SectionTitle>
                <Activity size={12} /> {tr('inspector.signalChartTitle')}
              </SectionTitle>
              <SignalChart />
              <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
                {tr('inspector.signalInfo1')}
                <br />
                {tr('inspector.signalInfo2')}
                <br />
                {tr('inspector.signalInfo3')}
                <br />
                {tr('inspector.signalInfo4')}
              </div>
            </div>

            <div>
              <SectionTitle>{tr('inspector.analysisInputsTitle')}</SectionTitle>
              <div className="space-y-1.5">
                {[
                  [tr('inspector.signal.player'), traceLabel(analysis?.stats?.player_trace, tr('inspector.trace.off'), tr),
                    tr('inspector.signal.playerDesc')],
                  [tr('inspector.signal.hit'), traceLabel(analysis?.stats?.hit_trace, tr('inspector.trace.off'), tr),
                    tr('inspector.signal.hitDesc')],
                  [tr('inspector.signal.motion'), tr('common.enabled'), tr('inspector.signal.motionDesc')],
                  [tr('inspector.signal.shuttle'), traceLabel(analysis?.stats?.shuttle_trace, tr('inspector.trace.disabled'), tr),
                    tr('inspector.signal.shuttleDesc')],
                ].map(([name, status, desc]) => (
                  <div key={name as string} className="panel-flat flex items-center gap-2 px-2.5 py-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-[11.5px] text-ink-100">{name}</div>
                      <div className="text-[10px] text-ink-500">{desc}</div>
                    </div>
                    <span
                      className={cn(
                        'shrink-0 rounded px-1.5 py-[2px] text-[10px]',
                        traceOffLabels.some((x) => String(status).includes(x))
                          ? 'bg-white/6 text-ink-400'
                          : 'bg-court-500/15 text-court-300',
                      )}
                    >
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {analysis?.stats && (
              <div>
                <SectionTitle>{tr('inspector.statsTitle')}</SectionTitle>
                <div className="space-y-1.5 text-[11.5px]">
                  {[
                    [tr('inspector.stat.rallyCount'), `${analysis.stats.count}`],
                    [tr('inspector.stat.activeDuration'), humanDuration(analysis.stats.active_duration || 0)],
                    [tr('inspector.stat.totalDuration'), humanDuration(analysis.stats.duration || 0)],
                    [tr('inspector.stat.activeRatio'), `${(((analysis.stats.active_duration || 0) / Math.max(1, analysis.stats.duration || 1)) * 100).toFixed(1)}%`],
                    [tr('inspector.stat.avgShots'), `${(analysis.stats.avg_shots || 0).toFixed(1)}`],
                    [tr('inspector.stat.maxShots'), `${analysis.stats.max_shots || 0}`],
                    [tr('inspector.stat.avgScore'), `${(analysis.stats.avg_score || 0).toFixed(1)}`],
                    [tr('inspector.stat.highScore'), `${analysis.stats.high_score_count || 0}${tr('inspector.stat.highScoreSuffix')}`],
                    [tr('inspector.stat.weights'), weightLabel],
                  ].map(([k, v]) => (
                    <div key={k as string} className="flex items-center justify-between gap-3">
                      <span className="text-ink-400">{k}</span>
                      <span className="mono text-ink-100">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
