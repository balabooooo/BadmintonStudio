import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { Sparkles, Activity, Wand2, AlertTriangle, Video, Crosshair, Ruler, Trash2, Images, ListChecks } from 'lucide-react'
import { cn, humanDuration } from '../lib/format'
import { api } from '../lib/api'
import type { AnalysisParams } from '../lib/types'
import { Button, Modal, Progress, SectionTitle, Segmented, Slider, Toggle, useConfirm } from './ui'
import SizeFilterPanel from './SizeFilterPanel'
import { WEIGHT_PRESETS_MAP, useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
import { jobStageLabel } from '../i18n/domain'

const VIEWPOINTS: { value: AnalysisParams['viewpoint']; labelKey: string; hintKey: string }[] = [
  { value: 'auto', labelKey: 'analysis.viewpoint.auto', hintKey: 'analysis.viewpoint.autoHint' },
  { value: 'rear', labelKey: 'analysis.viewpoint.rear', hintKey: 'analysis.viewpoint.rearHint' },
  { value: 'side', labelKey: 'analysis.viewpoint.side', hintKey: 'analysis.viewpoint.sideHint' },
  { value: 'elevated', labelKey: 'analysis.viewpoint.elevated', hintKey: 'analysis.viewpoint.elevatedHint' },
  { value: 'overhead', labelKey: 'analysis.viewpoint.overhead', hintKey: 'analysis.viewpoint.overheadHint' },
]

const SPEECH_MODELS: AnalysisParams['speech_model'][] = ['tiny', 'base', 'small', 'medium', 'large-v3']

/**
 * 语音口令检测的诊断行：有结果报命中数，缺依赖 / 缺模型给出提示。
 *
 * ``speechReady`` 是当前环境（`/api/env` 的 `caps.speech`）的实时能力位。旧分析留下
 * 的失败 trace 在依赖补装后不会自己消失；此时旧错误与现状矛盾，所以整段替换成
 * 「上次缺失、现已就绪」的提示，而不是把两句话拼在一起。
 */
function SpeechTraceLine({ trace, speechReady }: { trace: unknown; speechReady: boolean }) {
  const tr = useT()
  if (!trace || typeof trace !== 'object') return null
  const t = trace as Record<string, unknown>
  let text = ''
  let warn = false
  if (t.error || t.hint) {
    text = [t.error, t.hint].filter(Boolean).map(String).join('；')
    warn = true
  } else if (t.disabled) {
    text = String(t.disabled)
  } else if (t.available) {
    const model = t.model ? String(t.model).split(/[\\/]/).pop() : ''
    text = `${tr('analysis.speech.hitCount', { n: Number(t.count ?? 0) })}${model ? tr('analysis.speech.modelSuffix', { model }) : ''}`
  } else {
    return null
  }
  const stale = warn && speechReady
  if (stale) text = tr('analysis.speech.staleHint')
  return (
    <div className={cn('mt-2 text-[10.5px] leading-relaxed',
      stale ? 'text-court-400/90' : warn ? 'text-amber-glow/90' : 'text-ink-500')}>
      · {text}
    </div>
  )
}

export default function AnalysisDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const tr = useT()
  const params = useStore((s) => s.params)
  const setParams = useStore((s) => s.setParams)
  const runAnalysis = useStore((s) => s.runAnalysis)
  const runAnalysisBatch = useStore((s) => s.runAnalysisBatch)
  const project = useStore((s) => s.project)
  const env = useStore((s) => s.env)
  const refreshEnv = useStore((s) => s.refreshEnv)
  const media = useStore((s) => s.currentMedia())
  const openMediaPicker = useStore((s) => s.openMediaPicker)
  const selectMedia = useStore((s) => s.selectMedia)
  const analysis = useStore((s) => s.currentAnalysis())
  const jobs = useStore((s) => s.jobs)
  const weights = useStore((s) => s.weights)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const openCourtEditor = useStore((s) => s.setCourtEditorOpen)
  const presets = useStore((s) => s.presets)
  const refreshPresets = useStore((s) => s.refreshPresets)
  const applyPreset = useStore((s) => s.applyPreset)
  const deletePreset = useStore((s) => s.deletePreset)
  const [advanced, setAdvanced] = useState(false)
  const [scope, setScope] = useState<'all' | 'current'>('all')
  const confirm = useConfirm()
  const toast = useStore((s) => s.toast)

  // 打开时拉一次预设列表（保存后也靠它同步），并刷新环境能力位：
  // 用户可能在 app 运行期间才补装 faster-whisper，旧的能力位会让提示与现状不符。
  useEffect(() => {
    if (open) {
      void refreshPresets()
      void refreshEnv()
    }
  }, [open, refreshPresets, refreshEnv])

  const activeJob = useMemo(() => {
    const list = Object.values(jobs).filter((j) => j.kind === 'analyze')
    // 「取消当前」要打在真正在跑的任务上。批量分析会排队多个，按时间排序会先命中
    // 后面的排队任务，取消它等于什么都没做；所以 running 优先，其次才是最早的 queued。
    const runningJob = list.find((j) => j.status === 'running')
    if (runningJob) return runningJob
    return list
      .filter((j) => j.status === 'queued')
      .sort((a, b) => a.created_at - b.created_at)[0]
  }, [jobs])

  const lastJob = useMemo(
    () => Object.values(jobs).filter((j) => j.kind === 'analyze').sort((a, b) => b.created_at - a.created_at)[0],
    [jobs],
  )

  // 批量分析会同时入队多个 analyze 任务，这里统计整体进度用于「取消全部」。
  const activeAnalyze = useMemo(
    () =>
      Object.values(jobs)
        .filter((j) => j.kind === 'analyze' && (j.status === 'running' || j.status === 'queued'))
        .sort((a, b) => a.created_at - b.created_at),
    [jobs],
  )
  const mediaCount = project?.media.length ?? 0

  const done = analysis?.status === 'done'

  // 开启语音口令但一个短语都没填时，后端会因为 speech_phrases 为空而整段跳过，
  // 界面却看起来照常分析。这里直接判定为无效并在开始前拦住。
  const speechPhrases = (params.speech_phrases ?? []).map((p) => p.trim()).filter(Boolean)
  const speechInvalid = params.use_speech && speechPhrases.length === 0

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={tr('analysis.title')}
      subtitle={media ? `${media.name} · ${humanDuration(media.duration)} · ${media.width}×${media.height}` : undefined}
      width={720}
      footer={
        <div className="flex items-center justify-between gap-3">
          <div className="text-[11px] text-ink-500">
            {media && media.width >= 3000 ? (
              <span className="inline-flex items-center gap-1.5 text-amber-glow">
                <AlertTriangle size={12} /> {tr('analysis.slow4k')}
              </span>
            ) : (
              tr('analysis.localOnly')
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" data-tour="dlg-close" onClick={onClose}>
              {tr('common.close')}
            </Button>
            <Button
              variant="primary"
              data-tour="dlg-run"
              loading={!!activeJob}
              disabled={!media || !!activeJob || speechInvalid || (scope === 'all' && mediaCount === 0)}
              onClick={async () => {
                if (speechInvalid) {
                  toast({ kind: 'warn', title: tr('analysis.speech.needPhrase') })
                  return
                }
                if (scope === 'all') await runAnalysisBatch()
                else await runAnalysis()
              }}
            >
              <Sparkles size={14} />
              {scope === 'all'
                ? tr('analysis.analyzeAll', { n: mediaCount })
                : done
                  ? tr('analysis.reanalyzeCurrent')
                  : tr('analysis.analyzeCurrent')}
            </Button>
          </div>
        </div>
      }
    >
      {/* 进行中 */}
      <AnimatePresence>
        {activeJob && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 overflow-hidden"
          >
            <div className="panel-flat border-court-500/25 bg-court-500/[0.07] px-4 py-3.5">
              <div className="flex items-center gap-2">
                <span className="spin h-3.5 w-3.5 rounded-full border-[2px] border-court-400 border-t-transparent" />
                <span className="text-[12.5px] font-medium text-court-200">
                  {jobStageLabel(activeJob.stage) || activeJob.stage || tr('analysis.processing')}
                </span>
                <span className="mono ml-auto text-[12px] text-court-300">
                  {Math.round(activeJob.progress * 100)}%
                </span>
              </div>
              <Progress value={activeJob.progress} className="mt-2.5" />
              <div className="mt-2 text-[11px] text-ink-400">{activeJob.message}</div>
              <div className="mt-2 flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => api.cancelJob(activeJob.id)}
                >
                  {tr('analysis.cancelCurrent')}
                </Button>
                {activeAnalyze.length > 1 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      void Promise.all(activeAnalyze.map((j) => api.cancelJob(j.id)))
                    }}
                  >
                    {tr('analysis.cancelAll', { n: activeAnalyze.length })}
                  </Button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {lastJob?.status === 'error' && !activeJob && (
        <div className="mb-4 rounded-xl border border-rose-hot/30 bg-rose-hot/10 px-4 py-3">
          <div className="flex items-center gap-2 text-[12.5px] font-medium text-rose-hot">
            <AlertTriangle size={13} /> {tr('analysis.lastFailed')}
          </div>
          <div className="mono mt-1.5 max-h-[120px] overflow-y-auto text-[10.5px] whitespace-pre-wrap text-rose-hot/80">
            {lastJob.error || lastJob.message}
          </div>
        </div>
      )}

      {/* 分析范围：默认整个工程，也可以只跑当前素材 */}
      <SectionTitle>
        <ListChecks size={12} /> {tr('analysis.scopeTitle')}
      </SectionTitle>
      <div data-tour="dlg-scope" className="mb-4 flex flex-wrap gap-1.5">
        {([
          { value: 'all' as const, label: tr('analysis.scopeAll', { n: mediaCount }), hint: tr('analysis.scopeAllHint') },
          { value: 'current' as const, label: tr('analysis.scopeCurrent'), hint: media?.name || tr('analysis.scopeCurrentHint') },
        ]).map((o) => (
          <button
            key={o.value}
            onClick={() => setScope(o.value)}
            className={cn(
              'max-w-[280px] rounded-lg border px-3 py-2 text-left transition-all',
              scope === o.value
                ? 'border-court-500/50 bg-court-500/12'
                : 'border-white/8 bg-white/[0.025] hover:border-white/18',
            )}
          >
            <div className={cn('text-[12px] font-medium', scope === o.value ? 'text-court-200' : 'text-ink-100')}>
              {o.label}
            </div>
            <div className="mt-0.5 truncate text-[10.5px] text-ink-400">{o.hint}</div>
          </button>
        ))}
      </div>
      {scope === 'all' && (
        <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-2.5 text-[11px] leading-relaxed text-ink-500">
          {tr('analysis.scopeAllNote')}
        </div>
      )}
      {scope === 'current' && media && (
        <div className="mb-4 flex items-center gap-2.5 rounded-xl border border-white/8 bg-white/[0.02] px-3 py-2">
          {media.poster ? (
            <img src={api.assetUrl(media.poster)} alt="" className="h-9 w-16 shrink-0 rounded object-cover" />
          ) : (
            <div className="grid h-9 w-16 shrink-0 place-items-center rounded bg-ink-850 text-ink-600">
              <Video size={14} />
            </div>
          )}
          <div className="min-w-0 flex-1 truncate text-[11.5px] text-ink-200" title={media.path}>
            {media.name}
          </div>
          <Button
            size="sm"
            variant="subtle"
            onClick={() =>
              openMediaPicker('select', (mid) => {
                selectMedia(mid)
                setScope('current')
              })
            }
          >
            {tr('mediaPicker.change')}
          </Button>
        </div>
      )}

      {/* 场景预设：看截图自己挑，套用只是写入参数/标定，不会自动重跑 */}
      {presets.length > 0 && (
        <>
          <SectionTitle>
            <Images size={12} /> {tr('analysis.presetsTitle')}
          </SectionTitle>
          <div className="mb-4">
            <div className="flex gap-2 overflow-x-auto pb-1">
              {presets.map((p) => (
                <div
                  key={p.id}
                  className="group w-[150px] shrink-0 overflow-hidden rounded-xl border border-white/8 bg-white/[0.025]"
                >
                  <div className="relative h-[84px] bg-ink-850">
                    {p.preview ? (
                      <img src={api.assetUrl(p.preview)} alt="" className="h-full w-full object-cover" loading="lazy" />
                    ) : (
                      <div className="grid h-full place-items-center text-ink-700">
                        <Images size={20} />
                      </div>
                    )}
                  </div>
                  <div className="p-1.5">
                    <div className="truncate text-[11.5px] font-medium text-ink-100" title={p.name}>
                      {p.name}
                    </div>
                    <div className="truncate text-[10px] text-ink-500" title={`${p.source.project_name} · ${p.source.media_name}`}>
                      {p.source.media_name || tr('analysis.unknownMedia')}
                    </div>
                    <div className="mt-1.5 flex items-center gap-1">
                      <Button size="sm" variant="primary" className="flex-1" onClick={() => applyPreset(p)}>
                        {tr('analysis.applyPreset')}
                      </Button>
                      <button
                        onClick={async () => {
                          const ok = await confirm({
                            title: tr('analysis.deletePresetTitle', { name: p.name }),
                            desc: tr('analysis.deletePresetDesc'),
                            danger: true,
                          })
                          if (ok) await deletePreset(p.id)
                        }}
                        title={tr('analysis.deletePreset')}
                        className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-ink-500 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-white/8 hover:text-rose-hot"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-1 text-[10.5px] leading-relaxed text-ink-500">
              {tr('analysis.presetsNote')}
            </div>
          </div>
        </>
      )}

      {done && analysis?.stats?.mock && (
        <div
          data-tour-mock-badge
          className="mb-2 inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-300"
        >
          <span aria-hidden="true">🧪</span>
          {tr('tour.mock.badge')}
        </div>
      )}

      {done && analysis && (
        <div className="mb-4 grid grid-cols-4 gap-2">
          {[
            [tr('analysis.stat.rallies'), analysis.stats?.count ?? 0],
            [tr('analysis.stat.activeDuration'), humanDuration(analysis.stats?.active_duration ?? 0)],
            [tr('analysis.stat.totalShots'), analysis.stats?.total_shots ?? 0],
            [tr('analysis.stat.avgScore'), (analysis.stats?.avg_score ?? 0).toFixed(1)],
          ].map(([k, v]) => (
            <div key={k as string} className="panel-flat px-3 py-2">
              <div className="text-[10.5px] text-ink-400">{k}</div>
              <div className="tabular mt-0.5 text-[15px] font-semibold text-white">{v}</div>
            </div>
          ))}
        </div>
      )}

      {/* 分析模块 */}
      <SectionTitle>
        <Wand2 size={12} /> {tr('analysis.modulesTitle')}
      </SectionTitle>
      <div data-tour="dlg-modules" className="mb-4 grid gap-1.5 md:grid-cols-2">
        <Toggle
          checked={params.use_audio}
          onChange={(v) => setParams({ use_audio: v })}
          label={tr('analysis.toggle.audio.label')}
          hint={tr('analysis.toggle.audio.hint')}
        />
        <Toggle
          checked={params.use_motion}
          onChange={(v) => setParams({ use_motion: v })}
          label={tr('analysis.toggle.motion.label')}
          hint={tr('analysis.toggle.motion.hint')}
        />
        <Toggle
          checked={params.use_players}
          onChange={(v) => setParams({ use_players: v })}
          label={tr('analysis.toggle.players.label')}
          hint={tr('analysis.toggle.players.hint')}
        />
        <Toggle
          checked={params.use_pose}
          onChange={(v) => setParams({ use_pose: v })}
          label={tr('analysis.toggle.pose.label')}
          hint={tr('analysis.toggle.pose.hint')}
        />
        <Toggle
          checked={params.use_shuttle}
          onChange={(v) => setParams({ use_shuttle: v })}
          label={tr('analysis.toggle.shuttle.label')}
          hint={tr('analysis.toggle.shuttle.hint')}
        />
        <Toggle
          checked={params.use_speech}
          onChange={(v) => setParams({ use_speech: v })}
          label={tr('analysis.toggle.speech.label')}
          hint={tr('analysis.toggle.speech.hint')}
        />
      </div>

      {params.use_speech && (
        <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
          <div className="text-[11.5px] leading-relaxed text-ink-400">
            {tr('analysis.speech.desc')}
          </div>
          {speechInvalid && (
            <div className="mt-2 flex items-start gap-1.5 text-[11px] leading-relaxed text-amber-glow">
              <AlertTriangle size={12} className="mt-[1px] shrink-0" />
              <span>{tr('analysis.speech.needPhrase')}</span>
            </div>
          )}
          <div className="mt-2.5 flex flex-wrap items-start gap-x-4 gap-y-2">
            <div className="flex items-center gap-2">
              {[0, 1].map((i) => (
                <input
                  key={i}
                  className="field w-[110px]"
                  value={params.speech_phrases?.[i] ?? ''}
                  maxLength={3}
                  placeholder={i === 0 ? tr('analysis.speech.phrasePlaceholder1') : tr('analysis.speech.phrasePlaceholder2')}
                  onChange={(e) => {
                    const next = [params.speech_phrases?.[0] ?? '', params.speech_phrases?.[1] ?? '']
                    next[i] = e.target.value.replace(/\s/g, '').slice(0, 3)
                    setParams({ speech_phrases: next })
                  }}
                />
              ))}
            </div>
            <div className="max-w-[300px] flex-1">
              <Slider
                label={tr('analysis.speech.bonusLabel')}
                value={params.speech_bonus_points}
                min={0}
                max={30}
                step={1}
                onChange={(v) => setParams({ speech_bonus_points: v })}
                format={(v) => tr('analysis.speech.bonusFormat', { n: v })}
                hint={tr('analysis.speech.bonusHint')}
              />
            </div>
          </div>
          <div className="mt-2.5 flex flex-wrap items-end gap-x-4 gap-y-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-ink-400">{tr('analysis.speech.modelLabel')}</span>
              <select
                className="field w-[120px]"
                value={params.speech_model}
                onChange={(e) => setParams({ speech_model: e.target.value as AnalysisParams['speech_model'] })}
              >
                {SPEECH_MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <span className="text-[10px] text-ink-500">{tr('analysis.speech.modelHint')}</span>
            </label>
            <Toggle
              checked={params.speech_fuzzy}
              onChange={(v) => setParams({ speech_fuzzy: v })}
              label={tr('analysis.speech.fuzzyLabel')}
              hint={tr('analysis.speech.fuzzyHint')}
            />
          </div>
          {analysis && (
            <SpeechTraceLine
              trace={analysis.stats?.speech_trace}
              speechReady={!!env?.caps?.speech}
            />
          )}
        </div>
      )}

      {params.use_shuttle && (
        <div className="mb-4 flex items-center gap-3 rounded-xl border border-amber-glow/25 bg-amber-glow/[0.07] px-3.5 py-3">
          <AlertTriangle size={14} className="shrink-0 text-amber-glow" />
          <div className="min-w-0 flex-1">
            <div className="text-[11.5px] leading-relaxed text-amber-glow/90">
              {tr('analysis.shuttleBudget.desc')}
            </div>
            <div className="mt-2 max-w-[320px]">
              <Slider
                label={tr('analysis.shuttleBudget.label')}
                value={params.shuttle_budget_seconds === 0 ? 1800 : params.shuttle_budget_seconds}
                min={60}
                max={1800}
                step={30}
                // 后端用 0 表示「不设预算＝全片」；滑到最右要真的发 0，否则 1800 秒
                // 在长视频上不是全片。
                onChange={(v) => setParams({ shuttle_budget_seconds: v >= 1800 ? 0 : v })}
                format={(v) => (v >= 1800 ? tr('analysis.shuttleBudget.full') : tr('analysis.shuttleBudget.minutes', { n: Math.round(v / 60) }))}
              />
            </div>
          </div>
        </div>
      )}

      {/* 机位与场地标定 */}
      <SectionTitle>
        <Video size={12} /> {tr('analysis.viewpoint.title')}
      </SectionTitle>
      <div data-tour="dlg-viewpoint" className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
        <div className="text-[11.5px] leading-relaxed text-ink-400">
          {tr('analysis.viewpoint.desc')}
        </div>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {VIEWPOINTS.map((v) => (
            <button
              key={v.value}
              onClick={() => setParams({ viewpoint: v.value })}
              title={tr(v.hintKey)}
              className={cn(
                'rounded-lg border px-2.5 py-1.5 text-left transition-all',
                params.viewpoint === v.value
                  ? 'border-court-500/50 bg-court-500/12'
                  : 'border-white/8 bg-white/[0.025] hover:border-white/18',
              )}
            >
              <div className={cn('text-[11.5px] font-medium',
                params.viewpoint === v.value ? 'text-court-200' : 'text-ink-100')}>
                {tr(v.labelKey)}
              </div>
              <div className="mt-0.5 text-[10px] text-ink-500">{tr(v.hintKey)}</div>
            </button>
          ))}
        </div>
        <div className="mt-2.5 flex flex-wrap items-center gap-x-4">
          <Toggle
            checked={params.auto_calibrate}
            onChange={(v) => setParams({ auto_calibrate: v })}
            label={tr('analysis.autoCalibrate.label')}
            hint={tr('analysis.autoCalibrate.hint')}
          />
          <Button variant="ghost" size="sm" onClick={() => openCourtEditor(true)}>
            <Crosshair size={13} />
            {manualPoly ? tr('analysis.manualCourt.redo', { n: manualPoly.length }) : tr('analysis.manualCourt.set')}
          </Button>
          {manualPoly && (
            <span className="text-[11px] text-court-300">
              {tr('analysis.manualCourt.inUse')}
              {manualPoly.length > 4 && <> {tr('analysis.manualCourt.polygon', { n: manualPoly.length })}</>}
            </span>
          )}
        </div>
        <div className="mt-2.5 max-w-[360px]">
          <div className="mb-1.5 text-[12px] text-ink-200" title={tr('analysis.segmentMode.hint')}>
            {tr('analysis.segmentMode.label')}
          </div>
          <Segmented
            value={params.segment_mode}
            onChange={(v) => setParams({ segment_mode: v })}
            options={[
              { value: 'auto', label: tr('analysis.segmentMode.player'), hint: tr('analysis.segmentMode.hint') },
              { value: 'activity', label: tr('analysis.segmentMode.activity') },
              { value: 'hybrid', label: tr('analysis.segmentMode.hybrid') },
            ]}
          />
        </div>
      </div>

      {done && analysis?.calibration && (
        <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
            <span className="text-ink-400">{tr('analysis.calibration.viewpointLabel')}</span>
            <span className="font-medium text-court-200">
              {analysis.calibration.viewpoint_label}
            </span>
            {analysis.calibration.ok && (
              <span className="text-ink-500">
                {tr('analysis.calibration.courtInfo', {
                  color: analysis.calibration.court_color,
                  pct: Math.round((analysis.calibration.court_area_ratio ?? 0) * 100),
                })}
                {(analysis.calibration.point_count ?? 4) > 4 &&
                  ` ${tr('analysis.calibration.borderPolygon', { n: analysis.calibration.point_count })}`}
                {analysis.calibration.source === 'manual' && ` ${tr('analysis.calibration.manual')}`}
              </span>
            )}
          </div>
          {(analysis.calibration.notes?.length ?? 0) > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {analysis.calibration.notes.map((n, i) => (
                <li key={i} className="text-[11px] leading-relaxed text-ink-500">
                  · {n}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* 人物框尺寸筛选 */}
      <SectionTitle>
        <Ruler size={12} /> {tr('analysis.sizeFilterTitle')}
      </SectionTitle>
      {/* SizeFilterPanel takes no props, so the tour anchor wraps it here. */}
      <div data-tour="dlg-size-filter">
        <SizeFilterPanel />
      </div>

      {/* 评分口径 */}
      <SectionTitle>
        <Sparkles size={12} /> {tr('analysis.weightTitle')}
      </SectionTitle>
      <div data-tour="dlg-weights" className="mb-4 flex flex-wrap gap-1.5">
        {WEIGHT_PRESETS_MAP.map((w) => (
          <button
            key={w.value}
            onClick={() => useStore.getState().setWeights(w.value)}
            className={cn(
              'rounded-lg border px-3 py-2 text-left transition-all',
              weights === w.value
                ? 'border-court-500/50 bg-court-500/12'
                : 'border-white/8 bg-white/[0.025] hover:border-white/18',
            )}
          >
            <div className={cn('text-[12px] font-medium', weights === w.value ? 'text-court-200' : 'text-ink-100')}>
              {tr(w.labelKey)}
            </div>
            <div className="mt-0.5 text-[10.5px] text-ink-400">{tr(w.hintKey)}</div>
          </button>
        ))}
      </div>

      {/* 关键参数 */}
      <SectionTitle right={
        <button onClick={() => setAdvanced((v) => !v)} className="text-[11px] text-court-300 hover:underline">
          {advanced ? tr('analysis.advanced.collapse') : tr('analysis.advanced.expand')}
        </button>
      }>
        <Activity size={12} /> {tr('analysis.advanced.title')}
      </SectionTitle>
      <div data-tour="dlg-params" className="grid gap-x-5 gap-y-1 md:grid-cols-2">
        <Slider
          label={tr('analysis.slider.gap.label')}
          value={params.gap_seconds}
          min={0.5}
          max={8}
          step={0.1}
          onChange={(v) => setParams({ gap_seconds: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint={tr('analysis.slider.gap.hint')}
        />
        <Slider
          label={tr('analysis.slider.minRally.label')}
          value={params.min_rally_seconds}
          min={0.5}
          max={12}
          step={0.5}
          onChange={(v) => setParams({ min_rally_seconds: v })}
          format={(v) => `${v}s`}
          hint={tr('analysis.slider.minRally.hint')}
        />
        <Slider
          label={tr('analysis.slider.preRoll.label')}
          value={params.pre_roll}
          min={0}
          max={4}
          step={0.1}
          onChange={(v) => setParams({ pre_roll: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint={tr('analysis.slider.preRoll.hint')}
        />
        <Slider
          label={tr('analysis.slider.hitTail.label')}
          value={params.hit_tail_seconds}
          min={0.2}
          max={2.5}
          step={0.1}
          onChange={(v) => setParams({ hit_tail_seconds: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint={tr('analysis.slider.hitTail.hint')}
        />
        <Slider
          label={tr('analysis.slider.postRoll.label')}
          value={params.post_roll}
          min={0}
          max={4}
          step={0.1}
          onChange={(v) => setParams({ post_roll: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint={tr('analysis.slider.postRoll.hint')}
        />

        {advanced && (
          <>
            <Slider
              label={tr('analysis.slider.hitSensitivity.label')}
              value={params.hit_sensitivity}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setParams({ hit_sensitivity: v })}
              format={(v) => v.toFixed(2)}
              hint={tr('analysis.slider.hitSensitivity.hint')}
            />
            <Slider
              label={tr('analysis.slider.poseGate.label')}
              value={params.pose_gate_threshold}
              min={0.05}
              max={0.5}
              step={0.01}
              onChange={(v) => setParams({ pose_gate_threshold: v })}
              format={(v) => v.toFixed(2)}
              hint={tr('analysis.slider.poseGate.hint')}
            />
            <Slider
              label={tr('analysis.slider.poseGateWindow.label')}
              value={params.pose_gate_window}
              min={0.15}
              max={0.6}
              step={0.05}
              onChange={(v) => setParams({ pose_gate_window: v })}
              format={(v) => `${v.toFixed(2)}s`}
              hint={tr('analysis.slider.poseGateWindow.hint')}
            />
            <Slider
              label={tr('analysis.slider.maxRally.label')}
              value={params.max_rally_seconds}
              min={10}
              max={240}
              step={5}
              onChange={(v) => setParams({ max_rally_seconds: v })}
              format={(v) => `${v}s`}
              hint={tr('analysis.slider.maxRally.hint')}
            />
            <Slider
              label={tr('analysis.slider.sampleFps.label')}
              value={params.sample_fps}
              min={5}
              max={15}
              step={1}
              onChange={(v) => setParams({ sample_fps: v })}
              format={(v) => `${v} fps`}
              hint={tr('analysis.slider.sampleFps.hint')}
            />
          </>
        )}
      </div>
    </Modal>
  )
}
