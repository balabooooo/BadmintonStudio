import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Sparkles,
  Star,
  Eye,
  EyeOff,
  Filter,
  FilterX,
  Plus,
  Search,
  TrendingUp,
  Clock,
  Target,
  Zap,
  Scissors,
  Clapperboard,
  ListFilter,
  Check,
  Info,
  Mic,
  Settings2,
} from 'lucide-react'
import { cn, humanDuration, scoreColor, scoreGrade, tagColor } from '../lib/format'
import { Badge, Button, Empty, Segmented, Slider, Toggle, Tooltip, useConfirm } from './ui'
import {
  DEFAULT_FILTER,
  FILM_COVERED_RATIO,
  WEIGHT_PRESETS_MAP as WEIGHT_PRESETS,
  collectRallies,
  filterRallies,
  rallyFilmCoverage,
  useStore,
} from '../store/useStore'
import { useT } from '../i18n/useT'
import { tagLabel } from '../i18n/domain'
import type { AnalysisParams, Rally } from '../lib/types'
import type { RallyRef } from '../store/useStore'

// 与后端 pipeline.SPEECH_MATCH_TOLERANCE 保持一致：喊口令通常发生在回合结束后一点点，
// 所以判定时允许越过回合边界 1 秒。
const SPEECH_MATCH_TOLERANCE = 1.0

interface SpeechEvent {
  t: number
  phrase: string
}

/**
 * 统计本回合命中的口令及各自被喊出的次数。
 *
 * 加分是「每个不同口令只加一次」，但卡片上要展示实际喊了几次，所以次数从
 * stats.speech.events 里按时间窗重新数（事件缺失时退化为 1 次）。
 */
function speechHits(rally: Rally, events: SpeechEvent[] | undefined): { phrase: string; count: number }[] {
  const phrases = rally.features.speech_phrases ?? []
  if (!phrases.length) return []
  const counts = new Map<string, number>()
  for (const e of events ?? []) {
    const t = Number(e?.t)
    const p = String(e?.phrase ?? '')
    if (!p || !Number.isFinite(t)) continue
    if (t < rally.start - SPEECH_MATCH_TOLERANCE || t > rally.end + SPEECH_MATCH_TOLERANCE) continue
    counts.set(p, (counts.get(p) ?? 0) + 1)
  }
  return phrases.map((p) => ({ phrase: p, count: Math.max(1, counts.get(p) ?? 0) }))
}

// 邻场抑制强度 ↔ 门控阈值：滑杆向右 = 更严格（阈值更高，剔除更多邻场击球声）。
// 区间取 [0.08, 0.38]，让默认阈值 0.22 大致落在滑杆中间（适中）。
const GATE_MIN = 0.08
const GATE_MAX = 0.38
function gateStrengthToThreshold(s: number): number {
  return GATE_MAX - (GATE_MAX - GATE_MIN) * Math.max(0, Math.min(1, s))
}
function gateThresholdToStrength(t: number): number {
  return Math.max(0, Math.min(1, (GATE_MAX - t) / (GATE_MAX - GATE_MIN)))
}

function ScoreBadge({ rally, size = 34 }: { rally: Rally; size?: number }) {
  const c = scoreColor(rally.scores.total)
  return (
    <div
      className="grid shrink-0 place-items-center rounded-lg font-bold tabular"
      style={{
        width: size,
        height: size,
        color: c,
        background: `${c.replace('rgb', 'rgba').replace(')', ' / 0.14)')}`,
        boxShadow: `inset 0 0 0 1px ${c.replace('rgb', 'rgba').replace(')', ' / 0.4)')}`,
        fontSize: size * 0.42,
      }}
    >
      {rally.scores.total.toFixed(0)}
    </div>
  )
}

function RallyCard({
  rally,
  selected,
  onSelect,
  mediaName,
}: {
  rally: RallyRef
  selected: boolean
  onSelect: () => void
  mediaName?: string
}) {
  const patchRally = useStore((s) => s.patchRally)
  const addClip = useStore((s) => s.addClipFromRally)
  const tr = useT()
  // 这个回合在成片里已经有片段了吗（按原片时间范围 + 归属素材判断，见 rallyFilmCoverage）
  const inFilm = useStore((s) =>
    s.project
      ? rallyFilmCoverage(
          s.project.timeline.tracks.flatMap((t) => t.clips),
          rally,
          rally.media_id,
        ).ratio >= FILM_COVERED_RATIO
      : false,
  )
  // 语音口令命中：从该回合所属素材的分析结果里取检测事件，数出每个口令被喊了几次。
  const speechEvents = useStore(
    (s) => s.project?.analyses[rally.media_id]?.stats?.speech?.events as SpeechEvent[] | undefined,
  )
  const speech = useMemo(() => speechHits(rally, speechEvents), [rally, speechEvents])
  // 口令本身已经会作为 tag 写进 rally.tags，这里把它从通用标签行里剔掉，避免和口令行重复。
  const speechSet = useMemo(() => new Set(rally.features.speech_phrases ?? []), [rally.features.speech_phrases])
  const shownTags = useMemo(() => rally.tags.filter((t) => !speechSet.has(t)), [rally.tags, speechSet])
  const c = scoreColor(rally.scores.total)

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: rally.keep ? 1 : 0.42, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
      onClick={onSelect}
      className={cn(
        'group relative cursor-pointer rounded-xl border px-2.5 py-2.5 transition-all',
        selected
          ? 'border-court-500/55 bg-court-500/10'
          : 'border-white/7 bg-white/[0.025] hover:border-white/16 hover:bg-white/[0.05]',
      )}
    >
      <div className="flex gap-2.5">
        <ScoreBadge rally={rally} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-[12.5px] font-medium text-white">{tr('rally.index', { index: rally.index })}</span>
            {mediaName && (
              <span
                className="max-w-[92px] truncate rounded bg-white/8 px-1.5 py-[1px] text-[9.5px] font-normal text-ink-300"
                title={mediaName}
              >
                {mediaName}
              </span>
            )}
            <span className="text-[10.5px] font-semibold" style={{ color: c }}>
              {scoreGrade(rally.scores.total)}
            </span>
            {rally.starred && <Star size={11} className="text-amber-glow" fill="currentColor" />}
          </div>
          <div className="mono mt-0.5 flex items-center gap-2 text-[10.5px] text-ink-400">
            <span className="inline-flex items-center gap-1">
              <Clock size={9} /> {rally.duration.toFixed(1)}s
            </span>
            <span className="inline-flex items-center gap-1">
              <Target size={9} /> {tr('rally.shots', { n: rally.features.shot_count })}
            </span>
            {rally.features.tempo > 0 && (
              <span className="inline-flex items-center gap-1">
                <Zap size={9} /> {rally.features.tempo.toFixed(1)}/s
              </span>
            )}
            {inFilm && (
              <span className="inline-flex items-center gap-1 font-medium text-court-300">
                <Clapperboard size={9} /> {tr('rally.inFilm')}
              </span>
            )}
          </div>
          {speech.length > 0 && (
            <Tooltip
              width={300}
              content={tr('rally.speechHitTooltip', { bonus: rally.features.speech_bonus })}
            >
              <div className="mt-1.5 flex flex-wrap items-center gap-1 text-[10px] text-amber-glow">
                <Mic size={10} />
                <span className="font-medium">{tr('rally.speechHit')}</span>
                {speech.map((s) => (
                  <span key={s.phrase} className="rounded bg-amber-glow/15 px-1.5 py-[1px] font-medium">
                    {tr('rally.speechHitPhrase', { phrase: s.phrase, n: s.count })}
                  </span>
                ))}
              </div>
            </Tooltip>
          )}
          {shownTags.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {shownTags.slice(0, 3).map((t) => (
                <Badge key={t} color={tagColor(t)}>
                  {tagLabel(t)}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 悬浮操作 */}
      <div className="absolute top-2 right-2 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
        <Tooltip content={rally.starred ? tr('rally.unstar') : tr('rally.star')}>
          <button
            aria-label={rally.starred ? tr('rally.unstarAria') : tr('rally.starAria')}
            title={rally.starred ? tr('rally.unstarAria') : tr('rally.starAria')}
            onClick={(e) => {
              e.stopPropagation()
              patchRally(rally.id, { starred: !rally.starred })
            }}
            className="grid h-6 w-6 place-items-center rounded-md bg-black/40 text-ink-300 hover:text-amber-glow"
          >
            <Star size={11} fill={rally.starred ? 'currentColor' : 'none'} />
          </button>
        </Tooltip>
        <Tooltip
          width={300}
          content={rally.keep ? tr('rally.keepTooltip') : tr('rally.excludeTooltip')}
        >
          <button
            aria-label={rally.keep ? tr('rally.excludeAria') : tr('rally.keepAria')}
            title={rally.keep ? tr('rally.excludeAria') : tr('rally.keepAria')}
            onClick={(e) => {
              e.stopPropagation()
              patchRally(rally.id, { keep: !rally.keep })
            }}
            className="grid h-6 w-6 place-items-center rounded-md bg-black/40 text-ink-300 hover:text-white"
          >
            {rally.keep ? <Eye size={11} /> : <EyeOff size={11} />}
          </button>
        </Tooltip>
        <Tooltip
          width={300}
          content={inFilm ? tr('rally.locateTooltip') : tr('rally.addToFilmTooltip')}
        >
          <button
            aria-label={inFilm ? tr('rally.locateAria') : tr('rally.addToFilmAria')}
            title={inFilm ? tr('rally.alreadyInFilmTitle') : tr('rally.addToFilmTitle')}
            onClick={(e) => {
              e.stopPropagation()
              // 已经在成片里时 store 会拦住重复加入，并选中那一段（时间线会滚过去）
              addClip(rally)
            }}
            className={cn(
              'grid h-6 w-6 place-items-center rounded-md',
              inFilm
                ? 'bg-court-500/25 text-court-200 hover:bg-court-500/40'
                : 'bg-black/40 text-ink-300 hover:bg-court-500/30 hover:text-court-200',
            )}
          >
            {inFilm ? <Check size={12} /> : <Plus size={11} />}
          </button>
        </Tooltip>
      </div>

      {/* 分项条形 */}
      <div className="mt-2 flex gap-[3px]">
        {(
          [
            ['weight.dim.length', rally.scores.length],
            ['weight.dim.intensity', rally.scores.intensity],
            ['weight.dim.technique', rally.scores.technique],
            ['weight.dim.excitement', rally.scores.excitement],
          ] as const
        ).map(([k, v]) => (
          <Tooltip key={k} content={`${tr(k)} ${v.toFixed(0)}`} className="flex-1">
            <div className="h-[3px] w-full overflow-hidden rounded-full bg-white/8">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${v}%`, background: scoreColor(v) }}
              />
            </div>
          </Tooltip>
        ))}
      </div>
    </motion.div>
  )
}

export default function RallyPanel() {
  const tr = useT()
  const analysis = useStore((s) => s.currentAnalysis())
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const scope = useStore((s) => s.rallyScope)
  const setRallyScope = useStore((s) => s.setRallyScope)
  const previewFiltered = useStore((s) => s.previewFiltered)
  const filter = useStore((s) => s.filter)
  const setFilter = useStore((s) => s.setFilter)
  const weights = useStore((s) => s.weights)
  const params = useStore((s) => s.params)
  const setParams = useStore((s) => s.setParams)
  const resegment = useStore((s) => s.resegment)
  const rebuildHits = useStore((s) => s.rebuildHits)
  const rescore = useStore((s) => s.rescore)
  const rescoreAll = useStore((s) => s.rescoreAll)
  const busy = useStore((s) => s.busy)
  const selectRally = useStore((s) => s.selectRally)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const bulk = useStore((s) => s.bulkRallies)
  const autoCut = useStore((s) => s.autoCut)
  const confirm = useConfirm()
  const [showFilter, setShowFilter] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [sort, setSort] = useState<'index' | 'score' | 'duration' | 'shots'>('index')
  // 邻场抑制滑杆：拖动过程中防抖触发重切分（毫秒级），做到「即时生效」
  const gateTimer = useRef<number | null>(null)
  const applyGate = useCallback(
    (patch: Partial<AnalysisParams>) => {
      setParams(patch)
      if (gateTimer.current) window.clearTimeout(gateTimer.current)
      gateTimer.current = window.setTimeout(() => {
        void resegment(patch)
      }, 220)
    },
    [resegment, setParams],
  )
  const rawCount = analysis?.signals?.hit_times_raw?.length ?? 0
  const keptCount = analysis?.signals?.hit_times?.length ?? 0
  const preGateCount = rawCount || (analysis?.stats?.hit_trace?.count as number | undefined) || 0
  const poseTrace = analysis?.stats?.pose_trace as
    | { gate_skipped?: string; forced?: boolean; keep_ratio?: number; regate?: string }
    | undefined
  const gateActive = analysis?.status === 'done' && preGateCount > 0

  // 范围内全部回合（未过筛选）：scope 感知，供统计、标签、保留前 40% 用
  const all = useMemo(() => collectRallies(project, scope, mediaId), [project, scope, mediaId])
  // 当前范围内出现的标签全集（不受展示条数限制），用于判断已选标签是否仍然适用。
  const tagCounts = useMemo(() => {
    const s = new Map<string, number>()
    all.forEach((r) => r.tags.forEach((t) => s.set(t, (s.get(t) ?? 0) + 1)))
    return s
  }, [all])
  // 切换素材 / 范围后，若已选标签在新范围里不存在，筛选会一直命中 0 个回合。
  // 这里自动剔除缺失的标签，保留分数、时长等在跨素材时仍适用的条件。
  useEffect(() => {
    if (!all.length || !filter.tags.length) return
    const next = filter.tags.filter((t) => tagCounts.has(t))
    if (next.length !== filter.tags.length) setFilter({ tags: next })
  }, [all, filter.tags, tagCounts, setFilter])
  const visible = useMemo(() => {
    const l = filterRallies(all, filter)
    if (sort === 'score') l.sort((a, b) => b.scores.total - a.scores.total)
    else if (sort === 'duration') l.sort((a, b) => b.duration - a.duration)
    else if (sort === 'shots') l.sort((a, b) => b.features.shot_count - a.features.shot_count)
    // 全部素材时不再按 index 排：不同素材的 #1、#2 会互相穿插。
    // collectRallies 已经按「素材顺序 → 回合顺序」排好，保持分组更适合挑回合。
    else if (scope !== 'all') l.sort((a, b) => a.index - b.index)
    return l
  }, [all, filter, sort, scope])

  const allTags = useMemo(() => {
    const top = [...tagCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)
    // 已选但没排进前 12 的标签也要显示，否则用户看不到、也点不掉当前筛选。
    const shown = new Set(top.map(([t]) => t))
    const extra = filter.tags
      .filter((t) => tagCounts.has(t) && !shown.has(t))
      .map((t) => [t, tagCounts.get(t) ?? 0] as [string, number])
    return [...top, ...extra]
  }, [tagCounts, filter.tags])

  // 是否有任何筛选条件生效：用于高亮「清空筛选」按钮。
  const filterActive = useMemo(
    () =>
      filter.minScore !== DEFAULT_FILTER.minScore ||
      filter.maxScore !== DEFAULT_FILTER.maxScore ||
      filter.minDuration !== DEFAULT_FILTER.minDuration ||
      filter.maxDuration !== DEFAULT_FILTER.maxDuration ||
      filter.minShots !== DEFAULT_FILTER.minShots ||
      filter.minConfidence !== DEFAULT_FILTER.minConfidence ||
      filter.starredOnly ||
      filter.keepOnly ||
      filter.tags.length > 0 ||
      filter.search.trim() !== '',
    [filter],
  )

  const analyzedCount = useMemo(
    () => (project ? project.media.filter((m) => project.analyses[m.id]?.status === 'done').length : 0),
    [project],
  )

  // 某个评分口径是否已经对所有已分析素材统一重算过（后端 stats.weights = cross:<口径>）
  const isApplied = useCallback(
    (w: string) => {
      if (!project) return false
      const mids = project.media
        .filter((m) => project.analyses[m.id]?.status === 'done')
        .map((m) => m.id)
      if (!mids.length) return false
      const expect = `cross:${w}`
      return mids.every((mid) => project.analyses[mid]?.stats?.weights === expect)
    },
    [project],
  )
  const unifiedApplied = isApplied(weights)
  const activePreset = WEIGHT_PRESETS.find((w) => w.value === weights)

  if (!all.length && analysis?.status !== 'done') {
    return (
      <Empty
        icon={<Sparkles size={30} />}
        title={tr('rally.emptyTitle')}
        desc={tr('rally.emptyDesc')}
      />
    )
  }

  const totalSeconds = visible.reduce((a, r) => a + r.duration, 0)

  // 成片现状：给「筛选 → 成片 → 导出」这条链路一个可见的数字，
  // 否则很容易以为「筛选出来就已经是成片了」。
  const filmClips = project?.timeline.tracks.flatMap((t) => t.clips) ?? []
  const filmSeconds = filmClips.reduce((a, c) => a + (c.src_out - c.src_in) / c.speed, 0)
  // 只数「范围里的回合」有几个进了成片：按原片时间范围 + 归属素材判断，
  // 这样重新分析后旧片段还挂着老 id 也能认出来，顺带避免分子超过分母。
  const filmRallies = all.reduce(
    (n, r) => n + (rallyFilmCoverage(filmClips, r, r.media_id).ratio >= FILM_COVERED_RATIO ? 1 : 0),
    0,
  )

  return (
    <div className="flex h-full flex-col">
      {/* 固定顶栏：搜索 + 范围/排序 + 统计 + 当前评分口径。
          挑回合是主战场，控制项尽量少，其余收进下方折叠区。 */}
      <div className="shrink-0 border-b border-white/7 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <div className="relative min-w-0 flex-1">
            <Search size={13} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-500" />
            <input
              value={filter.search}
              onChange={(e) => setFilter({ search: e.target.value })}
              placeholder={tr('rally.searchPlaceholder')}
              className="field py-1.5 pl-7 text-[12px]"
            />
          </div>
          <Tooltip content={tr('rally.filterTooltip')}>
            <Button
              variant={showFilter ? 'outline' : 'ghost'}
              size="icon"
              onClick={() => setShowFilter((v) => !v)}
            >
              <Filter size={13} />
            </Button>
          </Tooltip>
          <Button
            variant={filterActive ? 'outline' : 'ghost'}
            size="sm"
            className={cn('shrink-0', filterActive && 'text-court-300')}
            disabled={!filterActive}
            onClick={() => setFilter(DEFAULT_FILTER)}
          >
            <FilterX size={13} />
            {tr('rally.clearFilter')}
          </Button>
        </div>

        <div className="mt-2 flex items-center gap-2">
          {analyzedCount > 1 && (
            <Segmented
              size="sm"
              className="flex-1 [&>button]:flex-1"
              value={scope}
              onChange={setRallyScope}
              options={[
                { value: 'current', label: tr('rally.scopeCurrent') },
                { value: 'all', label: tr('rally.scopeAll', { n: analyzedCount }) },
              ]}
            />
          )}
          <Segmented
            size="sm"
            className={cn('shrink-0', analyzedCount <= 1 && 'flex-1 [&>button]:flex-1')}
            value={sort}
            onChange={setSort}
            options={[
              { value: 'index', label: tr('rally.sortIndex') },
              { value: 'score', label: tr('rally.sortScore') },
              { value: 'duration', label: tr('rally.sortDuration') },
              { value: 'shots', label: tr('rally.sortShots') },
            ]}
          />
        </div>

        <div className="mt-2 flex items-center gap-2 text-[10.5px] text-ink-500">
          <span>
            {tr('rally.matchLabel')} <span className="mono font-semibold text-court-300">{visible.length}</span>{' '}
            {tr('rally.matchTotal', { n: all.length })}
          </span>
          <span className="text-ink-600">·</span>
          <span>{tr('rally.totalPlayingTime', { time: humanDuration(totalSeconds) })}</span>
          <Tooltip
            width={300}
            content={
              <span className="whitespace-pre-line">{tr('rally.gradeHelp')}</span>
            }
          >
            <span className="cursor-help underline decoration-dotted underline-offset-2">
              {tr('rally.gradeTitle')}
            </span>
          </Tooltip>
          <span className="flex-1" />
          <Tooltip content={tr('rally.scoreBy')} width={220}>
            <span className="cursor-help rounded-md bg-court-500/15 px-1.5 py-[1px] text-[10.5px] text-court-300">
              {tr(activePreset?.labelKey ?? 'weight.balanced.label')}
            </span>
          </Tooltip>
        </div>
      </div>

      {/* 筛选面板：默认折叠，展开时限制高度自行滚动 */}
      <AnimatePresence initial={false}>
        {showFilter && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="shrink-0 overflow-hidden border-b border-white/7"
          >
            <div className="max-h-[42vh] overflow-y-auto px-3 py-2.5">
              <div className="space-y-2.5 rounded-xl border border-white/7 bg-white/[0.025] px-3 py-2.5">
                <Slider
                  label={tr('rally.minScore')}
                  value={filter.minScore}
                  min={0}
                  max={100}
                  step={1}
                  onChange={(v) => setFilter({ minScore: v })}
                  format={(v) => tr('rally.minScoreFormat', { v })}
                />
                <Slider
                  label={tr('rally.minDuration')}
                  value={filter.minDuration}
                  min={0}
                  max={60}
                  step={0.5}
                  onChange={(v) => setFilter({ minDuration: v })}
                  format={(v) => `${v}s`}
                />
                <Slider
                  label={tr('rally.minShots')}
                  value={filter.minShots}
                  min={0}
                  max={40}
                  step={1}
                  onChange={(v) => setFilter({ minShots: v })}
                  format={(v) => tr('rally.shots', { n: v })}
                />
                <Slider
                  label={tr('rally.minConfidence')}
                  value={filter.minConfidence}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => setFilter({ minConfidence: v })}
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                  hint={tr('rally.confidenceHint')}
                />
                <div className="flex flex-wrap gap-1.5 pt-0.5">
                  <button
                    onClick={() => setFilter({ starredOnly: !filter.starredOnly })}
                    className={cn(
                      'rounded-md px-2 py-1 text-[11px] transition-colors',
                      filter.starredOnly ? 'bg-amber-glow/20 text-amber-glow' : 'bg-white/6 text-ink-300 hover:bg-white/12',
                    )}
                  >
                    ★ {tr('rally.onlyStarred')}
                  </button>
                  <button
                    onClick={() => setFilter({ keepOnly: !filter.keepOnly })}
                    className={cn(
                      'rounded-md px-2 py-1 text-[11px] transition-colors',
                      filter.keepOnly ? 'bg-court-500/20 text-court-300' : 'bg-white/6 text-ink-300 hover:bg-white/12',
                    )}
                  >
                    {tr('rally.keepOnly')}
                  </button>
                  {allTags.map(([t, n]) => {
                    const on = filter.tags.includes(t)
                    return (
                      <button
                        key={t}
                        onClick={() =>
                          setFilter({ tags: on ? filter.tags.filter((x) => x !== t) : [...filter.tags, t] })
                        }
                        className={cn(
                          'rounded-md px-2 py-1 text-[11px] transition-all',
                          on && 'font-semibold',
                        )}
                        style={{
                          background: on ? tagColor(t) : 'rgb(255 255 255 / 0.06)',
                          color: on ? 'var(--color-ink-950)' : 'var(--color-ink-300)',
                          boxShadow: on ? `0 0 0 2px ${tagColor(t)}66` : undefined,
                        }}
                      >
                        {tagLabel(t)} <span className="opacity-60">{n}</span>
                      </button>
                    )
                  })}
                </div>

                {/* 一键预设 */}
                <div className="flex flex-wrap gap-1.5 border-t border-white/7 pt-2.5">
                  <span className="self-center text-[10.5px] tracking-wide text-ink-500">{tr('rally.quickFilters')}</span>
                  {[
                    { label: tr('rally.presetTop20'), patch: { minScore: 70, minShots: 0, minDuration: 0, starredOnly: false, tags: [] } },
                    { label: tr('rally.presetManyShots'), patch: { minShots: 10, minScore: 0, minDuration: 0, tags: [] } },
                    { label: tr('rally.presetLong'), patch: { minDuration: 15, minScore: 0, minShots: 0, tags: [] } },
                    { label: tr('rally.presetAll'), patch: { ...DEFAULT_FILTER } },
                  ].map((p) => (
                    <button
                      key={p.label}
                      onClick={() => setFilter(p.patch as never)}
                      className="rounded-md bg-white/6 px-2 py-1 text-[11px] text-ink-200 transition-colors hover:bg-court-500/20 hover:text-court-200"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 列表：主区域 */}
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2.5">
        <AnimatePresence initial={false}>
          {visible.map((r) => (
            <RallyCard
              key={r.id}
              rally={r}
              selected={r.id === selectedRallyId}
              onSelect={() => selectRally(r.id)}
              mediaName={scope === 'all' ? r.media_name : undefined}
            />
          ))}
        </AnimatePresence>
        {!visible.length && (
          <div className="flex flex-col items-center gap-2.5 py-10 text-center text-[12px] text-ink-500">
            <span>{tr('rally.noMatches')}</span>
            {filterActive && (
              <Button variant="outline" size="sm" onClick={() => setFilter(DEFAULT_FILTER)}>
                <FilterX size={12} />
                {tr('rally.clearFilter')}
              </Button>
            )}
          </div>
        )}
      </div>

      {/* 底部：成片动作 + 批量操作，主 CTA 始终可见 */}
      <div className="shrink-0 border-t border-white/7 px-3 py-2">
        {/* 筛选 ≠ 成片：把这条链路直接摊开，
            否则很容易以为「筛出来的就是成片了，为什么还要点加入」。 */}
        <div className="flex items-start gap-2 rounded-xl border border-flux-400/20 bg-flux-400/[0.05] px-2.5 py-2">
          <Clapperboard size={12} className="mt-[1px] shrink-0 text-flux-400" />
          <div className="min-w-0 flex-1 text-[10.5px] leading-relaxed text-ink-300">
            <span>
              {tr('rally.filmLabel')}
              <b className="text-court-300">{filmRallies}</b>
              {tr('rally.filmRalliesOf', { total: all.length })}{' '}
              <b className="mono text-ink-200">{humanDuration(filmSeconds)}</b>
            </span>
            <div className="text-ink-500">{tr('rally.filmNote')}</div>
          </div>
        </div>

        <div className="mt-2 flex gap-2">
          <Tooltip
            width={300}
            content={tr('rally.rebuildTooltip', { n: visible.length })}
          >
            <Button
              variant="primary"
              size="sm"
              className="flex-1"
              onClick={() => autoCut({ mode: 'replace' })}
              disabled={!visible.length}
            >
              <Sparkles size={12} />
              {tr('rally.rebuildButton', { n: visible.length })}
            </Button>
          </Tooltip>
          <Tooltip
            width={290}
            content={tr('rally.appendTooltip')}
          >
            <Button variant="outline" size="sm" onClick={() => autoCut({ mode: 'append' })} disabled={!visible.length}>
              <Plus size={12} />
            </Button>
          </Tooltip>
          <Tooltip
            width={290}
            content={tr('rally.previewFilteredTooltip')}
          >
            <Button
              variant={previewFiltered ? 'outline' : 'ghost'}
              size="sm"
              className={previewFiltered ? 'text-court-300' : ''}
              disabled={!visible.length}
              onClick={() => {
                const s = useStore.getState()
                if (s.previewFiltered) s.setPreviewFiltered(false)
                else s.previewFilteredRallies()
              }}
            >
              <ListFilter size={12} />
            </Button>
          </Tooltip>
        </div>

        <div className="mt-2 flex items-center gap-2 border-t border-white/7 pt-2">
          <Tooltip content={tr('rally.bulkKeepTooltip')}>
            <Button
              variant="ghost"
              size="sm"
              disabled={!visible.length}
              onClick={() => bulk({ keep: true }, { useFilter: true })}
            >
              <Eye size={12} /> {tr('rally.bulkKeep')}
            </Button>
          </Tooltip>
          <Tooltip content={tr('rally.bulkExcludeTooltip')}>
            <Button
              variant="ghost"
              size="sm"
              disabled={!visible.length}
              onClick={() => bulk({ keep: false }, { useFilter: true })}
            >
              <EyeOff size={12} /> {tr('rally.bulkExclude')}
            </Button>
          </Tooltip>
          <div className="flex-1" />
          <Tooltip content={tr('rally.top40Tooltip')}>
            <Button
              variant="ghost"
              size="sm"
              disabled={!all.length}
              onClick={async () => {
                const sorted = [...all].sort((a, b) => b.scores.total - a.scores.total)
                const keepN = Math.ceil(sorted.length * 0.4)
                const ok = await confirm({
                  title: tr('rally.top40ConfirmTitle', { n: keepN, total: all.length }),
                  desc: tr('rally.top40ConfirmDesc', { n: sorted.length - keepN }),
                  danger: true,
                })
                if (!ok) return
                const ids = sorted.slice(0, keepN).map((r) => r.id)
                // 两次批量写必须串行：并发调用各自会 openProject 刷新，可能互相
                // 覆盖，导致其中一半更新丢失。
                await bulk({ keep: true }, { ids })
                const rest = sorted.slice(keepN).map((r) => r.id)
                if (rest.length) await bulk({ keep: false }, { ids: rest })
              }}
            >
              <TrendingUp size={12} /> {tr('rally.top40')}
            </Button>
          </Tooltip>
        </div>
      </div>

      {/* 分析设置：评分口径 + 切分调参 + 邻场抑制（默认折叠） */}
      <div className="shrink-0 border-t border-white/7 px-3 py-2">
        <button
          onClick={() => setShowSettings((v) => !v)}
          className="flex w-full items-center justify-between text-[11px] text-ink-200"
        >
          <span className="flex items-center gap-1.5">
            <Settings2 size={11} className="text-court-300" />
            {tr('rally.settings')}
          </span>
          <span className="text-ink-500">{showSettings ? tr('rally.collapse') : tr('rally.expand')}</span>
        </button>
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="max-h-[46vh] space-y-3 overflow-y-auto pt-3">
                {scope === 'all' && (
                  <div
                    className={cn(
                      'flex items-start gap-2 rounded-xl border px-2.5 py-2',
                      unifiedApplied ? 'border-court-500/25 bg-court-500/[0.06]' : 'border-amber-glow/20 bg-amber-glow/[0.05]',
                    )}
                  >
                    <Info size={12} className={cn('mt-[1px] shrink-0', unifiedApplied ? 'text-court-300' : 'text-amber-glow')} />
                    <div className="min-w-0 flex-1 text-[10.5px] leading-relaxed text-ink-300">
                      <span>
                        {unifiedApplied ? tr('rally.unifiedDone') : tr('rally.unifiedHint')}
                      </span>
                      <button
                        onClick={() => rescoreAll(weights)}
                        disabled={busy.rescore || unifiedApplied}
                        className={cn(
                          'mt-1 block rounded-md px-2 py-1 text-[10.5px] transition-colors',
                          unifiedApplied
                            ? 'cursor-default bg-court-500/15 text-court-300'
                            : 'bg-amber-glow/15 text-amber-glow hover:bg-amber-glow/25 disabled:opacity-50',
                        )}
                      >
                        {busy.rescore
                          ? tr('rally.rescoring')
                          : unifiedApplied
                            ? tr('rally.unifiedApplied')
                            : tr('rally.unifiedButton')}
                      </button>
                    </div>
                  </div>
                )}

                <div className="border-t border-white/7 pt-2.5">
                  <div className="mb-1.5 flex items-center gap-1.5">
                    <span className="text-[10.5px] tracking-wide text-ink-500">{tr('rally.scoreBy')}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {WEIGHT_PRESETS.map((w) => (
                      <Tooltip
                        key={w.value}
                        width={330}
                        content={
                          <div>
                            <div className="mb-1 font-medium text-court-300">{tr(w.labelKey)}</div>
                            <div className="text-ink-400">{tr(w.hintKey)}</div>
                            <div className="mt-2 text-ink-300">{tr('rally.weightHow')}</div>
                            <div className="mt-0.5 text-ink-300">{tr(w.detail.summaryKey)}</div>
                            <div className="mt-2 text-ink-300">{tr('rally.weightDims')}</div>
                            <div className="mt-1 space-y-1">
                              {w.detail.weights.map(([name, pct]) => (
                                <div key={name} className="flex items-center gap-2">
                                  <span className="w-8 shrink-0 text-[10.5px] text-ink-400">{tr(name)}</span>
                                  <span className="h-[4px] flex-1 overflow-hidden rounded-full bg-white/10">
                                    <span
                                      className="block h-full rounded-full bg-court-400"
                                      style={{ width: `${(pct / 42) * 100}%` }}
                                    />
                                  </span>
                                  <span className="mono w-8 shrink-0 text-right text-[10px] text-ink-300">{pct}%</span>
                                </div>
                              ))}
                            </div>
                            <div className="mt-2 border-t border-white/10 pt-1.5 text-[10.5px] leading-relaxed text-ink-500">
                              {tr('rally.weightFormula')}
                            </div>
                          </div>
                        }
                      >
                        <button
                          onClick={() => (scope === 'all' ? rescoreAll(w.value) : rescore(w.value))}
                          disabled={
                            busy.rescore ||
                            (scope === 'all' ? isApplied(w.value) : analysis?.stats?.weights === w.value)
                          }
                          className={cn(
                            'rounded-md px-1.5 py-[3px] text-[10.5px] whitespace-nowrap transition-colors',
                            'disabled:cursor-not-allowed disabled:opacity-40',
                            weights === w.value ? 'bg-court-500/20 text-court-300' : 'bg-white/5 text-ink-400 hover:bg-white/10',
                          )}
                        >
                          {tr(w.labelKey)}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                  <div className="mt-1.5 text-[10px] leading-relaxed text-ink-500">
                    {tr('rally.weightFooter')}
                  </div>
                </div>

                {/* 切分调参：只重跑切分，毫秒级 */}
                <div className="border-t border-white/7 pt-2.5">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-ink-200">
                    <Scissors size={11} className="text-court-300" />
                    {tr('rally.splitTitle')}
                    <span className="text-ink-500">{tr('rally.splitInstant')}</span>
                  </div>
                  <div className="space-y-2.5">
                    <Slider
                      label={tr('rally.splitGranularity')}
                      value={params.split_sensitivity}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={(v) => setParams({ split_sensitivity: v })}
                      format={(v) => (v < 0.3 ? tr('rally.granCoarse') : v > 0.7 ? tr('rally.granFine') : tr('rally.granMedium'))}
                      hint={tr('rally.granHint')}
                    />
                    <Slider
                      label={tr('rally.gapLabel')}
                      value={params.gap_seconds}
                      min={0.5}
                      max={8}
                      step={0.1}
                      onChange={(v) => setParams({ gap_seconds: v })}
                      format={(v) => `${v.toFixed(1)}s`}
                      hint={tr('rally.gapHint')}
                    />
                    <Slider
                      label={tr('rally.minRally')}
                      value={params.min_rally_seconds}
                      min={0.5}
                      max={12}
                      step={0.5}
                      onChange={(v) => setParams({ min_rally_seconds: v })}
                      format={(v) => `${v}s`}
                      hint={tr('rally.minRallyHint')}
                    />

                    {/* 邻场击球声抑制：门控阈值即时重算 */}
                    <div className="space-y-2.5 rounded-xl border border-white/7 bg-black/20 px-3 py-2.5">
                      <div className="flex items-center gap-1.5 text-[11px] text-ink-200">
                        <Target size={11} className="text-court-300" />
                        {tr('rally.gateTitle')}
                        <span className="text-ink-500">{tr('rally.splitInstant')}</span>
                      </div>
                      {/* 击球声总开关：关闭后走纯视觉切分（球员运动 + 画面运动），秒级重切可做 A/B */}
                      <Toggle
                        checked={params.use_audio}
                        onChange={(v) => applyGate({ use_audio: v })}
                        label={tr('rally.audioToggle')}
                        hint={tr('rally.audioToggleHint')}
                      />
                      {params.use_audio ? (
                        <>
                          <Slider
                            label={tr('rally.gateThreshold')}
                            value={gateThresholdToStrength(params.pose_gate_threshold)}
                            min={0}
                            max={1}
                            step={0.05}
                            onChange={(v) => applyGate({ pose_gate_threshold: gateStrengthToThreshold(v) })}
                            format={(v) =>
                              v < 0.34
                                ? tr('rally.gateLenient')
                                : v > 0.66
                                  ? tr('rally.gateStrict')
                                  : tr('rally.gateMedium')
                            }
                            hint={tr('rally.gateThresholdHint')}
                          />
                          <Toggle
                            checked={params.pose_gate_force}
                            onChange={(v) => applyGate({ pose_gate_force: v })}
                            label={tr('rally.gateForce')}
                            hint={tr('rally.gateForceHint')}
                          />
                          {gateActive ? (
                            <div className="text-[10px] leading-relaxed text-ink-500">
                              {tr('rally.gateStats', {
                                raw: preGateCount,
                                kept: keptCount,
                                dropped: Math.max(0, preGateCount - keptCount),
                              })}
                              {poseTrace?.gate_skipped ? (
                                <span className="text-amber-glow">
                                  {' '}· {tr('rally.gateSkipped', { reason: poseTrace.gate_skipped })}
                                </span>
                              ) : null}
                              {poseTrace?.forced ? (
                                <span className="text-amber-glow"> · {tr('rally.gateForced')}</span>
                              ) : null}
                            </div>
                          ) : (
                            <div className="flex items-center justify-between gap-2 text-[10px] leading-relaxed text-ink-500">
                              <span>{tr('rally.gateNeedRebuild')}</span>
                              <Button
                                variant="outline"
                                size="sm"
                                loading={busy.resegment}
                                onClick={() => rebuildHits()}
                              >
                                {tr('rally.gateRebuild')}
                              </Button>
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="text-[10px] leading-relaxed text-ink-500">
                          {tr('rally.audioOff')}
                        </div>
                      )}
                    </div>

                    <div className="flex gap-2">
                      <Button
                        variant="subtle"
                        size="sm"
                        className="flex-1"
                        loading={busy.resegment}
                        onClick={() => resegment()}
                      >
                        {tr('rally.applyResegment')}
                      </Button>
                      <Tooltip content={tr('rally.finerTooltip')}>
                        <Button
                          variant="outline"
                          size="sm"
                          loading={busy.resegment}
                          onClick={() => resegment({ split_sensitivity: Math.min(1, params.split_sensitivity + 0.2) })}
                        >
                          {tr('rally.finer')}
                        </Button>
                      </Tooltip>
                    </div>
                    <div className="text-[10px] leading-relaxed text-ink-500">
                      {tr('rally.splitSummary', {
                        n: all.length,
                        avg: all.length ? (all.reduce((a, r) => a + r.duration, 0) / all.length).toFixed(1) : 0,
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
