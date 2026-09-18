import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState } from 'react'
import {
  Sparkles,
  Star,
  Eye,
  EyeOff,
  Filter,
  Plus,
  Search,
  Sliders,
  TrendingUp,
  Clock,
  Target,
  Zap,
  Scissors,
  Clapperboard,
  ListFilter,
  Check,
} from 'lucide-react'
import { cn, humanDuration, scoreColor, scoreGrade, tagColor } from '../lib/format'
import { Badge, Button, Empty, Segmented, Slider, Tooltip } from './ui'
import {
  DEFAULT_FILTER,
  FILM_COVERED_RATIO,
  WEIGHT_PRESETS_MAP as WEIGHT_PRESETS,
  filterRallies,
  rallyFilmCoverage,
  useStore,
} from '../store/useStore'
import type { Rally } from '../lib/types'

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

function RallyCard({ rally, selected, onSelect }: { rally: Rally; selected: boolean; onSelect: () => void }) {
  const patchRally = useStore((s) => s.patchRally)
  const addClip = useStore((s) => s.addClipFromRally)
  // 这个回合在成片里已经有片段了吗（按原片时间范围判断，见 rallyFilmCoverage）
  const inFilm = useStore((s) =>
    s.project
      ? rallyFilmCoverage(
          s.project.timeline.tracks.flatMap((t) => t.clips),
          rally,
        ).ratio >= FILM_COVERED_RATIO
      : false,
  )
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
            <span className="text-[12.5px] font-medium text-white">回合 #{rally.index}</span>
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
              <Target size={9} /> {rally.features.shot_count} 拍
            </span>
            {rally.features.tempo > 0 && (
              <span className="inline-flex items-center gap-1">
                <Zap size={9} /> {rally.features.tempo.toFixed(1)}/s
              </span>
            )}
            {inFilm && (
              <span className="inline-flex items-center gap-1 font-medium text-court-300">
                <Clapperboard size={9} /> 已成片
              </span>
            )}
          </div>
          {rally.tags.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {rally.tags.slice(0, 3).map((t) => (
                <Badge key={t} color={tagColor(t)}>
                  {t}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 悬浮操作 */}
      <div className="absolute top-2 right-2 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
        <Tooltip content={rally.starred ? '取消标记' : '标记重点'}>
          <button
            aria-label={rally.starred ? '取消标记重点' : '标记重点'}
            title={rally.starred ? '取消标记重点' : '标记重点'}
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
          content={
            rally.keep
              ? '排除此回合：它会从筛选结果里消失、也不会被「按筛选成片」选中。\n这一步只改候选，不动已经排好的成片。'
              : '保留此回合：让它重新进入筛选候选。\n这一步只改候选，不动已经排好的成片。'
          }
        >
          <button
            aria-label={rally.keep ? '排除此回合' : '保留此回合'}
            title={rally.keep ? '排除此回合' : '保留此回合'}
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
          content={
            inFilm
              ? '这个回合已经在成片里了（每个回合只加入一次）。\n点一下会跳到成片里的那一段，想改长度就拖它的两端。'
              : '加入成片：把这个回合剪进下面的成片轨，导出时按顺序输出。\n只想看看、不动成片的话，单击这张卡片即可。'
          }
        >
          <button
            aria-label={inFilm ? '已在成片里，点击定位' : '加入成片'}
            title={inFilm ? '已在成片里（每个回合只加入一次）' : '加入成片'}
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
            ['长度', rally.scores.length],
            ['强度', rally.scores.intensity],
            ['技术', rally.scores.technique],
            ['精彩', rally.scores.excitement],
          ] as const
        ).map(([k, v]) => (
          <Tooltip key={k} content={`${k} ${v.toFixed(0)}`} className="flex-1">
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
  const analysis = useStore((s) => s.currentAnalysis())
  const project = useStore((s) => s.project)
  const previewFiltered = useStore((s) => s.previewFiltered)
  const filter = useStore((s) => s.filter)
  const setFilter = useStore((s) => s.setFilter)
  const weights = useStore((s) => s.weights)
  const params = useStore((s) => s.params)
  const setParams = useStore((s) => s.setParams)
  const resegment = useStore((s) => s.resegment)
  const busy = useStore((s) => s.busy)
  const selectRally = useStore((s) => s.selectRally)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const bulk = useStore((s) => s.bulkRallies)
  const autoCut = useStore((s) => s.autoCut)
  const [showFilter, setShowFilter] = useState(true)
  const [showSplit, setShowSplit] = useState(false)
  const [sort, setSort] = useState<'index' | 'score' | 'duration' | 'shots'>('index')

  const all = analysis?.rallies ?? []
  const visible = useMemo(() => {
    const l = filterRallies(all, filter)
    if (sort === 'score') l.sort((a, b) => b.scores.total - a.scores.total)
    else if (sort === 'duration') l.sort((a, b) => b.duration - a.duration)
    else if (sort === 'shots') l.sort((a, b) => b.features.shot_count - a.features.shot_count)
    else l.sort((a, b) => a.index - b.index)
    return l
  }, [all, filter, sort])

  const allTags = useMemo(() => {
    const s = new Map<string, number>()
    all.forEach((r) => r.tags.forEach((t) => s.set(t, (s.get(t) ?? 0) + 1)))
    return [...s.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)
  }, [all])

  if (!analysis || analysis.status !== 'done') {
    return (
      <Empty
        icon={<Sparkles size={30} />}
        title="还没有分析结果"
        desc="点击「AI 分析」让引擎自动剔除捡球、走动等无效片段，切出每个回合并评分。"
      />
    )
  }

  const totalSeconds = visible.reduce((a, r) => a + r.duration, 0)

  // 成片现状：给「筛选 → 成片 → 导出」这条链路一个可见的数字，
  // 否则很容易以为「筛选出来就已经是成片了」。
  const filmClips = project?.timeline.tracks.flatMap((t) => t.clips) ?? []
  const filmSeconds = filmClips.reduce((a, c) => a + (c.src_out - c.src_in) / c.speed, 0)
  // 只数「这次分析里的回合」有几个进了成片：按原片时间范围判断，
  // 这样重新分析后旧片段还挂着老 id 也能认出来，顺带避免分子超过分母。
  const filmRallies = all.reduce(
    (n, r) => n + (rallyFilmCoverage(filmClips, r).ratio >= FILM_COVERED_RATIO ? 1 : 0),
    0,
  )

  return (
    <div className="flex h-full flex-col">
      {/*
        筛选栏。它内容很长（滑杆 + 标签 + 预设 + 口径），必须限制高度并自己滚动，
        否则会把下面的回合列表挤成 0 高——列表才是挑回合的主战场。
      */}
      <div className="min-h-0 max-h-[46%] shrink overflow-y-auto border-b border-white/7 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <div className="relative min-w-0 flex-1">
            <Search size={13} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-500" />
            <input
              value={filter.search}
              onChange={(e) => setFilter({ search: e.target.value })}
              placeholder="搜索回合 / 标签…"
              className="field py-1.5 pl-7 text-[12px]"
            />
          </div>
          <Tooltip content="筛选条件">
            <Button
              variant={showFilter ? 'outline' : 'ghost'}
              size="icon"
              onClick={() => setShowFilter((v) => !v)}
            >
              <Filter size={13} />
            </Button>
          </Tooltip>
          <Tooltip content="重置筛选">
            <Button variant="ghost" size="icon" onClick={() => setFilter(DEFAULT_FILTER)}>
              <Sliders size={13} />
            </Button>
          </Tooltip>
        </div>

        <div className="mt-2 flex items-center justify-between gap-2">
          <div className="text-[11px] text-ink-400">
            匹配 <span className="mono font-semibold text-court-300">{visible.length}</span> / {all.length} 个回合
          </div>
          <Segmented
            size="sm"
            value={sort}
            onChange={setSort}
            options={[
              { value: 'index', label: '顺序' },
              { value: 'score', label: '评分' },
              { value: 'duration', label: '时长' },
              { value: 'shots', label: '拍数' },
            ]}
          />
        </div>
        <div className="mt-1.5 flex items-center gap-2 text-[10.5px] text-ink-500">
          <span>
            共 <b className="text-court-300">{humanDuration(totalSeconds)}</b> 打球时间
          </span>
          <Tooltip
            width={300}
            content={
              <span>
                等级是按总分划的：{'\n'}
                <b>S</b> ≥85 精彩　<b>A</b> ≥72 很好　<b>B</b> ≥58 不错{'\n'}
                <b>C</b> ≥42 一般　<b>D</b> &lt;42 建议剔除{'\n\n'}
                条数多的时候大部分会落在 C/B，属于正常分布。
              </span>
            }
          >
            <span className="cursor-help underline decoration-dotted underline-offset-2">
              S/A/B/C/D 是什么 ⓘ
            </span>
          </Tooltip>
        </div>

        <AnimatePresence>
          {showFilter && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className="overflow-hidden"
            >
              <div className="mt-3 space-y-2.5 rounded-xl border border-white/7 bg-white/[0.025] px-3 py-2.5">
                <Slider
                  label="最低评分"
                  value={filter.minScore}
                  min={0}
                  max={100}
                  step={1}
                  onChange={(v) => setFilter({ minScore: v })}
                  format={(v) => `${v} 分`}
                />
                <Slider
                  label="最短时长"
                  value={filter.minDuration}
                  min={0}
                  max={60}
                  step={0.5}
                  onChange={(v) => setFilter({ minDuration: v })}
                  format={(v) => `${v}s`}
                />
                <Slider
                  label="最少拍数"
                  value={filter.minShots}
                  min={0}
                  max={40}
                  step={1}
                  onChange={(v) => setFilter({ minShots: v })}
                  format={(v) => `${v} 拍`}
                />
                <Slider
                  label="最低置信度"
                  value={filter.minConfidence}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => setFilter({ minConfidence: v })}
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                  hint="置信度反映这一段信号干不干净（球员跟踪稳不稳、运动曲线清不清晰）。\n调高可以滤掉那些「AI 也不太确定」的片段。"
                />
                <div className="flex flex-wrap gap-1.5 pt-0.5">
                  <button
                    onClick={() => setFilter({ starredOnly: !filter.starredOnly })}
                    className={cn(
                      'rounded-md px-2 py-1 text-[11px] transition-colors',
                      filter.starredOnly ? 'bg-amber-glow/20 text-amber-glow' : 'bg-white/6 text-ink-300 hover:bg-white/12',
                    )}
                  >
                    ★ 仅重点
                  </button>
                  <button
                    onClick={() => setFilter({ keepOnly: !filter.keepOnly })}
                    className={cn(
                      'rounded-md px-2 py-1 text-[11px] transition-colors',
                      filter.keepOnly ? 'bg-court-500/20 text-court-300' : 'bg-white/6 text-ink-300 hover:bg-white/12',
                    )}
                  >
                    仅保留
                  </button>
                  {allTags.map(([t, n]) => {
                    const on = filter.tags.includes(t)
                    return (
                      <button
                        key={t}
                        onClick={() =>
                          setFilter({ tags: on ? filter.tags.filter((x) => x !== t) : [...filter.tags, t] })
                        }
                        className="rounded-md px-2 py-1 text-[11px] transition-all"
                        style={{
                          background: on ? `${tagColor(t)}2a` : 'rgb(255 255 255 / 0.06)',
                          color: on ? tagColor(t) : 'var(--color-ink-300)',
                        }}
                      >
                        {t} <span className="opacity-60">{n}</span>
                      </button>
                    )
                  })}
                </div>

                {/* 一键预设 */}
                <div className="flex flex-wrap gap-1.5 border-t border-white/7 pt-2.5">
                  <span className="self-center text-[10.5px] tracking-wide text-ink-500">快速筛选</span>
                  {[
                    { label: '精彩 TOP 20%', patch: { minScore: 70, minShots: 0, minDuration: 0, starredOnly: false, tags: [] } },
                    { label: '多拍 (≥10)', patch: { minShots: 10, minScore: 0, minDuration: 0, tags: [] } },
                    { label: '长回合 (≥15s)', patch: { minDuration: 15, minScore: 0, minShots: 0, tags: [] } },
                    { label: '全部', patch: { ...DEFAULT_FILTER } },
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
            </motion.div>
          )}
        </AnimatePresence>

        {/* 筛选 ≠ 成片：把这条链路直接摊开，
            否则很容易以为「筛出来的就是成片了，为什么还要点加入」。 */}
        <div className="mt-2 flex items-start gap-2 rounded-xl border border-flux-400/20 bg-flux-400/[0.05] px-2.5 py-2">
          <Clapperboard size={12} className="mt-[1px] shrink-0 text-flux-400" />
          <div className="min-w-0 flex-1 text-[10.5px] leading-relaxed text-ink-300">
            <span>
              成片：<b className="text-court-300">{filmRallies}</b>/{all.length} 个回合 ·{' '}
              <b className="mono text-ink-200">{humanDuration(filmSeconds)}</b>
            </span>
            <div className="text-ink-500">筛选只挑候选，不改动成片；导出只认下面那条成片轨。</div>
          </div>
        </div>

        <div className="mt-2 flex gap-2">
          <Tooltip
            width={300}
            content={`按当前筛选把这 ${visible.length} 个回合依次排进成片。\n会用筛选结果重建整条成片轨（你手工拖过的位置会被覆盖，可以 Ctrl+Z 撤销）。`}
          >
            <Button
              variant="primary"
              size="sm"
              className="flex-1"
              onClick={() => autoCut({ mode: 'replace' })}
              disabled={!visible.length}
            >
              <Sparkles size={12} />
              用筛选结果重建成片 ({visible.length})
            </Button>
          </Tooltip>
          <Tooltip
            width={290}
            content={'只把还没进成片的回合追加到成片末尾。\n已经在成片里的回合会自动跳过（一个回合只排一次）。'}
          >
            <Button variant="outline" size="sm" onClick={() => autoCut({ mode: 'append' })} disabled={!visible.length}>
              <Plus size={12} />
            </Button>
          </Tooltip>
          <Tooltip
            width={290}
            content={'只看筛选片段：从第一个筛选出来的回合开始连续播放，\n中间的捡球、走动会自动跳过。再点一次关闭。'}
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

        {/* 切分调参：只重跑切分，毫秒级 */}
        <div className="mt-2 rounded-xl border border-white/7 bg-white/[0.025] px-3 py-2.5">
          <button
            onClick={() => setShowSplit((v) => !v)}
            className="flex w-full items-center justify-between text-[11px] text-ink-200"
          >
            <span className="flex items-center gap-1.5">
              <Scissors size={11} className="text-court-300" />
              回合切分调参
              <span className="text-ink-500">（即时生效，不用重新分析）</span>
            </span>
            <span className="text-ink-500">{showSplit ? '收起' : '展开'}</span>
          </button>
          <AnimatePresence>
            {showSplit && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-2.5 pt-3">
                  <div>
                    <Slider
                      label="切分粒度"
                      value={params.split_sensitivity}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={(v) => setParams({ split_sensitivity: v })}
                      format={(v) => (v < 0.3 ? '偏粗（回合长）' : v > 0.7 ? '偏细（回合短）' : '适中')}
                      hint="连续训练时两个回合之间只停几秒，活跃度不会明显下陷，光靠阈值会把几个回合粘成一条。调细会在活跃度低谷处切得更积极。"
                    />
                    <Slider
                      label="间隔判定"
                      value={params.gap_seconds}
                      min={0.5}
                      max={8}
                      step={0.1}
                      onChange={(v) => setParams({ gap_seconds: v })}
                      format={(v) => `${v.toFixed(1)}s`}
                      hint="两个候选回合靠得比这个更近就合并成一个"
                    />
                    <Slider
                      label="最短回合"
                      value={params.min_rally_seconds}
                      min={0.5}
                      max={12}
                      step={0.5}
                      onChange={(v) => setParams({ min_rally_seconds: v })}
                      format={(v) => `${v}s`}
                      hint="短于这个时长的片段会被丢弃"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="subtle"
                      size="sm"
                      className="flex-1"
                      loading={busy.resegment}
                      onClick={() => resegment()}
                    >
                      应用并重新切分
                    </Button>
                    <Tooltip content="只调整切分粒度后重切，其余参数保持">
                      <Button
                        variant="outline"
                        size="sm"
                        loading={busy.resegment}
                        onClick={() => resegment({ split_sensitivity: Math.min(1, params.split_sensitivity + 0.2) })}
                      >
                        再细一档
                      </Button>
                    </Tooltip>
                  </div>
                  <div className="text-[10px] leading-relaxed text-ink-500">
                    当前共 {all.length} 个回合，平均 {all.length ? (all.reduce((a, r) => a + r.duration, 0) / all.length).toFixed(1) : 0} 秒。
                    如果发现某个回合里其实有好几段交锋，把「切分粒度」调细再点应用。
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="mt-2 flex items-center gap-1.5">
          <span className="shrink-0 text-[10.5px] tracking-wide text-ink-500">按什么打分</span>
          <div className="flex flex-wrap gap-1">
            {WEIGHT_PRESETS.map((w) => (
              <Tooltip
                key={w.value}
                width={330}
                content={
                  <div>
                    <div className="mb-1 font-medium text-court-300">{w.label}</div>
                    <div className="text-ink-400">{w.hint}</div>
                    <div className="mt-2 text-ink-300">它怎么算：</div>
                    <div className="mt-0.5 text-ink-300">{w.detail.summary}</div>
                    <div className="mt-2 text-ink-300">五个分项的权重</div>
                    <div className="mt-1 space-y-1">
                      {w.detail.weights.map(([name, pct]) => (
                        <div key={name} className="flex items-center gap-2">
                          <span className="w-8 shrink-0 text-[10.5px] text-ink-400">{name}</span>
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
                      总分 = 上面五项加权求和，再乘一个「分析置信度」折扣
                      （球员跟踪稳、运动曲线清晰就接近满分，来回都检不到人就打折）。
                    </div>
                  </div>
                }
              >
                <button
                  onClick={() => useStore.getState().rescore(w.value)}
                  className={cn(
                    'rounded-md px-1.5 py-[3px] text-[10.5px] whitespace-nowrap transition-colors',
                    weights === w.value ? 'bg-court-500/20 text-court-300' : 'bg-white/5 text-ink-400 hover:bg-white/10',
                  )}
                >
                  {w.label}
                </button>
              </Tooltip>
            ))}
          </div>
        </div>
        <div className="mt-1.5 text-[10px] leading-relaxed text-ink-500">
          鼠标移到选项上看它的计分配方。切换只重新算分，不会动切分和你的手工调整。
        </div>
      </div>

      {/* 列表 */}
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2.5">
        <AnimatePresence initial={false}>
          {visible.map((r) => (
            <RallyCard
              key={r.id}
              rally={r}
              selected={r.id === selectedRallyId}
              onSelect={() => selectRally(r.id)}
            />
          ))}
        </AnimatePresence>
        {!visible.length && (
          <div className="py-10 text-center text-[12px] text-ink-500">没有符合筛选条件的回合</div>
        )}
      </div>

      {/* 底部批量 */}
      <div className="flex shrink-0 items-center gap-2 border-t border-white/7 px-3 py-2">
        <Tooltip content="把筛选出的回合全部标记为保留">
          <Button
            variant="ghost"
            size="sm"
            disabled={!visible.length}
            onClick={() => bulk({ keep: true }, { useFilter: true })}
          >
            <Eye size={12} /> 全部保留
          </Button>
        </Tooltip>
        <Tooltip content="把筛选出的回合全部排除">
          <Button
            variant="ghost"
            size="sm"
            disabled={!visible.length}
            onClick={() => bulk({ keep: false }, { useFilter: true })}
          >
            <EyeOff size={12} /> 全部排除
          </Button>
        </Tooltip>
        <div className="flex-1" />
        <Tooltip content="按评分取前 40%">
          <Button
            variant="ghost"
            size="sm"
            disabled={!all.length}
            onClick={async () => {
              const sorted = [...all].sort((a, b) => b.scores.total - a.scores.total)
              const keepN = Math.ceil(sorted.length * 0.4)
              const ids = sorted.slice(0, keepN).map((r) => r.id)
              // 两次批量写必须串行：并发调用各自会 openProject 刷新，可能互相
              // 覆盖，导致其中一半更新丢失。
              await bulk({ keep: true }, { ids })
              const rest = sorted.slice(keepN).map((r) => r.id)
              if (rest.length) await bulk({ keep: false }, { ids: rest })
            }}
          >
            <TrendingUp size={12} /> 保留前 40%
          </Button>
        </Tooltip>
      </div>
    </div>
  )
}
