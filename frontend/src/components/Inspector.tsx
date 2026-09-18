import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Star, Eye, EyeOff, Plus, Check, Trash2, Scissors, Gauge, Timer, Target } from 'lucide-react'
import { cn, humanDuration, scoreColor, scoreGrade, tagColor, timecode } from '../lib/format'
import { Badge, Button, ScoreBar, ScoreRing, SectionTitle, Segmented, Slider, Tooltip } from './ui'
import { useStore } from '../store/useStore'

/* ------------------------------------------------------------------ 信号图 */

function SignalChart({ height = 92 }: { height?: number }) {
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
        <span>AI 活跃度曲线（绿色区域为识别出的回合，颜色=评分）</span>
        <span className="mono">{timecode(dur, false)}</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 检查器 */

const WEIGHT_LABEL: Record<string, string> = {
  balanced: '均衡',
  highlight: '精彩集锦',
  long_rally: '多拍回合',
  technique: '技术动作',
  training: '训练复盘',
}

/** 把后端的模块状态字典翻译成一句人话。 */
function traceLabel(trace: any, offText: string): string {
  if (!trace || typeof trace !== 'object') return offText
  if (trace.disabled) return '未启用'
  if (trace.error) return '出错'
  if (trace.skipped) return '已忽略'
  if (typeof trace.tracks === 'number') return `${trace.tracks} 条轨迹`
  if (typeof trace.count === 'number') return `${trace.count} 次击球`
  if (Array.isArray(trace.active_ids)) return `${trace.active_ids.length} 个 ID`
  return '已启用'
}

export default function Inspector() {
  const analysis = useStore((s) => s.currentAnalysis())
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const selectedClipId = useStore((s) => s.selectedClipId)
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
  /** 这个回合是不是已经在成片轨里了（一个回合只加入一次） */
  const inFilm = useMemo(
    () => (rally ? project?.timeline.tracks.some((t) => t.clips.some((c) => c.rally_id === rally.id)) ?? false : false),
    [project, rally],
  )

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
            { value: 'rally', label: '回合' },
            { value: 'clip', label: '片段' },
            { value: 'info', label: '信号' },
          ]}
        />
        <div className="mt-1.5 text-[10px] leading-relaxed text-ink-500">
          页签不会因为你切换回合、片段而自动跳走。
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {tab === 'rally' &&
          (rally ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <ScoreRing score={rally.scores.total} size={58} color={scoreColor(rally.scores.total)} label={scoreGrade(rally.scores.total)} />
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] font-semibold text-white">回合 #{rally.index}</div>
                  <div className="mono mt-0.5 text-[11px] text-ink-400">
                    {timecode(rally.start, false)} → {timecode(rally.end, false)}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {rally.tags.map((t) => (
                      <Badge key={t} color={tagColor(t)}>
                        {t}
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
                  {rally.starred ? '已标记' : '标记'}
                </Button>
                <Button variant="ghost" size="sm" className="flex-1" onClick={() => patchRally(rally.id, { keep: !rally.keep })}>
                  {rally.keep ? <Eye size={12} /> : <EyeOff size={12} />}
                  {rally.keep ? '保留' : '已排除'}
                </Button>
                <Tooltip
                  width={300}
                  className="flex-1"
                  content={
                    inFilm ? (
                      <span>
                        <b className="text-court-300">这个回合已经在成片里了</b>
                        {'\n\n'}
                        每个回合只加入一次，重复点不会再加一段。
                        点一下会跳到成片里的那一段，想改长度就拖它的两端。
                      </span>
                    ) : (
                      <span>
                        <b className="text-court-300">加入成片</b>
                        {'\n\n'}
                        把这个回合按「剪辑区间」追加到下面的<b>成片轨</b>末尾。
                        导出时输出的就是成片轨里的内容，所以「加入成片」= 决定这一段要不要出现在成片里。
                        {'\n\n'}
                        只想先看看、不动成片的话，单击回合卡片或时间线上方的色块就行，那是纯定位预览。
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
                    {inFilm ? '已在成片' : '加入成片'}
                  </Button>
                </Tooltip>
              </div>

              <div>
                <SectionTitle>
                  评分构成
                </SectionTitle>
                <div className="mb-1.5 text-[10.5px] leading-relaxed text-ink-500">
                  鼠标移到每一项可看它是怎么算的
                </div>
                <div className="space-y-2.5">
                  {(
                    [
                      ['长度', rally.scores.length,
                        '看拍数与时长。\n用饱和曲线：8 拍和 20 拍差别很大，但 40 拍和 52 拍差别不大，\n避免「越长越占榜首」。'],
                      ['强度', rally.scores.intensity,
                        '看球员跑动速度、画面运动峰值、整体节奏，\n以及回合最后 1/3 的节奏（末段提速会加分）。'],
                      ['技术', rally.scores.technique,
                        '看球速、击球力度、羽毛球在画面中出现的持续性。\n没开羽毛球跟踪时这一项主要由击球力度决定。'],
                      ['精彩', rally.scores.excitement,
                        '长度 × 强度 × 末段提速 × 多拍的综合。\n做集锦时优先看这一项。'],
                      ['画面', rally.scores.production,
                        '看清晰度、镜头抖动、主体在画面里够不够大。\n画面太糊或主体太小的片段不值得留。'],
                    ] as const
                  ).map(([k, v, hint]) => (
                    <Tooltip key={k} content={hint} side="left" width={280} block>
                      <div className="w-full cursor-help">
                        <div className="mb-1 flex justify-between text-[11px]">
                          <span className="text-ink-300 underline decoration-dotted decoration-ink-600 underline-offset-2">
                            {k}
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
                      总分 = 上面五项按当前评分口径加权求和，
                      再乘一个「分析置信度」折扣。
                      {'\n\n'}
                      置信度反映这一段的信号有多干净：球员跟踪稳、运动曲线清晰就高；
                      来回都检不到人就低。想换算法就点左侧「评分口径」。
                    </span>
                  }
                >
                  <div className="mt-2 cursor-help text-[10.5px] text-ink-500 underline decoration-dotted underline-offset-2">
                    总分 {rally.scores.total.toFixed(1)} 是怎么来的 ⓘ
                  </div>
                </Tooltip>
              </div>

              <div>
                <SectionTitle>客观数据</SectionTitle>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    ['时长', `${rally.duration.toFixed(2)}s`, Timer],
                    ['拍数', `${rally.features.shot_count}`, Target],
                    ['节奏', `${rally.features.tempo.toFixed(2)}/s`, Gauge],
                    ['置信度', `${(rally.features.confidence * 100).toFixed(0)}%`, Activity],
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
                  <SectionTitle right={<span className="text-[10.5px] text-ink-500">点击定位</span>}>
                    逐拍时间轴
                  </SectionTitle>
                  <div className="flex flex-wrap gap-1">
                    {rally.shots.map((s, i) => (
                      <Tooltip key={i} content={`第 ${i + 1} 拍 · ${s.time.toFixed(2)}s · 置信 ${(s.confidence * 100).toFixed(0)}%`}>
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
                          {i === 0 ? '发' : i === 1 ? '接' : i + 1}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                  <div className="mt-1.5 text-[10.5px] text-ink-500">
                    发球 {rally.serve_time ? `${rally.serve_time.toFixed(2)}s` : '—'} · 接发球{' '}
                    {rally.receive_time ? `${rally.receive_time.toFixed(2)}s` : '—'}
                  </div>
                </div>
              )}

              <div>
                <SectionTitle>剪辑区间</SectionTitle>
                <div className="mb-1.5 text-[10.5px] leading-relaxed text-ink-500">
                  这是从<b className="text-ink-300">原片</b>上裁下来的范围（原片时间）。
                  用「按筛选自动剪辑」生成时间线时，用的就是这里的入点/出点。
                  <br />
                  已经放进时间线的片段，请在下方的「片段」里调。
                </div>
                <div className="space-y-2">
                  <Slider
                    label="入点"
                    value={rally.clip_start}
                    min={0}
                    max={Math.max(0.1, rally.end)}
                    step={0.05}
                    onChange={(v) => patchRally(rally.id, { clip_start: v })}
                    format={(v) => timecode(v, false)}
                  />
                  <Slider
                    label="出点"
                    value={rally.clip_end}
                    min={Math.min(rally.start, rally.clip_start)}
                    max={(analysis?.stats?.duration as number) ?? rally.clip_end + 30}
                    step={0.05}
                    onChange={(v) => patchRally(rally.id, { clip_end: v })}
                    format={(v) => timecode(v, false)}
                  />
                  <div className="flex gap-1.5 pt-0.5">
                    {(
                      [
                        ['紧', 0.2, 0.4, '去掉准备与收尾，只留干净的交锋'],
                        ['标准', 0.8, 1.2, '保留一点发球准备与死球收尾'],
                        ['宽松', 1.8, 2.6, '前后各多留一段，适合看节奏'],
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
                <SectionTitle>备注</SectionTitle>
                <textarea
                  value={rally.note}
                  onChange={(e) => patchRally(rally.id, { note: e.target.value })}
                  rows={2}
                  placeholder="记录这一回合的战术观察…"
                  className="field text-[12px]"
                />
              </div>
            </div>
          ) : (
            <div className="px-2 py-10 text-center text-[12px] text-ink-500">
              在左侧回合列表或时间线上点选一个回合
            </div>
          ))}

        {tab === 'clip' &&
          (clip ? (
            <div className="space-y-4">
              <div className="rounded-xl border border-flux-400/25 bg-flux-400/[0.06] px-3 py-2.5">
                <div className="text-[11.5px] font-medium text-flux-400">这是时间线上的一个片段</div>
                <div className="mt-1 text-[10.5px] leading-relaxed text-ink-400">
                  它引用原片的一段，并决定<b className="text-ink-200">在成片里出现在第几秒、有多长、多快</b>。
                  改速度会改变它在时间线上占的长度（2× 就占一半），这是正常的。
                  <br />
                  想改「取原片哪一段」请用上面的「回合 → 剪辑区间」。
                </div>
              </div>

              <div>
                <div className="text-[13.5px] font-semibold text-white">{clip.label || '片段'}</div>
                <div className="mono mt-0.5 text-[11px] text-ink-400">
                  原片 {timecode(clip.src_in, false)} → {timecode(clip.src_out, false)}
                  {' · '}
                  素材时长 {(clip.src_out - clip.src_in).toFixed(2)}s
                </div>
                <div className="mono mt-0.5 text-[11px] text-ink-500">
                  成片位置 {timecode(clip.tl_start, false)} → {timecode(clip.tl_start + (clip.src_out - clip.src_in) / clip.speed, false)}
                  {' · '}
                  占 {( (clip.src_out - clip.src_in) / clip.speed).toFixed(2)}s
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
                  分割
                </Button>
                <Button variant="ghost" size="sm" className="flex-1" onClick={() => removeClip(clip.id)}>
                  <Trash2 size={12} className="text-rose-hot/80" />
                  删除
                </Button>
              </div>

              <div className="space-y-2.5">
                <Slider
                  label="播放速度"
                  value={clip.speed}
                  min={0.25}
                  max={4}
                  step={0.05}
                  onStart={() => pushHistory()}
                  onChange={(v) => updateClip(clip.id, { speed: Number(v.toFixed(2)) }, false)}
                  format={(v) => `${v.toFixed(2)}×`}
                  hint="改变速度会同时改变它在时间线上占的长度：2× 变一半，0.5× 变两倍"
                />
                <Slider
                  label="音量"
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
                <div className="mb-1.5 text-[10.5px] text-ink-500">快捷变速</div>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    [0.35, '0.35× 超慢放'],
                    [0.5, '0.5× 慢放'],
                    [1, '原速'],
                    [1.5, '1.5×'],
                    [2, '2× 快放'],
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
                  想改位置直接在上方时间线里拖动片段即可，这里不再重复给一个「起点」输入框。
                </div>
              </div>
            </div>
          ) : (
            <div className="px-2 py-10 text-center text-[12px] text-ink-500">在时间线上点选一个片段</div>
          ))}

        {tab === 'info' && (
          <div className="space-y-4">
            <div>
              <SectionTitle>
                <Activity size={12} /> AI 活跃度曲线
              </SectionTitle>
              <SignalChart />
              <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
                这是 AI 判断「现在有没有在打球」的综合曲线，由几路信号加权而成：
                <br />
                <b className="text-ink-300">球员跑动</b>（最可靠）、
                <b className="text-ink-300">画面运动</b>、
                <b className="text-ink-300">击球声</b>、
                <b className="text-ink-300">羽毛球出现</b>。
                <br />
                每路信号会自己算可信度：比如杂音大的球馆里，击球声那一路会被自动压到接近 0，
                改由球员跑动主导。
                <br />
                绿色区域 = AI 切出来的回合，颜色深浅代表评分；点击可跳转。
              </div>
            </div>

            <div>
              <SectionTitle>这次分析用了什么</SectionTitle>
              <div className="space-y-1.5">
                {[
                  ['球员跑动', traceLabel(analysis?.stats?.player_trace, '关'),
                    '球员检测与多目标跟踪'],
                  ['击球声', traceLabel(analysis?.stats?.hit_trace, '关'),
                    '球拍触球的中高频瞬态'],
                  ['画面运动', '已启用', '帧间运动能量与场地热区'],
                  ['羽毛球轨迹', traceLabel(analysis?.stats?.shuttle_trace, '未启用'),
                    '默认关闭，长视频很慢'],
                ].map(([name, status, desc]) => (
                  <div key={name as string} className="panel-flat flex items-center gap-2 px-2.5 py-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-[11.5px] text-ink-100">{name}</div>
                      <div className="text-[10px] text-ink-500">{desc}</div>
                    </div>
                    <span
                      className={cn(
                        'shrink-0 rounded px-1.5 py-[2px] text-[10px]',
                        String(status).includes('关') || String(status).includes('未启用')
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
                <SectionTitle>分析统计</SectionTitle>
                <div className="space-y-1.5 text-[11.5px]">
                  {[
                    ['识别回合数', `${analysis.stats.count}`],
                    ['有效时长', humanDuration(analysis.stats.active_duration || 0)],
                    ['素材总长', humanDuration(analysis.stats.duration || 0)],
                    ['有效占比', `${(((analysis.stats.active_duration || 0) / Math.max(1, analysis.stats.duration || 1)) * 100).toFixed(1)}%`],
                    ['平均拍数', `${(analysis.stats.avg_shots || 0).toFixed(1)}`],
                    ['最多拍数', `${analysis.stats.max_shots || 0}`],
                    ['平均分', `${(analysis.stats.avg_score || 0).toFixed(1)}`],
                    ['高分回合', `${analysis.stats.high_score_count || 0} (≥70)`],
                    ['评分口径', WEIGHT_LABEL[analysis.stats.weights as string] || analysis.stats.weights || '均衡'],
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
