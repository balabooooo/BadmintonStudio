import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  BarChart3,
  Camera,
  Image as ImageIcon,
  Lightbulb,
  Loader2,
  Ruler,
  ScanSearch,
  Trash2,
} from 'lucide-react'
import { api } from '../lib/api'
import type { AnalysisParams, PlayerProbeFrame, PlayerSizeStats } from '../lib/types'
import { cn } from '../lib/format'
import { Button, Segmented, Slider } from './ui'
import { useStore } from '../store/useStore'

/** 直方图分箱与横轴上限：与后端 `players.SIZE_HIST_BINS / SIZE_HIST_MAX` 对齐。 */
const BINS = 24
const H_MAX = 0.5
/** 最多保留多少张抓到的帧（够对比了，再多只会让缩略图条挤成一团）。 */
const MAX_FRAMES = 40
/** 相对口径下小于这个参考框高就不做判断（与后端 `SizeFilter.thresholds` 一致）。 */
const REF_MIN = 0.02

/** 相对口径下的推荐下限：同帧最大框的 42%（与后端自适应门限同源）。 */
const REL_MIN_DEFAULT = 0.42
/** 绝对口径下的推荐下限（占画面高度）。低于 5% 的人框在 1080p 上不到 54 像素高。 */
const ABS_MIN_DEFAULT = 0.05

const KEEP_COLOR = '#38e0a2'
const DROP_COLOR = '#f87171'

/** 一个框没通过筛选的原因（与后端 ``players.SizeFilter.keep`` 同一套判据，null = 通过）。
 *
 * 复刻这套判据是为了让界面「所见即所得」：拖滑杆时画面里的框立刻变色、
 * 直方图立刻重算，都不需要重新跑一遍检测。两边必须保持一致，改一边就要改另一边。
 */
function dropReason(s: number[], p: AnalysisParams): string | null {
  if (p.player_size_mode === 'off' || s.length < 2) return null
  const h = s[0]
  const area = s[1]
  const ref = s[2] ?? 0
  let lo = p.player_min_height
  let hi = p.player_max_height
  if (p.player_size_mode === 'relative') {
    // 参考框高太小说明这一帧没有可信的「球员尺度」，此时不做判断
    if (!(ref > REF_MIN)) return null
    lo = p.player_min_height * ref
    hi = p.player_max_height > 0 ? p.player_max_height * ref : 0
  }
  if (lo > 0 && h < lo) return '低于下限'
  if (hi > 0 && h > hi) return '超过上限'
  if (p.player_min_area > 0 && area < p.player_min_area) return '面积过小'
  if (p.player_max_area > 0 && area > p.player_max_area) return '面积过大'
  return null
}

/** 一个框是否通过当前筛选条件。 */
function keepBox(s: number[], p: AnalysisParams): boolean {
  return dropReason(s, p) === null
}

/** 归一化框 → 判据需要的三元组 ``[框高, 框面积, 同帧参考框高]``。 */
function boxSample(b: number[], ref: number): number[] {
  const w = Math.max(0, b[2] - b[0])
  const h = Math.max(0, b[3] - b[1])
  return [h, w * h, ref]
}

/* ------------------------------------------------------------------ 帧画面 + 框 */

/**
 * 把某一帧的人框画在**画面本身**上：绿色实线＝会被保留，红色虚线＝会被筛掉。
 *
 * 这是这一块最直观的地方：数字（框高 12%）没法让人判断「那到底是谁」，
 * 画出来一眼就知道「被筛掉的是看台上的人」还是「把真球员筛掉了」。
 * 颜色随阈值实时变化（判定在本地做，不动后端）。
 */
function FrameView({
  frame,
  aspect,
  params,
}: {
  frame: PlayerProbeFrame | null
  aspect: number
  params: AnalysisParams
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const measure = () => {
      const r = el.getBoundingClientRect()
      setBox({ w: r.width, h: r.height })
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const imgUrl = frame?.image ? api.assetUrl(frame.image, 86400) : null

  // object-contain：画面按自身比例缩放后居中，和预览播放器同一套映射
  const containerAspect = box.w / Math.max(1, box.h)
  let vw = box.w
  let vh = box.h
  if (box.w > 0 && box.h > 0) {
    if (containerAspect > aspect) {
      vh = box.h
      vw = vh * aspect
    } else {
      vw = box.w
      vh = vw / aspect
    }
  }
  const ox = (box.w - vw) / 2
  const oy = (box.h - vh) / 2

  const items = useMemo(() => {
    if (!frame) return []
    return frame.boxes.map((b, i) => {
      const s = boxSample(b, frame.ref)
      const reason = dropReason(s, params)
      return { b, i, h: s[0], area: s[1], reason }
    })
  }, [frame, params])

  const keptN = items.filter((x) => !x.reason).length

  return (
    <div>
      <div
        ref={wrapRef}
        className="relative w-full overflow-hidden rounded-lg border border-white/8 bg-black"
        style={{ aspectRatio: `${aspect}` }}
      >
        {imgUrl ? (
          <>
            <img
              src={imgUrl}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            />
            {/* 量到容器尺寸之后再画框：否则首帧会以 0 尺寸渲染一次（闪一下） */}
            {box.w > 0 && box.h > 0 && (
              <svg className="absolute inset-0 h-full w-full">
              {items.map(({ b, i, h, reason }) => {
                const x1 = ox + b[0] * vw
                const y1 = oy + b[1] * vh
                const x2 = ox + b[2] * vw
                const y2 = oy + b[3] * vh
                const color = reason ? DROP_COLOR : KEEP_COLOR
                const label = `#${i + 1} ${(h * 100).toFixed(1)}%${reason ? ` ${reason}` : ''}`
                return (
                  <g key={i}>
                    <rect
                      x={x1}
                      y={y1}
                      width={Math.max(1, x2 - x1)}
                      height={Math.max(1, y2 - y1)}
                      fill={color}
                      fillOpacity={reason ? 0.10 : 0.14}
                      stroke={color}
                      strokeWidth={2}
                      strokeDasharray={reason ? '5 3' : undefined}
                    >
                      <title>{`框 ${i + 1}：高 ${(h * 100).toFixed(1)}% · 面积 ${(100 * (b[2] - b[0]) * (b[3] - b[1])).toFixed(2)}%${reason ? ` · 会被筛掉（${reason}）` : ' · 会被保留'}`}</title>
                    </rect>
                    <text
                      x={x1 + 2}
                      y={Math.max(10, y1 - 3)}
                      fontSize={11}
                      fill={color}
                      stroke="#000"
                      strokeWidth={3}
                      paintOrder="stroke"
                      style={{ fontWeight: 600 }}
                    >
                      {label}
                    </text>
                  </g>
                )
              })}
              </svg>
            )}
          </>
        ) : (
          <div className="grid h-full w-full place-items-center px-6 text-center text-[11.5px] leading-relaxed text-ink-500">
            {frame
              ? '这一帧没有存到画面（只拿到了框）'
              : '点「抓这一帧」把当前时间的画面抓出来，框会直接画在画面上'}
          </div>
        )}
      </div>

      {frame && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px]">
          <span className="mono text-ink-300">t = {frame.t.toFixed(2)}s</span>
          <span className="text-ink-400">
            这一帧 <span className="mono text-ink-200">{items.length}</span> 个框 ·
            保留 <span className="mono" style={{ color: KEEP_COLOR }}>{keptN}</span> ·
            筛掉 <span className="mono" style={{ color: DROP_COLOR }}>{items.length - keptN}</span>
          </span>
          {frame.ref > 0 && (
            <span className="text-ink-500">参考框高（同帧最大）{(frame.ref * 100).toFixed(1)}%</span>
          )}
          <span className="inline-flex items-center gap-1 text-ink-500">
            <span className="inline-block h-[3px] w-4 rounded-full" style={{ background: KEEP_COLOR }} />保留
            <span
              className="ml-1 inline-block h-[3px] w-4 rounded-full"
              style={{ background: `repeating-linear-gradient(90deg, ${DROP_COLOR} 0 4px, transparent 4px 7px)` }}
            />
            筛掉
          </span>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 直方图 */

/** 框尺寸分布直方图：每根柱子按「保留 / 丢掉」两段着色，阈值线只作参照。 */
function SizeHistogram({
  sample,
  params,
  relative,
}: {
  sample: number[][]
  params: AnalysisParams
  relative: boolean
}) {
  const { counts, kcounts, kept, total, maxCount, xMax } = useMemo(() => {
    const xMax = relative ? 1.0 : H_MAX
    const counts = new Array<number>(BINS).fill(0)
    const kcounts = new Array<number>(BINS).fill(0)
    let kept = 0
    for (const s of sample) {
      const v = relative ? (s[2] > REF_MIN ? s[0] / s[2] : -1) : s[0]
      if (v < 0) continue // 这一帧没有可信参考尺度，后端也不筛，先不计入分布
      const bin = Math.min(BINS - 1, Math.max(0, Math.floor((v / xMax) * BINS)))
      counts[bin] += 1
      if (keepBox(s, params)) {
        kept += 1
        kcounts[bin] += 1
      }
    }
    const total = counts.reduce((a, b) => a + b, 0)
    return { counts, kcounts, kept, total, maxCount: Math.max(1, ...counts), xMax }
  }, [sample, params, relative])

  const pct = (v: number) => Math.min(100, Math.max(0, (v / xMax) * 100))
  const active = params.player_size_mode !== 'off'

  return (
    <div>
      <div className="relative h-[86px] w-full overflow-hidden rounded-lg border border-white/8 bg-ink-950/60">
        <div className="absolute inset-0 flex items-end gap-[1px] px-[2px] pb-[1px]">
          {counts.map((c, i) => {
            const keepFrac = c > 0 ? kcounts[i] / c : 0
            return (
              <div
                key={i}
                className="flex-1 overflow-hidden rounded-t-[2px] bg-ink-600/50"
                style={{ height: `${(c / maxCount) * 100}%` }}
                title={`${c} 个框，保留 ${kcounts[i]}`}
              >
                <div
                  className="w-full bg-court-400/80"
                  style={{ height: `${keepFrac * 100}%`, marginTop: `${(1 - keepFrac) * 100}%` }}
                />
              </div>
            )
          })}
        </div>
        {total === 0 && (
          <div className="absolute inset-0 grid place-items-center text-[11px] text-ink-500">
            还没有数据：先跑一次分析，或抓一帧看看
          </div>
        )}
        {active && params.player_min_height > 0 && (
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-court-300"
            style={{ left: `${pct(params.player_min_height)}%` }}
            title={`下限 ${params.player_min_height}`}
          />
        )}
        {active && params.player_max_height > 0 && (
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-amber-glow"
            style={{ left: `${pct(params.player_max_height)}%` }}
            title={`上限 ${params.player_max_height}`}
          />
        )}
      </div>
      <div className="mt-1 flex items-center justify-between text-[10.5px] text-ink-500">
        <span>0</span>
        <span>{relative ? '横轴：框高 ÷ 同帧最大框高' : '横轴：框高（占画面高度，比例）'}</span>
        <span>{relative ? '1.0' : H_MAX.toFixed(1)}</span>
      </div>
      {total > 0 && (
        <div className="mt-1 text-[11px] text-ink-400">
          按当前条件保留 <span className="mono text-court-300">{kept}</span> / {total} 个框
          <span className="text-ink-500">（{((kept / Math.max(1, total)) * 100).toFixed(0)}%）</span>
          {active && kept === total && (
            <span className="text-amber-glow"> · 目前一个都没筛掉，可以把下限调高一点</span>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 面板 */

/** 抓到的帧按时间合并（同一时刻只留最新的那次结果）。 */
function mergeFrames(prev: PlayerProbeFrame[], add: PlayerProbeFrame[]): PlayerProbeFrame[] {
  const byKey = new Map<string, PlayerProbeFrame>()
  for (const f of prev) byKey.set(f.t.toFixed(1), f)
  for (const f of add) byKey.set(f.t.toFixed(1), f)
  return [...byKey.values()].sort((a, b) => a.t - b.t).slice(0, MAX_FRAMES)
}

/**
 * 人物框尺寸筛选面板。
 *
 * 解决什么：原有的尺寸门限是**自适应**的（每帧最大框的 90 分位当参考尺度），
 * 在「球员是画面里最大的人」这种素材上够用，但两种素材会失灵：
 *
 * 1. **看台 / 观众比球员更靠近镜头** —— 参考尺度被最大的观众框劫持，
 *    真球员反而成了「偏小」的那一批；
 * 2. **全景 / 鱼眼畸变严重** —— 同一个球员在画面中心和边角的框高能差一倍以上，
 *    「多大才算球员」在画面不同位置本来就不是同一个数。
 *
 * 这两种情况只能让用户来指认，所以面板给出三层信息：
 *
 * 1. **画面本身**：手动选一帧（或抓当前播放位置）把检测框画上去，
 *    绿色＝会保留、红色虚线＝会筛掉 —— 「筛掉的到底是观众还是球员」一眼可见，
 *    拖阈值时实时变色；
 * 2. **分布**：筛选前所有框的框高直方图（按保留/丢掉两段着色）+ 保留比例；
 * 3. **两种口径**：绝对比例 / 同帧相对（后者对畸变更稳），外加面积上下限。
 */
export default function SizeFilterPanel() {
  const params = useStore((s) => s.params)
  const setParams = useStore((s) => s.setParams)
  const project = useStore((s) => s.project)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const currentTime = useStore((s) => s.currentTime)
  const toast = useStore((s) => s.toast)

  const [frames, setFrames] = useState<PlayerProbeFrame[]>([])
  const [sel, setSel] = useState(0)
  const [meta, setMeta] = useState<{ aspect: number; elapsed: number; device: string } | null>(null)
  // 初值取预览的播放位置：用户通常是在预览里看到某一帧觉得「就是这里」，
  // 才打开这个弹窗来调阈值，所以直接对到那个位置最省事。
  // 用惰性初始化而不是 effect：弹窗关闭时会卸载子树，每次打开都是一次新挂载，
  // 所以这里读到的就是「打开那一刻」的播放位置。
  const [time, setTime] = useState(() => {
    const t = useStore.getState().currentTime || 0
    return Math.max(0, Math.round(t * 2) / 2)
  })
  const [busy, setBusy] = useState(false)

  const stats = (analysis?.stats as Record<string, any> | undefined)?.player_trace
    ?.size_filter as PlayerSizeStats | undefined
  const relative = params.player_size_mode === 'relative'
  const ref = stats?.ref ?? frames[0]?.ref ?? 0

  // 分布样本：优先用「自己抓的帧」（和画面上看到的完全一致），否则用上次分析记录的
  const sample = useMemo(() => {
    if (frames.length) {
      const out: number[][] = []
      for (const f of frames) for (const b of f.boxes) out.push(boxSample(b, f.ref))
      return out
    }
    return stats?.sample ?? []
  }, [frames, stats])

  const aspect = meta?.aspect ?? (media && media.height > 0 ? media.width / media.height : 16 / 9)
  const duration = media?.duration ?? 0
  const cur = frames[Math.min(sel, Math.max(0, frames.length - 1))] ?? null

  const runProbe = useCallback(
    async (body: { count?: number; at_time?: number }, selectT?: number) => {
      if (!project || !media) return
      setBusy(true)
      try {
        const res = await api.playerProbe(project.id, {
          media_id: media.id,
          params,
          court_poly: manualPoly ?? undefined,
          save_frames: true,
          ...body,
        })
        if (res.error || !res.frames.length) {
          toast({
            kind: 'warn',
            title: '没抓到画面',
            detail: res.error || '这一帧没读出画面，换个时间再试',
          })
          return
        }
        setMeta({ aspect: res.aspect, elapsed: res.elapsed, device: res.device })
        const list = mergeFrames(frames, res.frames)
        setFrames(list)
        if (selectT !== undefined) {
          const k = list.findIndex((f) => Math.abs(f.t - selectT) < 0.06)
          setSel(k >= 0 ? k : Math.max(0, list.findIndex((f) => f.t >= selectT)))
        } else {
          setSel(0)
        }
        if (body.at_time === undefined) {
          toast({
            kind: 'success',
            title: `抓到 ${res.frames.length} 帧`,
            detail: `共 ${res.frames.reduce((a, f) => a + f.boxes.length, 0)} 个框 · 用了 ${res.elapsed.toFixed(1)}s · ${res.device}`,
          })
        }
      } catch (e) {
        toast({ kind: 'error', title: '试测失败', detail: String(e) })
      } finally {
        setBusy(false)
      }
    },
    [project, media, params, manualPoly, frames, toast],
  )

  /** 用实测分布给一个推荐下限：多少比例的人框能被留下。 */
  const applyRecommend = () => {
    if (params.player_size_mode === 'relative') {
      setParams({ player_min_height: REL_MIN_DEFAULT, player_max_height: 0 })
    } else if (ref > 0) {
      setParams({ player_min_height: Math.round(ref * 0.42 * 1000) / 1000, player_max_height: 0 })
    } else {
      setParams({ player_min_height: ABS_MIN_DEFAULT, player_max_height: 0 })
    }
    toast({
      kind: 'info',
      title: '已填入推荐阈值',
      detail: params.player_size_mode === 'relative'
        ? '同帧最大框的 42%：比它更小的人框基本不是比赛球员'
        : '按实测参考尺度（每帧最大框的 90 分位）折算，与自动门限同源',
    })
  }

  return (
    <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[12.5px] font-medium text-ink-100">
          <Ruler size={13} className="text-court-300" /> 人物框尺寸筛选
        </div>
        <Segmented
          size="sm"
          value={params.player_size_mode}
          onChange={(v) => {
            // 换口径时把下限换成该口径下的合理默认值：0.05 在「绝对」下是
            // 画面高度的 5%，在「相对」下却是「比最大的那个人小 95%」，
            // 直接把旧数字带过去会得到一个几乎不筛任何东西的阈值。
            setParams({
              player_size_mode: v,
              player_min_height: v === 'relative' ? REL_MIN_DEFAULT
                : v === 'absolute' ? ABS_MIN_DEFAULT
                  : params.player_min_height,
              player_max_height: 0,
            })
          }}
          options={[
            { value: 'off', label: '不筛', hint: '只保留原有的自适应尺寸门限' },
            { value: 'absolute', label: '绝对比例', hint: '按框高占画面高度的比例筛' },
            { value: 'relative', label: '同帧相对', hint: '按框高 ÷ 同帧最大框高筛，畸变素材更稳' },
          ]}
        />
      </div>

      <div className="mt-2 text-[11px] leading-relaxed text-ink-400">
        只在检测阶段丢掉「明显不是这场比赛球员」的人框。看台比球员更靠前、
        或者全景镜头把画面边角的人拉大/缩小的时候，自动门限会被带偏，这时用手动阈值最有效。
        {params.player_size_mode === 'relative' && (
          <span className="text-court-300"> 当前口径在每一帧里重新归一，免疫「位置不同框大小不同」。</span>
        )}
      </div>

      {/* ---- 画面核对：手动选帧，把框画上去 ---- */}
      <div className="mt-2.5 rounded-xl border border-white/8 bg-black/25 px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-200">
            <ImageIcon size={12} className="text-court-300" /> 帧核对
          </span>
          <Button
            variant="primary"
            size="sm"
            disabled={busy || !media}
            onClick={() => runProbe({ at_time: time }, time)}
          >
            {busy ? <Loader2 size={12} className="spin" /> : <Camera size={12} />}
            {busy ? '检测中…' : '抓这一帧'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={!media || currentTime <= 0}
            onClick={() => setTime(currentTime)}
            title="把滑杆对到预览播放头的位置"
          >
            用播放头 {currentTime > 0 ? `${currentTime.toFixed(1)}s` : ''}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={busy || !media}
            onClick={() => runProbe({ count: 10 })}
            title="在整条视频上均匀抽 10 帧，一次看清各类人框的大小分布"
          >
            <ScanSearch size={12} /> 均匀试测 10 帧
          </Button>
          {frames.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setFrames([])
                setSel(0)
              }}
              title="清掉抓到的帧"
            >
              <Trash2 size={12} /> 清空 {frames.length} 帧
            </Button>
          )}
          {meta && (
            <span className="text-[10.5px] text-ink-500">
              最近一次 {meta.elapsed.toFixed(1)}s · {meta.device}
            </span>
          )}
        </div>

        <div className="mt-2 max-w-[520px]">
          <Slider
            label="手动选帧"
            value={time}
            min={0}
            max={Math.max(1, Math.round(duration))}
            step={0.5}
            onChange={setTime}
            disabled={!media || duration <= 0}
            format={(v) => `${v.toFixed(1)}s`}
            hint="拖到想看的位置，再点「抓这一帧」；缩略图条里也可以直接点"
          />
        </div>

        {frames.length > 0 && (
          <div className="mt-2 flex gap-1.5 overflow-x-auto pb-1">
            {frames.map((f, i) => (
              <button
                key={f.t.toFixed(1)}
                onClick={() => setSel(i)}
                title={`${f.t.toFixed(2)}s · ${f.boxes.length} 个框（点击查看）`}
                className={cn(
                  'relative h-[46px] w-[82px] shrink-0 overflow-hidden rounded-md border transition-all',
                  i === sel ? 'border-court-400 ring-1 ring-court-400/40' : 'border-white/10 hover:border-white/30',
                )}
              >
                {f.image ? (
                  <img src={api.assetUrl(f.image, 86400)} alt="" className="h-full w-full object-cover" />
                ) : (
                  <span className="grid h-full w-full place-items-center bg-ink-900 text-[10px] text-ink-500">
                    无图
                  </span>
                )}
                <span className="mono absolute right-0 bottom-0 rounded-tl bg-black/70 px-1 text-[9px] text-ink-200">
                  {f.t.toFixed(1)}s
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="mt-2">
          <FrameView frame={cur} aspect={aspect} params={params} />
        </div>
        <div className="mt-1.5 text-[10.5px] leading-relaxed text-ink-500">
          拖下面的阈值滑杆，画面里的框会**立刻**变色（判定在本地做，不用重跑检测）：
          <span style={{ color: KEEP_COLOR }}> 绿实线＝会保留</span>、
          <span style={{ color: DROP_COLOR }}> 红虚线＝会筛掉</span>，
          鼠标停在框上能看到它的框高与面积。
        </div>
      </div>

      {/* ---- 阈值 ---- */}
      <div className="mt-2.5 grid gap-x-5 gap-y-1 md:grid-cols-2">
        <Slider
          label="框高下限"
          value={params.player_min_height}
          min={0}
          max={relative ? 1 : 0.4}
          step={relative ? 0.01 : 0.005}
          onChange={(v) => setParams({ player_min_height: v })}
          format={(v) => (relative ? `最大框的 ${(v * 100).toFixed(0)}%` : `画面高的 ${(v * 100).toFixed(1)}%`)}
          hint="比它更小的框在检测阶段就被丢掉（观众、隔壁场地、远处的人）"
          disabled={params.player_size_mode === 'off'}
        />
        <Slider
          label="框高上限"
          value={params.player_max_height}
          min={0}
          max={relative ? 2 : 1}
          step={relative ? 0.01 : 0.01}
          onChange={(v) => setParams({ player_max_height: v })}
          format={(v) => (v <= 0 ? '不限' : relative ? `最大框的 ${(v * 100).toFixed(0)}%` : `画面高的 ${(v * 100).toFixed(0)}%`)}
          hint="0 = 不限；用来挡掉「整个人贴到镜头前」的误检"
          disabled={params.player_size_mode === 'off'}
        />
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          disabled={params.player_size_mode === 'off'}
          onClick={applyRecommend}
          title="按实测分布填一个保守的下限"
        >
          <Lightbulb size={12} /> 用推荐值
        </Button>
        <span className="inline-flex items-center gap-1.5 text-[10.5px] text-ink-500">
          <BarChart3 size={11} />
          {frames.length
            ? `分布来自自己抓的 ${frames.length} 帧`
            : stats && stats.total > 0
              ? `上次分析的分布（${stats.frames} 帧 · ${stats.total} 个框${
                  stats.dropped ? ` · 已筛掉 ${stats.dropped}` : ''
                }）`
              : '暂无实测分布'}
        </span>
      </div>

      <div className="mt-2.5">
        <SizeHistogram sample={sample} params={params} relative={relative} />
      </div>

      {!frames.length && stats && stats.total > 0 && stats.active && (
        <div className="mt-1.5 text-[10.5px] text-ink-500">
          上次分析实际保留了 {stats.kept} / {stats.total} 个框
          {stats.dropped > 0 && `（筛掉 ${((stats.dropped / stats.total) * 100).toFixed(0)}%）`}
          {stats.ref > 0 && ` · 参考尺度 ${stats.ref.toFixed(3)}`}
        </div>
      )}

      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] text-ink-500 hover:text-ink-300">
          面积上下限（畸变把框拉宽时更准）
        </summary>
        <div className="mt-2 grid gap-x-5 gap-y-1 md:grid-cols-2">
          <Slider
            label="框面积下限"
            value={params.player_min_area}
            min={0}
            max={0.08}
            step={0.001}
            onChange={(v) => setParams({ player_min_area: v })}
            format={(v) => (v <= 0 ? '不限' : v.toFixed(3))}
            hint="归一化框面积（占画面面积的比例），0 = 不限"
            disabled={params.player_size_mode === 'off'}
          />
          <Slider
            label="框面积上限"
            value={params.player_max_area}
            min={0}
            max={0.3}
            step={0.005}
            onChange={(v) => setParams({ player_max_area: v })}
            format={(v) => (v <= 0 ? '不限' : v.toFixed(3))}
            hint="用来挡掉贴到镜头前的人；畸变会把框拉宽，这时比框高更灵"
            disabled={params.player_size_mode === 'off'}
          />
        </div>
      </details>
    </div>
  )
}
