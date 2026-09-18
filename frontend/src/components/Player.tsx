import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronFirst,
  ChevronLast,
  Volume2,
  VolumeX,
  Repeat,
  Gauge,
  Sparkles,
  Maximize2,
  ListFilter,
  Crosshair,
} from 'lucide-react'
import { api } from '../lib/api'
import { cn, scoreColor, timecode } from '../lib/format'
import { Badge, Button, Tooltip } from './ui'
import { filterRallies, useStore } from '../store/useStore'

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4]

const COURT_COLORS: Record<string, string> = {
  green: '#4ade80',
  blue: '#60a5fa',
  yellow: '#fbbf24',
  cyan: '#22d3ee',
  orange: '#fb923c',
  red: '#f87171',
  unknown: '#a3a3a3',
}

/** 在预览画面上画出 AI 识别到的球场范围。
 *
 * 视频用 `object-contain` 显示，所以画面实际占据的区域是**居中的等比矩形**，
 * 用一个同样按比例定位的绝对层叠上去，归一化坐标就能直接映射到像素。
 *
 * 画的是**多边形**（4~24 点）：全景 / 鱼眼素材的场地边界是弯的，
 * 用四个角画出来会让人以为「AI 圈错了」，其实只是形状描述不够。
 * 多边形之外还压了一层半透明遮罩 —— 「它到底在看哪一块」一眼就能看出来，
 * 而这正是标定最容易出错的地方（把看台当成场地）。
 */
function CourtOverlay({
  polygon,
  mediaAspect,
  color,
  label,
}: {
  polygon: number[][]
  mediaAspect: number
  color: string
  label: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = ref.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setBox({ w: r.width, h: r.height })
    })
    ro.observe(el)
    const r = el.getBoundingClientRect()
    setBox({ w: r.width, h: r.height })
    return () => ro.disconnect()
  }, [])

  if (!polygon || polygon.length < 3 || box.w <= 0 || box.h <= 0) return null

  // object-contain：画面按自身比例缩放后居中
  const containerAspect = box.w / Math.max(1, box.h)
  let vw = box.w
  let vh = box.h
  if (containerAspect > mediaAspect) {
    vh = box.h
    vw = vh * mediaAspect
  } else {
    vw = box.w
    vh = vw / mediaAspect
  }
  const ox = (box.w - vw) / 2
  const oy = (box.h - vh) / 2
  const px = (x: number) => ox + x * vw
  const py = (y: number) => oy + y * vh
  const pts = polygon.map(([x, y]) => `${px(x)},${py(y)}`).join(' ')
  const stroke = COURT_COLORS[color] ?? COURT_COLORS.unknown
  // 外圈矩形 + 多边形，用 evenodd 把多边形挖空 = 只压暗场地之外
  const maskPath =
    `M ${ox},${oy} H ${ox + vw} V ${oy + vh} H ${ox} Z ` +
    polygon.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${px(x)},${py(y)}`).join(' ') +
    ' Z'

  return (
    <div ref={ref} className="pointer-events-none absolute inset-0">
      <svg className="h-full w-full">
        <path d={maskPath} fill="#000" fillOpacity={0.42} fillRule="evenodd" />
        <polygon
          points={pts}
          fill={stroke}
          fillOpacity={0.06}
          stroke={stroke}
          strokeOpacity={0.9}
          strokeWidth={2}
          strokeDasharray="7 4"
        />
        {polygon.map(([x, y], i) => (
          <circle
            key={i}
            cx={px(x)}
            cy={py(y)}
            r={polygon.length > 8 ? 2 : 3}
            fill={stroke}
            fillOpacity={0.9}
          />
        ))}
        <text x={px(polygon[0][0]) + 6} y={py(polygon[0][1]) - 8} fill={stroke} fontSize={11} opacity={0.95}>
          AI 识别场地 · {label}
          {polygon.length > 4 ? ` · ${polygon.length} 点` : ''}
        </text>
      </svg>
    </div>
  )
}


export default function Player() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const playing = useStore((s) => s.playing)
  const currentTime = useStore((s) => s.currentTime)
  const setPlaying = useStore((s) => s.setPlaying)
  const seek = useStore((s) => s.seek)
  const syncTime = useStore((s) => s.syncTime)
  const previewMode = useStore((s) => s.previewMode)
  const setPreviewMode = useStore((s) => s.setPreviewMode)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const selectedClipId = useStore((s) => s.selectedClipId)
  const selectRally = useStore((s) => s.selectRally)
  const toast = useStore((s) => s.toast)
  const previewFiltered = useStore((s) => s.previewFiltered)
  const setPreviewFiltered = useStore((s) => s.setPreviewFiltered)
  const filter = useStore((s) => s.filter)
  // AI 识别到的球场范围：可以叠在画面上，用来核对「它到底在看哪块场地」
  const calibration = analysis?.calibration ?? null
  const showCourt = useStore((s) => s.showCourtOverlay)
  const setShowCourtOverlay = useStore((s) => s.setShowCourtOverlay)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const setCourtEditorOpen = useStore((s) => s.setCourtEditorOpen)

  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(1)
  const [speed, setSpeed] = useState(1)
  // 回合循环默认关：开着的时候一旦选中某个回合，播放就再也离不开它，
  // 用户会以为播放器坏了。需要反复看一个回合时再手动打开。
  const [loop, setLoop] = useState(false)
  const [showSpeed, setShowSpeed] = useState(false)
  const [ready, setReady] = useState(false)

  const timeline = project?.timeline
  const clips = useMemo(
    () => (timeline?.tracks?.[0]?.clips ?? []).slice().sort((a, b) => a.tl_start - b.tl_start),
    [timeline],
  )

  /** 成片模式：把时间线时间映射回素材内时间。 */
  const mapTimeline = useCallback(
    (t: number): { src: number; clipIndex: number } | null => {
      if (!clips.length) return null
      for (let i = 0; i < clips.length; i++) {
        const c = clips[i]
        const dur = (c.src_out - c.src_in) / c.speed
        if (t >= c.tl_start && t < c.tl_start + dur) {
          return { src: c.src_in + (t - c.tl_start) * c.speed, clipIndex: i }
        }
      }
      const last = clips[clips.length - 1]
      return { src: last.src_out, clipIndex: clips.length - 1 }
    },
    [clips],
  )

  const rallyAtTime = useMemo(() => {
    if (!analysis) return null
    return analysis.rallies.find((r) => currentTime >= r.start && currentTime <= r.end) ?? null
  }, [analysis, currentTime])

  const selectedRally = useMemo(
    () => analysis?.rallies.find((r) => r.id === selectedRallyId) ?? null,
    [analysis, selectedRallyId],
  )

  const activeRally = selectedRally ?? rallyAtTime

  /** 当前筛选出来的回合，按原片时间排出播放顺序 */
  const filteredRallies = useMemo(() => {
    const list = filterRallies(analysis?.rallies ?? [], filter)
    return list.slice().sort((a, b) => a.start - b.start)
  }, [analysis, filter])

  const filteredIndex = useMemo(
    () => filteredRallies.findIndex((r) => currentTime >= r.start && currentTime <= r.end),
    [filteredRallies, currentTime],
  )

  /**
   * 「只看筛选片段」的落点：不在任何筛选回合里（捡球、走动），
   * 或者已经播到当前回合的尾巴，就跳到下一个筛选回合开头。
   * 已经是最后一个了返回 'end'，让播放停下来。
   */
  const filteredTarget = useCallback(
    (t: number): number | 'end' | null => {
      if (!previewFiltered || !filteredRallies.length) return null
      const cur = filteredRallies.find((r) => t >= r.start && t <= r.end)
      const next = filteredRallies.find((r) => r.start > t + 0.02)
      if (!cur) return next ? next.start : 'end'
      if (t >= cur.end - 0.06) return next ? next.start : 'end'
      return null
    },
    [previewFiltered, filteredRallies],
  )

  // ------------------------------------------------ 同步 video 与状态
  useEffect(() => {
    setReady(false)
  }, [mediaId, previewMode])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (playing) v.play().catch(() => setPlaying(false))
    else v.pause()
  }, [playing, setPlaying, previewMode])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.playbackRate = speed
  }, [speed])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = muted
    v.volume = volume
  }, [muted, volume])

  // 播放头由 store 驱动 -> 若与 video 差异较大则校正
  useEffect(() => {
    const v = videoRef.current
    if (!v || !ready) return
    let target = currentTime
    if (previewMode === 'timeline') {
      const m = mapTimeline(currentTime)
      target = m ? m.src : 0
    }
    // 暂停时的容差要小得多：逐帧按钮一次只走 1/30 秒，
    // 阈值给 0.34 的话点十几下画面都不动（只有播放头在走）。
    const tol = playing ? 0.34 : 0.012
    if (Math.abs(v.currentTime - target) > tol) {
      try {
        v.currentTime = target
      } catch {
        /* ignore */
      }
    }
  }, [currentTime, ready, previewMode, mapTimeline, playing])

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (!v || !ready) return
    let t = v.currentTime
    if (previewMode === 'timeline') {
      // 素材时间 -> 时间线时间
      for (const c of clips) {
        const dur = (c.src_out - c.src_in) / c.speed
        if (t >= c.src_in && t < c.src_out) {
          t = c.tl_start + (t - c.src_in) / c.speed
          // 快进到片段末尾时跳到下一段
          if (t >= c.tl_start + dur - 0.06 && playing) {
            const idx = clips.indexOf(c)
            const next = clips[idx + 1]
            if (next) {
              v.currentTime = next.src_in
              seek(next.tl_start)
              return
            }
          }
          break
        }
      }
    } else if (previewFiltered && playing) {
      // 只看筛选片段：跳过空档。timeupdate 大约 4 次/秒，
      // 所以在回合末尾提前 60ms 就起跳，避免看到下一段捡球的画面。
      const jump = filteredTarget(v.currentTime)
      if (jump === 'end') {
        setPlaying(false)
        return
      }
      if (jump !== null) {
        try {
          v.currentTime = jump
        } catch {
          /* ignore */
        }
        seek(jump)
        return
      }
    }
    // 用 syncTime 而不是 seek：拖动播放头期间 video 会以旧时间继续触发
    // timeupdate，直接回写会把播放头拽回去，表现为「跳变」。
    syncTime(t)
  }

  const jumpRally = (dir: 1 | -1) => {
    if (!analysis?.rallies.length) return
    const list = analysis.rallies
    if (dir === 1) {
      const nxt = list.find((r) => r.start > currentTime + 0.05)
      if (nxt) {
        selectRally(nxt.id)
        seek(nxt.start)
      }
    } else {
      const prev = [...list].reverse().find((r) => r.end < currentTime - 0.35)
      if (prev) {
        selectRally(prev.id)
        seek(prev.start)
      }
    }
  }

  // 选中回合时自动定位
  const lastSelected = useRef<string | null>(null)
  useEffect(() => {
    if (!selectedRally || lastSelected.current === selectedRally.id) return
    lastSelected.current = selectedRally.id
    seek(selectedRally.start)
  }, [selectedRally, seek])

  // 回合循环（默认关闭）
  useEffect(() => {
    if (!loop || !activeRally || !playing) return
    if (currentTime > activeRally.end + 0.15) {
      seek(activeRally.clip_start ?? activeRally.start)
    }
  }, [currentTime, activeRally, loop, playing, seek])

  /** 当前选中的片段在成片时间轴上的范围（用于给分割提示兜错） */
  const selectedClipRange = useMemo(() => {
    if (previewMode !== 'timeline') return null
    const c = clips.find((x) => x.id === selectedClipId)
    if (!c) return null
    return { start: c.tl_start, end: c.tl_start + (c.src_out - c.src_in) / c.speed }
  }, [previewMode, selectedClipId, clips])

  const splitHint = useMemo(() => {
    if (!selectedClipId || !selectedClipRange) return null
    return currentTime >= selectedClipRange.start && currentTime <= selectedClipRange.end
  }, [selectedClipId, selectedClipRange, currentTime])

  const src = mediaId && project ? api.proxyUrl(project.id, mediaId) : undefined
  const duration = previewMode === 'timeline' ? timeline?.duration ?? 0 : media?.duration ?? 0

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* 工具条 */}
      <div className="flex items-center gap-2 px-3 py-2">
        <div className="flex items-center gap-1.5">
          <Button
            variant={previewMode === 'source' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => setPreviewMode('source')}
          >
            源片
          </Button>
          <Button
            variant={previewMode === 'timeline' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => {
              setPreviewMode('timeline')
              // 成片预览本身就是「按成片顺序播」，两个模式叠加只会互相打架
              setPreviewFiltered(false)
            }}
            disabled={!clips.length}
          >
            成片预览
            {clips.length > 0 && (
              <span className="mono ml-1 text-[10px] text-court-300">{clips.length}</span>
            )}
          </Button>
          <Tooltip
            side="top"
            width={300}
            content={
              previewFiltered
                ? `只看筛选片段：开\n跳过没被筛选出来的部分，连着播这 ${filteredRallies.length} 个回合`
                : `只看筛选片段\n播放时自动跳过筛选之外的捡球、走动，只连着放筛选出来的 ${filteredRallies.length} 个回合`
            }
          >
            <Button
              variant={previewFiltered ? 'outline' : 'ghost'}
              size="sm"
              disabled={!filteredRallies.length}
              onClick={() => {
                if (previewFiltered) setPreviewFiltered(false)
                else useStore.getState().previewFilteredRallies()
              }}
              className={previewFiltered ? 'text-court-300' : ''}
            >
              <ListFilter size={12} />
              只看筛选
            </Button>
          </Tooltip>
        </div>

        <div className="flex-1" />

        <AnimatePresence>
          {activeRally && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              className="flex items-center gap-2 rounded-lg border border-white/8 bg-white/4 px-2.5 py-1"
            >
              <Sparkles size={11} style={{ color: scoreColor(activeRally.scores.total) }} />
              <span className="text-[11.5px] text-ink-200">
                回合 #{activeRally.index}
              </span>
              <span
                className="mono text-[12px] font-semibold"
                style={{ color: scoreColor(activeRally.scores.total) }}
              >
                {activeRally.scores.total.toFixed(0)}
              </span>
              <span className="text-[10.5px] text-ink-500">
                {activeRally.features.shot_count} 拍 · {activeRally.duration.toFixed(1)}s
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 画面：让视频按自身比例铺满整个区域（object-contain），
          容器本身不再留多余的黑色边距 */}
      <div className="relative mx-3 flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-white/8 bg-black">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            className="block h-full w-full object-contain"
            onLoadedMetadata={() => setReady(true)}
            onTimeUpdate={onTimeUpdate}
            onClick={() => setPlaying(!playing)}
            playsInline
          />
        ) : (
          <div className="text-[12.5px] text-ink-500">请先导入素材</div>
        )}

        {calibration?.ok && showCourt && media && media.width > 0 && (
          <CourtOverlay
            polygon={calibration.polygon?.length ? calibration.polygon : calibration.quad}
            mediaAspect={media.width / Math.max(1, media.height)}
            color={calibration.court_color}
            label={calibration.viewpoint_label}
          />
        )}

        {!ready && src && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-ink-950/60">
            <div className="text-[12px] text-ink-300">正在准备预览用的小尺寸副本…</div>
          </div>
        )}

        {/* 角标 */}
        <div className="pointer-events-none absolute top-2.5 left-3 flex items-center gap-2">
          <Badge className="bg-black/50 backdrop-blur">{media?.width}×{media?.height}</Badge>
          {media && media.fps > 0 && <Badge className="bg-black/50 backdrop-blur">{media.fps.toFixed(2)} fps</Badge>}
          {media && (
            <button
              onClick={() => setCourtEditorOpen(true)}
              title={
                manualPoly
                  ? `已手动标定场地（${manualPoly.length} 个点），点击可修改`
                  : calibration?.ok
                    ? `AI 识别机位：${calibration.viewpoint_label}。识别不准的话点这里手动标一次`
                    : '手动标出球场范围（AI 找不准时用这个）'
              }
              className="pointer-events-auto"
            >
              <Badge
                color={manualPoly ? '#38e0a2' : calibration?.ok ? COURT_COLORS[calibration.court_color] ?? undefined : '#f59e0b'}
                className={cn('backdrop-blur', manualPoly || calibration?.ok ? '' : 'bg-black/50')}
              >
                <Crosshair size={9} />
                {manualPoly ? `手动标定 ${manualPoly.length} 点` : calibration?.ok ? `机位 ${calibration.viewpoint_label}` : '标定场地'}
              </Badge>
            </button>
          )}
          {calibration?.ok && (
            <button
              onClick={() => setShowCourtOverlay(!showCourt)}
              title={showCourt ? '隐藏识别到的球场范围' : '在画面上显示识别到的球场范围（场地外会压暗）'}
              className="pointer-events-auto"
            >
              <Badge className={cn('backdrop-blur', showCourt ? 'bg-court-500/25 text-court-200' : 'bg-black/50')}>
                {showCourt ? '隐藏范围' : '显示范围'}
              </Badge>
            </button>
          )}
        </div>
        <div className="pointer-events-none absolute right-3 bottom-2.5 flex items-center gap-2">
          {loop && activeRally && (
            <Badge color="#38e0a2" className="backdrop-blur">
              正在循环 回合 #{activeRally.index}
            </Badge>
          )}
          {previewMode === 'timeline' && selectedClipId && (
            <Badge color={splitHint ? '#5c9dff' : '#6b7787'} className="backdrop-blur">
              {splitHint ? '按 S 在此分割选中片段' : '播放头不在选中片段内'}
            </Badge>
          )}
          {rallyAtTime && previewMode === 'source' && (
            <Badge color={scoreColor(rallyAtTime.scores.total)} className="backdrop-blur">
              回合进行中
            </Badge>
          )}
          {previewMode === 'source' && previewFiltered && (
            <Badge color="#5c9dff" className="backdrop-blur">
              <ListFilter size={9} />
              只看筛选片段
              {filteredRallies.length > 0 && ` ${filteredIndex >= 0 ? filteredIndex + 1 : '-'}/${filteredRallies.length}`}
            </Badge>
          )}
        </div>

        {/* 全屏按钮。绝对定位放在外层 div 上：
            如果交给 Tooltip 的 span（它是 0×0 的 relative 盒子），
            absolute 的包含块就变成那个 span，按钮会跑到画面正中间。 */}
        <div className="absolute top-2.5 right-3">
          <Tooltip content="全屏" side="left">
            <button
              onClick={() => videoRef.current?.parentElement?.requestFullscreen?.()}
              className="grid h-7 w-7 place-items-center rounded-md bg-black/45 text-ink-300 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
            >
              <Maximize2 size={13} />
            </button>
          </Tooltip>
        </div>
      </div>

      {/* 传输控制 */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <Tooltip content="上一回合" side="top">
          <Button variant="ghost" size="icon" onClick={() => jumpRally(-1)}>
            <ChevronFirst size={15} />
          </Button>
        </Tooltip>
        <Tooltip content="后退一帧 (←)" side="top">
          <Button variant="ghost" size="icon" onClick={() => seek(currentTime - 1 / (media?.fps || 30))}>
            <SkipBack size={15} />
          </Button>
        </Tooltip>
        <button
          onClick={() => setPlaying(!playing)}
          className={cn(
            'grid h-9 w-9 place-items-center rounded-full transition-all active:scale-95',
            playing ? 'bg-white/12 text-white' : 'bg-gradient-to-b from-court-400 to-court-600 text-ink-950',
          )}
        >
          {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
        </button>
        <Tooltip content="前进一帧 (→)" side="top">
          <Button variant="ghost" size="icon" onClick={() => seek(currentTime + 1 / (media?.fps || 30))}>
            <SkipForward size={15} />
          </Button>
        </Tooltip>
        <Tooltip content="下一回合" side="top">
          <Button variant="ghost" size="icon" onClick={() => jumpRally(1)}>
            <ChevronLast size={15} />
          </Button>
        </Tooltip>

        <div className="mono ml-2 text-[12px] text-ink-200 tabular">
          <span className="text-white">{timecode(currentTime, false)}</span>
          <span className="text-ink-500"> / {timecode(duration, false)}</span>
          <span className="ml-2 text-[10.5px] text-ink-500">
            f{Math.floor((currentTime % 1) * (media?.fps || 30))}
          </span>
        </div>

        <div className="flex-1" />

        <Tooltip content={loop ? '正在循环当前回合 · 点击关闭' : '循环播放当前回合'} side="top">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => {
              const next = !loop
              setLoop(next)
              if (next) toast({ kind: 'info', title: '已开启回合循环', detail: '播放到回合结束会自动跳回开头' })
            }}
            className={loop ? 'text-court-300' : ''}
          >
            <Repeat size={14} />
          </Button>
        </Tooltip>

        <div className="relative">
          <Tooltip content="播放速度" side="top">
            <Button variant="ghost" size="sm" onClick={() => setShowSpeed((v) => !v)}>
              <Gauge size={13} />
              <span className="mono">{speed}×</span>
            </Button>
          </Tooltip>
          <AnimatePresence>
            {showSpeed && (
              <>
                {/* 点击空白处关闭。用 portal 挂到 body：
                    页面外壳 .anim-in 带 transform 动画，留在里面的话 fixed 会以它为参照，
                    遮罩只盖住 main，点顶栏/导航栏关不掉这个弹层。 */}
                {createPortal(
                  <div className="fixed inset-0 z-40" onClick={() => setShowSpeed(false)} />,
                  document.body,
                )}
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  className="panel absolute right-0 bottom-full z-50 mb-1.5 flex flex-col p-1"
                >
                  {SPEEDS.map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setSpeed(s)
                        setShowSpeed(false)
                      }}
                      className={cn(
                        'mono rounded-md px-3 py-1 text-[11.5px] transition-colors',
                        speed === s ? 'bg-court-500/20 text-court-300' : 'text-ink-300 hover:bg-white/8',
                      )}
                    >
                      {s}×
                    </button>
                  ))}
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>

        <Tooltip content={muted ? '取消静音' : '静音'} side="top">
          <Button variant="ghost" size="icon" onClick={() => setMuted((v) => !v)}>
            {muted || volume === 0 ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </Button>
        </Tooltip>
        <input
          type="range"
          min={0}
          max={1}
          step={0.02}
          value={muted ? 0 : volume}
          onChange={(e) => {
            setVolume(parseFloat(e.target.value))
            setMuted(false)
          }}
          className="w-[78px]"
        />
      </div>
    </div>
  )
}
